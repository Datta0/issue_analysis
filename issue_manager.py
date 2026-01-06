import os
import datetime
import json
import yaml
import concurrent.futures
from typing import List, Dict, Any, Optional
from github import Github, Issue, GithubException
from tqdm import tqdm
from similarity import HybridEngine

CACHE_FILE = 'issues_cache.json'
EMBEDDINGS_CACHE = 'embeddings_cache.pkl'

class IssueManager:
    def __init__(self, config_path: str = 'config.yaml', token: Optional[str] = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.mode = self.config.get('mode', 'analysis')
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found in environment or arguments.")

        # Disable retries to fail fast on Rate Limit (enable Offline Mode)
        self.g = Github(self.token, retry=0)
        self.repo_name = self.config.get('repo_name')
        self.repo = None

        # Local Cache
        self.issues_map: Dict[int, Dict] = {} # Map number -> issue dict (not Object to save serialization)
        self.last_fetch: Optional[datetime.datetime] = None

        # Initialize Similarity Engine
        sim_config = self.config.get('similarity', {})
        if sim_config.get('enabled', True):
            self.similarity_engine = HybridEngine(
                model_name=sim_config.get('model_name', 'BAAI/bge-base-en-v1.5'),
                alpha=sim_config.get('alpha', 0.5),
                device=sim_config.get('device'),
                batch_size=sim_config.get('batch_size', 32)
            )
        else:
            self.similarity_engine = None

    def connect_repo(self, repo_name: str):
        self.repo_name = repo_name
        if self.repo is None: # Only connect if not already connected (e.g. reused)
             print(f"Connecting to repository: {repo_name}...")
             try:
                self.repo = self.g.get_repo(repo_name)
             except Exception as e:
                print(f"Warning: Could not connect to repo (Offline/Rate Limit). Operations will use cache only. Error: {e}")
                self.repo = None

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    self.issues_map = {int(k): v for k, v in data.get('issues', {}).items()}
                    last_fetch_iso = data.get('last_fetch')
                    if last_fetch_iso:
                        self.last_fetch = datetime.datetime.fromisoformat(last_fetch_iso)
                print(f"Loaded {len(self.issues_map)} issues from cache.")
            except Exception as e:
                print(f"Error loading cache: {e}")

    def _save_cache(self):
        data = {
            'issues': self.issues_map,
            'last_fetch': self.last_fetch.isoformat() if self.last_fetch else None
        }
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(data, f)
            print("Saved issues cache.")
        except Exception as e:
             print(f"Error saving cache: {e}")

    def _serialize_issue(self, issue: Issue.Issue) -> Dict:
        """Convert GitHub Issue object to minimal dict for caching and logic."""
        return {
            'number': issue.number,
            'title': issue.title,
            'body': issue.body,
            'state': issue.state,
            'created_at': issue.created_at.isoformat(),
            'updated_at': issue.updated_at.isoformat(),
            'html_url': issue.html_url,
            'user_login': issue.user.login,
            # Placeholder for analysis fields
            'last_commenter': None,
            'last_comment_date': None
        }

    def fetch_issues(self):
        self._load_cache()

        fetched = []
        if self.repo:
            # Decide fetch strategy
            fetch_kwargs = {'state': 'open'}
            if self.last_fetch:
                print(f"Fetching issues updated since {self.last_fetch.isoformat()}...")
                fetch_kwargs['since'] = self.last_fetch
            else:
                print("Fetching ALL open issues (First Run)...")

            try:
                fetched = list(self.repo.get_issues(**fetch_kwargs))
                print(f"Fetched {len(fetched)} updated/new issues.")
            except Exception as e:
                print(f"Warning: Failed to fetch updates (Rate Limit/Offline?). Using cache. Error: {e}")

            # Update Cache (Merge)
            for issue in fetched:
                # If we simply overwrite, we lose 'last_commenter'.
                # Only overwrite if updated_at changed OR it's new.
                # actually 'fetched' only contains updated ones, so we MUST overwrite content,
                # BUT we should be careful.
                # If it's a new update, the comments might have changed, so we reset last_commenter to None
                # to force re-fetch in analysis.
                # So simple overwrite is correct for 'fetched' items.
                self.issues_map[int(issue.number)] = self._serialize_issue(issue)

            if fetched:
                self.last_fetch = datetime.datetime.now(datetime.timezone.utc)
                self._save_cache()
        else:
            print("Skipping fetch (Offline Mode). Using cache.")

        return fetched

    def _is_core_member(self, username: str) -> bool:
        return username in self.config.get('core_members', [])

    def fetch_last_comment_info(self, issue_number: int) -> Dict:
        """
        Network call to get last commenter. Dedicated for threading.
        Retries could be added here.
        """
        if not self.repo:
            return {'last_commenter': None, 'error': 'Offline'}

        try:
            gh_issue = self.repo.get_issue(issue_number)
            comments = list(gh_issue.get_comments())
            if comments:
                last_comment = comments[-1]
                return {
                    'last_commenter': last_comment.user.login,
                    'last_comment_date': last_comment.created_at.isoformat()
                }
            else:
                 return {
                    'last_commenter': gh_issue.user.login, # Creator
                    'last_comment_date': gh_issue.created_at.isoformat()
                 }
        except Exception as e:
            # Fallback or Log
            # print(f"Error fetching comments for #{issue_number}: {e}")
            return {'last_commenter': None, 'error': str(e)}

    def analyze_issue_logic(self, issue_data: Dict) -> Dict[str, Any]:
        """Pure logic analysis using cached/fetched data."""
        now = datetime.datetime.now(datetime.timezone.utc)

        # Parse dates
        updated_at = datetime.datetime.fromisoformat(issue_data['updated_at']).replace(tzinfo=datetime.timezone.utc)
        created_at = datetime.datetime.fromisoformat(issue_data['created_at']).replace(tzinfo=datetime.timezone.utc)

        inactivity_days = (now - updated_at).days
        last_commenter = issue_data.get('last_commenter') or 'N/A'
        is_last_core = self._is_core_member(last_commenter)

        result = {
            'number': issue_data['number'],
            'title': issue_data['title'],
            'created_at': created_at,
            'days': inactivity_days,
            'last_commenter': last_commenter,
            'action': 'NONE',
            'reason': 'active'
        }

        rules = self.config.get('rules', {})

        # Rule 1: Core Member Follow-up
        core_rule = rules.get('core_followup', {})
        if core_rule.get('enabled') and is_last_core:
            if inactivity_days >= core_rule.get('close_days', 21):
                result.update({'action': 'CLOSE', 'reason': 'core_no_response'})
                return result
            elif inactivity_days >= core_rule.get('warn_days', 14):
                result.update({'action': 'WARN', 'reason': 'core_no_response_warn'})
                return result

        # Rule 2: General Stale
        stale_rule = rules.get('general_stale', {})
        if stale_rule.get('enabled') and not is_last_core:
            if inactivity_days >= stale_rule.get('close_days', 84):
                 result.update({'action': 'CLOSE', 'reason': 'stale_max'})
                 return result

            if inactivity_days >= stale_rule.get('warn_days', 28):
                created_delta = (now - created_at).days
                new_window = stale_rule.get('new_issue_window_days', 90)

                if created_delta < new_window:
                     result.update({'action': 'NOTIFY_TEAM', 'reason': 'stale_but_new'})
                     return result
                else:
                     result.update({'action': 'WARN', 'reason': 'stale_warn'})
                     return result

        return result

    def process_bi_analysis(self, issue_data: Dict) -> Dict:
        """Worker function for threading."""
        updated = False

        # 1. Fetch live comment info IF MISSING
        # If we already have last_commenter in cache (cached from previous run), skip API call
        if not issue_data.get('last_commenter') and self.repo:
            comment_info = self.fetch_last_comment_info(issue_data['number'])
            if comment_info.get('last_commenter'): # Only update if successful
                issue_data.update(comment_info)
                updated = True

        # 2. Logic (CPU Bound)
        analysis = self.analyze_issue_logic(issue_data)

        return {'analysis': analysis, 'issue_data': issue_data, 'updated': updated}

    def execute_action(self, issue_number: int, analysis: Dict[str, Any], verbose: bool = False):
        action = analysis['action']
        if action == 'NONE':
            return

        # Re-fetch object for action - ONLY IF ONLINE
        if not self.repo:
            # print(f"Skipping action (Offline): {action} #{issue_number}")
            return

        repo_issue = self.repo.get_issue(issue_number) # Re-fetch object for action

        # Detailed Log
        if verbose or self.mode == 'analysis':
             reason = analysis['reason']
             days_inactive = analysis['days']
             last_commenter = analysis.get('last_commenter', 'N/A')
             created_str = analysis['created_at'].strftime("%Y-%m-%d")

             print("-" * 60)
             print(f"Issue #{issue_number}: {analysis['title']}")
             print(f"Created: {created_str} | Inactive: {days_inactive} days")
             print(f"Last Responder: {last_commenter}")
             print(f"Action: {action} (Reason: {reason})")
             print("-" * 60)

        if self.mode == 'analysis':
            return

        messages = self.config.get('messages', {})
        reason = analysis['reason']

        try:
            if action == 'CLOSE':
                msg_key = 'core_followup_close' if 'core' in reason else 'general_stale_close'
                body = messages.get(msg_key, "Closing due to inactivity.")
                repo_issue.create_comment(body)
                repo_issue.edit(state='closed')
                if not verbose: print(f"Closed #{issue_number}")

            elif action == 'WARN':
                msg_key = 'core_followup_warn' if 'core' in reason else 'general_stale_warn'
                body = messages.get(msg_key, "Warning: Inactive.")
                repo_issue.create_comment(body)
                if not verbose: print(f"Warned #{issue_number}")

            elif action == 'NOTIFY_TEAM':
                template = messages.get('general_stale_notify_team', "Team attention needed.")
                repo_issue.create_comment(template)
                if not verbose: print(f"Notified Team #{issue_number}")

        except GithubException as e:
            print(f"  Error executing action: {e}")

    def run_process(self, repo_name: str, verbose: bool = False):
        self.connect_repo(repo_name)

        # 1. Fetch & Cache
        self.fetch_issues()

        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff_date = now - datetime.timedelta(days=365)

        # Filter working set
        working_issues = [
            i for i in self.issues_map.values()
            if i['state'] == 'open' and datetime.datetime.fromisoformat(i['created_at']).replace(tzinfo=datetime.timezone.utc) >= cutoff_date
        ]

        ignored_count = len([i for i in self.issues_map.values() if i['state'] == 'open']) - len(working_issues)

        # 2. Update Similarity Index
        if self.similarity_engine:
            cache_loaded = self.similarity_engine.load_cache(EMBEDDINGS_CACHE)
            # Naive re-index for working set to ensure sync
            docs = [{'number': i['number'], 'title': i['title'], 'body': i['body'], 'html_url': i['html_url']} for i in working_issues]
            self.similarity_engine.index(docs)
            self.similarity_engine.save_cache(EMBEDDINGS_CACHE)

        # 3. Parallel Analysis
        analyses = []
        cache_needs_save = False

        print(f"Analyzing {len(working_issues)} issues (Parallel)...")

        # REDUCED WORKERS to avoid Rate Limit
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_issue = {executor.submit(self.process_bi_analysis, issue): issue for issue in working_issues}

            for future in tqdm(concurrent.futures.as_completed(future_to_issue), total=len(working_issues), desc="Processing"):
                res = future.result()
                analysis = res['analysis']
                issue_data = res['issue_data']
                analyses.append(analysis)

                # Update Cache if data caused an update (fetched comments)
                if res['updated']:
                    self.issues_map[issue_data['number']] = issue_data
                    cache_needs_save = True

                # Check for Duplicates / Similarity if NEW issue
                issue_date = analysis['created_at']
                if (now - issue_date).days < 7 and self.similarity_engine:
                     sims = self.similarity_engine.find_similar(f"{analysis['title']}", top_k=2)
                     for s in sims:
                         if s['number'] != analysis['number'] and s['score'] > 0.85:
                             if verbose: print(f"  [POTENTIAL DUPLICATE] #{analysis['number']} similar to #{s['number']} ({s['score']:.2f})")

        if cache_needs_save:
            self._save_cache()

        # 4. Print Stats & Execute
        stats = {'IGNORED_OLD': ignored_count}
        for analysis in analyses:
            act = analysis['action']
            stats[act] = stats.get(act, 0) + 1

        print("\n" + "="*40)
        print("ANALYSIS SUMMARY")
        print("="*40)
        for act, count in stats.items():
            print(f"{act}: {count}")
        print("="*40 + "\n")

        if not verbose and self.mode == 'analysis':
             print("Run with --verbose to see detailed logs.")
             # No return here, allow similarity report to print

        for analysis in analyses:
             if analysis['action'] != 'NONE':
                  self.execute_action(analysis['number'], analysis, verbose=verbose)

        # 5. Similarity Report (Clusters)
        if self.similarity_engine and self.similarity_engine.corpus_embeddings is not None:
             self.generate_similarity_report()

    def generate_similarity_report(self):
        """
        Cluster issues based on embedding similarity to find common topics.
        """
        print("\n" + "="*40)
        print("SIMILARITY / CLUSTER ANALYSIS")
        print("="*40)

        # Access embeddings (Tensor on Device)
        try:
             import torch
             from sentence_transformers import util
        except ImportError:
             print("Torch/SentenceTransformers not available for report.")
             return

        embeddings = self.similarity_engine.corpus_embeddings
        documents = self.similarity_engine.documents

        if embeddings is None or len(documents) == 0:
             print("No embeddings to analyze.")
             return

        # Compute NxN Cosine Similarity
        # This can be large, but for <5k issues it's fine on GPU
        # Check size
        n = len(documents)
        print(f"Clustering {n} issues...")

        sim_matrix = util.cos_sim(embeddings, embeddings)

        # Threshold for "Same Topic"
        threshold = 0.75

        # Greedy Clustering
        # List of clusters. Each cluster is list of indices.
        clusters = []
        visited = set()

        # Move to CPU for loop logic
        sim_matrix = sim_matrix.cpu()

        for i in range(n):
            if i in visited:
                continue

            # Start new cluster
            cluster = [i]
            visited.add(i)

            # Find all neighbors > threshold
            for j in range(i + 1, n):
                if j in visited:
                    continue

                if sim_matrix[i][j] >= threshold:
                    cluster.append(j)
                    visited.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        # Sort clusters by size
        clusters.sort(key=len, reverse=True)

        print(f"Found {len(clusters)} clusters/topics involving {sum(len(c) for c in clusters)} issues.\n")

        # Print Top 10 Clusters
        print("Top Recurring Issues:")
        for idx, cluster in enumerate(clusters[:10]):
            primary_doc_idx = cluster[0] # Using first as representative
            title = documents[primary_doc_idx]['title']
            print(f"{idx+1}. [{len(cluster)} issues] Topic: '{title}'")
            # Print a few examples
            # for c_idx in cluster[:3]:
            #     print(f"   - #{documents[c_idx]['number']} {documents[c_idx]['title']}")
            print("")

    def find_similar_for_new(self, title: str, body: str):
        if not self.similarity_engine:
            print("Similarity engine not enabled.")
            return

        # Try loading cache if not in memory
        if not self.similarity_engine.documents:
            self.similarity_engine.load_cache(EMBEDDINGS_CACHE)

        query = f"{title} {body}"
        top_k = self.config.get('similarity', {}).get('top_k', 3)
        results = self.similarity_engine.find_similar(query, top_k=top_k)

        print(f"Found {len(results)} similar issues:")
        for res in results:
            print(f" - #{res['number']} {res['title']} (Score: {res['score']:.2f})")
            print(f"   {res['url']}")
