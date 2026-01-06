import os
import datetime
import json
import yaml
import concurrent.futures
from typing import List, Dict, Any, Optional
from github import Github, Issue, GithubException
from tqdm import tqdm
from similarity import HybridEngine

import shutil

CACHE_FILENAME = 'issues_cache.json'
EMBEDDINGS_FILENAME = 'embeddings_cache.pkl'

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

        # Paths (initialized later in setup_paths)
        self.output_dir = None
        self.cache_file = None
        self.embeddings_cache = None

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

    def setup_paths(self, repo_name: str):
        """Initialize output directory and cache paths based on repo name."""
        safe_repo = repo_name.replace('/', '_')
        self.output_dir = os.path.join('outputs', safe_repo)
        os.makedirs(self.output_dir, exist_ok=True)

        self.cache_file = os.path.join(self.output_dir, CACHE_FILENAME)
        self.embeddings_cache = os.path.join(self.output_dir, EMBEDDINGS_FILENAME)

        self._migrate_legacy_cache()

    def _migrate_legacy_cache(self):
        """Move root cache files to the new output directory if they exist."""
        # Check issues cache
        if os.path.exists(CACHE_FILENAME) and not os.path.exists(self.cache_file):
            print(f"Migrating legacy {CACHE_FILENAME} to {self.cache_file}...")
            shutil.move(CACHE_FILENAME, self.cache_file)

        # Check embeddings cache
        if os.path.exists(EMBEDDINGS_FILENAME) and not os.path.exists(self.embeddings_cache):
            print(f"Migrating legacy {EMBEDDINGS_FILENAME} to {self.embeddings_cache}...")
            shutil.move(EMBEDDINGS_FILENAME, self.embeddings_cache)

    def connect_repo(self, repo_name: str):
        self.repo_name = repo_name
        self.setup_paths(repo_name) # Ensure paths are set up

        if self.repo is None: # Only connect if not already connected (e.g. reused)
             print(f"Connecting to repository: {repo_name}...")
             try:
                self.repo = self.g.get_repo(repo_name)
             except Exception as e:
                print(f"Warning: Could not connect to repo (Offline/Rate Limit). Operations will use cache only. Error: {e}")
                self.repo = None

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
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
            with open(self.cache_file, 'w') as f:
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
            'comments_count': issue.comments,
            'unique_commenters': 1, # Default to 1 (creator) until we fetch comments
            # Placeholder for analysis fields
            'last_commenter': None,
            'last_comment_date': None
        }

    # ... (fetch_issues remains mostly same, just ensuring calls use _serialize_issue)

    # ...

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
        n = len(documents)
        print(f"Clustering {n} issues...")

        sim_matrix = util.cos_sim(embeddings, embeddings)

        # Threshold for "Same Topic"
        threshold = 0.85

        # Greedy Clustering
        clusters = []
        visited = set()

        # Move to CPU
        sim_matrix = sim_matrix.cpu()

        for i in range(n):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            for j in range(i + 1, n):
                if j in visited:
                    continue
                if sim_matrix[i][j] >= threshold:
                    cluster.append(j)
                    visited.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        # Smart Sorting: Impact Score = (Num Issues) + (Total Comments * 0.2)
        # We weigh issues higher, but comments add "heat"
        def get_cluster_stats(cluster_indices):
            num_issues = len(cluster_indices)
            # Note: documents might not have 'comments_count' if loaded from OLD cache
            # We assume cache is refreshed or we default to 0
            total_comments = sum(documents[idx].get('comments_count', 0) for idx in cluster_indices)
            unique_authors = len(set(documents[idx]['user_login'] for idx in cluster_indices)) # Approx users
            return num_issues, total_comments, unique_authors

        clusters.sort(key=lambda c: len(c) + (sum(documents[idx].get('comments_count', 0) for idx in c) * 0.1), reverse=True)

        print(f"Found {len(clusters)} clusters/topics involving {sum(len(c) for c in clusters)} issues.\n")

        # Print Top 10 Clusters
        print("Top Recurring Issues (Ranked by Impact):")
        for idx, cluster in enumerate(clusters[:10]):
            primary_doc_idx = cluster[0]
            title = documents[primary_doc_idx]['title']
            count, comments, authors = get_cluster_stats(cluster)

            print(f"{idx+1}. [{count} issues | {comments} comments | ~{authors} users] Topic: '{title}'")

            for c_idx in cluster[:5]:
                 doc = documents[c_idx]
                 c_count = doc.get('comments_count', 0)
                 print(f"   - #{doc['number']} ({c_count} comments) {doc['title']}")
            print("")

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
                # Calculate unique commenters
                unique_users = set(c.user.login for c in comments)
                unique_users.add(gh_issue.user.login) # Add creator

                return {
                    'last_commenter': last_comment.user.login,
                    'last_comment_date': last_comment.created_at.isoformat(),
                    'unique_commenters': len(unique_users)
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
        if verbose:
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
            cache_loaded = self.similarity_engine.load_cache(self.embeddings_cache)
            # Naive re-index for working set to ensure sync
            docs = [{
                'number': i['number'],
                'title': i['title'],
                'body': i['body'],
                'html_url': i['html_url'],
                'user_login': i['user_login'],
                'comments_count': i.get('comments_count', 0),
                'unique_commenters': i.get('unique_commenters', 1)
            } for i in working_issues]
            self.similarity_engine.index(docs)
            self.similarity_engine.save_cache(self.embeddings_cache)

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

        if not verbose and self.mode == 'analysis':
             print("Run with --verbose to see ALL proposed actions.\n")

        # ---------------------------------------------------------
        # NEW SECTION: Top Important Recent Issues (High Impact)
        # ---------------------------------------------------------
        reporting_config = self.config.get('reporting', {})
        if reporting_config.get('enabled', True):
            recent_days = reporting_config.get('recent_days', 60)
            top_n = reporting_config.get('top_n', 10)

            # Filter: Created recently OR (Updated recently AND High Impact)
            cutoff_recent = now - datetime.timedelta(days=recent_days)

            # Helper to calculate impact
            def calculate_impact(issue):
                users = issue.get('unique_commenters', 1)
                comments = issue.get('comments_count', 0)
                if comments is None: comments = 0 # Handle legacy cache
                return (users * 2) + comments

            filtered_issues = []
            for i in working_issues:
                created_dt = datetime.datetime.fromisoformat(i['created_at']).replace(tzinfo=datetime.timezone.utc)
                updated_dt = datetime.datetime.fromisoformat(i['updated_at']).replace(tzinfo=datetime.timezone.utc)

                # Criteria 1: Recently Created
                if created_dt >= cutoff_recent:
                    filtered_issues.append(i)
                    continue

                # Criteria 2: Old but Active & High Impact
                # "Recent past" can mean recently active.
                if updated_dt >= cutoff_recent:
                    score = calculate_impact(i)
                    # Use a threshold for "High Impact" on old issues to avoid noise
                    if score >= reporting_config.get('min_impact_score', 5):
                        filtered_issues.append(i)

            filtered_issues.sort(key=calculate_impact, reverse=True)

            print(f"\n" + "="*40)
            print(f"TOP {top_n} IMPORTANT RECENT ISSUES (High Impact)")
            print(f"Criteria: Created or Active in last {recent_days} days, ranked by Users & Comments")
            print("="*40)

            # Check for data quality warnings
            missing_stats = sum(1 for i in filtered_issues if i.get('comments_count') is None)
            if missing_stats > 0:
                 print(f"WARNING: {missing_stats} issues have missing stats (legacy cache). Run with GITHUB_TOKEN to refresh.")

            for rank, issue in enumerate(filtered_issues[:top_n]):
                users = issue.get('unique_commenters', 1)
                comments = issue.get('comments_count', 0)
                if comments is None: comments = "?"

                score = calculate_impact(issue)
                created_dt = datetime.datetime.fromisoformat(issue['created_at']).strftime("%Y-%m-%d")

                print(f"{rank+1}. [Score: {score}] #{issue['number']} {issue['title']}")
                print(f"    - {users} Users | {comments} Comments | Created: {created_dt}")
                print(f"    - {issue['html_url']}")
            print("")


        # ---------------------------------------------------------
        # Show Top 10 High-Impact Issues Requiring Action
        #  - Prioritize by Engagement (Unique Users / Comments)
        #  - Then by Age (Days Inactive)
        # ---------------------------------------------------------
        actionable = [a for a in analyses if a['action'] != 'NONE']

        # Sort by Impact: (UniqueUsers, TotalComments, DaysInactive) descending
        actionable.sort(key=lambda x: (
            x.get('unique_commenters', 0),
            x.get('comments_count', 0),
            x['days']
        ), reverse=True)

        if actionable:
            print(f"Top 10 Actionable Stale Issues (requiring bot action):")
            for analysis in actionable[:10]:
                 reason = analysis['reason']
                 days = analysis['days']
                 users = analysis.get('unique_commenters', 1)
                 comments = analysis.get('comments_count', 0)

                 print(f" - #{analysis['number']} [{analysis['action']}] {analysis['title']}")
                 print(f"   (Impact: ~{users} users, {comments} comments | Inactive: {days}d | Reason: {reason})")
            print("")

        # Execute Actions (Silent unless verbose)
        for analysis in actionable:
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

        # ---------------------------------------------------------
        # 1. Agglomerative Clustering (Hierarchical)
        # ---------------------------------------------------------
        n = len(documents)
        print(f"Clustering {n} issues (Agglomerative)...")

        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.manifold import TSNE
            import plotly.express as px
            import pandas as pd
            import numpy as np
        except ImportError:
             print("Missing dependencies (sklearn, plotly, pandas). Skipping advanced clustering.")
             return

        # Convert to numpy
        X = embeddings.cpu().numpy()

        # Distance threshold (High = larger clusters).
        # Using cosine distance, so dist = 1 - sim.
        # Our sim threshold was 0.85, so dist threshold is 0.15
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.25, # Slightly looser than 0.15 because Ward works differently
            metric='cosine',
            linkage='average' # Average linkage good for tight semantic groups
        )
        labels = model.fit_predict(X)

        # Group by Cluster ID
        clusters = {} # id -> list of indices
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        cluster_list = list(clusters.values())

        # Filter out singletons (noise) or keep them? Keep for map, filter for report.
        report_clusters = [c for c in cluster_list if len(c) > 1] # Only recurring topics

        # Smart Sorting: Impact Score
        def get_cluster_stats(cluster_indices):
            num_issues = len(cluster_indices)
            total_comments = sum(documents[idx].get('comments_count', 0) for idx in cluster_indices)
            total_unique_users = sum(documents[idx].get('unique_commenters', 1) for idx in cluster_indices)
            return num_issues, total_comments, total_unique_users

        report_clusters.sort(key=lambda c: len(c) + (sum(documents[idx].get('unique_commenters', 1) for idx in c) * 0.25), reverse=True)

        print(f"Found {len(report_clusters)} significant clusters/topics.\n")

        # Print Top 10 Clusters
        print("Top Recurring Issues (Ranked by Impact):")
        for idx, cluster in enumerate(report_clusters[:10]):
            # Find closest to centroid for title? Or just first?
            # Let's pick the one with most comments as 'representative'
            primary_doc_idx = max(cluster, key=lambda i: documents[i].get('comments_count', 0))
            title = documents[primary_doc_idx]['title']
            count, comments, users = get_cluster_stats(cluster)

            print(f"{idx+1}. [{count} issues | ~{users} unique users | {comments} comments] Topic: '{title}'")

            for c_idx in cluster[:3]:
                 doc = documents[c_idx]
                 c_count = doc.get('comments_count', 0)
                 print(f"   - #{doc['number']} ({c_count} comments) {doc['title']}")
            print("")

        # ---------------------------------------------------------
        # 2. Interactive t-SNE Visualization
        # ---------------------------------------------------------
        print("Generating t-SNE Map...")

        # Perplexity must be < n_samples
        perp = min(30, n - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, metric='cosine', init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(X)

        # Prepare Dataframe
        data = []
        for i in range(n):
            doc = documents[i]
            lbl = labels[i]
            is_noise = (lbl not in [k for k,v in clusters.items() if len(v) > 1])

            # Label Name (Representative)
            cluster_rep = "Misc / Single"
            if not is_noise:
                 # Find rep title for this cluster
                 rep_idx = max(clusters[lbl], key=lambda x: documents[x].get('comments_count', 0))
                 cluster_rep = documents[rep_idx]['title'][:50] + "..."

            data.append({
                'x': X_embedded[i, 0],
                'y': X_embedded[i, 1],
                'Title': doc['title'],
                'Issue': f"#{doc['number']}",
                'Comments': doc.get('comments_count', 0),
                'ClusterID': str(lbl) if not is_noise else "Noise",
                'Topic': cluster_rep,
                'User': doc['user_login']
            })

        df = pd.DataFrame(data)

        fig = px.scatter(
            df, x='x', y='y',
            color='Topic',
            hover_data=['Issue', 'Title', 'Comments', 'User'],
            title=f"GitHub Issue Landscape ({self.repo_name})",
            template='plotly_dark'
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))

        map_path = os.path.join(self.output_dir, "map.html")
        fig.write_html(map_path)
        print(f"Visualization saved to: {map_path}")

    def find_similar_for_new(self, title: str, body: str):
        if not self.similarity_engine:
            print("Similarity engine not enabled.")
            return
        # Try loading cache if not in memory
        if not self.similarity_engine.documents:
            self.similarity_engine.load_cache(self.embeddings_cache)

        query = f"{title} {body}"
        top_k = self.config.get('similarity', {}).get('top_k', 3)
        results = self.similarity_engine.find_similar(query, top_k=top_k)

        print(f"Found {len(results)} similar issues:")
        for res in results:
            print(f" - #{res['number']} {res['title']} (Score: {res['score']:.2f})")
            print(f"   {res['url']}")
