import argparse
import os
import sys
from issue_manager import IssueManager

def main():
    # Parent parser to share arguments between main and subcommands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parent_parser.add_argument('--token', help='GitHub Token (overrides env GITHUB_TOKEN)')
    parent_parser.add_argument('--repo', help='Target repository (owner/name)')
    parent_parser.add_argument('--verbose', action='store_true', help='Show detailed logs')

    parser = argparse.ArgumentParser(description="GitHub Issue Manager CLI")

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Base command to run the main logic (inactivity, etc.)
    # Inherit parents so arguments can be passed after 'run'
    run_parser = subparsers.add_parser('run', parents=[parent_parser], help='Run the issue management process')

    # Command to check similarity
    sim_parser = subparsers.add_parser('similar', parents=[parent_parser], help='Find similar issues')
    sim_parser.add_argument('--title', required=True, help='Issue title')
    sim_parser.add_argument('--body', default='', help='Issue body')

    args = parser.parse_args()

    # Resolve Token
    token = args.token or os.environ.get('GITHUB_TOKEN')
    if not token:
        print("Error: No GitHub Token provided. Set GITHUB_TOKEN env var or use --token.")
        sys.exit(1)

    # Resolve Repo
    try:
        manager = IssueManager(config_path=args.config, token=token)
    except Exception as e:
        print(f"Error initializing IssueManager: {e}")
        sys.exit(1)

    repo_name = args.repo or manager.config.get('repo_name')
    if not repo_name:
        print("Error: Target repository must be specified via --repo or config.yaml")
        sys.exit(1)

    if args.command == 'run':
        try:
            manager.run_process(repo_name, verbose=args.verbose)
        except Exception as e:
            print(f"Error running process: {e}")
            sys.exit(1)

    elif args.command == 'similar':
        try:
            manager.connect_repo(repo_name)
            issues = manager.list_issues()

            # Prepare docs for indexing
            docs = [{'number': i.number, 'title': i.title, 'body': i.body or "", 'html_url': i.html_url} for i in issues]
            if manager.similarity_engine:
                manager.similarity_engine.index(docs)
                manager.find_similar_for_new(args.title, args.body)
            else:
                print("Similarity engine is disabled in config.")

        except Exception as e:
            print(f"Error checking similarity: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
