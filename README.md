# GitHub Issue Manager (AI-Powered)

A powerful CLI tool to analyze, manage, and group GitHub issues using **AI Similarity Search** (Embeddings + BM25) and configurable rules.

## 🚀 Features

- **🧠 Semantic Clustering**: Groups similar issues (e.g., "Feature Requests", "Specific Bugs") using GPU-accelerated embeddings (`BAAI/bge-base-en-v1.5`).
- **📉 Stale Management**: Automatically identifies and tracks inactive issues based on configurable rules (Core Member follow-up vs. General Staleness).
- **⚡ Performance**:
    - **Caching**: Local JSON/Pickle cache for issues and embeddings to minimize API calls.
    - **Offline Mode**: Gracefully degrades to use cached data if GitHub API Rate Limits (403) are hit.
    - **Mulithreading**: Fetches comments in parallel (when fetching is safe).
- **📊 Reporting**: Generates a summary of actions (Close, Warn) and a "Top Recurring Issues" report sorted by impact.

## 🛠️ Setup

### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (Optional, but recommended for faster similarity indexing)
- GitHub Personal Access Token (PAT)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-org/github-issue-manager.git
    cd github-issue-manager
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This installs `torch`, `sentence-transformers`, `pygithub`, etc.*

3.  **Configure Environment**:
    Set your GitHub Token:
    ```bash
    export GITHUB_TOKEN="ghp_..."
    ```
    *(Alternatively, pass it via `--token` CLI argument).*

## ⚙️ Configuration

Edit `config.yaml` to customize behavior:

```yaml
repo_name: "unslothai/unsloth"
mode: "analysis"  # 'analysis' (Dry Run) or 'action' (Live Changes)

core_members:
  - "your_username"
  - "maintainer_2"

rules:
  core_followup:
    enabled: true
    warn_days: 14
    close_days: 21
  general_stale:
    close_days: 84
```

## 🏃 Usage

### 1. Analyze Issues (Dry Run)
Fetch issues, update cache, and print a summary report (Cluster Analysis + Action Plan).
```bash
python main.py run --repo owner/repo
```
*   **Offline Mode**: If the API is rate-limited, it will auto-switch to using the local cache `issues_cache.json`.
*   **Verbose**: Use `--verbose` to see decision logs for *every* issue.

### 2. Take Action (Close/Warn)
**Warning**: This will modify issues on GitHub!
1.  Set `mode: 'action'` in `config.yaml`.
2.  Run the command:
    ```bash
    python main.py run
    ```

### 3. Debug Similarity
Find issues similar to a query (using the same hybrid engine):
```bash
python main.py similar --title "Cuda Error" --body "Out of memory"
```

## 🏗️ Development

### Project Structure
- `main.py`: CLI Entry point.
- `issue_manager.py`: Core logic for fetching, caching, and analyzing.
- `similarity.py`: Hybrid Search Engine (Dense + BM25).
- `config.yaml`: Central config.

### Caching Strategy
- **Issues**: Saved to `issues_cache.json`. Deleted logic relies on cache invalidation (manual for now).
- **Embeddings**: Saved to `embeddings_cache.pkl`.
- **Note**: These files are gitignored (should be added to `.gitignore`).

### Known Limitations
- **Offline Limitations**: In offline mode, "Unique Users" defaults to 1 per issue as we cannot verify comment authors without API access.
