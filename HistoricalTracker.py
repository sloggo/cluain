import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def analyze_worktree(args):
    """Worker function to analyze a single commit in its own worktree."""
    worktree_path, commit_info, excluded_paths, threshold, minimum_block_size, model_name, max_threads, encode_batch_size, similarity_batch_size, device = args

    # Import here to avoid pickling issues with multiprocessing
    from SSCDAnalyser import SSCDAnalyser
    from pathlib import Path

    try:
        analyser = SSCDAnalyser(
            model_name=model_name,
            threshold=threshold,
            minimum_block_size=minimum_block_size,
            max_threads=max_threads,
            encode_batch_size=encode_batch_size,
            similarity_batch_size=similarity_batch_size,
            device=device
        )

        _, blocks, result = analyser.analyze(
            worktree_path,
            excluded_paths=excluded_paths or []
        )

        if not blocks:
            # Debug: check what files exist
            wt = Path(worktree_path)
            c_files = list(wt.rglob("*.c")) + list(wt.rglob("*.cpp")) + list(wt.rglob("*.h"))
            print(f"Warning: {commit_info['month']} - 0 blocks extracted, but found {len(c_files)} C/C++ files in {worktree_path}")

        return {
            "date": commit_info["date"],
            "month": commit_info["month"],
            "commit": commit_info["commit"],
            "metrics": result.get("metrics", {})
        }
    except Exception as e:
        import traceback
        print(f"Warning: Analysis failed for {commit_info['month']}: {e}")
        traceback.print_exc()
        return None


class HistoricalTracker:
    def __init__(self, analyser, clone_dir: str = "./repos", workers: int = 1):
        """
        Args:
            analyser: SSCDAnalyser instance
            clone_dir: Directory to clone repos into
            workers: Number of parallel workers (default 1 to avoid memory issues)
        """
        self.analyser = analyser
        self.clone_dir = Path(clone_dir).resolve()  # Use absolute path
        self.clone_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers

    def clone_repo(self, repo_url: str, repo_name: str) -> str:
        """Clone a repository from GitHub. Returns path to cloned repo."""
        repo_path = self.clone_dir / repo_name

        if repo_path.exists():
            print(f"Repository {repo_name} already exists at {repo_path}")
            return str(repo_path)

        full_url = f"https://{repo_url}"
        print(f"Cloning {repo_name} from {full_url}...")

        result = subprocess.run(
            ["git", "clone", full_url, str(repo_path)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone {repo_url}: {result.stderr}")

        print(f"Cloned {repo_name} to {repo_path}")
        return str(repo_path)

    def get_monthly_commits(self, repo_path: str, years: int = 5) -> list:
        """Get the first commit of each month for the past N years."""
        cutoff_date = datetime.now() - timedelta(days=years * 365)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

        # Don't use --reverse as it traverses from oldest commit and stops early
        result = subprocess.run(
            ["git", "-C", repo_path, "log", "--format=%H %ci", f"--after={cutoff_str}"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Warning: git log failed: {result.stderr}")
            return []

        commits_by_month = defaultdict(list)

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue

            commit_hash = parts[0]
            date_str = parts[1]

            try:
                commit_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                month_key = commit_date.strftime("%Y-%m")
                commits_by_month[month_key].append({
                    "hash": commit_hash,
                    "date": commit_date
                })
            except ValueError:
                continue

        # Take the earliest commit from each month
        monthly_commits = []
        for month_key in sorted(commits_by_month.keys()):
            # Sort commits in this month by date and take the first one
            month_commits = sorted(commits_by_month[month_key], key=lambda x: x["date"])
            first_commit = month_commits[0]
            monthly_commits.append({
                "date": first_commit["date"].strftime("%Y-%m-%d"),
                "month": month_key,
                "commit": first_commit["hash"]
            })

        return monthly_commits

    def create_worktree(self, repo_path: str, commit_hash: str, worktree_name: str) -> str:
        """Create a git worktree for a specific commit."""
        worktree_path = (self.clone_dir / "worktrees" / worktree_name).resolve()

        # Always try to remove via git first (handles git's internal tracking)
        subprocess.run(
            ["git", "-C", repo_path, "worktree", "remove", "--force", str(worktree_path)],
            capture_output=True,
            text=True
        )

        # Prune stale worktree references
        subprocess.run(
            ["git", "-C", repo_path, "worktree", "prune"],
            capture_output=True,
            text=True
        )

        # Force remove directory if it still exists
        if worktree_path.exists():
            shutil.rmtree(worktree_path)

        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["git", "-C", repo_path, "worktree", "add", "--detach", str(worktree_path), commit_hash],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Warning: Failed to create worktree for {commit_hash}: {result.stderr}")
            return None

        # Verify the worktree actually contains files
        if not worktree_path.exists() or not any(worktree_path.iterdir()):
            print(f"Warning: Worktree created but appears empty: {worktree_path}")
            return None

        return str(worktree_path)

    def remove_worktree(self, repo_path: str, worktree_path: str):
        """Remove a git worktree."""
        subprocess.run(
            ["git", "-C", repo_path, "worktree", "remove", "--force", worktree_path],
            capture_output=True,
            text=True
        )
        if Path(worktree_path).exists():
            shutil.rmtree(worktree_path)

    def cleanup_worktrees(self, repo_path: str):
        """Clean up all worktrees."""
        subprocess.run(
            ["git", "-C", repo_path, "worktree", "prune"],
            capture_output=True,
            text=True
        )
        worktrees_dir = self.clone_dir / "worktrees"
        if worktrees_dir.exists():
            shutil.rmtree(worktrees_dir)

    def track_repository(self, repo_name: str, repo_url: str, excluded_paths: list = None, years: int = 5) -> dict:
        """Track duplication history for a repository over time using parallel processing."""
        repo_path = self.clone_repo(repo_url, repo_name)
        monthly_commits = self.get_monthly_commits(repo_path, years)

        print(f"Found {len(monthly_commits)} monthly commits to analyze")
        print(f"Using {self.workers} parallel workers")

        history = {
            "repo_name": repo_name,
            "repo_url": repo_url,
            "analysis_config": {
                "threshold": self.analyser.threshold,
                "minimum_block_size": self.analyser.minimum_block_size
            },
            "generated_at": datetime.now().isoformat(),
            "snapshots": []
        }

        # Create worktrees for all commits
        print("Creating worktrees...")
        worktree_tasks = []
        for commit_info in monthly_commits:
            worktree_name = f"{repo_name}_{commit_info['month']}"
            worktree_path = self.create_worktree(repo_path, commit_info["commit"], worktree_name)
            if worktree_path:
                worktree_tasks.append((
                    worktree_path,
                    commit_info,
                    excluded_paths,
                    self.analyser.threshold,
                    self.analyser.minimum_block_size,
                    self.analyser.model.model_card_data.base_model if hasattr(self.analyser.model, 'model_card_data') else 'mchochlov/codebert-base-cd-ft',
                    getattr(self.analyser, 'max_threads', None),
                    getattr(self.analyser, 'encode_batch_size', 32),
                    getattr(self.analyser, 'similarity_batch_size', 500),
                    getattr(self.analyser, 'device', 'cpu')
                ))

        print(f"Analyzing {len(worktree_tasks)} commits in parallel...")

        # Process in parallel
        completed = 0
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(analyze_worktree, task): task for task in worktree_tasks}

            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result:
                    history["snapshots"].append(result)
                    print(f"[{completed}/{len(worktree_tasks)}] Completed {result['month']}")
                else:
                    print(f"[{completed}/{len(worktree_tasks)}] Failed")

        # Sort snapshots by date
        history["snapshots"].sort(key=lambda x: x["date"])

        # Cleanup
        print("Cleaning up worktrees...")
        self.cleanup_worktrees(repo_path)

        print(f"Completed analysis: {len(history['snapshots'])} snapshots collected")
        return history

    def save_history(self, history: dict, output_file: str):
        """Save history to JSON file."""
        with open(output_file, "w") as f:
            json.dump(history, f, indent=2)
        print(f"History saved to {output_file}")
