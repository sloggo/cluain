"""
Historical analysis - track code duplication over time.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from .git_utils import (
    clone_repo, create_worktree, remove_worktree,
    get_monthly_commits, prune_worktrees
)


def _analyse_commit(args):
    """Worker function to analyse a single commit (runs in subprocess)."""
    worktree_path, commit_info, excluded_paths, config = args

    from cluain.analyser import Analyser

    try:
        analyser = Analyser(
            threshold=config['threshold'],
            minimum_block_size=config['minimum_block_size'],
            device=config['device'],
            max_threads=config['max_threads'],
            encode_batch_size=config['encode_batch_size'],
            similarity_batch_size=config['similarity_batch_size']
        )

        result = analyser.analyse(worktree_path, excluded_paths or [])

        return {
            'date': commit_info['date'],
            'month': commit_info['month'],
            'commit': commit_info['commit'],
            'metrics': result.get('metrics', {})
        }
    except Exception as e:
        print(f"Warning: Analysis failed for {commit_info['month']}: {e}")
        return None


class HistoricalTracker:
    """Track code duplication history over time."""

    def __init__(self, config: dict, clone_dir: str = "./repos", workers: int = 1):
        """
        Args:
            config: Analyser configuration dict
            clone_dir: Directory to store cloned repos
            workers: Number of parallel workers
        """
        self.config = config
        self.clone_dir = Path(clone_dir).resolve()
        self.clone_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers

    def track_repository(
        self,
        repo_name: str,
        repo_url: str,
        excluded_paths: list = None,
        years: int = 5
    ) -> dict:
        """
        Track duplication history for a repository.

        Args:
            repo_name: Name for the repository
            repo_url: GitHub URL (without https://)
            excluded_paths: Paths to exclude
            years: How many years of history

        Returns:
            History dict with snapshots
        """
        repo_path = self._ensure_repo(repo_url, repo_name)
        commits = get_monthly_commits(str(repo_path), years)

        print(f"Found {len(commits)} monthly commits to analyse")
        print(f"Using {self.workers} workers")

        history = self._create_history_skeleton(repo_name, repo_url)

        tasks = self._prepare_worktree_tasks(repo_path, commits, excluded_paths)
        results = self._run_analysis(repo_path, tasks)

        history['snapshots'] = sorted(results, key=lambda x: x['date'])
        self._cleanup(repo_path)

        print(f"Completed: {len(history['snapshots'])} snapshots")
        return history

    def _ensure_repo(self, repo_url: str, repo_name: str) -> Path:
        """Clone repo if not exists, return path."""
        repo_path = self.clone_dir / repo_name
        if not repo_path.exists():
            print(f"Cloning {repo_name}...")
            clone_repo(repo_url, str(repo_path))
        return repo_path

    def _create_history_skeleton(self, repo_name: str, repo_url: str) -> dict:
        """Create empty history structure."""
        return {
            'repo_name': repo_name,
            'repo_url': repo_url,
            'analysis_config': self.config,
            'generated_at': datetime.now().isoformat(),
            'snapshots': []
        }

    def _prepare_worktree_tasks(
        self, repo_path: Path, commits: list, excluded_paths: list
    ) -> list:
        """Create worktrees and prepare analysis tasks."""
        print("Creating worktrees...")
        tasks = []

        for commit_info in commits:
            worktree_name = f"{repo_path.name}_{commit_info['month']}"
            worktree_path = self.clone_dir / "worktrees" / worktree_name

            if create_worktree(str(repo_path), commit_info['commit'], str(worktree_path)):
                tasks.append((
                    str(worktree_path),
                    commit_info,
                    excluded_paths,
                    self.config
                ))

        return tasks

    def _run_analysis(self, repo_path: Path, tasks: list) -> list:
        """Run analysis on all tasks."""
        print(f"Analysing {len(tasks)} commits...")
        results = []

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(_analyse_commit, t): t for t in tasks}

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result:
                    results.append(result)
                    print(f"[{i}/{len(tasks)}] {result['month']}")
                else:
                    print(f"[{i}/{len(tasks)}] Failed")

        return results

    def _cleanup(self, repo_path: Path):
        """Clean up worktrees."""
        print("Cleaning up...")
        prune_worktrees(str(repo_path))
        worktrees_dir = self.clone_dir / "worktrees"
        if worktrees_dir.exists():
            shutil.rmtree(worktrees_dir)

    def save_history(self, history: dict, output_file: str):
        """Save history to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved to {output_file}")
