"""
Git utilities for repository operations.
"""

import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


def get_changed_files(repo_path: str, base: str, head: str, extensions: set = None) -> list:
    """
    Get list of files changed between base and head.

    Args:
        repo_path: Path to git repository
        base: Base commit/branch
        head: Head commit/branch
        extensions: Set of extensions to filter (e.g., {'.c', '.cpp'})

    Returns:
        List of changed file paths (relative to repo)
    """
    if extensions is None:
        extensions = {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx'}

    result = subprocess.run(
        ["git", "-C", repo_path, "diff", "--name-only", f"{base}...{head}"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return []

    changed = []
    for line in result.stdout.strip().split('\n'):
        if line and Path(line).suffix in extensions:
            changed.append(line)

    return changed


def clone_repo(repo_url: str, dest_path: str) -> bool:
    """
    Clone a repository.

    Args:
        repo_url: URL to clone (without https://)
        dest_path: Destination path

    Returns:
        True if successful
    """
    full_url = f"https://{repo_url}"
    result = subprocess.run(
        ["git", "clone", full_url, dest_path],
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def create_worktree(repo_path: str, commit: str, worktree_path: str) -> bool:
    """
    Create a git worktree for a specific commit.

    Args:
        repo_path: Path to main repository
        commit: Commit hash to checkout
        worktree_path: Path for worktree

    Returns:
        True if successful
    """
    # Clean up any existing worktree
    subprocess.run(
        ["git", "-C", repo_path, "worktree", "remove", "--force", worktree_path],
        capture_output=True, text=True
    )
    subprocess.run(
        ["git", "-C", repo_path, "worktree", "prune"],
        capture_output=True, text=True
    )

    # Create new worktree
    result = subprocess.run(
        ["git", "-C", repo_path, "worktree", "add", "--detach", worktree_path, commit],
        capture_output=True, text=True
    )
    return result.returncode == 0


def remove_worktree(repo_path: str, worktree_path: str):
    """Remove a git worktree."""
    subprocess.run(
        ["git", "-C", repo_path, "worktree", "remove", "--force", worktree_path],
        capture_output=True, text=True
    )


def get_current_commit(repo_path: str) -> str:
    """Get current HEAD commit hash."""
    result = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "HEAD"],
        capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def checkout(repo_path: str, ref: str) -> bool:
    """Checkout a specific ref."""
    result = subprocess.run(
        ["git", "-C", repo_path, "checkout", ref],
        capture_output=True, text=True
    )
    return result.returncode == 0


def get_monthly_commits(repo_path: str, years: int = 5) -> list:
    """
    Get the first commit of each month for the past N years.

    Returns:
        List of dicts with date, month, commit keys
    """
    cutoff_date = datetime.now() - timedelta(days=years * 365)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")

    result = subprocess.run(
        ["git", "-C", repo_path, "log", "--format=%H %ci", f"--after={cutoff_str}"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
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

    # Take earliest commit from each month
    monthly_commits = []
    for month_key in sorted(commits_by_month.keys()):
        month_commits = sorted(commits_by_month[month_key], key=lambda x: x["date"])
        first = month_commits[0]
        monthly_commits.append({
            "date": first["date"].strftime("%Y-%m-%d"),
            "month": month_key,
            "commit": first["hash"]
        })

    return monthly_commits


def prune_worktrees(repo_path: str):
    """Prune stale worktree references."""
    subprocess.run(
        ["git", "-C", repo_path, "worktree", "prune"],
        capture_output=True, text=True
    )
