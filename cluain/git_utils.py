"""
Git utilities for repository operations.
"""

import subprocess
from pathlib import Path


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
