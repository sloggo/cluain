"""
Incremental historical analysis - only analyse changed files between commits.
"""

import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from .git_utils import (
    clone_repo, get_monthly_commits, prune_worktrees
)
from .parser import CodeParser
from .embeddings import CodeEmbedder
from .similarity import find_duplicates


class IncrementalTracker:
    """Fast historical tracking using incremental diffs."""

    def __init__(self, config: dict, clone_dir: str = "./repos"):
        self.config = config
        self.clone_dir = Path(clone_dir).resolve()
        self.clone_dir.mkdir(parents=True, exist_ok=True)

        self.parser = CodeParser(config.get('minimum_block_size', 150))
        self.embedder = CodeEmbedder(
            device=config.get('device', 'cpu'),
            max_threads=config.get('max_threads', 4),
            batch_size=config.get('encode_batch_size', 32)
        )
        self.threshold = config.get('threshold', 0.95)
        self.similarity_batch_size = config.get('similarity_batch_size', 500)

    def track_repository(
        self,
        repo_name: str,
        repo_url: str,
        excluded_paths: list = None,
        years: int = 5
    ) -> dict:
        """Track duplication history using incremental analysis."""
        repo_path = self._ensure_repo(repo_url, repo_name)
        commits = get_monthly_commits(str(repo_path), years)

        print(f"Found {len(commits)} monthly commits")
        print("Using incremental diff analysis")

        history = {
            'repo_name': repo_name,
            'repo_url': repo_url,
            'analysis_config': self.config,
            'generated_at': datetime.now().isoformat(),
            'snapshots': []
        }

        if not commits:
            return history

        # Cache: file_path -> (hash, block, embedding)
        block_cache = {}
        embedding_cache = {}

        prev_commit = None
        for i, commit_info in enumerate(commits):
            print(f"\n[{i+1}/{len(commits)}] {commit_info['month']} ({commit_info['commit'][:8]})")

            snapshot = self._analyse_commit_incremental(
                repo_path, commit_info, prev_commit,
                excluded_paths or [], block_cache, embedding_cache
            )

            if snapshot:
                history['snapshots'].append(snapshot)

            prev_commit = commit_info['commit']

        return history

    def _analyse_commit_incremental(
        self,
        repo_path: Path,
        commit_info: dict,
        prev_commit: str,
        excluded_paths: list,
        block_cache: dict,
        embedding_cache: dict
    ) -> dict:
        """Analyse a single commit, reusing cached data where possible."""
        commit = commit_info['commit']

        # Checkout this commit
        subprocess.run(
            ["git", "-C", str(repo_path), "checkout", "-f", commit],
            capture_output=True, text=True
        )

        if prev_commit is None:
            # First commit - full scan
            print("  Full scan (first commit)")
            blocks = self.parser.scan_directory(str(repo_path), excluded_paths)

            if not blocks:
                return self._empty_snapshot(commit_info)

            # Cache all blocks with file content hash
            for block in blocks:
                file_hash = self._hash_file(block['file'])
                cache_key = (block['file'], block['start_line'])
                block_cache[cache_key] = (file_hash, block)

            # Encode all
            embeddings = self.embedder.encode_blocks(blocks, show_progress=True)
            for idx, block in enumerate(blocks):
                cache_key = (block['file'], block['start_line'])
                embedding_cache[cache_key] = embeddings[idx]

        else:
            # Get changed files
            changed_files = self._get_changed_files(repo_path, prev_commit, commit)
            print(f"  {len(changed_files)} files changed")

            # Invalidate cache for changed files
            changed_file_set = set(str(repo_path / f) for f in changed_files)
            keys_to_remove = [k for k in block_cache if k[0] in changed_file_set]
            for k in keys_to_remove:
                del block_cache[k]
                if k in embedding_cache:
                    del embedding_cache[k]

            # Scan changed files only
            new_blocks = []
            for changed_file in changed_files:
                full_path = repo_path / changed_file
                if full_path.exists() and full_path.suffix in self.parser.EXTENSIONS:
                    file_blocks = self.parser.extract_functions(full_path)
                    for block in file_blocks:
                        cache_key = (block['file'], block['start_line'])
                        file_hash = self._hash_file(block['file'])
                        block_cache[cache_key] = (file_hash, block)
                        new_blocks.append((cache_key, block))

            # Encode only new blocks
            if new_blocks:
                print(f"  Encoding {len(new_blocks)} new/changed functions")
                new_codes = [b[1]['code'] for b in new_blocks]
                new_embeddings = self.embedder.encode(new_codes, show_progress=True)
                for idx, (cache_key, _) in enumerate(new_blocks):
                    embedding_cache[cache_key] = new_embeddings[idx]

            # Rebuild full block list from cache
            blocks = [data[1] for data in block_cache.values()]

        if not blocks:
            return self._empty_snapshot(commit_info)

        # Rebuild embeddings array from cache
        embeddings = np.array([
            embedding_cache[(b['file'], b['start_line'])]
            for b in blocks
            if (b['file'], b['start_line']) in embedding_cache
        ])

        # Filter blocks to match embeddings
        blocks = [
            b for b in blocks
            if (b['file'], b['start_line']) in embedding_cache
        ]

        if len(blocks) == 0:
            return self._empty_snapshot(commit_info)

        # Find duplicates
        duplicates = find_duplicates(
            blocks, embeddings,
            threshold=self.threshold,
            minimum_size=self.config.get('minimum_block_size', 150),
            batch_size=self.similarity_batch_size
        )

        metrics = self._compute_metrics(blocks, duplicates)
        print(f"  {len(blocks)} functions, {len(duplicates)} duplicates")

        return {
            'date': commit_info['date'],
            'month': commit_info['month'],
            'commit': commit_info['commit'],
            'metrics': metrics
        }

    def _get_changed_files(self, repo_path: Path, prev_commit: str, curr_commit: str) -> list:
        """Get files changed between two commits."""
        result = subprocess.run(
            ["git", "-C", str(repo_path), "diff", "--name-only", prev_commit, curr_commit],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return []
        return [f for f in result.stdout.strip().split('\n') if f]

    def _hash_file(self, file_path: str) -> str:
        """Get a simple hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                import hashlib
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""

    def _compute_metrics(self, blocks: list, duplicates: list) -> dict:
        """Compute duplication metrics."""
        duplicate_blocks = set()
        clone_types = {'T1': 0, 'T2': 0, 'T3': 0, 'T4': 0}

        for d in duplicates:
            duplicate_blocks.add(d['block1']['file'] + str(d['block1']['start_line']))
            duplicate_blocks.add(d['block2']['file'] + str(d['block2']['start_line']))
            clone_types[d.get('clone_type', 'T4')] += 1

        total = len(blocks)
        return {
            'total_blocks': total,
            'duplicate_pairs': len(duplicates),
            'duplicate_blocks': len(duplicate_blocks),
            'duplication_ratio': len(duplicate_blocks) / total if total > 0 else 0,
            'clone_types': clone_types
        }

    def _empty_snapshot(self, commit_info: dict) -> dict:
        """Return empty snapshot."""
        return {
            'date': commit_info['date'],
            'month': commit_info['month'],
            'commit': commit_info['commit'],
            'metrics': {'total_blocks': 0, 'duplicate_pairs': 0}
        }

    def _ensure_repo(self, repo_url: str, repo_name: str) -> Path:
        """Clone repo if not exists or is bare."""
        repo_path = self.clone_dir / repo_name
        needs_clone = False

        if not repo_path.exists():
            needs_clone = True
        else:
            # Check if it's a bare repo (no working directory)
            result = subprocess.run(
                ["git", "-C", str(repo_path), "config", "--get", "core.bare"],
                capture_output=True, text=True
            )
            if result.stdout.strip() == "true":
                print(f"Removing bare repo {repo_name}...")
                shutil.rmtree(repo_path)
                needs_clone = True

        if needs_clone:
            print(f"Cloning {repo_name}...")
            clone_repo(repo_url, str(repo_path))

        return repo_path

    def save_history(self, history: dict, output_file: str):
        """Save history to JSON."""
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved to {output_file}")
