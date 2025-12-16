"""
PR analyser - detect code duplication in pull requests.
"""

import sys
from pathlib import Path
import numpy as np

from .parser import CodeParser
from .embeddings import CodeEmbedder
from .similarity import compute_similarity_matrix, classify_clone_type
from .git_utils import get_changed_files


class PRAnalyser:
    """Analyse pull requests for introduced code duplication."""

    def __init__(self, threshold: float = 0.95, minimum_block_size: int = 100):
        self.threshold = threshold
        self.minimum_block_size = minimum_block_size
        self.parser = CodeParser(minimum_block_size)
        self.embedder = CodeEmbedder(device='cpu', max_threads=2)

    def analyse(self, repo_path: str, base: str, head: str, excluded_paths: list = None) -> dict:
        """
        Analyse a PR for code duplication against entire codebase.

        Args:
            repo_path: Path to git repository
            base: Base commit SHA
            head: Head commit SHA
            excluded_paths: Paths to exclude

        Returns:
            Analysis result dict
        """
        self._log(f"Analysing PR: {base[:8]}...{head[:8]}")

        changed_files = get_changed_files(repo_path, base, head)
        if not changed_files:
            return self._empty_result("No C/C++ file changes detected")

        self._log(f"Found {len(changed_files)} changed C/C++ files")

        # Scan entire codebase
        self._log("Scanning entire codebase...")
        repo = Path(repo_path).resolve()
        all_blocks = self.parser.scan_directory(str(repo), excluded_paths or [])

        if not all_blocks:
            return self._empty_result("No code blocks found in codebase")

        self._log(f"Found {len(all_blocks)} total functions")

        # Split blocks into changed vs other
        changed_file_set = set(str(repo / f) for f in changed_files)
        changed_blocks = [b for b in all_blocks if b['file'] in changed_file_set]

        self._log(f"  - {len(changed_blocks)} functions in changed files")

        if not changed_blocks:
            return self._empty_result("No functions found in changed files")

        # Find duplicates
        duplicates = self._find_pr_duplicates(
            all_blocks, changed_blocks, changed_file_set, repo
        )

        return self._build_result(all_blocks, changed_blocks, duplicates)

    def _find_pr_duplicates(
        self,
        all_blocks: list,
        changed_blocks: list,
        changed_file_set: set,
        repo: Path
    ) -> list:
        """Find duplicates between changed code and entire codebase."""
        self._log("Encoding all functions...")
        all_embeddings = self.embedder.encode_blocks(all_blocks)

        block_to_idx = {id(b): i for i, b in enumerate(all_blocks)}
        changed_indices = [block_to_idx[id(b)] for b in changed_blocks]
        changed_embeddings = all_embeddings[changed_indices]

        self._log("Computing similarities...")
        similarities = compute_similarity_matrix(changed_embeddings, all_embeddings)

        duplicates = []
        seen_pairs = set()

        for i, changed_block in enumerate(changed_blocks):
            changed_idx = changed_indices[i]
            top_indices = np.argsort(similarities[i])[::-1]

            for j in top_indices[:10]:
                sim = similarities[i][j]
                if sim < self.threshold:
                    break

                if j == changed_idx:
                    continue

                other_block = all_blocks[j]
                if changed_block['file'] == other_block['file']:
                    continue

                if changed_block['size'] < self.minimum_block_size:
                    continue
                if other_block['size'] < self.minimum_block_size:
                    continue

                pair_key = tuple(sorted([changed_idx, j]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                duplicates.append(self._format_duplicate(
                    changed_block, other_block, sim,
                    other_block['file'] in changed_file_set, repo
                ))

        duplicates.sort(key=lambda x: x['similarity'], reverse=True)
        return duplicates

    def _format_duplicate(
        self, block1: dict, block2: dict, sim: float, within_pr: bool, repo: Path
    ) -> dict:
        """Format a duplicate pair for output."""
        clone_type = classify_clone_type(block1['code'], block2['code'])

        def format_block(block):
            return {
                'file': str(Path(block['file']).relative_to(repo)),
                'lines': f"{block['start_line']}-{block['end_line']}",
                'size': block['size'],
                'preview': block['code'][:200] + '...' if len(block['code']) > 200 else block['code']
            }

        return {
            'similarity': float(sim),
            'clone_type': clone_type,
            'within_pr': within_pr,
            'new_code': format_block(block1),
            'existing_code': format_block(block2)
        }

    def _build_result(self, all_blocks: list, changed_blocks: list, duplicates: list) -> dict:
        """Build the analysis result."""
        clone_types = {'T1': 0, 'T2': 0, 'T3': 0, 'T4': 0}
        within_pr = 0
        against_existing = 0

        for d in duplicates:
            clone_types[d['clone_type']] += 1
            if d['within_pr']:
                within_pr += 1
            else:
                against_existing += 1

        return {
            'status': 'success' if not duplicates else 'warning',
            'message': f"Found {len(duplicates)} potential duplications" if duplicates else "No duplications detected",
            'summary': {
                'total_functions_in_codebase': len(all_blocks),
                'total_changed_functions': len(changed_blocks),
                'duplicate_count': len(duplicates),
                'duplicates_within_pr': within_pr,
                'duplicates_against_existing': against_existing,
                'clone_types': clone_types
            },
            'duplicates': duplicates[:20]
        }

    def _empty_result(self, message: str) -> dict:
        """Return empty result."""
        return {
            'status': 'success',
            'message': message,
            'duplicates': [],
            'summary': {'total_changed_functions': 0, 'duplicate_count': 0}
        }

    def _log(self, message: str):
        """Log to stderr to keep stdout clean for JSON output."""
        print(message, file=sys.stderr)
