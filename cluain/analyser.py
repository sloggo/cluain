"""
Main analyser - orchestrates parsing, embedding, and similarity detection.
"""

from datetime import datetime
from pathlib import Path
import numpy as np

from .parser import CodeParser
from .embeddings import CodeEmbedder
from .similarity import find_duplicates


class Analyser:
    """Main SSCD analyser for detecting code clones."""

    def __init__(
        self,
        threshold: float = 0.95,
        minimum_block_size: int = 80,
        device: str = 'cpu',
        max_threads: int = None,
        encode_batch_size: int = 32,
        similarity_batch_size: int = 500
    ):
        self.threshold = threshold
        self.minimum_block_size = minimum_block_size
        self.similarity_batch_size = similarity_batch_size

        self.parser = CodeParser(minimum_block_size)
        self.embedder = CodeEmbedder(
            device=device,
            max_threads=max_threads,
            batch_size=encode_batch_size
        )

    def analyse(self, root_dir: str, excluded_paths: list = None, repo_name: str = None) -> dict:
        """
        Analyse a codebase for code duplication.

        Args:
            root_dir: Root directory to scan
            excluded_paths: Paths to exclude
            repo_name: Optional repository name for output

        Returns:
            Analysis result dict with metrics and duplicates
        """
        print(f"Scanning codebase: {root_dir}")
        blocks = self.parser.scan_directory(root_dir, excluded_paths or [])
        print(f"Extracted {len(blocks)} code blocks")

        if not blocks:
            return self._empty_result(repo_name or Path(root_dir).name)

        print("Encoding code blocks...")
        embeddings = self.embedder.encode_blocks(blocks, show_progress=True)

        print("Finding duplicates...")
        duplicates = find_duplicates(
            blocks, embeddings,
            threshold=self.threshold,
            minimum_size=self.minimum_block_size,
            batch_size=self.similarity_batch_size
        )

        return self._build_result(
            repo_name or Path(root_dir).name,
            blocks, duplicates
        )

    def _empty_result(self, repo_name: str) -> dict:
        """Return empty result when no blocks found."""
        return {
            'repo_name': repo_name,
            'timestamp': datetime.now().isoformat(),
            'threshold': self.threshold,
            'metrics': {'total_blocks': 0},
            'duplicates': []
        }

    def _build_result(self, repo_name: str, blocks: list, duplicates: list) -> dict:
        """Build the analysis result dict."""
        metrics = self._compute_metrics(blocks, duplicates)

        return {
            'repo_name': repo_name,
            'timestamp': datetime.now().isoformat(),
            'threshold': self.threshold,
            'metrics': metrics,
            'similarity_distribution': self._get_distribution(duplicates),
            'top_duplicates': self._format_top_duplicates(duplicates[:50])
        }

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
            'avg_similarity': float(np.mean([d['similarity'] for d in duplicates])) if duplicates else 0,
            'clone_types': clone_types
        }

    def _get_distribution(self, duplicates: list) -> dict:
        """Get similarity distribution stats."""
        if not duplicates:
            return {}

        sims = [d['similarity'] for d in duplicates]
        return {
            'min': float(np.min(sims)),
            'max': float(np.max(sims)),
            'mean': float(np.mean(sims)),
            'median': float(np.median(sims)),
        }

    def _format_top_duplicates(self, duplicates: list) -> list:
        """Format duplicates for output."""
        return [
            {
                'similarity': d['similarity'],
                'clone_type': d.get('clone_type', 'T4'),
                'file1': d['block1']['file'],
                'file2': d['block2']['file'],
                'lines1': f"{d['block1']['start_line']}-{d['block1']['end_line']}",
                'lines2': f"{d['block2']['start_line']}-{d['block2']['end_line']}",
            }
            for d in duplicates
        ]
