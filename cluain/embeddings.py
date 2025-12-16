"""
Code embeddings using sentence-transformers with CodeBERT.
"""

import os
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = 'mchochlov/codebert-base-cd-ft'


class CodeEmbedder:
    """Generates embeddings for code blocks using CodeBERT."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = 'cpu',
        max_threads: int = None,
        batch_size: int = 32
    ):
        """
        Args:
            model_name: HuggingFace model for code embeddings
            device: Device to run model on ('cpu' or 'mps')
            max_threads: Limit CPU threads (None = use all)
            batch_size: Batch size for encoding
        """
        self._configure_threads(max_threads)

        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def _configure_threads(self, max_threads: int):
        """Configure thread limits for PyTorch/NumPy."""
        if max_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(max_threads)
            os.environ["MKL_NUM_THREADS"] = str(max_threads)
            torch.set_num_threads(max_threads)

    def encode(self, codes: list, show_progress: bool = False) -> 'numpy.ndarray':
        """
        Encode a list of code strings into embeddings.

        Args:
            codes: List of code strings
            show_progress: Whether to show progress bar

        Returns:
            NumPy array of embeddings (n_codes x embedding_dim)
        """
        return self.model.encode(
            codes,
            show_progress_bar=show_progress,
            batch_size=self.batch_size
        )

    def encode_blocks(self, blocks: list, show_progress: bool = False) -> 'numpy.ndarray':
        """
        Encode a list of code blocks into embeddings.

        Args:
            blocks: List of block dicts (must have 'code' key)
            show_progress: Whether to show progress bar

        Returns:
            NumPy array of embeddings
        """
        codes = [b['code'] for b in blocks]
        return self.encode(codes, show_progress)
