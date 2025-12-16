"""
Cluain - Semantic Source Code Duplication Detector

A tool for detecting semantically similar code clones in C/C++ codebases
using CodeBERT embeddings.
"""

from .analyser import Analyser
from .pr_analyser import PRAnalyser
from .historical import HistoricalTracker

__version__ = "0.1.0"
__all__ = ["Analyser", "PRAnalyser", "HistoricalTracker"]
