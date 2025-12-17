"""
Similarity computation and clone type classification.
"""

import re
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


KEYWORDS = {
    'if', 'else', 'while', 'for', 'do', 'switch', 'case', 'break', 'continue',
    'return', 'goto', 'typedef', 'struct', 'union', 'enum', 'class', 'public',
    'private', 'protected', 'virtual', 'static', 'const', 'volatile', 'extern',
    'inline', 'template', 'typename', 'namespace', 'using', 'try', 'catch',
    'throw', 'new', 'delete', 'sizeof', 'void', 'int', 'char', 'float', 'double',
    'long', 'short', 'unsigned', 'signed', 'bool', 'true', 'false', 'nullptr',
    'auto', 'register', 'default'
}


def compute_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of embeddings."""
    return cosine_similarity(embeddings1, embeddings2)


def normalize_code(code: str) -> str:
    """Remove whitespace and comments for T1 comparison."""
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code).strip()
    return code


def tokenize_code(code: str) -> list:
    """
    Extract normalized token sequence for T2 comparison.
    Replaces identifiers with placeholders while preserving keywords.
    """
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    tokens = re.findall(r'[a-zA-Z_]\w*|[0-9]+|[^\s\w]', code)

    normalized = []
    id_map = {}
    id_counter = 0

    for token in tokens:
        if token in KEYWORDS or not re.match(r'^[a-zA-Z_]', token):
            normalized.append(token)
        else:
            if token not in id_map:
                id_map[token] = f'ID{id_counter}'
                id_counter += 1
            normalized.append(id_map[token])

    return normalized


def classify_clone_type(code1: str, code2: str) -> str:
    """
    Classify clone pair into Type 1-4.

    T1: Exact (after whitespace/comment normalization)
    T2: Renamed identifiers (same token structure)
    T3: Near-miss (some statements added/removed/modified)
    T4: Semantic only (different structure, same meaning)
    """
    if normalize_code(code1) == normalize_code(code2):
        return 'T1'

    tokens1 = tokenize_code(code1)
    tokens2 = tokenize_code(code2)
    if tokens1 == tokens2:
        return 'T2'

    if tokens1 and tokens2:
        token_similarity = SequenceMatcher(None, tokens1, tokens2).ratio()
        if token_similarity > 0.7:
            return 'T3'

    return 'T4'


def find_duplicates(
    blocks: list,
    embeddings: np.ndarray,
    threshold: float = 0.95,
    minimum_size: int = 80,
    batch_size: int = 500
) -> list:
    """
    Find duplicate code blocks based on embedding similarity.

    Args:
        blocks: List of code block dicts
        embeddings: Corresponding embeddings array
        threshold: Minimum similarity to consider as duplicate
        minimum_size: Minimum block size in characters
        batch_size: Batch size for similarity computation

    Returns:
        List of duplicate pairs with similarity and clone type
    """
    duplicates = []
    seen = set()
    n = len(blocks)

    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        batch_similarities = cosine_similarity(embeddings[i:end_i], embeddings)

        for local_idx, global_idx in enumerate(range(i, end_i)):
            similarities = batch_similarities[local_idx]
            top_indices = np.argsort(similarities)[::-1]

            for neighbor_idx in top_indices[1:11]:
                similarity = similarities[neighbor_idx]

                if similarity < threshold:
                    break

                if blocks[global_idx]['file'] == blocks[neighbor_idx]['file']:
                    continue

                if blocks[global_idx]['size'] < minimum_size:
                    continue
                if blocks[neighbor_idx]['size'] < minimum_size:
                    continue

                pair = tuple(sorted([global_idx, neighbor_idx]))
                if pair in seen:
                    continue
                seen.add(pair)

                clone_type = classify_clone_type(
                    blocks[global_idx]['code'],
                    blocks[neighbor_idx]['code']
                )

                duplicates.append({
                    'similarity': float(similarity),
                    'clone_type': clone_type,
                    'block1': blocks[global_idx],
                    'block2': blocks[neighbor_idx]
                })

    duplicates.sort(key=lambda x: x['similarity'], reverse=True)
    return duplicates
