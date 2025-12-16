from sentence_transformers import SentenceTransformer
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json
from datetime import datetime
import os
import torch
import re


class SSCDAnalyser:
    def __init__(self, model_name='mchochlov/codebert-base-cd-ft', threshold=0.95, minimum_block_size=80,
                 max_threads=None, encode_batch_size=32, similarity_batch_size=500, device='cpu'):
        """
        Args:
            model_name: HuggingFace model for code embeddings
            threshold: Similarity threshold for duplicate detection (0-1)
            minimum_block_size: Minimum code block size in characters
            max_threads: Limit CPU threads (default: all cores). Set lower to reduce resource usage.
            encode_batch_size: Batch size for encoding (default 32). Lower = less memory.
            similarity_batch_size: Batch size for similarity computation (default 500). Lower = less memory.
            device: Device to run model on ('cpu' or 'mps'). Default 'cpu' to avoid GPU memory issues.
        """
        # Limit threads if specified
        self.max_threads = max_threads
        if max_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(max_threads)
            os.environ["MKL_NUM_THREADS"] = str(max_threads)
            torch.set_num_threads(max_threads)

        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.threshold = threshold
        self.minimum_block_size = minimum_block_size
        self.encode_batch_size = encode_batch_size
        self.similarity_batch_size = similarity_batch_size

        self.c_language = Language(tsc.language())
        self.cpp_language = Language(tscpp.language())

        self.c_parser = Parser(self.c_language)
        self.cpp_parser = Parser(self.cpp_language)

    def extract_code_blocks(self, file_path):
        """Extract functions from C/C++ files using tree-sitter"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            if file_path.suffix in ['.cpp', '.cc', '.cxx', '.hpp', '.hxx', '.h']:
                parser = self.cpp_parser
            else:
                parser = self.c_parser

            tree = parser.parse(content)
            root = tree.root_node

            blocks = []

            def traverse(node):
                if node.type == 'function_definition':
                    code = node.text.decode('utf-8', errors='ignore')
                    if len(code.strip()) > 50:
                        blocks.append({
                            'code': code,
                            'type': 'function',
                            'start_line': node.start_point[0] + 1,
                            'end_line': node.end_point[0] + 1,
                            'file': str(file_path),
                            'size': len(code)
                        })

                for child in node.children:
                    traverse(child)

            traverse(root)
            return blocks

        except Exception as e:
            return []

    def scan_codebase(self, root_dir, excluded_paths):
        """Extract all code blocks from C/C++ codebase"""
        root = Path(root_dir).resolve()
        excluded = [(root / ex.lstrip("/")).resolve() for ex in excluded_paths]

        all_blocks = []
        extensions = ['*.c', '*.cpp', '*.cc', '*.cxx', '*.h', '*.hpp', '*.hxx']

        for ext in extensions:
            for file_path in root.rglob(ext):
                file_path = file_path.resolve()

                skip = False
                for ex in excluded:
                    if ex.is_dir() and file_path.is_relative_to(ex):
                        skip = True
                        break

                    if file_path == ex:
                        skip = True
                        break

                if skip:
                    continue

                blocks = self.extract_code_blocks(file_path)
                all_blocks.extend(blocks)

        return all_blocks

    def find_duplicates(self, blocks):
        """Find duplicates using sklearn cosine_similarity"""
        if not blocks:
            return []

        codes = [b['code'] for b in blocks]
        print(f"Encoding {len(codes)} code blocks...")
        embeddings = self.model.encode(codes, show_progress_bar=True, batch_size=self.encode_batch_size)

        print("Computing similarity matrix...")
        duplicates = []
        seen = set()

        n = len(blocks)
        for i in range(0, n, self.similarity_batch_size):
            end_i = min(i + self.similarity_batch_size, n)

            batch_similarities = cosine_similarity(embeddings[i:end_i], embeddings)

            for local_idx, global_idx in enumerate(range(i, end_i)):
                similarities = batch_similarities[local_idx]
                top_indices = np.argsort(similarities)[::-1]

                for neighbor_idx in top_indices[1:11]:
                    similarity = similarities[neighbor_idx]

                    if similarity < self.threshold:
                        break

                    if blocks[global_idx]['file'] == blocks[neighbor_idx]['file']:
                        continue

                    if blocks[global_idx]['size'] < self.minimum_block_size or blocks[neighbor_idx]['size'] < self.minimum_block_size:
                        continue

                    pair = tuple(sorted([global_idx, neighbor_idx]))
                    if pair not in seen:
                        seen.add(pair)
                        clone_type = self.classify_clone_type(
                            blocks[global_idx]['code'],
                            blocks[neighbor_idx]['code'],
                            similarity
                        )
                        duplicates.append({
                            'similarity': float(similarity),
                            'clone_type': clone_type,
                            'block1': blocks[global_idx],
                            'block2': blocks[neighbor_idx]
                        })

            print(f"Processed {end_i}/{n} blocks...")

        duplicates.sort(key=lambda x: x['similarity'], reverse=True)
        return duplicates

    def normalize_code(self, code):
        """Remove whitespace and comments for T1 comparison."""
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code).strip()
        return code

    def tokenize_code(self, code):
        """Extract token sequence for T2 comparison (normalize identifiers)."""
        # Remove comments first
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # C/C++ keywords and types to preserve
        keywords = {
            'if', 'else', 'while', 'for', 'do', 'switch', 'case', 'break', 'continue',
            'return', 'goto', 'typedef', 'struct', 'union', 'enum', 'class', 'public',
            'private', 'protected', 'virtual', 'static', 'const', 'volatile', 'extern',
            'inline', 'template', 'typename', 'namespace', 'using', 'try', 'catch',
            'throw', 'new', 'delete', 'sizeof', 'void', 'int', 'char', 'float', 'double',
            'long', 'short', 'unsigned', 'signed', 'bool', 'true', 'false', 'nullptr',
            'auto', 'register', 'default'
        }

        # Tokenize: split on non-alphanumeric, keep operators
        tokens = re.findall(r'[a-zA-Z_]\w*|[0-9]+|[^\s\w]', code)

        # Normalize: replace non-keyword identifiers with placeholders
        normalized = []
        id_map = {}
        id_counter = 0

        for token in tokens:
            if token in keywords or not re.match(r'^[a-zA-Z_]', token):
                normalized.append(token)
            else:
                # It's an identifier - normalize it
                if token not in id_map:
                    id_map[token] = f'ID{id_counter}'
                    id_counter += 1
                normalized.append(id_map[token])

        return normalized

    def classify_clone_type(self, code1, code2, semantic_similarity):
        """
        Classify clone pair into Type 1-4.

        T1: Exact (after whitespace/comment normalization)
        T2: Renamed identifiers (same token structure)
        T3: Near-miss (some statements added/removed/modified)
        T4: Semantic only (different structure, same meaning)
        """
        # T1: Exact match after normalization
        norm1 = self.normalize_code(code1)
        norm2 = self.normalize_code(code2)
        if norm1 == norm2:
            return 'T1'

        # T2: Same structure with renamed identifiers
        tokens1 = self.tokenize_code(code1)
        tokens2 = self.tokenize_code(code2)
        if tokens1 == tokens2:
            return 'T2'

        # T3 vs T4: Based on token similarity
        # Calculate token-level similarity (Jaccard or sequence matching)
        if len(tokens1) > 0 and len(tokens2) > 0:
            # Use sequence matcher for ordered similarity
            from difflib import SequenceMatcher
            token_similarity = SequenceMatcher(None, tokens1, tokens2).ratio()

            # T3: High token similarity (>0.7) but not identical
            if token_similarity > 0.7:
                return 'T3'

        # T4: Semantic similarity only (caught by embeddings, but different structure)
        return 'T4'

    def compute_metrics(self, duplicates, blocks):
        """Calculate duplication metrics"""
        total_blocks = len(blocks)

        duplicate_blocks = set()
        for d in duplicates:
            duplicate_blocks.add(d['block1']['file'] + str(d['block1']['start_line']))
            duplicate_blocks.add(d['block2']['file'] + str(d['block2']['start_line']))

        # Count clone types
        clone_type_counts = {'T1': 0, 'T2': 0, 'T3': 0, 'T4': 0}
        for d in duplicates:
            clone_type = d.get('clone_type', 'T4')
            clone_type_counts[clone_type] += 1

        metrics = {
            'total_blocks': total_blocks,
            'duplicate_pairs': len(duplicates),
            'duplicate_blocks': len(duplicate_blocks),
            'duplication_ratio': len(duplicate_blocks) / total_blocks if total_blocks > 0 else 0,
            'avg_similarity': float(np.mean([d['similarity'] for d in duplicates])) if duplicates else 0,
            'clone_types': clone_type_counts
        }

        return metrics

    def get_similarity_distribution(self, duplicates):
        """Analyze distribution of similarities"""
        if not duplicates:
            return {}

        similarities = [d['similarity'] for d in duplicates]
        return {
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'mean': float(np.mean(similarities)),
            'median': float(np.median(similarities)),
            'std': float(np.std(similarities)),
            'bins': {
                '0.95-0.96': sum(1 for s in similarities if 0.95 <= s < 0.96),
                '0.96-0.97': sum(1 for s in similarities if 0.96 <= s < 0.97),
                '0.97-0.98': sum(1 for s in similarities if 0.97 <= s < 0.98),
                '0.98-0.99': sum(1 for s in similarities if 0.98 <= s < 0.99),
                '0.99-1.00': sum(1 for s in similarities if 0.99 <= s <= 1.00),
            }
        }

    def analyze(self, root_dir, repo_name=None, excluded_paths=None):
        """Full SSCD-style analysis for C/C++"""
        print(f"Scanning C/C++ codebase: {root_dir}")
        blocks = self.scan_codebase(root_dir, excluded_paths)
        print(f"Extracted {len(blocks)} code blocks")

        if not blocks:
            return [], [], {}

        duplicates = self.find_duplicates(blocks)
        metrics = self.compute_metrics(duplicates, blocks)

        # Add metadata
        result = {
            'repo_name': repo_name or Path(root_dir).name,
            'timestamp': datetime.now().isoformat(),
            'threshold': self.threshold,
            'metrics': metrics,
            'similarity_distribution': self.get_similarity_distribution(duplicates),
            'top_duplicates': [
                {
                    'similarity': d['similarity'],
                    'clone_type': d.get('clone_type', 'T4'),
                    'file1': d['block1']['file'],
                    'file2': d['block2']['file'],
                    'lines1': f"{d['block1']['start_line']}-{d['block1']['end_line']}",
                    'lines2': f"{d['block2']['start_line']}-{d['block2']['end_line']}",
                    'size1': d['block1']['size'],
                    'size2': d['block2']['size']
                }
                for d in duplicates[:50]  # Top 50
            ]
        }

        return duplicates, blocks, result

    def save_results(self, result, output_file='duplication_analysis.json'):
        """Save results to JSON"""
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output_file}")