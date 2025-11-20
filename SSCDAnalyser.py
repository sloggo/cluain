from sentence_transformers import SentenceTransformer
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json
from datetime import datetime


class SSCDAnalyser:
    def __init__(self, model_name='mchochlov/codebert-base-cd-ft', threshold=0.95):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

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

    def find_duplicates(self, blocks, batch_size=1000):
        """Find duplicates using sklearn cosine_similarity"""
        if not blocks:
            return []

        codes = [b['code'] for b in blocks]
        print(f"Encoding {len(codes)} code blocks...")
        embeddings = self.model.encode(codes, show_progress_bar=True, batch_size=32)

        print("Computing similarity matrix...")
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

                    if similarity < self.threshold:
                        break

                    if blocks[global_idx]['file'] == blocks[neighbor_idx]['file']:
                        continue

                    pair = tuple(sorted([global_idx, neighbor_idx]))
                    if pair not in seen:
                        seen.add(pair)
                        duplicates.append({
                            'similarity': float(similarity),
                            'block1': blocks[global_idx],
                            'block2': blocks[neighbor_idx]
                        })

            print(f"Processed {end_i}/{n} blocks...")

        duplicates.sort(key=lambda x: x['similarity'], reverse=True)
        return duplicates

    def compute_metrics(self, duplicates, blocks):
        """Calculate duplication metrics"""
        total_blocks = len(blocks)

        duplicate_blocks = set()
        for d in duplicates:
            duplicate_blocks.add(d['block1']['file'] + str(d['block1']['start_line']))
            duplicate_blocks.add(d['block2']['file'] + str(d['block2']['start_line']))

        metrics = {
            'total_blocks': total_blocks,
            'duplicate_pairs': len(duplicates),
            'duplicate_blocks': len(duplicate_blocks),
            'duplication_ratio': len(duplicate_blocks) / total_blocks if total_blocks > 0 else 0,
            'avg_similarity': float(np.mean([d['similarity'] for d in duplicates])) if duplicates else 0
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