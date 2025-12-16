"""
PR Clone Analyser - Detect code duplication introduced by pull requests.

Usage in GitHub Actions:
    python pr_analyse.py --repo . --base main --head feature-branch

Or with explicit diff:
    python pr_analyse.py --repo . --diff-file changes.diff
"""

import subprocess
import argparse
import json
import sys
from pathlib import Path
from SSCDAnalyser import SSCDAnalyser


class PRAnalyser:
    def __init__(self, analyser: SSCDAnalyser = None, similarity_threshold: float = 0.95):
        """
        Args:
            analyser: SSCDAnalyser instance (creates default if None)
            similarity_threshold: Minimum similarity to report as duplicate
        """
        self.analyser = analyser or SSCDAnalyser(
            threshold=similarity_threshold,
            minimum_block_size=100,
            device='cpu',
            max_threads=2
        )
        self.threshold = similarity_threshold

    def get_changed_files(self, repo_path: str, base: str, head: str) -> list:
        """Get list of C/C++ files changed between base and head."""
        result = subprocess.run(
            ["git", "-C", repo_path, "diff", "--name-only", f"{base}...{head}"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Error getting diff: {result.stderr}", file=sys.stderr)
            return []

        extensions = {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx'}
        changed = []
        for line in result.stdout.strip().split('\n'):
            if line and Path(line).suffix in extensions:
                changed.append(line)

        return changed

    def get_changed_functions(self, repo_path: str, base: str, head: str) -> list:
        """
        Extract functions that were added or modified in the PR.
        Returns list of code blocks from the HEAD version.
        """
        changed_files = self.get_changed_files(repo_path, base, head)
        if not changed_files:
            return []

        print(f"Found {len(changed_files)} changed C/C++ files", file=sys.stderr)

        # Checkout HEAD and extract functions from changed files
        changed_blocks = []
        repo = Path(repo_path).resolve()

        for file_path in changed_files:
            full_path = repo / file_path
            if full_path.exists():
                blocks = self.analyser.extract_code_blocks(full_path)
                for block in blocks:
                    block['pr_file'] = file_path  # Track which PR file it came from
                changed_blocks.extend(blocks)

        return changed_blocks

    def get_baseline_blocks(self, repo_path: str, base: str, excluded_paths: list = None) -> list:
        """
        Get all code blocks from the base branch.
        These are cached/precomputed in a real deployment.
        """
        # Stash any changes, checkout base, scan, then restore
        repo = Path(repo_path).resolve()

        # Get current HEAD
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True
        )
        original_head = result.stdout.strip()

        try:
            # Checkout base
            subprocess.run(
                ["git", "-C", str(repo), "checkout", base],
                capture_output=True, text=True
            )

            # Scan codebase
            blocks = self.analyser.scan_codebase(str(repo), excluded_paths or [])
            return blocks

        finally:
            # Restore original HEAD
            subprocess.run(
                ["git", "-C", str(repo), "checkout", original_head],
                capture_output=True, text=True
            )

    def analyse_pr(self, repo_path: str, base: str, head: str,
                   excluded_paths: list = None, baseline_blocks: list = None) -> dict:
        """
        Analyse a PR for introduced code duplication.

        Compares functions in changed files against the ENTIRE codebase (HEAD version),
        to find if new/modified code duplicates anything anywhere in the project.

        Args:
            repo_path: Path to git repository
            base: Base branch (e.g., 'main')
            head: Head branch or commit (e.g., 'feature-branch')
            excluded_paths: Paths to exclude from analysis
            baseline_blocks: Pre-computed baseline blocks (optional, for caching)

        Returns:
            Analysis result with detected duplications
        """
        print(f"Analysing PR: {base}...{head}", file=sys.stderr)

        # Get list of changed files
        changed_files = self.get_changed_files(repo_path, base, head)
        if not changed_files:
            return {
                'status': 'success',
                'message': 'No C/C++ file changes detected',
                'duplicates': [],
                'summary': {'total_changed_functions': 0}
            }

        print(f"Found {len(changed_files)} changed C/C++ files", file=sys.stderr)

        # Scan the ENTIRE codebase at HEAD (current state)
        print("Scanning entire codebase...", file=sys.stderr)
        repo = Path(repo_path).resolve()
        all_blocks = self.analyser.scan_codebase(str(repo), excluded_paths or [])

        if not all_blocks:
            return {
                'status': 'success',
                'message': 'No code blocks found in codebase',
                'duplicates': [],
                'summary': {'total_changed_functions': 0}
            }

        print(f"Found {len(all_blocks)} total functions in codebase", file=sys.stderr)

        # Identify which blocks are from changed files
        changed_file_set = set(str(repo / f) for f in changed_files)
        changed_blocks = [b for b in all_blocks if b['file'] in changed_file_set]
        other_blocks = [b for b in all_blocks if b['file'] not in changed_file_set]

        print(f"  - {len(changed_blocks)} functions in changed files", file=sys.stderr)
        print(f"  - {len(other_blocks)} functions in other files", file=sys.stderr)

        if not changed_blocks:
            return {
                'status': 'success',
                'message': 'No functions found in changed files',
                'duplicates': [],
                'summary': {'total_changed_functions': 0}
            }

        # Encode all blocks
        print("Encoding all functions...", file=sys.stderr)
        all_codes = [b['code'] for b in all_blocks]
        all_embeddings = self.analyser.model.encode(
            all_codes,
            show_progress_bar=False,
            batch_size=self.analyser.encode_batch_size
        )

        # Create index mapping
        block_to_idx = {id(b): i for i, b in enumerate(all_blocks)}

        # Compare changed functions against ALL other functions
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        print("Computing similarities...", file=sys.stderr)

        changed_indices = [block_to_idx[id(b)] for b in changed_blocks]
        changed_embeddings = all_embeddings[changed_indices]

        # Compare against entire codebase
        similarities = cosine_similarity(changed_embeddings, all_embeddings)

        # Find duplicates
        duplicates = []
        seen_pairs = set()

        for i, changed_block in enumerate(changed_blocks):
            changed_idx = changed_indices[i]
            top_indices = np.argsort(similarities[i])[::-1]

            for j in top_indices[:10]:  # Top 10 matches
                sim = similarities[i][j]
                if sim < self.threshold:
                    break

                # Skip self-comparison
                if j == changed_idx:
                    continue

                other_block = all_blocks[j]

                # Skip if same file
                if changed_block['file'] == other_block['file']:
                    continue

                # Skip if too small
                if changed_block['size'] < self.analyser.minimum_block_size:
                    continue
                if other_block['size'] < self.analyser.minimum_block_size:
                    continue

                # Avoid duplicate pairs
                pair_key = tuple(sorted([changed_idx, j]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                clone_type = self.analyser.classify_clone_type(
                    changed_block['code'],
                    other_block['code'],
                    sim
                )

                # Determine if the match is in changed files or existing code
                is_match_in_pr = other_block['file'] in changed_file_set

                duplicates.append({
                    'similarity': float(sim),
                    'clone_type': clone_type,
                    'within_pr': is_match_in_pr,  # True if both files are in the PR
                    'new_code': {
                        'file': str(Path(changed_block['file']).relative_to(repo)),
                        'lines': f"{changed_block['start_line']}-{changed_block['end_line']}",
                        'size': changed_block['size'],
                        'preview': changed_block['code'][:200] + '...' if len(changed_block['code']) > 200 else changed_block['code']
                    },
                    'existing_code': {
                        'file': str(Path(other_block['file']).relative_to(repo)),
                        'lines': f"{other_block['start_line']}-{other_block['end_line']}",
                        'size': other_block['size'],
                        'preview': other_block['code'][:200] + '...' if len(other_block['code']) > 200 else other_block['code']
                    }
                })

        # Sort by similarity
        duplicates.sort(key=lambda x: x['similarity'], reverse=True)

        # Compute summary
        clone_type_counts = {'T1': 0, 'T2': 0, 'T3': 0, 'T4': 0}
        within_pr_count = 0
        against_existing_count = 0

        for d in duplicates:
            clone_type_counts[d['clone_type']] += 1
            if d.get('within_pr', False):
                within_pr_count += 1
            else:
                against_existing_count += 1

        return {
            'status': 'success' if not duplicates else 'warning',
            'message': f"Found {len(duplicates)} potential duplications" if duplicates else "No duplications detected",
            'summary': {
                'total_functions_in_codebase': len(all_blocks),
                'total_changed_functions': len(changed_blocks),
                'duplicate_count': len(duplicates),
                'duplicates_within_pr': within_pr_count,
                'duplicates_against_existing': against_existing_count,
                'clone_types': clone_type_counts
            },
            'duplicates': duplicates[:20]  # Top 20 for CI output
        }

    def format_github_comment(self, result: dict) -> str:
        """Format analysis result as a GitHub PR comment."""
        if result['status'] == 'success' and not result['duplicates']:
            return "✅ **No code duplication detected in this PR**"

        summary = result['summary']
        lines = [
            "## 🔍 Code Duplication Analysis",
            "",
            f"Scanned **{summary.get('total_functions_in_codebase', 'N/A')}** functions in codebase, "
            f"**{summary['total_changed_functions']}** in this PR.",
            "",
            f"Found **{summary['duplicate_count']}** potential duplications:",
            f"- **{summary.get('duplicates_against_existing', 0)}** match existing code in the codebase",
            f"- **{summary.get('duplicates_within_pr', 0)}** are within this PR's changed files",
            "",
            "### Clone Type Breakdown",
            "| Type | Count | Description |",
            "|------|-------|-------------|",
            f"| T1 | {summary['clone_types']['T1']} | Exact copy |",
            f"| T2 | {summary['clone_types']['T2']} | Renamed identifiers |",
            f"| T3 | {summary['clone_types']['T3']} | Modified statements |",
            f"| T4 | {summary['clone_types']['T4']} | Semantic similarity |",
            "",
        ]

        if result['duplicates']:
            lines.extend([
                "### Top Duplications",
                ""
            ])

            for i, dup in enumerate(result['duplicates'][:5], 1):
                location_tag = "📁 within PR" if dup.get('within_pr', False) else "📦 existing code"
                lines.extend([
                    f"<details>",
                    f"<summary><b>{i}. {dup['clone_type']}</b> - {dup['similarity']:.1%} similarity ({location_tag})</summary>",
                    "",
                    f"**New code** in `{dup['new_code']['file']}` (lines {dup['new_code']['lines']})",
                    "```cpp",
                    dup['new_code']['preview'],
                    "```",
                    "",
                    f"**Similar to** `{dup['existing_code']['file']}` (lines {dup['existing_code']['lines']})",
                    "```cpp",
                    dup['existing_code']['preview'],
                    "```",
                    "</details>",
                    ""
                ])

        lines.append("---")
        lines.append("*Generated by [Cluain](https://github.com/your-repo/cluain) - Semantic Code Clone Detector*")

        return "\n".join(lines)

    def format_ci_output(self, result: dict) -> str:
        """Format as CI-friendly output with exit code guidance."""
        output = []
        output.append(f"=== Cluain PR Analysis ===")
        output.append(f"Changed functions: {result['summary']['total_changed_functions']}")
        output.append(f"Baseline functions: {result['summary'].get('total_baseline_functions', 'N/A')}")
        output.append(f"Duplications found: {result['summary']['duplicate_count']}")
        output.append("")

        if result['duplicates']:
            output.append("Clone types: " + ", ".join(
                f"{k}={v}" for k, v in result['summary']['clone_types'].items() if v > 0
            ))
            output.append("")
            output.append("Top duplications:")

            for i, dup in enumerate(result['duplicates'][:10], 1):
                output.append(
                    f"  {i}. [{dup['clone_type']}] {dup['similarity']:.1%} "
                    f"| {dup['new_code']['file']}:{dup['new_code']['lines']} "
                    f"<-> {dup['existing_code']['file']}:{dup['existing_code']['lines']}"
                )

        return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description='Analyse PR for code duplication')
    parser.add_argument('--repo', default='.', help='Path to git repository')
    parser.add_argument('--base', default='main', help='Base branch')
    parser.add_argument('--head', default='HEAD', help='Head branch/commit')
    parser.add_argument('--threshold', type=float, default=0.95, help='Similarity threshold')
    parser.add_argument('--exclude', nargs='*', default=[], help='Paths to exclude')
    parser.add_argument('--output', choices=['json', 'github', 'ci'], default='ci', help='Output format')
    parser.add_argument('--fail-on-duplicates', action='store_true', help='Exit with code 1 if duplicates found')

    args = parser.parse_args()

    analyser = PRAnalyser(similarity_threshold=args.threshold)
    result = analyser.analyse_pr(
        repo_path=args.repo,
        base=args.base,
        head=args.head,
        excluded_paths=args.exclude
    )

    if args.output == 'json':
        print(json.dumps(result, indent=2))
    elif args.output == 'github':
        print(analyser.format_github_comment(result))
    else:
        print(analyser.format_ci_output(result))

    if args.fail_on_duplicates and result['duplicates']:
        sys.exit(1)


if __name__ == '__main__':
    main()
