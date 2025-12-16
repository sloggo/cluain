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

        # Get functions changed in PR
        changed_blocks = self.get_changed_functions(repo_path, base, head)
        if not changed_blocks:
            return {
                'status': 'success',
                'message': 'No C/C++ function changes detected',
                'duplicates': [],
                'summary': {'total_changed_functions': 0}
            }

        print(f"Extracted {len(changed_blocks)} functions from changed files", file=sys.stderr)

        # Get baseline (or use provided cache)
        if baseline_blocks is None:
            print("Scanning baseline codebase...", file=sys.stderr)
            baseline_blocks = self.get_baseline_blocks(repo_path, base, excluded_paths)

        if not baseline_blocks:
            return {
                'status': 'success',
                'message': 'No baseline code to compare against',
                'duplicates': [],
                'summary': {'total_changed_functions': len(changed_blocks)}
            }

        print(f"Comparing against {len(baseline_blocks)} baseline functions", file=sys.stderr)

        # Encode both sets
        changed_codes = [b['code'] for b in changed_blocks]
        baseline_codes = [b['code'] for b in baseline_blocks]

        print("Encoding changed functions...", file=sys.stderr)
        changed_embeddings = self.analyser.model.encode(
            changed_codes,
            show_progress_bar=False,
            batch_size=self.analyser.encode_batch_size
        )

        print("Encoding baseline functions...", file=sys.stderr)
        baseline_embeddings = self.analyser.model.encode(
            baseline_codes,
            show_progress_bar=False,
            batch_size=self.analyser.encode_batch_size
        )

        # Compare changed against baseline
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        print("Computing similarities...", file=sys.stderr)
        similarities = cosine_similarity(changed_embeddings, baseline_embeddings)

        # Find duplicates
        duplicates = []
        for i, changed_block in enumerate(changed_blocks):
            top_indices = np.argsort(similarities[i])[::-1]

            for j in top_indices[:5]:  # Top 5 matches
                sim = similarities[i][j]
                if sim < self.threshold:
                    break

                baseline_block = baseline_blocks[j]

                # Skip if same file (likely same function before/after)
                if Path(changed_block['file']).name == Path(baseline_block['file']).name:
                    continue

                # Skip if too small
                if changed_block['size'] < self.analyser.minimum_block_size:
                    continue

                clone_type = self.analyser.classify_clone_type(
                    changed_block['code'],
                    baseline_block['code'],
                    sim
                )

                duplicates.append({
                    'similarity': float(sim),
                    'clone_type': clone_type,
                    'new_code': {
                        'file': changed_block.get('pr_file', changed_block['file']),
                        'lines': f"{changed_block['start_line']}-{changed_block['end_line']}",
                        'size': changed_block['size'],
                        'preview': changed_block['code'][:200] + '...' if len(changed_block['code']) > 200 else changed_block['code']
                    },
                    'existing_code': {
                        'file': baseline_block['file'],
                        'lines': f"{baseline_block['start_line']}-{baseline_block['end_line']}",
                        'size': baseline_block['size'],
                        'preview': baseline_block['code'][:200] + '...' if len(baseline_block['code']) > 200 else baseline_block['code']
                    }
                })

        # Sort by similarity
        duplicates.sort(key=lambda x: x['similarity'], reverse=True)

        # Compute summary
        clone_type_counts = {'T1': 0, 'T2': 0, 'T3': 0, 'T4': 0}
        for d in duplicates:
            clone_type_counts[d['clone_type']] += 1

        return {
            'status': 'success' if not duplicates else 'warning',
            'message': f"Found {len(duplicates)} potential duplications" if duplicates else "No duplications detected",
            'summary': {
                'total_changed_functions': len(changed_blocks),
                'total_baseline_functions': len(baseline_blocks),
                'duplicate_count': len(duplicates),
                'clone_types': clone_type_counts
            },
            'duplicates': duplicates[:20]  # Top 20 for CI output
        }

    def format_github_comment(self, result: dict) -> str:
        """Format analysis result as a GitHub PR comment."""
        if result['status'] == 'success' and not result['duplicates']:
            return "✅ **No code duplication detected in this PR**"

        lines = [
            "## 🔍 Code Duplication Analysis",
            "",
            f"Found **{result['summary']['duplicate_count']}** potential duplications",
            "",
            "### Clone Type Breakdown",
            "| Type | Count | Description |",
            "|------|-------|-------------|",
            f"| T1 | {result['summary']['clone_types']['T1']} | Exact copy |",
            f"| T2 | {result['summary']['clone_types']['T2']} | Renamed identifiers |",
            f"| T3 | {result['summary']['clone_types']['T3']} | Modified statements |",
            f"| T4 | {result['summary']['clone_types']['T4']} | Semantic similarity |",
            "",
        ]

        if result['duplicates']:
            lines.extend([
                "### Top Duplications",
                ""
            ])

            for i, dup in enumerate(result['duplicates'][:5], 1):
                lines.extend([
                    f"<details>",
                    f"<summary><b>{i}. {dup['clone_type']}</b> - {dup['similarity']:.1%} similarity</summary>",
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
