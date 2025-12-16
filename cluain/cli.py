"""
Command-line interface for Cluain.
"""

import argparse
import json
import sys

from .pr_analyser import PRAnalyser
from .formatters import get_formatter


def main_pr():
    """CLI entry point for PR analysis."""
    parser = argparse.ArgumentParser(description='Analyse PR for code duplication')
    parser.add_argument('--repo', default='.', help='Path to git repository')
    parser.add_argument('--base', default='main', help='Base branch/commit')
    parser.add_argument('--head', default='HEAD', help='Head branch/commit')
    parser.add_argument('--threshold', type=float, default=0.95, help='Similarity threshold')
    parser.add_argument('--exclude', nargs='*', default=[], help='Paths to exclude')
    parser.add_argument('--output', choices=['json', 'github', 'ci'], default='ci')
    parser.add_argument('--fail-on-duplicates', action='store_true')

    args = parser.parse_args()

    analyser = PRAnalyser(threshold=args.threshold)
    result = analyser.analyse(
        repo_path=args.repo,
        base=args.base,
        head=args.head,
        excluded_paths=args.exclude
    )

    formatter = get_formatter(args.output)
    print(formatter(result))

    if args.fail_on_duplicates and result.get('duplicates'):
        sys.exit(1)


def main_analyse():
    """CLI entry point for full codebase analysis."""
    parser = argparse.ArgumentParser(description='Analyse codebase for code duplication')
    parser.add_argument('path', help='Path to codebase')
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--min-size', type=int, default=80)
    parser.add_argument('--exclude', nargs='*', default=[])
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')

    args = parser.parse_args()

    from .analyser import Analyser

    analyser = Analyser(
        threshold=args.threshold,
        minimum_block_size=args.min_size,
        device='cpu'
    )

    result = analyser.analyse(args.path, excluded_paths=args.exclude)

    output = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main_pr()
