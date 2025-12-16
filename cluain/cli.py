"""
Command-line interface for Cluain.
"""

import argparse
import json
import sys


def main_pr():
    """CLI entry point for PR analysis."""
    from .pr_analyser import PRAnalyser
    from .formatters import get_formatter

    parser = argparse.ArgumentParser(description='Analyse PR for code duplication')
    parser.add_argument('--repo', default='.', help='Path to git repository')
    parser.add_argument('--base', default='main', help='Base branch/commit')
    parser.add_argument('--head', default='HEAD', help='Head branch/commit')
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--exclude', nargs='*', default=[])
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
    """CLI entry point for codebase analysis."""
    from .analyser import Analyser

    parser = argparse.ArgumentParser(description='Analyse codebase for duplication')
    parser.add_argument('path', help='Path to codebase')
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--min-size', type=int, default=80)
    parser.add_argument('--exclude', nargs='*', default=[])
    parser.add_argument('--output', '-o', help='Output file')

    args = parser.parse_args()

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
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(output)


def main_historical():
    """CLI entry point for historical analysis."""
    parser = argparse.ArgumentParser(description='Track duplication history')
    parser.add_argument('repo_url', help='Repository URL (e.g., github.com/user/repo)')
    parser.add_argument('--name', help='Repository name (default: from URL)')
    parser.add_argument('--years', type=int, default=3, help='Years of history')
    parser.add_argument('--workers', type=int, default=2, help='Parallel workers (ignored if --fast)')
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--min-size', type=int, default=150)
    parser.add_argument('--exclude', nargs='*', default=['/tests', '/test', '/bench'])
    parser.add_argument('--output', '-o', default='history.json', help='Output file')
    parser.add_argument('--clone-dir', default='./repos', help='Clone directory')
    parser.add_argument('--fast', action='store_true', help='Use incremental diff analysis (faster)')

    args = parser.parse_args()

    config = {
        'threshold': args.threshold,
        'minimum_block_size': args.min_size,
        'device': 'cpu',
        'max_threads': 4,
        'encode_batch_size': 32,
        'similarity_batch_size': 500
    }

    repo_name = args.name or args.repo_url.split('/')[-1]

    if args.fast:
        from .incremental import IncrementalTracker
        tracker = IncrementalTracker(config, clone_dir=args.clone_dir)
    else:
        from .historical import HistoricalTracker
        tracker = HistoricalTracker(config, clone_dir=args.clone_dir, workers=args.workers)

    history = tracker.track_repository(
        repo_name=repo_name,
        repo_url=args.repo_url,
        excluded_paths=args.exclude,
        years=args.years
    )
    tracker.save_history(history, args.output)


def main_batch():
    """CLI entry point for batch analysis of multiple repos."""
    from .batch import run_batch_analysis

    parser = argparse.ArgumentParser(description='Batch analyse repositories from CSV')
    parser.add_argument('csv_file', help='Path to repositories.csv')
    parser.add_argument('--output-dir', default='./results', help='Output directory')
    parser.add_argument('--clone-dir', default='./repos', help='Clone directory')
    parser.add_argument('--years', type=int, default=3, help='Years of history')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers')
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--min-size', type=int, default=150)
    parser.add_argument('--exclude', nargs='*', default=['/tests', '/test', '/bench'])
    parser.add_argument('--fast', action='store_true', help='Use incremental diff analysis (faster)')

    args = parser.parse_args()

    config = {
        'threshold': args.threshold,
        'minimum_block_size': args.min_size,
        'device': 'cpu',
        'max_threads': 4,
        'encode_batch_size': 32,
        'similarity_batch_size': 500
    }

    run_batch_analysis(
        csv_path=args.csv_file,
        output_dir=args.output_dir,
        clone_dir=args.clone_dir,
        config=config,
        workers=args.workers,
        years=args.years,
        excluded_paths=args.exclude,
        fast=args.fast
    )


if __name__ == '__main__':
    main_pr()
