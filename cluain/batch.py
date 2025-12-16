"""
Batch analysis - run historical analysis on multiple repositories.
"""

import csv
import json
from pathlib import Path
from datetime import datetime

from .historical import HistoricalTracker


def load_repositories(csv_path: str) -> list:
    """
    Load repositories from CSV file.

    Expected format: name,url (no header)
    """
    repos = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                repos.append({
                    'name': row[0].strip(),
                    'url': row[1].strip()
                })
    return repos


def run_batch_analysis(
    csv_path: str,
    output_dir: str = "./results",
    clone_dir: str = "./repos",
    config: dict = None,
    workers: int = 1,
    years: int = 3,
    excluded_paths: list = None,
    fast: bool = False
) -> dict:
    """
    Run historical analysis on all repositories in CSV.

    Args:
        csv_path: Path to repositories.csv
        output_dir: Directory for output JSON files
        clone_dir: Directory for cloned repos
        config: Analyser configuration
        workers: Parallel workers per repo
        years: Years of history to analyse
        excluded_paths: Paths to exclude (applied to all repos)
        fast: Use incremental diff analysis (faster)

    Returns:
        Summary dict with results for each repo
    """
    if config is None:
        config = {
            'threshold': 0.95,
            'minimum_block_size': 150,
            'device': 'cpu',
            'max_threads': 4,
            'encode_batch_size': 32,
            'similarity_batch_size': 500
        }

    if excluded_paths is None:
        excluded_paths = ['/tests', '/test', '/bench', '/examples', '/docs']

    repos = load_repositories(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Batch Analysis ===")
    print(f"Repositories: {len(repos)}")
    print(f"Years: {years}")
    print(f"Output: {output_path}")
    print(f"Mode: {'incremental (fast)' if fast else 'full'}")
    print()

    if fast:
        from .incremental import IncrementalTracker
        tracker = IncrementalTracker(config, clone_dir=clone_dir)
    else:
        tracker = HistoricalTracker(config, clone_dir=clone_dir, workers=workers)

    summary = {
        'started_at': datetime.now().isoformat(),
        'config': config,
        'repositories': []
    }

    for i, repo in enumerate(repos, 1):
        print(f"\n[{i}/{len(repos)}] {repo['name']}")
        print("=" * 50)

        try:
            history = tracker.track_repository(
                repo_name=repo['name'],
                repo_url=repo['url'],
                excluded_paths=excluded_paths,
                years=years
            )

            output_file = output_path / f"{repo['name']}_history.json"
            tracker.save_history(history, str(output_file))

            # Add to summary
            latest = history['snapshots'][-1] if history['snapshots'] else {}
            summary['repositories'].append({
                'name': repo['name'],
                'status': 'success',
                'snapshots': len(history['snapshots']),
                'latest_metrics': latest.get('metrics', {})
            })

        except Exception as e:
            print(f"Error analysing {repo['name']}: {e}")
            summary['repositories'].append({
                'name': repo['name'],
                'status': 'error',
                'error': str(e)
            })

    summary['completed_at'] = datetime.now().isoformat()

    # Save summary
    summary_file = output_path / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    return summary
