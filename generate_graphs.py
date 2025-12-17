#!/usr/bin/env python3
"""
Generate graphs showing clone type growth over time for each repository.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def load_history(json_path: Path) -> dict:
    """Load history JSON file."""
    with open(json_path) as f:
        return json.load(f)


def plot_repo_history(history: dict, output_path: Path):
    """Generate clone type growth graph for a repository."""
    repo_name = history.get('repo_name', 'Unknown')
    snapshots = history.get('snapshots', [])

    if not snapshots:
        print(f"  No snapshots for {repo_name}, skipping")
        return

    dates = []
    t1_counts = []
    t2_counts = []
    t3_counts = []
    t4_counts = []
    total_blocks = []

    for snap in snapshots:
        try:
            date = datetime.strptime(snap['date'], '%Y-%m-%d')
            metrics = snap.get('metrics', {})
            clone_types = metrics.get('clone_types', {})

            dates.append(date)
            t1_counts.append(clone_types.get('T1', 0))
            t2_counts.append(clone_types.get('T2', 0))
            t3_counts.append(clone_types.get('T3', 0))
            t4_counts.append(clone_types.get('T4', 0))
            total_blocks.append(metrics.get('total_blocks', 0))
        except (KeyError, ValueError) as e:
            continue

    if not dates:
        print(f"  No valid data for {repo_name}, skipping")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'{repo_name} - Code Clone Analysis Over Time', fontsize=14, fontweight='bold')

    ax1.stackplot(dates, t1_counts, t2_counts, t3_counts, t4_counts,
                  labels=['T1 (Exact)', 'T2 (Renamed)', 'T3 (Near-miss)', 'T4 (Semantic)'],
                  colors=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'],
                  alpha=0.8)
    ax1.set_ylabel('Number of Clone Pairs')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Clone Types Over Time')

    ax2_twin = ax2.twinx()

    ax2.plot(dates, total_blocks, 'b-', linewidth=2, label='Total Functions')
    ax2.set_ylabel('Total Functions', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    dup_ratios = []
    for snap in snapshots:
        metrics = snap.get('metrics', {})
        ratio = metrics.get('duplication_ratio', 0) * 100
        dup_ratios.append(ratio)

    if len(dup_ratios) == len(dates):
        ax2_twin.plot(dates, dup_ratios, 'r--', linewidth=2, label='Duplication %')
        ax2_twin.set_ylabel('Duplication Ratio (%)', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        ax2_twin.set_ylim(0, max(max(dup_ratios) * 1.2, 10))

    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Codebase Size and Duplication Ratio')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    results_dir = Path('results')
    if not results_dir.exists():
        print("Error: results/ directory not found")
        sys.exit(1)

    history_files = list(results_dir.glob('*_history.json'))

    if not history_files:
        print("No history files found in results/")
        sys.exit(1)

    print(f"Found {len(history_files)} history files")
    print()

    for json_path in sorted(history_files):
        repo_name = json_path.stem.replace('_history', '')
        print(f"Processing {repo_name}...")

        history = load_history(json_path)
        output_path = results_dir / f"{repo_name}_clones.png"
        plot_repo_history(history, output_path)

    print()
    print("Done! Graphs saved to results/")


if __name__ == '__main__':
    main()
