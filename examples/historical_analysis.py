"""
Example: Track code duplication history over time.
"""

from cluain import HistoricalTracker

if __name__ == "__main__":
    # Configuration for the analyser
    config = {
        'threshold': 0.95,
        'minimum_block_size': 150,
        'device': 'cpu',
        'max_threads': 4,
        'encode_batch_size': 32,
        'similarity_batch_size': 500
    }

    tracker = HistoricalTracker(
        config=config,
        clone_dir="./repos",
        workers=2  # Parallel workers (adjust based on RAM)
    )

    history = tracker.track_repository(
        repo_name="spdlog",
        repo_url="github.com/gabime/spdlog",
        excluded_paths=["/tests", "/bench", "/example"],
        years=3
    )

    tracker.save_history(history, "spdlog_history.json")

    # Print summary
    print("\n=== Summary ===")
    for snapshot in history['snapshots'][-5:]:
        metrics = snapshot['metrics']
        print(f"{snapshot['month']}: {metrics.get('duplicate_pairs', 0)} duplicates, "
              f"ratio: {metrics.get('duplication_ratio', 0):.2%}")
