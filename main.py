from SSCDAnalyser import SSCDAnalyser
from HistoricalTracker import HistoricalTracker

if __name__ == "__main__":
    analyser = SSCDAnalyser(
        minimum_block_size=150,
        max_threads=4,
        encode_batch_size=32,
        similarity_batch_size=500,
        device='cpu'
    )
    tracker = HistoricalTracker(analyser, clone_dir="./repos", workers=2)

    history = tracker.track_repository(
        repo_name="seasocks",
        repo_url="github.com/mattgodbolt/seasocks",
        excluded_paths=["/tests", "/bench", "/example"],
        years=5
    )
    tracker.save_history(history, "spdlog_history.json")
