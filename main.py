"""
Example usage of Cluain for codebase analysis.
"""

from cluain import Analyser
import json

if __name__ == "__main__":
    analyser = Analyser(
        threshold=0.95,
        minimum_block_size=150,
        device='cpu',
        max_threads=4,
        encode_batch_size=32,
        similarity_batch_size=500
    )

    result = analyser.analyse(
        root_dir="./repos/spdlog",
        excluded_paths=["/tests", "/bench", "/example"]
    )

    with open("analysis_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Found {result['metrics']['duplicate_pairs']} duplicate pairs")
    print(f"Clone types: {result['metrics']['clone_types']}")
