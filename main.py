from SSCDAnalyser import SSCDAnalyser

analyser = SSCDAnalyser(minimum_block_size=150)

duplicates, blocks, results = analyser.analyze('ezEngine', excluded_paths=["/Documentation"])

analyser.save_results(results, "cpp-httplib.json")