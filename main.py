from SSCDAnalyser import SSCDAnalyser

analyser = SSCDAnalyser()

duplicates, blocks, results = analyser.analyze('cpp-httplib', excluded_paths=["/test", "/example"])

analyser.save_results(results, "cpp-httplib.json")