from SSCDAnalyser import SSCDAnalyser

analyser = SSCDAnalyser()

duplicates, blocks, results = analyser.analyze('cpp-httplib')

analyser.save_results(results, "cpp-httplib.json")