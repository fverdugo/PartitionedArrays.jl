
# test sequential processing
HPCG_benchmark(1, 104, 104, 104, total_runtime = 10, report_results = false)

# test parallel processing
HPCG_benchmark(4, 16, 16, 16, total_runtime = 10, report_results = false)