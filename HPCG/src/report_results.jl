"""
	report_results(np, times, levels, ref_max_iters, opt_max_iters, nr_cg_sets, norm_data, geom) -> file output

	# Arguments 
	- `np`: number of processes.
	- `times`: reported times from all processes.
	- `levels`: levels of recursion of the preconditioner
	- `ref_max_iters`: max iterations of the reference algorithm.
	- `opt_max_iters`: max iterations of the optimised algorithm.
	- `nr_cg_sets`: number of times the optimised version is run.
	- `norm_data`: convergence data of all cg sets.
	- `geom`: struct contianing geometry data.

	# Output

	- file output.

"""
function report_results(np, times, levels, ref_max_iters, opt_max_iters, nr_cg_sets, norm_data, geom; output_type = "txt", output_folder = "results")
	fniters = nr_cg_sets * opt_max_iters
	fnrow = geom.nrows[levels]
	fnnz = geom.nnz[levels]
	mg_data = Dict()
	## FLOPS calculations
	fnops_ddot = (3.0 * fniters + nr_cg_sets) * 2.0 * fnrow # 3 ddots with nrow adds and nrow mults
	fnops_waxpby = (3.0 * fniters + nr_cg_sets) * 2.0 * fnrow # 3 WAXPBYs with nrow adds and nrow mults
	fnops_sparsemv = (fniters + nr_cg_sets) * 2.0 * fnnz # 1 SpMV with nnz adds and nnz mults

	fnops_precond = 0.0
	for i in 2:levels
		nr_non_zeros_level = geom.nnz[i]
		fnops_precond += fniters * 4.0 * nr_non_zeros_level # number of presmoother flops
		fnops_precond += fniters * 2.0 * nr_non_zeros_level # cost of fine grid residual calculation
		fnops_precond += fniters * 4.0 * nr_non_zeros_level  # number of postsmoother flops
	end
	fnops_precond += fniters * 4.0 * geom.nnz[1] # One symmetric GS sweep at the coarsest level
	fnops = fnops_ddot + fnops_waxpby + fnops_sparsemv + fnops_precond
	frefnops = fnops * (ref_max_iters / opt_max_iters)

	#======================== Memory bandwidth model =======================================#
	# Read/Write counts come from implementation of CG in CG.cpp (include 1 extra for the CG preamble ops)
	fnreads_ddot = (3.0 * fniters + nr_cg_sets) * 2.0 * fnrow * sizeof(Float64) # 3 ddots with 2 nrow reads
	fnwrites_ddot = (3.0 * fniters + nr_cg_sets) * sizeof(Float64) # 3 ddots with 1 write
	fnreads_waxpby = (3.0 * fniters + nr_cg_sets) * 2.0 * fnrow * sizeof(Float64) # 3 WAXPBYs with nrow adds and nrow mults
	fnwrites_waxpby = (3.0 * fniters + nr_cg_sets) * fnrow * sizeof(Float64) # 3 WAXPBYs with nrow adds and nrow mults
	fnreads_sparsemv = (fniters + nr_cg_sets) * (fnnz * (sizeof(Float64) + sizeof(Int64)) + fnrow * sizeof(Float64)) # 1 SpMV with nnz reads of values, nnz reads indices,
	# plus nrow reads of x
	fnwrites_sparsemv = (fniters + nr_cg_sets) * fnrow * sizeof(Float64) # 1 SpMV nrow writes
	# Op counts from the multigrid preconditioners
	fnreads_precond = 0.0
	fnwrites_precond = 0.0

	for i in 2:levels
		nr_non_zeros_level = geom.nnz[i]
		fnrow_level = geom.nrows[i]
		mg_data["level_$i"] = Dict("non_zeros" => nr_non_zeros_level, "nr_equations" => fnrow_level)
		fnreads_precond += 1 * fniters * (2.0 * nr_non_zeros_level * (sizeof(Float64) + sizeof(Int64)) + fnrow_level * sizeof(Float64)) # number of presmoother reads
		fnwrites_precond += 1 * fniters * nr_non_zeros_level * sizeof(Float64) # number of presmoother writes
		fnreads_precond += fniters * (nr_non_zeros_level * (sizeof(Float64) + sizeof(Int64)) + fnrow_level * sizeof(Float64)) # Number of reads for fine grid residual calculation
		fnwrites_precond += fniters * nr_non_zeros_level * sizeof(Float64) # Number of writes for fine grid residual calculation
		fnreads_precond += 1 * fniters * (2.0 * nr_non_zeros_level * (sizeof(Float64) + sizeof(Int64)) + fnrow_level * sizeof(Float64))  # number of postsmoother reads
		fnwrites_precond += 1 * fniters * nr_non_zeros_level * sizeof(Float64)  # number of postsmoother writes
	end

	fnnz_bottom = geom.nnz[1]
	fnrow_bottom = geom.nrows[1]
	mg_data["level_1"] = Dict("non_zeros" => fnnz_bottom, "nr_equations" => fnrow_bottom)
	fnreads_precond += fniters * (2.0 * fnnz_bottom * (sizeof(Float64) + sizeof(Int64)) + fnrow_bottom * sizeof(Float64)) # One symmetric GS sweep at the coarsest level
	fnwrites_precond += fniters * fnrow_bottom * sizeof(Float64) # One symmetric GS sweep at the coarsest level
	fnreads = fnreads_ddot + fnreads_waxpby + fnreads_sparsemv + fnreads_precond
	fnwrites = fnwrites_ddot + fnwrites_waxpby + fnwrites_sparsemv + fnwrites_precond
	frefnreads = fnreads * (ref_max_iters) / (opt_max_iters)
	frefnwrites = fnwrites * (ref_max_iters) / (opt_max_iters)

	# ======================== Memory usage model ======================================= #
	main_times = zeros(Float64, 10)
	mean_times = zeros(Float64, 10)
	max_times = zeros(Float64, 10)


	# max time of the process with the slowest total time.
	transposed_times = collect(eachrow(reduce(hcat, times)))
	max_times = times[argmax(transposed_times[10])]
	main_times = times[1]
	mean_times = mean(times, dims = 1)[1]

	function time_dict(times)
		return OrderedDict("setup" => times[10],
			"total" => times[1],
			"DDOT" => times[2],
			"WAXPBY" => times[3],
			"SPMV" => times[4],
			"allreduce" => times[5],
			"MG" => times[6],
			"halo_time" => times[7],
			"opt_time" => times[8],
			"ref_time" => times[9])
	end

	# Write results to file. 
	isdir(output_folder) || mkdir(output_folder)
	if output_type == "json"
		filename_json = output_folder * "/hpcg-benchmark_results" * Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * ".json"
		json_dict = OrderedDict()
		json_dict["procs"] = np
		json_dict["main_times"] = time_dict(main_times)
		json_dict["mean_times"] = time_dict(mean_times)
		json_dict["max_times"] = time_dict(max_times)
		json_dict["nr_equations"] = fnrow
		json_dict["non_zeors"] = fnnz
		json_dict["multigrid_data"] = mg_data
		json_dict["geometry"] = OrderedDict("npx" => geom.npx,
			"npy" => geom.npy,
			"npz" => geom.npz,
			"gnx" => geom.npx * geom.nx,
			"gny" => geom.npy * geom.ny,
			"gnz" => geom.npz * geom.nz,
			"nx" => geom.nx,
			"ny" => geom.ny,
			"nz" => geom.nz)
		json_dict["iter_data"] = OrderedDict("ref_iters_set" => ref_max_iters,
			"opt_iters_set" => opt_max_iters,
			"ref_iters_total" => ref_max_iters * nr_cg_sets,
			"opt_iters_total" => opt_max_iters * nr_cg_sets)
		json_dict["reproducibility_data"] = OrderedDict("mean" => Statistics.mean(norm_data), "var" => Statistics.var(norm_data))
		json_dict["flops"] = OrderedDict("DDOT" => fnops_ddot,
			"WAXPBY" => fnops_waxpby,
			"SpMV" => fnops_sparsemv,
			"MG" => fnops_precond,
			"Total" => fnops,
			"Total_conv" => frefnops)

		times = main_times
		json_dict["GB/s"] = OrderedDict("Read" => (fnreads / times[1] / 1.0E9),
			"Write" => (fnwrites / times[1] / 1.0E9),
			"Total" => ((fnreads + fnwrites) / (times[1]) / 1.0E9),
			"Total_conv_opt" => ((frefnreads + frefnwrites) / (times[1] + nr_cg_sets * (times[8] / 10.0 + times[10] / 10.0)) / 1.0E9))

		totalGflops = frefnops / (times[1] + nr_cg_sets * (times[8] / 10.0 + times[10] / 10.0)) / 1.0E9
		json_dict["GFLOP/s"] = OrderedDict("DDOT" => (fnops_ddot / times[2] / 1.0E9),
			"WAXPBY" => (fnops_waxpby / times[3] / 1.0E9),
			"SpMV" => (fnops_sparsemv / times[4] / 1.0E9),
			"MG" => (fnops_precond / times[6] / 1.0E9),
			"Total" => (fnops / times[1] / 1.0E9),
			"Total_conv" => (frefnops / times[1] / 1.0E9),
			"Total_conv_opt" => totalGflops)
		json_dict["Overview"] = OrderedDict("GFLOP/s" => totalGflops, "time" => times[1])

		open(filename_json, "w") do file
			JSON.print(file, json_dict)
		end
	elseif output_type == "txt"
		filename = output_folder * "/hpcg-benchmark_results" * Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * ".txt"
		open(filename, "w") do file
			println(file, "########## Problem Summary  ##########")
			println(file, "Number of Procs: ", np)
			println(file, "Main setup time: ", main_times[10])
			println(file, "Mean setup time: ", mean_times[10])
			println(file, "Max setup time: ", max_times[10])
			println(file, "Number of Equations: ", fnrow)
			println(file, "Number of Nonzero Terms: ", fnnz)
			println(file, "")

			println(file, "Problem geometry:")
			println(file, "npx: ", geom.npx)
			println(file, "npy: ", geom.npy)
			println(file, "npz: ", geom.npz)
			println(file, "gnx: ", geom.npx * geom.nx)
			println(file, "gny: ", geom.npy * geom.ny)
			println(file, "gnz: ", geom.npz * geom.nz)
			println(file, "nx: ", geom.nx)
			println(file, "ny: ", geom.ny)
			println(file, "nz: ", geom.nz)
			println(file, "")

			println(file, "########## Iterations Summary  ##########")
			println(file, "Reference CG iterations per set = ", ref_max_iters)
			println(file, "Optimized CG iterations per set = ", opt_max_iters)
			println(file, "Total number of reference iterations = ", ref_max_iters * nr_cg_sets)
			println(file, "Optimized CG iterations per set = ", opt_max_iters * nr_cg_sets)
			println(file, "")

			println(file, "########## Reproducibility Summary  ##########")
			println(file, "Scaled residual mean = ", Statistics.mean(norm_data))
			println(file, "Scaled residual variance = ", Statistics.var(norm_data))
			println(file, "")

			println(file, "########## Performance Summary (times in sec) ##########")

			println(file, "Benchmark Time Summary:")
			println(file, "Main times:")
			println(file, "Optimization phase = ", main_times[8])
			println(file, "DDOT = ", main_times[2])
			println(file, "WAXPBY = ", main_times[3])
			println(file, "SpMV = ", main_times[4])
			println(file, "MG = ", main_times[6])
			println(file, "Total = ", main_times[1])

			println(file, "")
			println(file, "Mean times:")
			println(file, "Optimization phase = ", mean_times[8])
			println(file, "DDOT = ", mean_times[2])
			println(file, "WAXPBY = ", mean_times[3])
			println(file, "SpMV = ", mean_times[4])
			println(file, "MG = ", mean_times[6])
			println(file, "Total = ", mean_times[1])

			println(file, "")
			println(file, "Max times:")
			println(file, "Optimization phase = ", max_times[8])
			println(file, "DDOT = ", max_times[2])
			println(file, "WAXPBY = ", max_times[3])
			println(file, "SpMV = ", max_times[4])
			println(file, "MG = ", max_times[6])
			println(file, "Total = ", max_times[1])

			println(file, "")
			println(file, "Floating Point Operations:")
			println(file, "Raw DDOT = ", fnops_ddot)
			println(file, "Raw WAXPBY = ", fnops_waxpby)
			println(file, "Raw SpMV = ", fnops_sparsemv)
			println(file, "Raw MG = ", fnops_precond)
			println(file, "Total = ", fnops)
			println(file, "Total with convergence overhead = ", frefnops)

			times = main_times
			println(file, "")
			println(file, "GB/s Summary:")
			println(file, "Raw Read B/W = ", fnreads / times[1] / 1.0E9)
			println(file, "Raw Write B/W = ", fnwrites / times[1] / 1.0E9)
			println(file, "Raw Total B/W = ", (fnreads + fnwrites) / (times[1]) / 1.0E9)
			println(file, "Total with convergence and optimization phase overhead = ", (frefnreads + frefnwrites) / (times[1] + nr_cg_sets * (times[8] / 10.0 + times[10] / 10.0)) / 1.0E9)

			println(file, "")
			println(file, "GFLOP/s Summary:")
			println(file, "Raw DDOT = ", fnops_ddot / times[2] / 1.0E9)
			println(file, "Raw WAXPBY = ", fnops_waxpby / times[3] / 1.0E9)
			println(file, "Raw SpMV = ", fnops_sparsemv / times[4] / 1.0E9)
			println(file, "Raw MG = ", fnops_precond / times[6] / 1.0E9)
			println(file, "Raw Total = ", fnops / times[1] / 1.0E9)
			println(file, "Total with convergence overhead =", frefnops / times[1] / 1.0E9)

			totalGflops = frefnops / (times[1] + nr_cg_sets * (times[8] / 10.0 + times[10] / 10.0)) / 1.0E9
			println(file, "Total with convergence and optimization phase overhead = ", totalGflops)

			println(file, "")
			println(file, "User Optimization Overheads:")
			println(file, "Optimization phase time (sec) = ", times[8])
			println(file, "Optimization phase time vs reference SpMV+MG time = ", times[8] / times[9])

			println(file, "")
			println(file, "HPCG result is VALID with a GFLOP/s rating of: ", totalGflops)
			#println(file, "GFLOP/s Summary::Total with convergence and optimization phase overhead", frefnops/timing_data[1])
			println(file, "Results are valid but execution time (sec) is ", times[1])
		end
	end
end
