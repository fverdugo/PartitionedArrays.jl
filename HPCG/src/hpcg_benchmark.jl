"""
    hpcg_benchmark(distribute, np, nx, ny, nz; total_runtime = 60, output_type = "txt", output_folder = "results") -> output to file

    High performance congjugate gradient benchmark. 

    Consists of 3 phases 
        - Reference phase: get tolerance of reference algorithm after 50 iterations.
        - Optimisation phase: run optimised version until refrence tolerance is achieved.
        - Measuring phase: run the optimised version multiple times until the set total runtime.

    # Arguments

    - `distribute`: method of distribution (mpi or debug).
    - `np`: number of processes.
    - `nx`: points in the x direction for each process.
    - `ny`: points in the y direction for each process.
    - `nz`: points in the z direction for each process.
    - `total_runtime`: desired total runtime (official time requirement is 1800).
    - `output_type`: output results to txt or json.
    - `output_folder`: location of output.

    # Output

    - file output.
"""
function hpcg_benchmark(distribute, np, nx, ny, nz; total_runtime = 60, output_type = "txt", output_folder = "results")
    ranks = distribute(LinearIndices((np,)))
    timing_data = zeros(Float64, 10)
    ref_timing_data = zeros(Float64, 10)
    opt_timing_data = zeros(Float64, 10)
    ref_max_iters = 50

    timing_data[10] = @elapsed begin # CG setup time
        l = 4
        S, geom = pc_setup(np, ranks, l, nx, ny, nz)
        x = similar(S.x[l])
        b = S.r[l]
    end

    ### Reference CG Timing Phase ###
    nr_of_cg_sets = 2
    totalNiters_ref = 0
    statevars = CGStateVariables(zero(x), similar(x), similar(x))
    iters = 0
    normr0 = 0
    normr = 0
    for i in 1:nr_of_cg_sets
        x .= 0
        x, ref_timing_data, normr0, normr, iters = ref_cg!(x, S.A_vec[l], b, ref_timing_data, maxiter = ref_max_iters, tolerance = 0.0, Pl = S, statevars = statevars)
        totalNiters_ref += iters
    end

    ref_tol = normr / normr0

    ### Optimized CG Setup Phase ### (only relevant after optimising the algorithm with potential convergence loss)
    opt_max_iters = 10 * ref_max_iters
    opt_worst_time = 0.0
    opt_n_iters = ref_max_iters
    for i in 1:nr_of_cg_sets
        last_cummulative_time = opt_timing_data[1]
        x .= 0
        x, opt_timing_data, normr0, normr, iters = opt_cg!(x, S.A_vec[l], b, opt_timing_data, maxiter = opt_max_iters, tolerance = ref_tol, Pl = S, statevars = statevars) # Change ref_cg calls below to own optimised version.

        if iters > opt_n_iters # take largest number of iterations to guarantee convergence.
            opt_n_iters = iters
        end

        current_time = opt_timing_data[1] - last_cummulative_time
        if current_time > opt_worst_time # Save worst time.
            opt_worst_time = current_time
        end
    end

    # All reduce for worst time
    r = reduction(max, map(rank -> opt_worst_time, ranks); destination = :all)
    map(r) do r
        opt_worst_time = r
    end

    ### Optimized CG Timing Phase ###
    nr_of_cg_sets = Int64(div(total_runtime, opt_worst_time, RoundUp))
    opt_tolerance = 0.0
    norm_data = zeros(Float64, nr_of_cg_sets)
    for i in 1:nr_of_cg_sets
        x .= 0
        x, timing_data, normr0, normr, iters = opt_cg!(x, S.A_vec[l], b, timing_data, maxiter = opt_n_iters, tolerance = opt_tolerance, Pl = S, statevars = statevars)
        norm_data[i] = normr / normr0
    end

    # collect all timing data from procs.
    timing_data_buf = gather(map(rank -> timing_data, ranks); destination = MAIN)
    all_timing_data = zeros(Float64, (4, 10))
    map_main(timing_data_buf) do t
        all_timing_data = t
    end

    map_main(ranks) do _
        report_results(np, all_timing_data, l, ref_max_iters, opt_n_iters, nr_of_cg_sets, norm_data, geom, output_type = output_type, output_folder = output_folder)
    end
end

"""
    hpcg_benchmark_mpi(np, nx, ny, nz; total_runtime = 60, output_type = "txt", output_folder = "results") -> output to file

    Run the benchmark using MPI.

    # Arguments

    - `np`: number of processes
    - `nx`: points in the x direction for each process
    - `ny`: points in the y direction for each process
    - `nz`: points in the z direction for each process
    - `total_runtime`: desired total runtime (official requirement is 1800)
    - `output_type`: output results to txt or json.
    - `output_folder`: location of output.

    # Output

    - file output.
"""
function hpcg_benchmark_mpi(np, nx, ny, nz; total_runtime = 60, output_type = "txt", output_folder = "results")
    with_mpi() do distribute
        hpcg_benchmark(distribute, np, nx, ny, nz, total_runtime = total_runtime, output_type = output_type, output_folder = output_folder)
    end
end

"""
    hpcg_benchmark_debug(np, nx, ny, nz; total_runtime = 60, output_type = "txt", output_folder = "results") -> output to file

    Run the benchmark using debug array.

    # Arguments

    - `np`: number of processes
    - `nx`: points in the x direction for each process
    - `ny`: points in the y direction for each process
    - `nz`: points in the z direction for each process
    - `total_runtime`: desired total runtime (official requirement is 1800)
    - `output_type`: output results to txt or json.
    - `output_folder`: location of output.

    # Output

    - file output.
"""
function hpcg_benchmark_debug(np, nx, ny, nz; total_runtime = 60, output_type = "txt", output_folder = "results")
    with_debug() do distribute
        hpcg_benchmark(distribute, np, nx, ny, nz, total_runtime = total_runtime, output_type = output_type, output_folder = output_folder)
    end
end
