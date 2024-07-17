using PartitionedArrays
using PartitionedSolvers
using LinearAlgebra
using Test
using SparseArrays
using IterativeSolvers
using BenchmarkTools
using DelimitedFiles

include("cg.jl")
include("report_results.jl")

struct Mg_preconditioner
	f2c::Vector{Vector{Int64}}
	A_vec::Vector{SparseMatrixCSC}
	b::Vector{Float64}
end

# Build sparse matrix.
function build_matrix(nx, ny, nz)
	row_count = nx * ny * nz

	@assert row_count > 0

	non_zeros_per_row = 27
	b = zeros(Float64, row_count)

	vec_size = row_count * non_zeros_per_row
	col_vec = zeros(Int64, vec_size)
	row_vec = zeros(Int64, vec_size)
	val_vec = zeros(Float64, vec_size)

	current_vec_index = 0
	for iz in 1:nz
		for iy in 1:ny
			for ix in 1:nx
				current_row = ((iz - 1) * nx * ny + (iy - 1) * nx + (ix - 1)) + 1
				non_zeros_in_row_count = 0
				# for each value get 27 point stencil
				for sz in -1:1
					if iz + sz > 0 && iz + sz < nz + 1
						for sy in -1:1
							if iy + sy > 0 && iy + sy < ny + 1
								for sx in -1:1
									if ix + sx > 0 && ix + sx < nx + 1
										curcol = current_row + sz * nx * ny + sy * nx + sx
										current_vec_index += 1

										if curcol == current_row
											val_vec[current_vec_index] = 26.0
										else
											val_vec[current_vec_index] = -1.0
										end
										col_vec[current_vec_index] = curcol
										row_vec[current_vec_index] = current_row
										non_zeros_in_row_count += 1
									end
								end
							end
						end
					end
				end
				b[current_row] = 27.0 - non_zeros_in_row_count
			end
		end
	end
	A = sparse(view(col_vec, 1:current_vec_index), view(row_vec, 1:current_vec_index), view(val_vec, 1:current_vec_index))
	A, b
end

# Create a mapping between the fine and coarse levels.
function restrict_operator(nx, ny, nz)
	@assert (nx % 2 == 0) && (ny % 2 == 0) && (nz % 2 == 0)
	nxc = div(nx, 2)
	nyc = div(ny, 2)
	nzc = div(nz, 2)
	localNumberOfRows = nxc * nyc * nzc
	f2cOperator = zeros(Int64, localNumberOfRows)
	for izc in 1:nzc
		izf = 2 * (izc - 1)
		for iyc in 1:nyc
			iyf = 2 * (iyc - 1)
			for ixc in 1:nxc
				ixf = 2 * (ixc - 1)
				currentCoarseRow = (izc - 1) * nxc * nyc + (iyc - 1) * nxc + (ixc - 1) + 1
				currentFineRow = izf * nx * ny + iyf * nx + ixf
				f2cOperator[currentCoarseRow] = currentFineRow + 1
			end
		end
	end
	return f2cOperator
end

# Setup the preconditioner, create A and b for all levels.
function pc_setup(level, nx, ny, nz)
	row_count = nx * ny * nz
	A_vec = Vector{SparseMatrixCSC}(undef, level)
	f2c = Vector{Vector{Int64}}(undef, level - 1)
	b = zeros(Float64, row_count)
	for i in 1:level
		if i == 1
			A, b = build_matrix(nx, ny, nz)
		else
			A, _ = build_matrix(nx, ny, nz)
		end
		if i != level
			f2c[(level-i)] = restrict_operator(nx, ny, nz)
		end
		A_vec[(level-i)+1] = A
		nx = div(nx, 2)
		ny = div(ny, 2)
		nz = div(nz, 2)
	end
	Mg_preconditioner(f2c, A_vec, b)
end

# Calls the preconditioner with the required recursion level.
function pc_solve!(x, state::Mg_preconditioner, b)
	level = length(state.A_vec)
	multigrid_preconditioner!(state, b, x, level)
	x
end

# Function called by the cg algorithm for the preconditioning.
function LinearAlgebra.ldiv!(x, P::Mg_preconditioner, b)
	println(b[1])
	x .= 0
	pc_solve!(x, P, b)
	println(x[1])
	x
end

# Restrict vector and b - Ax
function restrict!(r_c, r_f, Axf, f2c)
	for (i, v) in pairs(f2c)
		@inbounds r_c[i] = r_f[v] - Axf[v]
	end
	r_c
end

# Prolongate by adding all the values using the f2c mapping. 
function prolongate!(x_f, x_c, f2c)
	for (i, v) in pairs(f2c)
		@inbounds x_f[v] += x_c[i]
	end
	x_f
end



# Recursive multigrid preconditioner.
function multigrid_preconditioner!(s, b, x, level)
	solver = PartitionedSolvers.gauss_seidel(; iters = 1)
	S = setup(solver, x, s.A_vec[level], b)

	if level == 1
		solve!(x, S, b) # bottom solve
	else
		solve!(x, S, b) # presmooth
		Axf = s.A_vec[level] * x
		r_c = zeros(Float64, length(s.f2c[level-1]))
		r_c = restrict!(r_c, b, Axf, s.f2c[level-1])
		x_c = similar(r_c)
		x_c .= 0
		multigrid_preconditioner!(s, r_c, x_c, level - 1)
		prolongate!(x, x_c, s.f2c[level-1])
		solve!(x, S, b) # post smooth
	end
	x
end



function HPCG_sequential(nx, ny, nz; total_runtime = 60)
	timing_data = zeros(Float64, 10)
	ref_timing_data = zeros(Float64, 10)
	opt_timing_data = zeros(Float64, 10)
	ref_max_iters = 50

	timing_data[10] = @elapsed begin # CG setup time
		levels = 4
		S = pc_setup(levels, nx, ny, nz)
		x = similar(S.b)
		x .= 0
	end

	### Reference CG Timing Phase ###
	nr_calls = 1
	totalNiters_ref = 0
	log = true
	history = ConvergenceHistory(partial = !log)
	for i in 1:nr_calls
		x .= 0
		x, history, ref_timing_data = ref_cg!(x, S.A_vec[levels], S.b, ref_timing_data, maxiter = ref_max_iters, abstol = 0.0, Pl = S, log = log)
		totalNiters_ref += history.iters
	end
	res_norm = history.data[:resnorm]
	ref_tol = last(res_norm) / res_norm[1]

	show(x[1:10])
	return

	### Optimized CG Setup Phase ### (only relevant after optimising the algorithm with potential convergence loss)
	# Change ref_cg calls below to own optimised version.

	opt_max_iters = 10 * ref_max_iters
	opt_worst_time = 0.0
	opt_n_iters = ref_max_iters
	for i in 1:nr_calls
		last_cummulative_time = opt_timing_data[1]
		x .= 0
		x, history, opt_timing_data = ref_cg!(x, S.A_vec[levels], S.b, opt_timing_data, maxiter = opt_max_iters, reltol = ref_tol, Pl = S, log = log)

		if history.iters > opt_n_iters # take largest number of iterations to guarantee convergence.
			opt_n_iters = history.iters
		end

		current_time = opt_timing_data[1] - last_cummulative_time
		if current_time > opt_worst_time # Save worst time.
			opt_worst_time = current_time
		end
	end

	### Optimized CG Timing Phase ###

	# Run the algorithm multiple times to get a high enough total runtime.
	nr_of_cg_sets = Int64(div(total_runtime, opt_worst_time, RoundUp))

	opt_tolerance = 0.0
	norm_data = zeros(Float64, nr_of_cg_sets)
	for i in 1:nr_of_cg_sets
		x .= 0
		x, history, timing_data = ref_cg!(x, S.A_vec[levels], S.b, timing_data, maxiter = opt_n_iters, reltol = opt_tolerance, Pl = S, log = log)
		res_norm = history.data[:resnorm]
		norm_data[i] = last(res_norm) / res_norm[1]
	end
	total_rows = nx * ny * nz
	report_results(1, timing_data, S, levels, ref_max_iters, opt_n_iters, nr_of_cg_sets, norm_data)
end

HPCG_sequential(32, 32, 16, total_runtime = 10)

# np = 1


# nx = 32
# ny = 32
# nz = 16
# levels = 4
# log = true
# S = pc_setup(levels, nx, ny, nz)
# x = similar(S.b)
# x .= 0
# #ref_timing_data = zeros(Float64, 10)


# pc_solve!(x, S, S.b)
# #x, history, ref_timing_data = ref_cg!(x, S.A_vec[levels], S.b, ref_timing_data, maxiter = 3, abstol = 0.0, log = log)
# x = collect(x)
# show(x[1:10])
