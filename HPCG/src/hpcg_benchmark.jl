using PartitionedArrays
using PartitionedSolvers
using LinearAlgebra
using Test
using SparseArrays
using IterativeSolvers
using BenchmarkTools
using DelimitedFiles

include("timed_cg.jl")
include("report_results.jl")
include("compute_optimal_xyz.jl")

struct Mg_preconditioner
	f2c::Vector{Vector{Int64}}
	A_vec::Vector{PSparseMatrix}
	gs_states::Vector{Any}
	r::Vector{PVector}
	x::Vector{PVector}
	Axf::Vector{PVector}
	levels::Int64
end

struct Mg_preconditioner_seq
	f2c::Vector{Vector{Int64}}
	A_vec::Vector{SparseMatrixCSC}
	gs_states::Vector{Any}
	r::Vector{Vector}
	x::Vector{Vector}
	Axf::Vector{Vector}
	levels::Int64
end

mutable struct Geometry
	nx::Int64
	ny::Int64
	nz::Int64
	npx::Int64
	npy::Int64
	npz::Int64
	nnz::Vector{Int64}
	nrows::Vector{Int64}
end

function build_pmatrix(nx, ny, nz, gnx, gny, gnz, gix0, giy0, giz0)
	row_count = nx * ny * nz

	@assert row_count > 0

	non_zeros_per_row = 27
	b = zeros(Float64, row_count)
	row_b = zeros(Int64, row_count)

	col_vec = zeros(Int64, row_count * non_zeros_per_row)
	row_vec = zeros(Int64, row_count * non_zeros_per_row)
	val_vec = zeros(Float64, row_count * non_zeros_per_row)

	current_vec_index = 0
	for iz in 1:nz
		giz = giz0 + iz - 1
		for iy in 1:ny
			giy = giy0 + iy - 1
			for ix in 1:nx
				gix = gix0 + ix - 1
				current_row = ((iz - 1) * nx * ny + (iy - 1) * nx + (ix - 1)) + 1
				current_global_row = (giz - 1) * gnx * gny + (giy - 1) * gnx + (gix - 1) + 1
				non_zeros_in_row_count = 0
				# for each value get 27 point stencil
				for sz in -1:1
					if giz + sz > 0 && giz + sz < gnz + 1
						for sy in -1:1
							if giy + sy > 0 && giy + sy < gny + 1
								for sx in -1:1
									if gix + sx > 0 && gix + sx < gnx + 1
										curcol = current_global_row + sz * gnx * gny + sy * gnx + sx
										current_vec_index += 1

										if curcol == current_global_row
											val_vec[current_vec_index] = 26.0
										else
											val_vec[current_vec_index] = -1.0
										end
										col_vec[current_vec_index] = curcol
										row_vec[current_vec_index] = current_global_row
										non_zeros_in_row_count += 1
									end
								end
							end
						end
					end
				end
				row_b[current_row] = current_global_row
				b[current_row] = 27.0 - non_zeros_in_row_count
			end
		end
	end
	first(col_vec, current_vec_index), first(row_vec, current_vec_index), first(val_vec, current_vec_index), b, row_b
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
function pc_setup(np, ranks, level, nx, ny, nz)
	row_count = nx * ny * nz
	A_vec = Vector{PSparseMatrix}(undef, level)
	f2c = Vector{Vector{Int64}}(undef, level - 1)
	r = Vector{PVector}(undef, level)
	x = Vector{PVector}(undef, level)
	gs_states = Vector{Any}(undef, level)
	npx, npy, npz = compute_optimal_shape_XYZ(np)
	nnz_vec = Vector{Int64}(undef, level)
	nrows_vec = Vector{Int64}(undef, level)
	solver = additive_schwarz(PartitionedSolvers.gauss_seidel(; iters = 1))
	tnx = nx
	tny = ny
	tnz = nz

	for i ∈ reverse(1:level)
		gnx = npx * tnx
		gny = npy * tny
		gnz = npz * tnz
		row_partition = uniform_partition(ranks, (npx, npy, npz), (gnx, gny, gnz))
		cis = CartesianIndices((gnx, gny, gnz))
		#show((npx, npy, npz, gnx, gny, gnz))
		IJVb = map(row_partition) do my_rows
			gix0, giy0, giz0 = Tuple(cis[first(my_rows)])
			I, J, V, b, I_b = build_pmatrix(tnx, tny, tnz, gnx, gny, gnz, gix0, giy0, giz0)
			I, J, V, b, I_b
		end
		I, J, V, b, I_b = tuple_of_arrays(IJVb)

		col_partition = row_partition
		A = psparse(I, J, V, row_partition, col_partition) |> fetch

		row_partition = partition(axes(A, 2))
		b = pvector(I_b, b, row_partition) |> fetch
		consistent!(b) |> wait
		r[i] = b
		x[i] = pzeros(Float64, row_partition)

		A_vec[i] = A
		gs_states[i] = setup(solver, x[i], A_vec[i], b)

		if i != 1
			f2c[i-1] = restrict_operator(tnx, tny, tnz)
		end
		nrows_vec[i] = size(A, 1)
		nnz_vec[i] = nnz(A)
		tnx = div(tnx, 2)
		tny = div(tny, 2)
		tnz = div(tnz, 2)
	end
	Axf = similar(x)
	Mg_preconditioner(f2c, A_vec, gs_states, r, x, Axf, level), Geometry(nx, ny, nz, npx, npy, npz, nnz_vec, nrows_vec)
end

# Calls the preconditioner with the required recursion level.
function pc_solve!(x, state::Mg_preconditioner, b)
	multigrid_preconditioner!(state, b, x, state.levels)
	x
end

# Function called by the cg algorithm for the preconditioning.
function LinearAlgebra.ldiv!(x, P::Mg_preconditioner, b)
	pc_solve!(x, P, b)
	x
end

# Restrict vector and b - Ax
function restrict!(r_c, r_f, Axf, f2c)
	for (i, v) in pairs(f2c)
		r_c[i] = r_f[v] - Axf[v]
	end
	r_c
end

# Prolongate by adding all the values using the f2c mapping. 
function prolongate!(x_f, x_c, f2c)
	for (i, v) in pairs(f2c)
		x_f[v] += x_c[i]
	end
	x_f
end

# Restrict vector and b - Ax
function p_restrict!(r_c, r_f, Axf, f2c)
	map(local_values(r_f), local_values(Axf), local_values(r_c)) do rf_local, Axf_local, rc_local
		restrict!(rc_local, rf_local, Axf_local, f2c)
	end
	r_c
end

# Prolongate by adding all the values using the f2c mapping. 
function p_prolongate!(x_f, x_c, f2c)
	map(local_values(x_f), local_values(x_c)) do xf_local, xc_local
		prolongate!(xf_local, xc_local, f2c)
	end
	x_f
end

# # Recursive multigrid preconditioner.
function multigrid_preconditioner!(s, b, x, level)
	x .= 0

	if level == 1
		solve!(x, s.gs_states[level], b) # bottom solve
	else
		solve!(x, s.gs_states[level], b) # presmooth
		s.Axf[level] = s.A_vec[level] * x
		s.r[level-1] .= 0
		p_restrict!(s.r[level-1], b, s.Axf[level], s.f2c[level-1])
		multigrid_preconditioner!(s, s.r[level-1], s.x[level-1], level - 1)
		p_prolongate!(x, s.x[level-1], s.f2c[level-1])
		solve!(x, s.gs_states[level], b) # post smooth
	end
	x
end

function HPCG_parallel(distribute, np, nx, ny, nz; total_runtime = 60)
	ranks = distribute(LinearIndices((np,)))

	timing_data = zeros(Float64, 10)
	ref_timing_data = zeros(Float64, 10)
	opt_timing_data = zeros(Float64, 10)
	ref_max_iters = 50

	timing_data[10] = @elapsed begin # CG setup time
		levels = 4
		S, geom = pc_setup(np, ranks, levels, nx, ny, nz)
		x = similar(S.x[levels])
		b = S.r[levels]
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
		@time x, ref_timing_data, normr0, normr, iters = ref_cg!(x, S.A_vec[levels], b, ref_timing_data, maxiter = ref_max_iters, tolerance = 0.0, Pl = S, statevars = statevars)
		totalNiters_ref += iters
	end
	ref_tol = normr / normr0
	### Optimized CG Setup Phase ### (only relevant after optimising the algorithm with potential convergence loss)
	# Change ref_cg calls below to own optimised version.
	opt_max_iters = 10 * ref_max_iters
	opt_worst_time = 0.0
	iters = 0
	normr0 = 0
	normr = 0
	opt_n_iters = ref_max_iters
	for i in 1:nr_of_cg_sets
		last_cummulative_time = opt_timing_data[1]
		x .= 0
		x, opt_timing_data, normr0, normr, iters = ref_cg!(x, S.A_vec[levels], b, opt_timing_data, maxiter = opt_max_iters, tolerance = ref_tol, Pl = S, statevars = statevars)

		if iters > opt_n_iters # take largest number of iterations to guarantee convergence.
			opt_n_iters = iters
		end

		current_time = opt_timing_data[1] - last_cummulative_time
		if current_time > opt_worst_time # Save worst time.
			opt_worst_time = current_time
		end
	end

	# all reduce for worst time
	r = reduction(max, map(rank -> opt_worst_time, ranks); destination = :all)
	map(r) do r
		opt_worst_time = r
	end

	### Optimized CG Timing Phase ###

	# Run the algorithm multiple times to get a high enough total runtime.
	nr_of_cg_sets = Int64(div(total_runtime, opt_worst_time, RoundUp))
	iters = 0
	normr0 = 0
	normr = 0
	opt_tolerance = 0.0
	norm_data = zeros(Float64, nr_of_cg_sets)
	for i in 1:nr_of_cg_sets
		x .= 0
		x, timing_data, normr0, normr, iters = ref_cg!(x, S.A_vec[levels], b, timing_data, maxiter = opt_n_iters, tolerance = opt_tolerance, Pl = S, statevars = statevars)
		norm_data[i] = normr / normr0
	end

	map_main(ranks) do _
		report_results(np, timing_data, levels, ref_max_iters, opt_n_iters, nr_of_cg_sets, norm_data, geom)
	end
end

function hpcg_benchmark()
	with_mpi() do distribute
		HPCG_parallel(distribute, 4, 104, 104, 104, total_runtime = 60)
	end
end

# with_debug() do distribute
# 	HPCG_parallel(distribute, 4, 16, 16, 16, total_runtime = 10)
# end

hpcg_benchmark()
