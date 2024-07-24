include("compute_optimal_xyz.jl")
include("sparse_matrix.jl")

"""
	Mg_preconditioner

	Contains all the data needed by the multi-grid preconditioner.

	# Arguments

	- `f2c`: mappings between the different levels of the preconditioner
	- `A_vec`: sparse matrices of each level of the preconditioner
	- `gs_states`: gauss seidel solver setup state for each level of the preconditioner
	- `r`: residual of each level of the preconditioner
	- `x`: initial guess of each level of the preconditioner
	- `Axf`: pre-allocation for the A*x of each level of the preconditioner
	- `l`: number of levels in the preconditioner

"""
mutable struct Mg_preconditioner{A, B, C, D, E, F, G}
	f2c::Vector{A}
	A_vec::Vector{B}
	gs_states::Vector{C}
	r::Vector{D}
	x::Vector{E}
	Axf::Vector{F}
	l::G

	function Mg_preconditioner(f2c, A_vec, gs_states, r, x, Axf, l)
		A = typeof(f2c[1])
		B = typeof(A_vec[1])
		C = typeof(gs_states[1])
		D = typeof(r[1])
		E = typeof(x[1])
		F = typeof(Axf[1])
		G = typeof(l)
		new{A, B, C, D, E, F, G}(f2c, A_vec, gs_states, r, x, Axf, l)
	end
end

"""
	Geometry

	Collect data about the geometry of the problem.

	# Arguments

	- `nx`: points in the x direction for each process
	- `ny`: points in the y direction for each process
	- `nz`: points in the z direction for each process
	- `npx`: parts in the x direction 
	- `npy`: parts in the y direction 
	- `npz`: parts in the z direction 
	- `nnz`: number of non zeroes in sparse matrices of the preconditioner
	- `nrows`: number of rows in each of the sparse matrices of the preconditioner
"""
struct Geometry
	nx::Int64
	ny::Int64
	nz::Int64
	npx::Int64
	npy::Int64
	npz::Int64
	nnz::Vector{Int64}
	nrows::Vector{Int64}
end

"""
	restrict_operator(nx, ny, nz) -> f2c

	Creates a mapping for vector x to a coarse vector x corresponding to a 
		mapping for a matrix to a matrix that is half the size in all directions.

	# Arguments

	- `nx`: points in the x direction for each process
	- `ny`: points in the y direction for each process
	- `nz`: points in the z direction for each process

	# Output

	- `f2c`: fine to coarse mapping vector.
"""
function restrict_operator(nx, ny, nz)
	@assert (nx % 2 == 0) && (ny % 2 == 0) && (nz % 2 == 0)
	nxc = div(nx, 2)
	nyc = div(ny, 2)
	nzc = div(nz, 2)
	local_number_of_rows = nxc * nyc * nzc
	f2c = zeros(Int32, local_number_of_rows)
	for izc in 1:nzc
		izf = 2 * (izc - 1)
		for iyc in 1:nyc
			iyf = 2 * (iyc - 1)
			for ixc in 1:nxc
				ixf = 2 * (ixc - 1)
				current_coarse_row = (izc - 1) * nxc * nyc + (iyc - 1) * nxc + (ixc - 1) + 1
				current_fine_row = izf * nx * ny + iyf * nx + ixf
				f2c[current_coarse_row] = current_fine_row + 1
			end
		end
	end
	return f2c
end


"""
	pc_setup(np, ranks, l, nx, ny, nz) -> Mg_preconditioner, Geometry

	Function initializes all the sparse matrices and vectors of the preconditioner,
	and collects data about the geometry for reporting.

	# Arguments

	- `np`: number of processes
	- `ranks`: distribute object of the processes
	- `l`: number of levels for the multi-grid preconditioner
	- `nx`: points in the x direction for each process
	- `ny`: points in the y direction for each process
	- `nz`: points in the z direction for each process

	# Output

	- `Mg_preconditioner`: struct containing preconditioner data
	- `Geometry`: struct containing geometry data
"""
function pc_setup(np, ranks, l, nx, ny, nz)
	A_vec = Vector{PSparseMatrix}(undef, l)
	f2c = Vector{Vector{Int32}}(undef, l - 1)
	r = Vector{PVector}(undef, l)
	x = Vector{PVector}(undef, l)
	Axf = Vector{PVector}(undef, l)
	gs_states = Vector{PartitionedSolvers.Preconditioner}(undef, l)
	npx, npy, npz = compute_optimal_shape_XYZ(np)
	nnz_vec = Vector{Int64}(undef, l)
	nrows_vec = Vector{Int64}(undef, l)
	solver = additive_schwarz(PartitionedSolvers.gauss_seidel(; iters = 1))
	tnx = nx
	tny = ny
	tnz = nz

	for i ∈ reverse(1:l)
		gnx = npx * nx
		gny = npy * ny
		gnz = npz * nz
		A_vec[i], r[i] = build_p_matrix(ranks, nx, ny, nz, gnx, gny, gnz, npx, npy, npz)

		x[i] = similar(r[i])
		Axf[i] = similar(r[i])
		Axf[i] .= 0
		x[i] .= 0
		gs_states[i] = setup(solver, x[i], A_vec[i], r[i])

		if i != 1
			f2c[i-1] = restrict_operator(nx, ny, nz)
		end
		nrows_vec[i] = size(A_vec[i], 1)
		nnz_vec[i] = PartitionedArrays.nnz(A_vec[i])
		nx = div(nx, 2)
		ny = div(ny, 2)
		nz = div(nz, 2)
	end
	Mg_preconditioner(f2c, A_vec, gs_states, r, x, Axf, l), Geometry(tnx, tny, tnz, npx, npy, npz, nnz_vec, nrows_vec)
end

"""
	LinearAlgebra.ldiv!(x, P::Mg_preconditioner, b) -> x

	Function called by the cg algorithm for the preconditioning,
		which in turn calls the preconditioner solver.

	# Arguments 
	- `P`: Mg_preconditioner state object
	- `b`: right-hand side
	- `x`: initial guess, updated in-place

	# Output

	- `x`: approximated solution.
"""
function LinearAlgebra.ldiv!(x, P::Mg_preconditioner, b)
	pc_solve!(x, P, b, P.l)
	x
end

"""
	restrict!(r_c, r_f, Axf, f2c) -> r_c

	Restrict vector r_f to r_c using the mapping in f2c and subtracts Axf.

	# Arguments

	- `r_c`: the coarse r vector, updated in-place
	- `r_f`: the fine x vector
	- `f2c`: vector containing the mapping from fine to coarse
	- `Axf`: vector containing A * x

	# Output

	- `r_c`: coarse residual.
"""
function restrict!(r_c, r_f, Axf, f2c)
	for (i, v) in pairs(f2c)
		r_c[i] = r_f[v] - Axf[v]
	end
	r_c
end

"""
	prolongate!(x_f, x_c, f2c) -> x_f

	Prolongate maps values from the coarse grid to the fine grid using the f2c mapping.

	# Arguments

	- `x_f`: the fine x vector, updated in-place
	- `x_c`: the coarse x vector
	- `f2c`: vector containing the mapping from fine to coarse

	# Output

	- `x_f`: fine approximated solution.
"""
function prolongate!(x_f, x_c, f2c)
	for (i, v) in pairs(f2c)
		x_f[v] += x_c[i]
	end
	x_f
end

"""
	p_restrict!(r_c, r_f, Axf, f2c) -> r_c

	Distributed restrict maps local values and calls sequential restrict!().

	# Arguments

	- `r_c`: the coarse r pvector, updated in-place
	- `r_f`: the fine x pvector
	- `f2c`: vector containing the mapping from fine to coarse
	- `Axf`: pvector containing A * x

	# Output

	- `r_c`: coarse residual.
"""
function p_restrict!(r_c, r_f, Axf, f2c)
	map(local_values(r_f), local_values(Axf), local_values(r_c)) do rf_local, Axf_local, rc_local
		restrict!(rc_local, rf_local, Axf_local, f2c)
	end
	r_c
end


"""
	p_prolongate!(x_f, x_c, f2c) -> x_f

	Distributed prolongate maps local values and calls sequential prolongate!().

	# Arguments

	- `x_f`: the fine x pvector, updated in-place
	- `x_c`: the coarse x pvector
	- `f2c`: vector containing the mapping from fine to coarse

	# Output

	- `x_f`: fine approximated solution.
"""
function p_prolongate!(x_f, x_c, f2c)
	map(local_values(x_f), local_values(x_c)) do xf_local, xc_local
		prolongate!(xf_local, xc_local, f2c)
	end
	x_f
end


"""
	pc_solve!(s, b, x, l) -> x

	# Arguments 

	- `s`: Mg_preconditioner state object
	- `b`: right-hand side
	- `x`: initial guess, updated in-place
	- `l`: levels of recursion

	# Output

	- `x`: approximated solution.
"""
function pc_solve!(x, s, b, l)
	x .= 0

	if l == 1
		solve!(x, s.gs_states[l], b) # bottom solve
	else
		solve!(x, s.gs_states[l], b) # presmooth
		mul!(s.Axf[l], s.A_vec[l], x)
		s.r[l-1] .= 0
		p_restrict!(s.r[l-1], b, s.Axf[l], s.f2c[l-1])
		pc_solve!(s.x[l-1], s, s.r[l-1], l - 1)
		p_prolongate!(x, s.x[l-1], s.f2c[l-1])
		solve!(x, s.gs_states[l], b) # post smooth
	end
	x
end

