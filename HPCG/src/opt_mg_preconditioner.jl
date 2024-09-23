

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
struct opt_Mg_preconditioner{A, B, C, D, E, F, G}
	f2c::Vector{A}
	A_vec::Vector{B}
	gs_states::Vector{C}
	r::Vector{D}
	x::Vector{E}
	Axf::Vector{F}
	l::G

	function opt_Mg_preconditioner(f2c, A_vec, gs_states, r, x, Axf, l)
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

function opt_generate_problem(ranks, npx, npy, npz, nx, ny, nz, solver)
	gnx = npx * nx
	gny = npy * ny
	gnz = npz * nz
	A, r = build_p_matrix(ranks, nx, ny, nz, gnx, gny, gnz, npx, npy, npz)
	x = similar(r)
	Axf = similar(r)
	Axf .= 0
	x .= 0
	gs_state = setup(solver, x, A, r)
	return A, r, x, Axf, gs_state
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
function opt_pc_setup(np, ranks, l, nx, ny, nz)
	f2c_vec = Vector{Vector{Int32}}(undef, l - 1)
	r = Vector{PVector}(undef, l)
	x = Vector{PVector}(undef, l)
	Axf = Vector{PVector}(undef, l)
	gs_states = Vector{PartitionedSolvers.Preconditioner}(undef, l)
	npx, npy, npz = compute_optimal_shape_XYZ(np)
	nnz_vec = Vector{Int64}(undef, l)
	nrows_vec = Vector{Int64}(undef, l)
	solver = PartitionedSolvers.additive_schwarz_correction_2(gauss_seidel(; iters = 1))

	# create top problem 
	A, r, x, Axf, gs_state = opt_generate_problem(ranks, npx, npy, npz, nx, ny, nz, solver)
	A_vec = Vector{typeof(A)}(undef, l)
	r_vec = Vector{typeof(r)}(undef, l)
	x_vec = Vector{typeof(r)}(undef, l)
	Axf_vec = Vector{typeof(r)}(undef, l)
	gs_states = Vector{typeof(gs_state)}(undef, l)

	A_vec[l] = A
	r_vec[l] = r
	x_vec[l] = x
	Axf_vec[l] = Axf
	gs_states[l] = gs_state
	nrows_vec[l] = size(A, 1)
	nnz_vec[l] = PartitionedArrays.nnz(A)

	tnx = nx
	tny = ny
	tnz = nz

	# create lower levels
	for i ∈ reverse(1:l-1)
		f2c_vec[i] = restrict_operator(nx, ny, nz)
		nx = div(nx, 2)
		ny = div(ny, 2)
		nz = div(nz, 2)
		A, r, x, Axf, gs_state = opt_generate_problem(ranks, npx, npy, npz, nx, ny, nz, solver)
		A_vec[i] = A
		r_vec[i] = r
		x_vec[i] = x
		Axf_vec[i] = Axf
		gs_states[i] = gs_state
		nrows_vec[i] = size(A, 1)
		nnz_vec[i] = PartitionedArrays.nnz(A)

	end
	opt_Mg_preconditioner(f2c_vec, A_vec, gs_states, r_vec, x_vec, Axf_vec, l), Geometry(tnx, tny, tnz, npx, npy, npz, nnz_vec, nrows_vec)
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
function LinearAlgebra.ldiv!(x, P::opt_Mg_preconditioner, b)
	fill!(x, zero(eltype(x)))
	opt_pc_solve!(x, P, b, P.l; zero_guess = true)
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
function opt_restrict!(r_c, r_f, Axf, f2c)
	for (i, v) in collect(enumerate(f2c))
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
function opt_prolongate!(x_f, x_c, f2c)
	for (i, v) in collect(enumerate(f2c))
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
function opt_p_restrict!(r_c, r_f, Axf, f2c)
	map(local_values(r_f), local_values(Axf), local_values(r_c)) do rf_local, Axf_local, rc_local
		opt_restrict!(rc_local, rf_local, Axf_local, f2c)
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
function opt_p_prolongate!(x_f, x_c, f2c)
	map(local_values(x_f), local_values(x_c)) do xf_local, xc_local
		opt_prolongate!(xf_local, xc_local, f2c)
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
function opt_pc_solve!(x, s::opt_Mg_preconditioner, b, l; zero_guess = false)
	if l == 1
		solve!(x, s.gs_states[l], b; zero_guess) # bottom solve
	else
		solve!(x, s.gs_states[l], b; zero_guess) # presmoother 
		mul!(s.Axf[l], s.A_vec[l], x)
		opt_p_restrict!(s.r[l-1], b, s.Axf[l], s.f2c[l-1])
		s.x[l-1] .= 0.0
		opt_pc_solve!(s.x[l-1], s, s.r[l-1], l - 1; zero_guess = true)
		opt_p_prolongate!(x, s.x[l-1], s.f2c[l-1])
		consistent!(x) |> wait
		solve!(x, s.gs_states[l], b) # post smooth
	end
	x
end
