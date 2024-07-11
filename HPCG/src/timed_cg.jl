"""
	Conjugate gradient solver from IterativeSolvers with benchmark timing added.
"""

import Base: iterate
using Printf
using PartitionedArrays

export cg, ref_cg!, PCGIterable, cg_iterator!, CGStateVariables

mutable struct PCGIterable{precT, matT, solT, vecT, numT <: Real, paramT <: Number}
	Pl::precT
	A::matT
	x::solT
	r::vecT
	c::vecT
	u::vecT
	tol::numT
	residual0::numT
	residual::numT
	ρ::paramT
	maxiter::Int
	timing_data::Vector{Float64}
end

@inline converged(it::Union{CGIterable, PCGIterable}) = it.residual / it.residual0 ≤ it.tol

@inline start(it::Union{CGIterable, PCGIterable}) = 0

@inline done(it::Union{CGIterable, PCGIterable}, iteration::Int) = iteration ≥ it.maxiter || converged(it)

#####################
# Preconditioned CG #
#####################

function iterate(it::PCGIterable, iteration::Int = start(it))
	# Check for termination first
	if done(it, iteration)
		return nothing
	end


	it.timing_data[1] += @elapsed begin # total time
		# Apply left preconditioner
		it.timing_data[6] += @elapsed ldiv!(it.c, it.Pl, it.r)

		ρ_prev = it.ρ

		it.timing_data[2] += @elapsed it.ρ = dot(it.c, it.r)

		# u := c + βu (almost an axpy)
		β = it.ρ / ρ_prev
		it.timing_data[3] += @elapsed it.u .= it.c .+ β .* it.u

		# c = A * u
		it.timing_data[4] += @elapsed mul!(it.c, it.A, it.u)
		it.timing_data[2] += @elapsed uc = dot(it.u, it.c)
		α = it.ρ / uc

		# Improve solution and residual
		it.timing_data[3] += @elapsed it.x .+= α .* it.u
		it.timing_data[3] += @elapsed it.r .-= α .* it.c

		it.timing_data[2] += @elapsed it.residual = norm(it.r)
	end
	# Return the residual at item and iteration number as state
	it.residual, iteration + 1
end

# Utility functions

"""
Intermediate CG state variables to be used inside cg and cg!. `u`, `r` and `c` should be of the same type as the solution of `cg` or `cg!`.
```
struct CGStateVariables{T,Tx<:AbstractArray{T}}
	u::Tx
	r::Tx
	c::Tx
end
```
"""
struct CGStateVariables{T, Tx <: AbstractArray{T}}
	u::Tx
	r::Tx
	c::Tx
end

function cg_iterator!(x, A, b, timing_data, Pl = Identity();
	tolerance::Float64 = 0.0,
	maxiter::Int = size(A, 2),
	statevars::CGStateVariables = CGStateVariables(zero(x), similar(x), similar(x)))
	u = statevars.u
	r = statevars.r
	c = statevars.c
	u .= zero(eltype(x))
	copyto!(r, b)

	# Compute r with an MV-product.
	mul!(c, A, x)
	r .-= c
	residual = norm(r)

	return PCGIterable(Pl, A, x, r, c, u,
		tolerance, residual, residual, one(eltype(x)),
		maxiter, timing_data,
	)
end

"""
	cg(A, b; kwargs...) -> x, [history]

Same as [`cg!`](@ref), but allocates a solution vector `x` initialized with zeros.
"""
cg(A, b; kwargs...) = cg!(zerox(A, b), A, b; initially_zero = true, kwargs...)

"""
	cg!(x, A, b; kwargs...) -> x, [history]

# Arguments

- `x`: Initial guess, will be updated in-place;
- `A`: linear operator;
- `b`: right-hand side.

## Keywords

- `statevars::CGStateVariables`: Has 3 arrays similar to `x` to hold intermediate results;
- `initially_zero::Bool`: If `true` assumes that `iszero(x)` so that one
  matrix-vector product can be saved when computing the initial
  residual vector;
- `Pl = Identity()`: left preconditioner of the method. Should be symmetric,
  positive-definite like `A`;
- `abstol::Real = zero(real(eltype(b)))`,
  `reltol::Real = sqrt(eps(real(eltype(b))))`: absolute and relative
  tolerance for the stopping condition
  `|r_k| ≤ max(reltol * |r_0|, abstol)`, where `r_k ≈ A * x_k - b`
  is approximately the residual in the `k`th iteration.
  !!! note
	  The true residual norm is never explicitly computed during the iterations
	  for performance reasons; it may accumulate rounding errors.
- `maxiter::Int = size(A,2)`: maximum number of iterations;
- `verbose::Bool = false`: print method information;
- `log::Bool = false`: keep track of the residual norm in each iteration.

# Output

**if `log` is `false`**

- `x`: approximated solution.

**if `log` is `true`**

- `x`: approximated solution.
- `ch`: convergence history.

**ConvergenceHistory keys**

- `:tol` => `::Real`: stopping tolerance.
- `:resnom` => `::Vector`: residual norm at each iteration.
"""
function ref_cg!(x, A, b, timing_data;
	tolerance::Float64 = 0.0,
	maxiter::Int = size(A, 2),
	statevars::CGStateVariables = CGStateVariables(zero(x), similar(x), similar(x)),
	Pl = Identity(),
	kwargs...)

	# Actually perform CG
	iterable = cg_iterator!(x, A, b, timing_data, Pl; tolerance = tolerance, maxiter = maxiter,
		statevars = statevars, kwargs...)
	iters = 0
	for (iteration, item) ∈ enumerate(iterable)
		iters += 1
	end

	iterable.x, iterable.timing_data, iterable.residual0, iterable.residual, iters
end
