"""
    Conjugate gradient solver from IterativeSolvers with benchmark timing integrated.
"""
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

@inline converged(it::PCGIterable) = it.residual / it.residual0 ≤ it.tol

@inline start(it::PCGIterable) = 0

@inline done(it::PCGIterable, iteration::Int) = iteration ≥ it.maxiter || converged(it)

"""
    Intermediate CG state variables to be used inside cg and cg!. 
    `u`, `r` and `c` should be of the same type as the solution of `cg` or `cg!`.
    Used in opt_cg and ref_cg.
"""
struct CGStateVariables{T, Tx <: AbstractArray{T}}
    u::Tx
    r::Tx
    c::Tx
end

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
        it.timing_data[4] += @elapsed mul_no_lat!(it.c, it.A, it.u)
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
    ref_cg!(x, A, b; kwargs...) -> x

# Arguments

- `x`: Initial guess, will be updated in-place;
- `A`: linear operator;
- `b`: right-hand side.

## Keywords

- `statevars::CGStateVariables`: Has 3 arrays similar to `x` to hold intermediate results;
- `Pl = Identity()`: left preconditioner of the method. Should be symmetric,
  positive-definite like `A`;
- `maxiter::Int = size(A,2)`: maximum number of iterations;

# Output

- `x`: approximated solution.


"""
function ref_cg!(x, A, b, timing_data;
    tolerance::Float64 = 0.0,
    maxiter::Int = size(A, 2),
    statevars::CGStateVariables = CGStateVariables(zero(x), similar(x), similar(x)),
    Pl = Identity())

    iterable = cg_iterator!(x, A, b, timing_data, Pl; tolerance = tolerance, maxiter = maxiter,
        statevars = statevars)
    iters = 0

    for (iteration, item) ∈ enumerate(iterable)
        iters += 1
    end

    iterable.x, iterable.timing_data, iterable.residual0, iterable.residual, iters
end
