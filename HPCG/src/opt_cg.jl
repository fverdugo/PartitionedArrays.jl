"""
    opt_cg!(x, A, b; kwargs...) -> x

    This version can be changed to implement optimisations.

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
function opt_cg!(x, A, b, timing_data;
    tolerance::Float64 = 0.0,
    maxiter::Int = size(A, 2),
    statevars::CGStateVariables = CGStateVariables(zero(x), similar(x), similar(x)),
    Pl = Identity())

    return ref_cg!(x, A, b, timing_data, maxiter = maxiter, tolerance = tolerance, Pl = Pl, statevars = statevars)
end
