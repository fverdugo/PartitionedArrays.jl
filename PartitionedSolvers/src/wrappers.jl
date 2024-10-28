
function LinearAlgebra_lu(p)
    @assert uses_mutable_types(p)
    F = lu(matrix(p))
    function update(F,A)
        lu!(F,A)
        F
    end
    function step(x,F,b,phase=:start;kwargs...)
        ldiv!(x,F,b)
        phase = :stop
        x,F,phase
    end
    uses_initial_guess = Val(false)
    linear_solver(update,step,p,F;uses_initial_guess)
end

function IterativeSolvers_cg(p;kwargs...)
    A = matrix(p)
    function update(state,A)
        A
    end
    function step(x,A,b,phase=:start;kwargs...)
        IterativeSolvers.cg!(x,A,b;kwargs...)
        phase = :stop
        x,A,phase
    end
    linear_solver(update,step,p,A)
end

function NLsolve_nlsolve_setup(p)
    function f!(r,x)
        update(p,residual=r,jacobian=nothing,solution=x)
        r
    end
    function j!(j,x)
        update(p,residual=nothing,jacobian=j,solution=x)
        j
    end
    function fj!(r,j,x)
        update(p,residual=r,jacobian=j,solution=x)
        r,j
    end
    df = NLsolve.OnceDifferentiable(f!,j!,fj!,solution(p),residual(p),jacobian(p))
end

function NLsolve_nlsolve(p;kwargs...)
    @assert uses_mutable_types(p)
    workspace = NLsolve_nlsolve_setup(p)
    function update(workspace,p)
        workspace = NLsolve_nlsolve_setup(p)
    end
    function step(workspace,p,phase=:start;options...)
        if phase === :stop
            return nothing
        end
        df = workspace
        x = solution(p)
        result = NLsolve.nlsolve(df,x;kwargs...)
        copyto!(x,result.zero)
        phase = :stop
        workspace,p,phase
    end
    nonlinear_solver(update,step,p,workspace)
end

function NLsolve_nlsolve_linsolve(solver,p)
    x = solution(p)
    A = jacobian(p)
    r = residual(p)
    dx = similar(x,axes(A,2))
    lp = linear_problem(dx,A,r)
    ls = solver(lp)
    function linsolve(dx,A,b)
        ls = update(ls,matrix=A)
        ldiv!(dx,ls,b)
        dx
    end
end

