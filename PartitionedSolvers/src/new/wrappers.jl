
function LinearAlgebra_lu(p)
    @assert uses_mutable_types(p)
    F = lu(matrix(p))
    function update(F,A)
        lu!(F,A)
        F
    end
    function step(x,F,b,phase=:start)
        ldiv!(x,F,b)
        phase = :stop
        x,F,phase
    end
    uses_initial_guess = false
    PS.linear_solver(update,step,p,F;uses_initial_guess)
end

function NLSolvers_nlsolve_setup(p)
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

function NLSolvers_nlsolve(p;kwargs...)
    @assert uses_mutable_types(p)
    workspace = NLSolvers_nlsolve_setup(p)
    function update(workspace,p)
        workspace = NLSolvers_nlsolve_setup(p)
    end
    function step(workspace,p,phase=:start)
        if phase === :stop
            return nothing
        end
        df = workspace
        x = solution(p)
        result = NLsolve.nlsolve(df,x;kwargs...)
        copyto!(x,result.x)
        workspace,p,phase = :stop
    end
    nonlinear_solver(update,step,p,workspace)
end


