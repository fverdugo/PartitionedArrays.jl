# Interfaces for nonlinear solvers

# TODO add nullspace and block size
struct AffineOperator{A,B} <: Function
    matrix::A
    rhs::B
end

function matrix(lp::AffineOperator)
    lp.matrix
end

function rhs(lp::AffineOperator)
    lp.rhs
end

struct InplaceFunction{A,B} <: Function
    parent::A
    parent!::B
end

function (f::InplaceFunction)(x...)
    f.parent(x...)
end

function inplace(f::InplaceFunction)
    f.parent!
end

struct FunctionWithDerivative{A,B} <: Function
    parent::A
    derivative::B
end

function (f::FunctionWithDerivative)(x...)
    f.parent(x...)
end

function inplace(f::FunctionWithDerivative)
    inplace(f.parent)
end

function derivative(f::FunctionWithDerivative)
    f.derivative
end

function tangent(f::FunctionWithDerivative)
    function g(x...)
        r,r_s = f.parent(x...)
        j,j_s = f.derivative(x...)
        r .-= r
        t = AffineOperator(j,r)
        t_s = (;r_s,j_s)
        t,t_s
    end
    function g!(t,t_s,x...)
        r = rhs(t)
        j = matrix(t)
        (;r_s,j_s) = t_s
        f! = inplace(f.parent)
        df! = inplace(f.derivative)
        r,r_s = f!(r,r_s,x...)
        j,j_s = df!(j,j_s,x...)
        t = AffineOperator(j,r)
        t_s = (;r_s,j_s)
        t,t_s
    end
    InplaceFunction(g,g!)
end

abstract type AbstractNonlinearSolver <: AbstractType end

function nonlinear_solver(;
        solve! = nothing,
        finalize! = ls_setup->nothing,
        step! = nothing,
        is_iterative=Val(false),
    )
    @assert step! !== nothing || solve! !== nothing
    if step! === nothing && solve! !== nothing
        function default_step!(x,f,options)
            x,workspace = solve!(x,f,options)
            step = 0
            x,workspace,step+1
        end
        function default_step!(x,f,workspace,options,step=0)
            if step != 0
                return nothing
            end
            x,workspace = solve!(x,f,workspace,options)
            x,workspace,step+1
        end
        step! = default_step!
    end
    if step! !== nothing && solve! === nothing
        function default_solve!(x,f,options)
            next = step!(x,f,options)
            # Assumes that the solver advances one step
            # OK since solvers stop at the start of the
            # next step after convergence
            x,workspace,state = next
            while next !== nothing
                x,workspace,state = next
                next = step!(x,f,workspace,options,state)
            end
            x,workspace
        end
        function default_solve!(x,f,workspace,options)
            next = step!(x,f,workspace)
            while next !== nothing
                x,workspace,state = next
                next = step!(x,f,workspace,options,state)
            end
            x,workspace
        end
        solve! = default_solve!
    end
    traits = NonlinearSolverTraits(is_iterative)
    NonlinearSolver(solve!,step!,finalize!,traits)
end

struct NonlinearSolverTraits{C} <: AbstractType
    is_iterative::C
end

struct NonlinearSolver{A,B,C,D} <: AbstractNonlinearSolver
    solve!::A
    step!::B
    finalize!::C
    traits::D
end

function nonlinear_solver(s::AbstractNonlinearSolver)
    s
end

function is_iterative(a::NonlinearSolver)
    val_parameter(a.traits.is_iterative)
end

function solve!(solver::NonlinearSolver,x,f;kwargs...)
    options = nonlinar_solve_options(;kwargs...)
    x,workspace = solver.solve!(x,f,options)
    P = NonlinearSolverWorkspace(solver,workspace)
    x,P
end

struct NonlinearSolverWorkspace{A,B} <: AbstractType
    solver::A
    workspace::B
end

function nonlinar_solve_options(;zero_guess=false)
    options = (;zero_guess)
end

function solve!(x,P::NonlinearSolverWorkspace,f;kwargs...)
    options = nonlinar_solve_options(;kwargs...)
    x,workspace = P.solver.solve!(x,f,P.workspace,options)
    P = NonlinearSolverWorkspace(P.solver,workspace)
    x,P
end

function finalize!(P::NonlinearSolverWorkspace)
    P.solver.finalize!(P.workspace)
end

struct NonlinearSolverIterator{A} <: AbstractType
    params::A
end

function iterations!(solver::NonlinearSolver,x,f;kwargs...)
    options = nonlinar_solve_options(;kwargs...)
    params = (;options,x,f,solver)
    NonlinearSolverIterator(params)
end

function iterations!(x,P::NonlinearSolverWorkspace,f;kwargs...)
    options = nonlinar_solve_options(;kwargs...)
    solver = P.solver
    workspace = P.workspace
    params = (;options,x,f,solver,workspace)
    NonlinearSolverIterator(params)
end

function Base.iterate(a::NonlinearSolverIterator)
    options = a.params.options
    f = params.f
    x = a.params.x
    solver = params.solver
    if hasproperty(params,:workspace)
        workspace = params.workspace
        next = solver.step!(x,f,workspace,options)
    else
        next = solver.step!(x,f,options)
    end
    if next === nothing
        return nothing
    end
    x,workspace,state = next
    P = NonlinearSolverWorkspace(P.solver,workspace)
    (x,P),state
end

function Base.iterate(a::NonlinearSolverIterator,state)
    options = a.params.options
    x = a.params.x
    solver = a.params.solver
    workspace = a.params.workspace
    next = P.solver.step!(x,P.workspace,options,state)
    if next === nothing
        return nothing
    end
    x,workspace,state = next
    P = NonlinearSolverWorkspace(P.solver,workspace)
    (x,P),state
end

function newton_raphson(;
    linear_solver=lu_solver(),maxiters=1000,reltol=nothing,abstol=nothing)
    function step!(x,f,options)
        df = tangent(f)
        t,t_s = df(x)
        dx = similar(b,axes(A,2))
        P = setup(linear_solver,dx,A,b)
        b = rhs(t)
        T = real(eltype(b))
        current = zero(eltype(b))
        target = current
        iteration = 0
        workspace = (;P,dx,t,t_s,iteration,current,target)
        state = :start
        step!(x,f,workspace,state)
    end
    function step!(x,f,workspace,options,state=:restart)
        @assert state in (:start,:restart,:advance,:stop)
        if state === :stop
            return nothing
        end
        (;P,dx,t,t_s,iteration,current,target) = workspace
        if state !== :start
            df = tangent(f)
            df! = inplace(df)
            t,t_s = df!(t,t_s,x)
        end
        A = matrix(t)
        b = rhs(t)
        current = norm(b)
        if state===:start || state == :restart
            T = real(eltype(b))
            if abstol === nothing
                abstol = zero(T)
            end
            if reltol === nothing
                reltol = sqrt(eps(T))
            end
            target = max(reltol*current,abstol)
            iteration = 0
        end
        converged = current <= target
        if converged
            state = :stop
            workspace = (;P,dx,t,t_s,iteration,current,target)
            return x,workspace,state
        end
        if state !== :start
            P = update!(P,A)
        end
        dx = ldiv!(dx,P,b)
        x += dx
        iteration += 1
        tired = maxiters == iteration
        if tired
            state = :stop
        else
            state = :advance
        end
        workspace = (;P,dx,t,t_s,iteration,current,target)
        x, workspace, state
    end
    function finalize!(state)
        (;P) = workspace
        finalize!(P)
    end
    is_iterative = Val(true)
    nonlinear_solver(;step!,finalize!,is_iterative)
end



