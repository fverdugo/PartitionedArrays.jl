# Interfaces for nonlinear solvers

struct LinearProblem{A,B}
    matrix::A
    rhs::B
end

function matrix(lp::LinearProblem)
    lp.matrix
end

function rhs(lp::LinearProblem)
    lp.rhs
end

abstract type AbstractNonlinearOperator <: AbstractType end

function default_tangent(fop,jop)
    function tangent_setup(x)
        r,r! = setup(fop,x)
        j,j! = setup(jop,x)
        r .-= r
        LinearProblem(j,r), (;r!,j!)
    end
    function tangent_call!(Ab,state,x)
        (;r!,j!) = state
        b = rhs(Ab)
        A = matrix(Ab)
        b = r!(b,x)
        b .-= b
        A = j!(A,x)
        LinearProblem(A,b)
    end
    function tangent_update!(state,op)
        (;r!,j!) = state
        r! = update!(r!,op)
        j! = update!(j!,jacobian(op))
        (;r!,j!)
    end
    nonlinear_operator(setup=tangent_setup,call! =tangent_call!,update! =tangent_update!)
end

function nonlinear_operator(;
        setup,
        call!,
        update! = (state,op) -> state,
        jacobian = nothing,
        tangent = nothing,
    )
    if jacobian !== nothing && tangent === nothing
        aux = NonlinearOperator(setup,call!,update!,nothing)
        tangent = default_tangent(aux,jacobian)
    end
    extra = (;jacobian,tangent)
    op = NonlinearOperator(setup,call!,update!,extra)
end

struct NonlinearOperator{A,B,C,D} <: AbstractNonlinearOperator
    setup::A
    call!::B
    update!::C
    extra::D
end

struct NonlinearCall{A,B} <: AbstractType
    operator::A
    operator_setup::B
end

function (P::NonlinearCall)(y,x)
    P.operator.call!(y,P.operator_setup,x)
end

function setup(op::NonlinearOperator,x)
    y,state = op.setup(x)
    y, NonlinearCall(op,state)
end

function update!(P::NonlinearCall,op::NonlinearOperator)
    op_setup = P.operator.update!(P.operator_setup,op)
    NonlinearCall(op,op_setup)
end

function jacobian(op::NonlinearOperator)
    op.extra.jacobian
end

function tangent(op::NonlinearOperator)
    op.extra.tangent
end

abstract type AbstractNonlinearSolver <: AbstractType end

function nonlinear_solver(;
        setup,
        solve! = nothing,
        update!,
        finalize! = ls_setup->nothing,
        step! = nothing,
        returns_history = Val(false),
    )
    @assert step! !== nothing || solve! !== nothing
    if step! === nothing && solve! !== nothing
        step! = (x,solver_setup,options,step=0) -> begin
            if step !=0
                return nothing
            end
            x = solve!(x,solver_setup,options)
            x,step+1
        end
    end
    if step! !== nothing && solve! === nothing
        solve! = (x,solver_setup,options) -> begin
            next = step!(x,solver_setup,options)
            if next === nothing
                return x
            end
            x,step = next
            while true
                next = step!(x,solver_setup,options,step)
                if next === nothing
                    return x
                end
                x,step = next
            end
        end
    end
    traits = NonlinearSolverTraits(returns_history)
    NonlinearSolver(setup,solve!,update!,finalize!,step!,traits)
end

struct NonlinearSolverTraits{C} <: AbstractType
    returns_history::C
end

struct NonlinearSolver{A,B,C,D,E,F} <: AbstractNonlinearSolver
    setup::A
    solve!::B
    update!::C
    finalize!::D
    step!::E
    traits::F
end

function nonlinear_solver(s::AbstractNonlinearSolver)
    s
end

struct NonlinearPreconditioner{A,B} <: AbstractType
    solver::A
    solver_setup::B
end

function nonlinear_setup_options()
    options = (;)
end

function returns_history(a::NonlinearSolver)
    val_parameter(a.traits.returns_history)
end

function setup(solver::NonlinearSolver,f,x;kwargs...)
    options = nonlinear_setup_options(;kwargs...)
    solver_setup = solver.setup(f,x,options)
    NonlinearPreconditioner(solver,solver_setup)
end

function update!(P::NonlinearPreconditioner,f;kwargs...)
    options = nonlinear_setup_options(;kwargs...)
    solver_setup = P.solver.update!(P.solver_setup,f,options)
    NonlinearPreconditioner(P.solver,solver_setup)
end

function nonlinar_solve_options(;zero_guess=false,history=Val(false))
    options = (;zero_guess,history)
end

function solve!(x,P::NonlinearPreconditioner;kwargs...)
    options = nonlinar_solve_options(;kwargs...)
    next = P.solver.solve!(x,P.solver_setup,options)
    if returns_history(P.solver)
        x,log = next
    else
        x = next
        log = nothing
    end
    if val_parameter(options.history) == true
        return x, log
    else
        return x
    end
end

function finalize!(P::NonlinearPreconditioner)
    P.solver.finalize!(P.solver_setup)
end

struct NonlinearSolverIterator{A} <: AbstractType
    params::A
end

function iterations!(x,P::NonlinearPreconditioner;kwargs...)
    options = nonlinar_solve_options(;kwargs...)
    params = (;options,x,P)
    LinearSolverIterator(params)
end

function Base.iterate(a::NonlinearSolverIterator)
    P = a.params.P
    options = a.params.options
    x = a.params.x
    next = P.solver.step!(x,P.solver_setup,options)
    next
end

function Base.iterate(a::NonlinearSolverIterator,state)
    P = a.params.P
    options = a.params.options
    x = a.params.x
    next = P.solver.step!(x,P.solver_setup,options,state)
    next
end



