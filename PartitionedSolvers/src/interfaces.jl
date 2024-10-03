
abstract type AbstractType end
function Base.show(io::IO,data::AbstractType)
    print(io,"PartitionedSolvers.$(nameof(typeof(data)))(…)")
end

function default_nullspace(A)
    T = eltype(A)
    [ones(T,size(A,2))]
end

function default_nullspace(A::PSparseMatrix)
    col_partition = partition(axes(A,2))
    T = eltype(A)
    [ pones(T,col_partition) ]
end

abstract type AbstractLinearSolver <: AbstractType end

function linear_solver(;
        setup,
        solve!,
        update!,
        finalize! = ls_setup->nothing,
        step! = nothing,
        uses_nullspace = Val(false),
        uses_initial_guess = Val(true),
        returns_history = Val(false),
    )
    if step! === nothing
        step! = (x,ls_setup,b,options,step=0) -> begin
            if step !=0
                return nothing
            end
            x = solve!(x,ls_setup,b,options)
            x,step+1
        end
    end
    traits = LinearSolverTraits(uses_nullspace,uses_initial_guess,returns_history)
    LinearSolver(setup,solve!,update!,finalize!,step!,traits)
end

struct LinearSolverTraits{A,B,C} <: AbstractType
    uses_nullspace::A
    uses_initial_guess::B
    returns_history::C
end

struct LinearSolver{A,B,C,D,E,F} <: AbstractLinearSolver
    setup::A
    solve!::B
    update!::C
    finalize!::D
    step!::E
    traits::F
end

function linear_solver(s::LinearSolver)
    s
end

struct Preconditioner{A,B} <: AbstractType
    solver::A
    solver_setup::B
end

function setup_options(;nullspace=nothing)
    options = (;nullspace)
end

function nullspace(options)
    options.nullspace
end

function uses_nullspace(a::LinearSolver)
    val_parameter(a.traits.uses_nullspace)
end

function uses_initial_guess(a::LinearSolver)
    val_parameter(a.traits.uses_initial_guess)
end

function returns_history(a::LinearSolver)
    val_parameter(a.traits.returns_history)
end

function setup(solver::LinearSolver,x,A,b;kwargs...)
    options = setup_options(;kwargs...)
    solver_setup = solver.setup(x,A,b,options)
    Preconditioner(solver,solver_setup)
end

function update!(P::Preconditioner,A;kwargs...)
    options = setup_options(;kwargs...)
    P.solver.update!(P.solver_setup,A,options)
    P
end

function solve_options(;zero_guess=false,history=Val(false))
    options = (;zero_guess,history)
end

function solve!(x,P::Preconditioner,b;kwargs...)
    options = solve_options(;kwargs...)
    next = P.solver.solve!(x,P.solver_setup,b,options)
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

function LinearAlgebra.ldiv!(x,P::Preconditioner,b)
    if uses_initial_guess(P.solver)
        fill!(x,zero(eltype(x)))
    end
    solve!(x,P,b;zero_guess=true)
    x
end

function finalize!(P::Preconditioner)
    P.solver.finalize!(P.solver_setup)
end

function iterations!(x,P::Preconditioner,b;kwargs...)
    options = solve_options(;kwargs...)
    params = (;options,x,P,b)
    LinearSolverIterator(params)
end

struct LinearSolverIterator{A} <: AbstractType
    params::A
end

function Base.iterate(a::LinearSolverIterator)
    P = a.params.P
    options = a.params.options
    b = a.params.b
    x = a.params.x
    next = P.solver.step!(x,P.solver_setup,b,options)
    next
end

function Base.iterate(a::LinearSolverIterator,state)
    P = a.params.P
    options = a.params.options
    b = a.params.b
    x = a.params.x
    next = P.solver.step!(x,P.solver_setup,b,options,state)
    next
end

