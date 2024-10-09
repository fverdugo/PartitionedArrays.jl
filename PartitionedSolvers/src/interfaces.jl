
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
        solve! = nothing,
        update!,
        finalize! = workspace->nothing,
        step! = nothing,
        uses_nullspace = Val(false),
        uses_initial_guess = Val(true),
        is_iterative = Val(false),
    )
    @assert step! !== nothing || solve! !== nothing
    if step! === nothing && solve! !== nothing
        step! = (x,workspace,b,options,step=0) -> begin
            if step !=0
                return nothing
            end
            x, workspace = solve!(x,workspace,b,options)
            step += 1
            x,workspace,step
        end
    end
    if step! !== nothing && solve! === nothing
        solve! = (x,workspace,b,options) -> begin
            next = step!(x,workspace,b,options)
            while next !== nothing
                x,workspace,step = next
                next = step!(x,workspace,b,options,step)
            end
            (x,workspace)
        end
    end
    traits = LinearSolverTraits(
        uses_nullspace,
        uses_initial_guess,
        is_iterative
       )
    LinearSolver(setup,solve!,update!,finalize!,step!,traits)
end

struct LinearSolverTraits{A,B,C} <: AbstractType
    uses_nullspace::A
    uses_initial_guess::B
    is_iterative::C
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
    workspace::B
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

function is_iterative(a::LinearSolver)
    val_parameter(a.traits.is_iterative)
end

function setup(solver::LinearSolver,x,A,b;kwargs...)
    options = setup_options(;kwargs...)
    workspace = solver.setup(x,A,b,options)
    Preconditioner(solver,workspace)
end

function update!(P::Preconditioner,A;kwargs...)
    options = setup_options(;kwargs...)
    workspace = P.solver.update!(P.workspace,A,options)
    Preconditioner(P.solver,workspace)
end

function solve_options(;zero_guess=false)
    options = (;zero_guess)
end

function solve!(x,P::Preconditioner,b;kwargs...)
    options = solve_options(;kwargs...)
    x, workspace = P.solver.solve!(x,P.workspace,b,options)
    x, Preconditioner(P.solver,workspace)
end

function LinearAlgebra.ldiv!(x,P::Preconditioner,b)
    if uses_initial_guess(P.solver)
        fill!(x,zero(eltype(x)))
    end
    x, P = solve!(x,P,b;zero_guess=true)
    x
end

function finalize!(P::Preconditioner)
    P.solver.finalize!(P.workspace)
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
    next = P.solver.step!(x,P.workspace,b,options)
    if next === nothing
        return nothing
    end
    x,workspace,state = next
    P = Preconditioner(P.solver,workspace)
    (x,P), state
end

function Base.iterate(a::LinearSolverIterator,state)
    P = a.params.P
    options = a.params.options
    b = a.params.b
    x = a.params.x
    next = P.solver.step!(x,P.workspace,b,options,state)
    if next === nothing
        return nothing
    end
    x,workspace,state = next
    P = Preconditioner(P.solver,workspace)
    (x,P),state
end

function solve(solver::AbstractLinearSolver,A,b;kwargs...)
    x = similar(b,axes(A,2))
    fill!(x,zero(eltype(x)))
    solve!(solver,x,A,b;kwargs...)
end

function solve!(solver::AbstractLinearSolver,x,A,b;kwargs...)
    P = setup(solver,x,A,b)
    solve!(x,P,b;kwargs...)
end


