
function default_nullspace(A)
    T = eltype(A)
    [ones(T,size(A,2))]
end

function default_nullspace(A::PSparseMatrix)
    col_partition = partition(axes(A,2))
    T = eltype(A)
    [ pones(T,col_partition) ]
end

abstract type AbstractLinearSolver end

function linear_solver(;
        setup,
        solve!,
        update!,
        finalize! = ls_setup->nothing,
        uses_nullspace = false,
        uses_initial_guess = true,
    )
    LinearSolver(setup,solve!,update!,finalize!,uses_nullspace,uses_initial_guess)
end

struct LinearSolver{A,B,C,D} <: AbstractLinearSolver
    setup::A
    solve!::B
    update!::C
    finalize!::D
    uses_nullspace::Bool
    uses_initial_guess::Bool
end

function linear_solver(s::LinearSolver)
    s
end

struct Preconditioner{A,B}
    solver::A
    solver_setup::B
end

function setup_options(;nullspace=nothing)
    options = (;nullspace)
end

function nullspace(options)
    options.nullspace
end

function uses_nullspace(a)
    false
end

function uses_nullspace(a::LinearSolver)
    a.uses_nullspace
end

function uses_initial_guess(a)
    true
end

function uses_initial_guess(a::LinearSolver)
    a.uses_initial_guess
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

function solve_options(;zero_guess=false)
    options = (;zero_guess)
end

function solve!(x,P::Preconditioner,b;kwargs...)
    options = solve_options(;kwargs...)
    P.solver.solve!(x,P.solver_setup,b,options)
    x
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

