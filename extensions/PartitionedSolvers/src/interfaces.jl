
abstract type AbstractLinearSolver end

struct GenericLinearSolver{A,B,C,D} <: AbstractLinearSolver
    setup::A # (x,A,b) -> ls_setup
    solve!::B # (x,ls_setup,b) -> results
    setup!::C # (ls_setup,A) -> ls_setup
    finalize!::D # ls_setup -> nothing
end

setup(solver::GenericLinearSolver,x,A,b) = solver.setup(x,A,b)
solve!(solver::GenericLinearSolver,x,S,b) = solver.solve!(x,S,b)
setup!(solver::GenericLinearSolver,S,A) = solver.setup!(S,A)
finalize!(solver::GenericLinearSolver,S) = solver.finalize!(S)

function linear_solver(;
        setup,
        solve!,
        setup!,
        finalize! = ls_setup->nothing,
    )
    GenericLinearSolver(setup,solve!,setup!,finalize!)
end

struct Preconditioner{A,B}
    solver::A
    solver_setup::B
end

function LinearAlgebra.ldiv!(x,P::Preconditioner,b)
    fill!(x,zero(eltype(x)))
    solve!(P.solver,x,P.solver_setup,b)
    x
end

function preconditioner(x,A,b,solver)
    solver_setup =  setup(solver,x,A,b)
    Preconditioner(solver,solver_setup)
end

function preconditioner!(P::Preconditioner,A)
    setup!(P.solver,P.solver_setup,A)
    P
end

function finalize!(P::Preconditioner)
    PartitionedSolvers.finalize!(P.solver,P.solver_setup)
end

