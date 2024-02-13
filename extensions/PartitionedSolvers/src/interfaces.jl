
abstract type AbstractLinearProblem  end

function linear_problem(matrix,rhs,nullspace=nothing)
    GenericLinearProblem(matrix,rhs,nullspace)
end

struct GenericLinearProblem{A,B,C} <: AbstractLinearProblem
    matrix::A
    rhs::B
    nullspace::C
end

matrix(a::GenericLinearProblem) = a.matrix
rhs(a::GenericLinearProblem) = a.rhs
nullspace(a::GenericLinearProblem) = a.nullspace

function replace_matrix(a::GenericLinearProblem,A)
    linear_problem(
        A,
        rhs(a),
        nullspace(a))
end

function replace_rhs(a::GenericLinearProblem,b)
    linear_problem(
        matrix(a),
        b,
        nullspace(a))
end

abstract type AbstractLinearSolver end

function linear_solver(;
        setup,
        use!,
        setup!,
        finalize! = ls_setup->nothing,
    )
    GenericLinearSolver(setup,use!,setup!,finalize!)
end

struct GenericLinearSolver{A,B,C,D} <: AbstractLinearSolver
    setup::A
    use!::B
    setup!::C
    finalize!::D
end

setup(solver::GenericLinearSolver) = solver.setup
setup!(solver::GenericLinearSolver) = solver.setup!
use!(solver::GenericLinearSolver) = solver.use!
finalize!(solver::GenericLinearSolver) = solver.finalize!

struct Preconditioner{A,B,C}
    solver::A
    solver_setup::B
    problem::C
end

function preconditioner(solver,problem,x)
    solver_setup = setup(solver)(problem,x)
    Preconditioner(solver,solver_setup,problem)
end

function preconditioner!(P::Preconditioner,problem,x)
    setup!(P.solver)(problem,x,P.solver_setup)
    P
end

function LinearAlgebra.ldiv!(x,P::Preconditioner,b)
    fill!(x,zero(eltype(x)))
    problem = replace_rhs(P.problem,b)
    use!(P.solver)(problem,x,P.solver_setup)
    x
end

function finalize!(P::Preconditioner)
    PartitionedSolvers.finalize!(P.solver)(P.solver_setup)
end

