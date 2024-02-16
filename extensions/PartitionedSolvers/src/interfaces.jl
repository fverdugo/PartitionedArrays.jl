
function attach_nullspace(A,ns)
    MatrixWithNullspace(A,ns)
end

struct MatrixWithNullspace{A,B}
    matrix::A
    nullspace::B
end
matrix(a::MatrixWithNullspace) = a.matrix
nullspace(a::MatrixWithNullspace) = a.nullspace

matrix(a) = a
nullspace(a) = default_nullspace(a)

function default_nullspace(A)
    ones(size(A,1))
end

function default_nullspace(A::PSparseMatrix)
    map(default_nullspace,own_own_values(A))
end

abstract type AbstractLinearSolver end

function linear_solver(;
        setup,
        use!,
        setup!,
        finalize! = ls_setup->nothing,
    )
    LinearSolver(setup,use!,setup!,finalize!)
end

struct LinearSolver{A,B,C,D} <: AbstractLinearSolver
    setup::A
    use!::B
    setup!::C
    finalize!::D
end

setup(solver::LinearSolver) = solver.setup
setup!(solver::LinearSolver) = solver.setup!
use!(solver::LinearSolver) = solver.use!
finalize!(solver::LinearSolver) = solver.finalize!

struct Preconditioner{A,B}
    solver::A
    solver_setup::B
end

function preconditioner(solver,x,A,b)
    solver_setup = setup(solver)(x,A,b)
    Preconditioner(solver,solver_setup)
end

function preconditioner!(P::Preconditioner,A)
    setup!(P.solver)(P.solver_setup,A)
    P
end

function LinearAlgebra.ldiv!(x,P::Preconditioner,b)
    fill!(x,zero(eltype(x)))
    use!(P.solver)(x,P.solver_setup,b)
    x
end

function finalize!(P::Preconditioner)
    PartitionedSolvers.finalize!(P.solver)(P.solver_setup)
end

