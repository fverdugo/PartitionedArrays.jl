
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
        setup!,
        finalize! = ls_setup->nothing,
    )
    LinearSolver(setup,solve!,setup!,finalize!)
end

struct LinearSolver{A,B,C,D} <: AbstractLinearSolver
    setup::A
    solve!::B
    setup!::C
    finalize!::D
end

setup(solver::LinearSolver) = solver.setup
setup!(solver::LinearSolver) = solver.setup!
solve!(solver::LinearSolver) = solver.solve!
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
    solve!(P.solver)(x,P.solver_setup,b)
    x
end

function finalize!(P::Preconditioner)
    PartitionedSolvers.finalize!(P.solver)(P.solver_setup)
end

