
abstract type AbstractAlgebraicProblem end

function albegraic_problem(;
        domain::AbstractUnitRange,
        linearize,
        linearize! = nothing,
        rhs!,
        jacobian!)

    GenericAlgebraicProblem(
                            domain,
                            linearize,
                            linearize!
                            rhs!,
                            jacobian!,
                           )
end
struct GenericAlgebraicProblem{A,B,C,D,E} <: AbstractAlgebraicProblem
    domain::A
    linearize::B
    linearize!::C
    residual!::D
    jacobian!::E
end
domain(a::GenericAlgebraicProblem) = a.domain
linearize(a::GenericAlgebraicProblem,x) = a.linearize(x)
rhs!(a::GenericAlgebraicProblem,x,linear_problem) = a.rhs!(x,linear_problem)
jacobian!(a::GenericAlgebraicProblem,x,linear_problem) = a.jacobian!(x,linear_problem)
function linearize!(a::GenericAlgebraicProblem,x,linear_problem)
    if a.linearize! === nothing
        a.rhs!(x,linear_problem)
        a.jacobian!(x,linear_problem)
    else
        a.linearize!(x,linear_problem)
    end
    linear_problem
end

abstract type AbstractLinearProblem <: AbstractAlgebraicProblem end

function linear_problem(;linear_operator,rhs,null_space=nothing,workspace=nothing)
    GenericLinearProblem(linear_solver,rhs,null_space,workspace)
end
struct GenericLinearProblem{A,B,C,D} <: AbstractLinearProblem
    linear_operator::A
    rhs::B
    null_space::C
    workspace::D
end
linear_operator(a::GenericLinearProblem) = a.linear_operator
rhs(a::GenericLinearProblem) = a.rhs
null_space(a::GenericLinearProblem) = a.null_space
workspace(a::GenericLinearProblem) = a.workspace
function replace_rhs(a::GenericAlgebraicProblem,b)
    linear_problem(;
        linear_operator=linear_operator(a),
        rhs = b,
        null_space = null_space(a),
        workspace = workspace(a)
       )
end

domain(a::AbstractLinearProblem) = axes(linear_operator(a),2)
function linearize(a::AbstractLinearProblem,x)
    A = linear_operator(a)
    b = rhs(a) - A*x
    replace_rhs(a,b)
end
function rhs!(a::AbstractLinearProblem,x,linear_problem)
    A = linear_operator(a)
    b = rhs(a)
    r = rhs(linear_problem)
    mul!(r,A,x)
    r .= b .- r
    r
end
function jacobian!(a::AbstractLinearProblem,x,linear_problem)
    A = linear_operator(a)
    B = linear_operator(linear_problem)
    if A !== B
        copy!(B,A)
    end
    B
end

abstract type AbstractAlgerbaicSolver end

function algebraic_solver(;
        setup,
        solve!,
        setup!,
        finalize! = ls_setup->nothing,
    )
    GenericAlgebraicSolver(setup,solve!,setup!,finalize!)
end
struct GenericAlgebraicSolver{A,B,C,D} <: AbstractAlgebraicSolver
    setup::A
    solve!::B
    setup!::C
    finalize!::D
end
setup(solver::GenericAlgebraicSolver,x,problem) = solver.setup(x,problem)
setup!(solver::GenericAlgebraicSolver,x,problem,state) = solver.setup!(x,problem,state)
solve!(solver::GenericAlgebraicSolver,x,problem,state) = solver.solve!(x,problem,state)
finalize!(solver::GenericAlgebraicSolver,state) = solver.finalize!(state)

struct Preconditioner{A,B,C}
    solver::A
    solver_setup::B
    problem::C
end

function preconditioner(solver,x,problem)
    solver_setup = setup(solver,x,problem)
    Preconditioner(solver,solver_setup,problem)
end

function preconditioner!(P::Preconditioner,x,problem)
    setup!(P.solver,x,problem,P.solver_setup)
    P
end

function LinearAlgebra.ldiv!(x,P::Preconditioner,b)
    fill!(x,zero(eltype(x)))
    problem = replace_rhs(P.problem,b)
    solve!(P.solver,x,problem,P.solver_setup)
    x
end

function finalize!(P::Preconditioner)
    PartitionedSolvers.finalize!(P.solver,P.solver_setup)
end

