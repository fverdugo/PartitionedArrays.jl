
setup(a) = a.setup
solve!(a) = a.solve!
apply!(a) = a.apply!
setup!(a) = a.setup!
finalize!(a) = a.finalize!
residual!(a) = a.residual!
jacobian!(a) = a.jacobian!
residual_and_jacobian!(a) = a.residual_and_jacobian!

abstract type AbstractLinearSolver end

struct GenericLinearSolver{A,B,C,D,E} <: AbstractLinearSolver
    setup::A # (x,A,b) -> ls_setup
    solve!::B # (x,ls_setup,b) -> results
    setup!::C # (ls_setup,A) -> ls_setup
    finalize!::D # ls_setup -> nothing
    apply!::E # (x,ls_setup,b) -> x
end

function linear_solver(;
        setup,
        solve!,
        setup!,
        finalize! = ls_setup->nothing,
        apply! = (x,ls_setup,b) -> begin
            fill!(x,zero(eltype(x)))
            solve!(x,ls_setup,b)
            x
        end
    )
    GenericLinearSolver(setup,solve!,setup!,finalize!,apply!)
end

struct Preconditioner{A,B}
    solver::A
    solver_setup::B
end

function LinearAlgebra.ldiv!(x,P::Preconditioner,b)
    apply!(P.solver)(x,P.solver_setup,b)
    x
end

function preconditioner(x,A,b,solver)
    solver_setup =  setup(solver)(x,A,b)
    Preconditioner(solver,solver_setup)
end

function preconditioner!(P::Preconditioner,A)
    setup!(P.solver)(P.solver_setup,A)
    P
end

function finalize!(P::Preconditioner)
    PartitionedSolvers.finalize!(P.solver)(P.solver_setup)
end


