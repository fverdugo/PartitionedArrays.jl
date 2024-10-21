
function linear_problem(args...;nullspace=nothing,block_size=1)
    attributes = (;nullspace,block_size)
    LinearProblem(args...,attributes)
end

mutable struct LinearProblem{A,B,C,D}
    solution::A
    matrix::B
    rhs::C
    attributes::D
end

solution(a) = a.solution
matrix(a) = a.matrix
rhs(a) = a.rhs
attributes(a) = a.attributes

function update!(p::LinearProblem;kwargs...)
    @assert issubset(propertynames(kwargs),propertynames(p))
    if hasproperty(kwargs,:matrix)
        p.matrix = kwargs.matrix
    end
    if hasproperty(kwargs,:rhs)
        p.rhs = kwargs.rhs
    end
    if hasproperty(kwargs,:solution)
        p.solution = kwargs.solution
    end
    if hasproperty(kwargs,:attributes)
        p.attributes = kwargs.attributes
    end
    p
end

function nonlinear_problem(args...;nullspace=nothing,block_size=1)
    attributes = (;nullspace,block_size)
    NonlinearProblem(args...,attributes)
end

mutable struct NonlinearProblem{A,B,C,D,E}
    inplace::A
    solution::B
    residual::C
    jacobian::D
    attributes::E
end

inplace(a) = a.inplace
residual(a) = a.residual
jacobian(a) = a.jacobian

function update!(p::NonlinearProblem;kwargs...)
    @assert issubset(propertynames(kwargs),propertynames(p))
    if hasproperty(kwargs,:inplace)
        p.inplace = kwargs.inplace
    end
    if hasproperty(kwargs,:residual)
        p.residual = kwargs.residual
    end
    if hasproperty(kwargs,:jacobian)
        p.jacobian = kwargs.jacobian
    end
    if hasproperty(kwargs,:solution)
        p.solution = kwargs.solution
    end
    if hasproperty(kwargs,:attributes)
        p.attributes = kwargs.attributes
    end
    p
end

function ode_problem(args...;nullspace=nothing,block_size=1,constant_jacobian=false)
    attributes = (;nullspace,block_size,constant_jacobian)
    ODEProblem(args...,attributes)
end

mutable struct ODEProblem{A,B,C,D,E,F}
    inplace::A
    interval::B
    solution::C
    residual::D
    jacobian::E
    attributes::F
end

interval(a) = a.interval

function update!(p::ODEProblem;kwargs...)
    @assert issubset(propertynames(kwargs),propertynames(p))
    if hasproperty(kwargs,:inplace)
        p.inplace = kwargs.inplace
    end
    if hasproperty(kwargs,:residual)
        p.residual = kwargs.residual
    end
    if hasproperty(kwargs,:jacobian)
        p.jacobian = kwargs.jacobian
    end
    if hasproperty(kwargs,:solution)
        p.solution = kwargs.solution
    end
    if hasproperty(kwargs,:attributes)
        p.attributes = kwargs.attributes
    end
    if hasproperty(kwargs,:interval)
        p.interval = kwargs.interval
    end
    p
end

function solve!(solver)
    next = step!(solver)
    while next !== nothing
        solver,phase = next
        next = step!(solver,next)
    end
    solution(solver)
end

function history(solver)
    History(solver)
end

struct History{A}
    solver::A
end

function Base.iterate(a::History)
    next = step!(a.solver)
    if next === nothing
        return nothing
    end
    solution(problem(a.solver)), next
end

function Base.iterate(a::History,next)
    next = step!(a.solver,next)
    if next === nothing
        return nothing
    end
    solution(problem(a.solver)), next
end

struct Solver{A,B,C,D,E}
    problem::A
    step!::B
    update!::C
    finalize!::D
    attributes::E
end

uses_initial_guess(a) = a.uses_initial_guess

function solver(args...;uses_initial_guess=Val(true))
    attributes = (;uses_initial_guess)
    Solver(args...,attributes)
end

function step!(solver::Solver)
    solver.step!()
end

function step!(solver::Solver,state)
    solver.step!(state)
end

function update!(solver::Solver;kwargs...)
    solver.update!(;kwargs...)
    solver
end

function finalize!(solver::Solver)
    solver.step!()
end

function LinearAlgebra.ldiv!(x,solver::Solver,b)
    if uses_initial_guess(solver.attributes)
        fill!(x,zero(eltype(x)))
    end
    smooth!(x,solver,b)
    x
end

function smooth!(x,solver,b)
    update!(solver,solution=x,rhs=b)
    solve!(solver)
    x
end

function LinearAlgebra_lu(problem)
    F = lu(matrix(problem))
    function lu_step!(phase=:start)
        if phase === :stop
            return nothing
        end
        x = solution(problem)
        b = rhs(problem)
        ldiv!(x,F,b)
        phase = :stop
        phase
    end
    function lu_update!(;kwargs...)
        problem = update!(problem,kwargs...)
        if hasproperty(kwargs,:matrix)
            lu!(F,matrix(problem))
        end
        problem
    end
    function lu_finalize!()
        nothing
    end
    uses_initial_guess = false
    solver(problem,lu_step!,lu_update!,lu_finalize!;uses_initial_guess)
end

function newton_raphson(problem;solver)
    function nr_step!(phase=:start)
        A = jacobian(problem)
        b = residual(problem)
        update!(solver,matrix=A,rhs=b,solution=dx)
        solve!(solver)
        x .-= dx
        update!(problem,solution=x)
    end


end

