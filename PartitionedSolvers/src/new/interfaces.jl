
abstract type AbstractProblem <: AbstractType end
abstract type AbstractSolver <: AbstractType end
abstract type AbstractAge <: AbstractType end

#function update(;kwargs...)
#    function update_it(p)
#        update(p;kwargs...)
#    end
#end
#
#function update(s::AbstractSolver)
#    function update_solver(p)
#        update(s,p)
#    end
#end
#
#function update(p::AbstractProblem;kwargs...)
#    function update_problem(args...)
#        update(p,args...;kwargs...)
#    end
#end

function solve(solver)
    next = step(solver)
    while next !== nothing
        solver,state =  next
        next = step(solver,state)
    end
    solver
end

function history(solver)
    History(solver)
end

struct History{A}
    solver::A
end

function Base.iterate(a::History)
    next = step(a.solver)
    if next === nothing
        return nothing
    end
    solution(problem(a.solver)), next
end

function Base.iterate(a::History,next)
    s,state = next
    next = step(s,state)
    if next === nothing
        return nothing
    end
    solution(problem(s)), next
end

workspace(a) = a.workspace
solution(a) = a.solution
jacobian(a) = a.jacobian
residual(a) = a.residual
attributes(a) = a.attributes
problem(a) = a.problem
age(a) = a.age
matrix(a) = a.matrix
rhs(a) = a.rhs
uses_initial_guess(a) = val_parameter(a.uses_initial_guess)
constant_jacobian(a) = val_parameter(a.constant_jacobian)

solution(a::AbstractProblem) = solution(workspace(a))
jacobian(a::AbstractProblem) = jacobian(workspace(a))
residual(a::AbstractProblem) = residual(workspace(a))
age(a::AbstractProblem) = age(workspace(a))
matrix(a::AbstractProblem) = matrix(workspace(a))
rhs(a::AbstractProblem) = rhs(workspace(a))
problem(a::AbstractSolver) = problem(workspace(a))

abstract type AbstractLinearProblem <: AbstractProblem end

struct LinearProblemAge <: AbstractAge
    solution::Int
    matrix::Int
    rhs::Int
end

function linear_problem_age()
    LinearProblemAge(1,1,1)
end

function increment(a::LinearProblemAge;solution=0,matrix=0,rhs=0)
    LinearProblemAge(
                          a.solution + solution,
                          a.matrix + matrix,
                          a.rhs + rhs,
                         )
end

function linear_problem(solution,matrix,rhs;attributes...)
    age = linear_problem_age()
    workspace = (;solution,matrix,rhs,age)
    LinearProblem(workspace,attributes)
end

struct LinearProblem{A,B} <: AbstractLinearProblem
    workspace::A
    attributes::B
end

function update(p::LinearProblem;kwargs...)
    a = age(p)
    if haskey(kwargs,:matrix)
        A = kwargs.matrix
        a = increment(a,matrix=1)
    else
        A = matrix(p)
    end
    if haskey(kwargs,:rhs)
        b = kwargs.rhs
        a = increment(a,rhs=1)
    else
        b = rhs(p)
    end
    if haskey(kwargs,:solution)
        x = kwargs.solution
        a = increment(a,solution=1)
    else
        x = solution(p)
    end
    attrs = attributes(p)
    workspace = (;matrix=A,rhs=b,solution=x,attributes=attrs,age=a)
    LinearProblem(workspace)
end

abstract type AbstractLinearSolver <: AbstractSolver end

function LinearAlgebra.ldiv!(x,solver::AbstractLinearSolver,b)
    if uses_initial_guess(attributes(solver))
        fill!(x,zero(eltype(x)))
    end
    smooth!(x,solver,b;zero_guess=true)
    x
end

function smooth!(x,s::AbstractLinearSolver,b;kwargs...)
    p = problem(s)
    p = update(p;solution=x,rhs=b)
    s = update(s,p)
    s = solve(s)
    x = solution(s)
end

function linear_solver(update,step,workspace;
        uses_initial_guess = Val(true),
    )
    attributes = (;uses_initial_guess)
    matrix_age = age(matrix(problem(workspace)))
    LinearSolver(update,step,workspace,attributes,matrix_age)
end

struct LinearSolver <: AbstractLinearSolver
    update::A
    step::B
    workspace::C
    attributes::D
    matrix_age::Int
end

function update(s::LinearSolver,p::AbstractLinearProblem)
    matrix_age = s.matrix_age
    if matrix_age != matrix(age(p))
        workspace = s.update(s.workspace,matrix(p))
        matrix_age = matrix(age(p))
    end
    LinearSolver(s.update,s.step,workspace,matrix_age)
end

function step(s::LinearSolver)
    next = s.step(s.workspace)
    if next === nothing
        return nothing
    end
    workspace, state = next
    LinearSolver(s.update,s.step,workspace,s.matrix_age), state
end

function step(s::LinearSolver,state)
    next = s.step(s.workspace,state)
    if next === nothing
        return nothing
    end
    workspace, state = next
    LinearSolver(s.update,s.step,workspace,s.matrix_age), state
end

abstract type AbstractNonlinearProblem <: AbstractProblem end

#function update(p::AbstractNonlinearProblem;kwargs...)
#    function update_nonlinear_problem(x)
#        update(p,x;kwargs...)
#    end
#end

struct NonlinearProblemAge <: AbstractAge
    solution::Int
    residual::Int
    jacobian::Int
end

function nonlinear_problem_age()
    NonlinearProblemAge(0,0,0)
end

function increment(a::NonlinearProblemAge;solution=0,residual=0,jacobian=0)
    NonlinearProblemAge(
                          a.solution + solution,
                          a.residual + residual,
                          a.jacobian + jacobian,
                         )
end

function nonlinear_problem(update,workspace;attributes...)
    NonlinearProblem(update,workspace,attributes)
end

struct NonlinearProblem{A,B,C} <: AbstractNonlinearProblem
    update::A
    workspace::B
    attributes::C
end

function update(p::NonlinearProblem,x;kwargs...)
    workspace = p.update(p.workspace,x;kwargs...)
    NonlinearProblem(p.update,workspace)
end

abstract type AbstractNonlinearSolver <: AbstractType end

workspace(a::AbstractNonlinearProblem) = a.workspace

struct NonlinearSolver{A,B,C,D,E} <: AbstractNonlinearSolver
    step::B
    update::C
    workspace::D
    attributes::E
end

function update(s::NonlinearSolver,p::AbstractNonlinearProblem)
    workspace = s.update(s.workspace,p)
    NonlinearSolver(s.step,s.update,workspace,s.attributes)
end

function step(s::NonlinearSolver)
    next = s.step(s.workspace)
    if next === nothing
        return nothing
    end
    workspace, state = next
    NonlinearSolver(s.update,s.step,workspace,s.attributes), state
end

function step(s::NonlinearSolver,state)
    next = s.step(s.workspace,state)
    if next === nothing
        return nothing
    end
    workspace, state = next
    NonlinearSolver(s.update,s.step,workspace,s.attributes), state
end

abstract type AbstractODEProblem <: AbstractProblem end

#function update(p::AbstractODEProblem;kwargs...)
#    function update_ode_problem(x)
#        update(p,x;kwargs...)
#    end
#end

struct ODEProblemAge <: AbstractAge
    solution::Int
    residual::Int
    jacobian::Int
end

function ode_problem_age()
    ODEProblemAge(0,0,0)
end

function increment(a::ODEProblemAge;solution=0,residual=0,jacobian=0)
    ODEProblemAge(
                          a.solution + solution,
                          a.residual + residual,
                          a.jacobian + jacobian,
                         )
end

function ode_problem(update,workspace;constant_jacobian=Val(false))
    attributes = (;constant_jacobian)
    ODEProblem(update,workspace,attributes)
end

struct ODEProblem{A,B,C} <: AbstractODEProblem
    update::A
    workspace::B
    attributes::C
end

function update(p::ODEProblem,x;kwargs...)
    workspace = p.update(p.workspace,x;kwargs...)
    ODEProblem(p.update,workspace)
end

abstract type AbstractODESolver <: AbstractType end

workspace(a::AbstractODEProblem) = a.workspace

struct ODESolver{A,B,C,D,E} <: AbstractODESolver
    step::B
    update::C
    workspace::D
    attributes::E
end

function update(s::ODESolver,p::AbstractODEProblem)
    workspace = s.update(s.workspace,p)
    ODESolver(s.step,s.update,workspace,s.attributes)
end

function step(s::ODESolver)
    next = s.step(s.workspace)
    if next === nothing
        return nothing
    end
    workspace, state = next
    ODESolver(s.update,s.step,workspace,s.attributes), state
end

function step(s::ODESolver,state)
    next = s.step(s.workspace,state)
    if next === nothing
        return nothing
    end
    workspace, state = next
    ODESolver(s.update,s.step,workspace,s.attributes), state
end










function be(ode)
    t = first(interval(ode))
    w = workspace(ode)
    p = NonlinearProblem(w) do w,u
        x = (t,u,v)
        w = x |> update(ode,(1,1/d)) |> workspace
        w
    end
end











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

struct NonlinearProblem{A,B,C,D,E}
    inplace::A
    solution::B
    residual::C
    jacobian::D
    attributes::E
end

inplace(a) = a.inplace
residual(a) = a.residual
jacobian(a) = a.jacobian

#function update!(p::NonlinearProblem;kwargs...)
#    @assert issubset(propertynames(kwargs),propertynames(p))
#    if hasproperty(kwargs,:inplace)
#        p.inplace = kwargs.inplace
#    end
#    if hasproperty(kwargs,:residual)
#        p.residual = kwargs.residual
#    end
#    if hasproperty(kwargs,:jacobian)
#        p.jacobian = kwargs.jacobian
#    end
#    if hasproperty(kwargs,:solution)
#        p.solution = kwargs.solution
#    end
#    if hasproperty(kwargs,:attributes)
#        p.attributes = kwargs.attributes
#    end
#    p
#end

function ode_problem(args...;nullspace=nothing,block_size=1,constant_jacobian=false)
    attributes = (;nullspace,block_size,constant_jacobian)
    ODEProblem(args...,attributes)
end

struct ODEProblem{A,B,C,D,E,F}
    inplace::A
    interval::B
    solution::C
    residual::D
    jacobian::E
    attributes::F
end

interval(a) = a.interval

#function update!(p::ODEProblem;kwargs...)
#    @assert issubset(propertynames(kwargs),propertynames(p))
#    if hasproperty(kwargs,:inplace)
#        p.inplace = kwargs.inplace
#    end
#    if hasproperty(kwargs,:residual)
#        p.residual = kwargs.residual
#    end
#    if hasproperty(kwargs,:jacobian)
#        p.jacobian = kwargs.jacobian
#    end
#    if hasproperty(kwargs,:solution)
#        p.solution = kwargs.solution
#    end
#    if hasproperty(kwargs,:attributes)
#        p.attributes = kwargs.attributes
#    end
#    if hasproperty(kwargs,:interval)
#        p.interval = kwargs.interval
#    end
#    p
#end

function solve!(solver;kwargs...)
    next = step!(solver;kwargs...)
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
    kwargs2 = solver.update!(;kwargs...)
    update!(solver.problem,kwargs2...)
    solver
end

function finalize!(solver::Solver)
    solver.step!()
end

function LinearAlgebra.ldiv!(x,solver::Solver,b)
    if uses_initial_guess(solver.attributes)
        fill!(x,zero(eltype(x)))
    end
    smooth!(x,solver,b;zero_guess=true)
    x
end

function smooth!(x,solver,b;kwargs...)
    update!(solver,solution=x,rhs=b)
    solve!(solver;kwargs...)
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
        if hasproperty(kwargs,:matrix)
            lu!(F,kwargs.matrix)
        end
        kwargs
    end
    function lu_finalize!()
        nothing
    end
    uses_initial_guess = false
    solver(problem,lu_step!,lu_update!,lu_finalize!;uses_initial_guess)
end

function identity_solver(problem)
    function id_step!(phase=:start)
        if phase === :stop
            return nothing
        end
        x = solution(problem)
        b = rhs(problem)
        copyto!(x,b)
        phase = :stop
        phase
    end
    function id_update!(;kwargs...)
        kwargs
    end
    function id_finalize!()
        nothing
    end
    uses_initial_guess = false
    solver(problem,id_step!,id_update!,id_finalize!;uses_initial_guess)
end

function jacobi_correction(problem)
    Adiag = dense_diag(matrix(problem))
    function jc_step!(phase=:start)
        if phase === :stop
            return nothing
        end
        x = solution(problem)
        b = rhs(problem)
        x .= Adiag .\ b
        phase = :stop
        phase
    end
    function jc_update!(;kwargs...)
        if hasproperty(kwargs,:matrix)
            dense_diag!(Adiag,kwargs.matrix)
        end
        kwargs
    end
    function jc_finalize!()
        nothing
    end
    uses_initial_guess = false
    solver(problem,id_step!,id_update!,id_finalize!;uses_initial_guess)
end

function richardson(problem;
    P = identity_solver(problem),
    iterations = 10,
    omega = 1,
    )
    iteration = 0
    b = rhs(problem)
    A = matrix(problem)
    x = solution(problem)
    dx = similar(x,axes(A,2))
    r = similar(b,axes(A,1))
    function rc_step!(phase=:start;zero_guess=false)
        @assert phase in (:start,:stop,:advance)
        if phase === :stop
            return nothing
        end
        if phase === :start
            iteration = 0
            phase = :advance
        end
        b = rhs(problem)
        A = matrix(problem)
        x = solution(problem)
        dx .= x
        if zero_guess
            r .= .- b
        else
            mul!(r,A,dx)
            r .-= b
        end
        ldiv!(dx,P,r)
        x .-= omega .* dx
        iteration += 1
        if iteration == iterations
            phase = :stop
        end
        phase
    end
    function rc_update!(;kwargs...)
        kwargs
    end
    function rc_finalize!()
        nothing
    end
    solver(problem,id_step!,id_update!,id_finalize!)
end

function jacobi(problem;iterations=10,omega=1)
    P = jacobi_correction(problem)
    R = richardson(problem;P,iterations,omega)
    function ja_step!(args...)
        step!(R,args...)
    end
    function update!(;kwargs...)
        update!(P;kwargs...)
        update!(R;kwargs...)
        kwargs
    end
    function rc_finalize!()
        finalize!(P)
        finalize!(R)
    end
    solver(problem,ja_step!,ja_update!,ja_finalize!)
end

function convergence(;kwargs...)
    convergence(Float64;kwargs...)
end

function convergence(::Type{T};
    iterations = 1000,
    abs_res_tol = typemax(T),
    rel_res_tol = T(1e-12),
    abs_sol_tol = zero(T),
    res_norm = norm,
    sol_norm = dx -> maximum(abs,dx)
    ) where T
    (;iterations,abs_res_tol,rel_res_tol,abs_sol_tol,res_norm,sol_norm)
end

function verbosity(;
        level=0,
        prefix="")
    (;level,prefix)
end

function status(params)
    (;abs_res_tol,abs_sol_tol,rel_res_tol,iterations) = params
    res_target = zero(abs_res_tol)
    sol_target = zero(abs_sol_tol)
    iteration = 0
    status = Status(iterations,iteration,res_target,sol_target,res_error,sol_error)
    start!(status,res_error,sol_error)
    status
end

mutable struct Status{T}
    iterations::Int
    iteration::Int
    res_target::T
    sol_target::T
    res_error::T
    sol_error::T
end

function start!(status::Status,res_error,sol_error)
    res_target = min(abs_res_tol,res_error*rel_res_tol)
    sol_target = abs_sol_tol
    iteration = 0
end

function step!(status::Status,res_error,sol_error)
    status.iteration += 1
    status.res_error = res_error
    status.sol_error = sol_error
    status
end

function tired(status)
    status.iteration >= status.iterations
end

function converged(status)
    status.res_error <= status.res_target || status.sol_error <= status.sol_target
end

function print_progress(verbosity,status)
    if verbosity.level > 0
        s = verbosity.prefix
        @printf "%s%6i %6i %12.3e %12.3e\n" s status.iteration status.iterations status.res_error a.res_target
    end
end

#struct Usage
#    num_updates::Dict{Symbol,Int}
#end
#
#function usage()
#    num_updates = Dict{Symbol,Int}()
#    Usage(num_updates)
#end
#
#function start!(usage::Usage)
#    for k in keys(usage.num_updates)
#        usage.num_updates[k] = 0
#    end
#    usage
#end
#
#function update!(usage::Usage;kwargs...)
#    for k in propertynames(kwargs)
#        if ! haskey(usage.num_updates,k)
#            usage.num_updates[k] = 0
#        end
#        usage.num_updates[k] += 1
#    end
#    usage
#end

function newton_raphson(problem;
        solver=lp->LinearAlgebra_lu(lp),
        convergence = PartitionedSolvers.convergence(eltype(solution(problem))),
        verbosity = PartitionedSolvers.verbosity(),
    )
    x = solution(problem)
    J = jacobian(problem)
    r = residual(problem)
    dx = similar(x,axes(J,2))
    lp = linear_problem(dx,J,r)
    S = solver(lp)
    status = PartitionedSolvers.status(convergence)
    function nr_step!(phase=:start;kwargs...)
        @assert phase in (:start,:stop,:advance)
        if phase === :stop
            return nothing
        end
        x = solution(problem)
        J = jacobian(problem)
        r = residual(problem)
        rj! = inplace(problem)
        if phase === :start
            rj!(r,J,x)
            res_error = convergence.res_norm(r)
            sol_error = typemax(res_error)
            start!(status,res_error,sol_error)
            print_progress(verbosity,status)
            phase = :advance
        end
        update!(S,matrix=J)
        dx = solution(S)
        ldiv!(dx,S,b)
        x .-= dx
        rj!(r,J,x)
        res_error = convergence.res_norm(r)
        sol_error = convergence.sol_norm(dx)
        step!(status,res_error,sol_error)
        print_progress(verbosity,status)
        if converged(status) || tired(status)
            phase = :stop
        end
        phase
    end
    function nr_update!(;kwargs...)
        kwargs
    end
    function nr_finalize!()
        nothing
    end
    PartitionedSolvers.solver(problem,nr_step!,nr_update!,nr_finalize!)
end

function print_time_step(verbosity,t,tend)
    if verbosity.level > 0
        s = verbosity.prefix
        @printf "%s%12.3e %12.3e\n" s t tend
    end
end

function backward_euler(ode;
        dt = (interval(ode)[2]-interval(ode)[1])/100,
        solver = constant_jacobian(ode) ? LinearAlgebra_lu : newton_raphson,
        verbosity = PartitionedSolvers.verbosity(),
    )

    (t,u,v) = solution(ode)
    x = copy(u)
    J = jacobian(ode)
    r = residual(ode)
    attrs = attributes(problem)
    rj! = inplace(problem)
    if constant_jacobian(ode)
        rj!(r,j,(t,u,v),(1,1/dt))
        lp = linear_problem(x,J,r;attrs...)
        S = solver(lp)
    else
        nlp = nonlinear_problem(x,J,r;attrs...) do r,j,x
            v .= (x .- u) ./ dt
            rj!(r,j,(t,x,v),(1,1/dt))
        end
        S = solver(nlp)
    end
    function be_step!(phase=:start;kwargs...)
        @assert phase in (:start,:stop,:advance)
        if phase === :stop
            return nothing
        end
        J = jacobian(ode)
        r = residual(ode)
        tend = last(interval(ode))
        if phase === :start
            t = first(interval(ode))
            phase = :advance
            if constant_jacobian
                rj!(r,J,(t,u,v),(1,1/dt))
            end
            print_time_step(verbosity,t,tend)
        end
        x = solve!(S)
        v .= (x .- u) ./ dt
        u .= x
        t += dt
        if constant_jacobian(ode)
            rj!(r,nothing,(t,u,v),(1,1/dt))
        end
        print_time_step(verbosity,t,tend)
        if t >= tend
            phase = :stop
        end
    end
    function be_update!(;kwargs...)
        kwargs
    end
    function be_finalize!()
        nothing
    end
    PartitionedSolvers.solver(problem,be_step!,be_update!,be_finalize!)
end

