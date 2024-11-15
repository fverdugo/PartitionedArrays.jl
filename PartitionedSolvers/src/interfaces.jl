
abstract type AbstractType end
function Base.show(io::IO,data::AbstractType)
    print(io,"PartitionedSolvers.$(nameof(typeof(data)))(…)")
end

abstract type AbstractProblem <: AbstractType end
abstract type AbstractSolver <: AbstractType end
#abstract type AbstractAge <: AbstractType end

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

function solve(solver::AbstractSolver;kwargs...)
    solver, state = step(solver;kwargs...)
    while state !== :stop
        solver, state = step(solver,state)
    end
    solver
end

function history(solver::AbstractSolver;kwargs...)
    History(identity,solver,kwargs)
end

function history(f,solver::AbstractSolver;kwargs...)
    History(f,solver,kwargs)
end

function solve(p::AbstractProblem;kwargs...)
    s = default_solver(p)
    solve(s;kwargs...)
end

function history(p::AbstractProblem;kwargs...)
    s = default_solver(p)
    history(s;kwargs...)
end

function history(f,p::AbstractProblem;kwargs...)
    s = default_solver(p)
    history(f,s;kwargs...)
end

struct History{A,B,C}
    f::A
    solver::B
    kwargs::C
end

function Base.iterate(a::History)
    solver, state = step(a.solver;a.kwargs...)
    a.f(solver), (solver,state)
end

function Base.iterate(a::History,(solver,state))
    if state === :stop
        return nothing
    end
    solver, state = step(solver,state)
    a.f(solver), (solver,state)
end

solution(a) = a.solution
jacobian(a) = a.jacobian
residual(a) = a.residual
attributes(a) = a.attributes
problem(a) = a.problem
matrix(a) = a.matrix
rhs(a) = a.rhs
statement(a) = a.statement
workspace(a) = a.workspace
interval(a) = a.interval
coefficients(a) = a.coefficients
uses_initial_guess(a) = val_parameter(a.uses_initial_guess)
constant_jacobian(a) = val_parameter(a.constant_jacobian)
uses_mutable_types(a) = val_parameter(a.uses_mutable_types)

uses_initial_guess(a::AbstractSolver) = uses_initial_guess(attributes(a))
constant_jacobian(a::AbstractProblem) = constant_jacobian(attributes(a))
uses_mutable_types(a::AbstractProblem) = uses_mutable_types(attributes(a))

#solution(a::AbstractProblem) = solution(workspace(a))
#jacobian(a::AbstractProblem) = jacobian(workspace(a))
#residual(a::AbstractProblem) = residual(workspace(a))
#age(a::AbstractProblem) = age(workspace(a))
#matrix(a::AbstractProblem) = matrix(workspace(a))
#rhs(a::AbstractProblem) = rhs(workspace(a))

solution(a::AbstractSolver) = solution(problem(a))
matrix(a::AbstractSolver) = matrix(problem(a))
rhs(a::AbstractSolver) = rhs(problem(a))
residual(a::AbstractSolver) = residual(problem(a))
jacobian(a::AbstractSolver) = jacobian(problem(a))

abstract type AbstractLinearProblem <: AbstractProblem end

default_solver(p::AbstractLinearProblem) = LinearAlgebra_lu(p)

#struct LinearProblemAge <: AbstractAge
#    solution::Int
#    matrix::Int
#    rhs::Int
#end
#
#function linear_problem_age()
#    LinearProblemAge(1,1,1)
#end
#
#function increment(a::LinearProblemAge;solution=0,matrix=0,rhs=0)
#    LinearProblemAge(
#                          a.solution + solution,
#                          a.matrix + matrix,
#                          a.rhs + rhs,
#                         )
#end

function linear_problem(solution,matrix,rhs;uses_mutable_types=Val(true),nullspace=nothing)
    attributes = (;uses_mutable_types,nullspace)
    LinearProblem(solution,matrix,rhs,attributes)
end

struct LinearProblem{A,B,C,D} <: AbstractLinearProblem
    solution::A
    matrix::B
    rhs::C
    attributes::D
end

nullspace(a::LinearProblem) = a.attributes.nullspace

function update(p::LinearProblem;kwargs...)
    data = (;kwargs...)
    if hasproperty(data,:matrix)
        A = data.matrix
    else
        A = matrix(p)
    end
    if hasproperty(data,:rhs)
        b = data.rhs
    else
        b = rhs(p)
    end
    if hasproperty(data,:solution)
        x = data.solution
    else
        x = solution(p)
    end
    if hasproperty(data,:attributes)
        attrs = data.attributes
    else
        attrs = attributes(p)
    end
    LinearProblem(x,A,b,attrs)
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
    s = update(s,solution=x,rhs=b)
    s = solve(s;kwargs...)
    x
end

function linear_solver(args...;
        uses_initial_guess = Val(true),
    )
    attributes = (;uses_initial_guess)
    LinearSolver(args...,attributes)
end

struct LinearSolver{A,B,C,D,E} <: AbstractLinearSolver
    update::A
    step::B
    problem::C
    workspace::D
    attributes::E
end

function update(s::LinearSolver;problem=s.problem,kwargs...)
    p = update(problem;kwargs...)
    workspace = s.workspace
    if haskey(kwargs,:matrix) || problem !== s.problem
        workspace = s.update(workspace,matrix(p))
    end
    LinearSolver(s.update,s.step,p,workspace,s.attributes)
end

function step(s::LinearSolver;kwargs...)
    p = s.problem
    x = solution(p)
    b = rhs(p)
    next = s.step(x,s.workspace,b;kwargs...)
    #if next === nothing
    #    return nothing
    #end
    x,workspace,state = next
    p = update(p,solution=x)
    s = LinearSolver(s.update,s.step,p,workspace,s.attributes)
    s,state
end

function step(s::LinearSolver,state)
    p = s.problem
    x = solution(p)
    b = rhs(p)
    next = s.step(x,s.workspace,b,state)
    #if next === nothing
    #    return nothing
    #end
    x,workspace,state = next
    p = update(p,solution=x)
    s = LinearSolver(s.update,s.step,p,workspace,s.attributes)
    s,state
end

function preconditioner(solver,p)
    dx = similar(solution(p),axes(matrix(p),2))
    r = similar(rhs(p),axes(matrix(p),1))
    dp = update(p,solution=dx,rhs=r)
    solver(dp)
end

abstract type AbstractNonlinearProblem <: AbstractProblem end

default_solver(p::AbstractNonlinearProblem) = newton_raphson(p)

#function update(p::AbstractNonlinearProblem;kwargs...)
#    function update_nonlinear_problem(x)
#        update(p,x;kwargs...)
#    end
#end

#struct NonlinearProblemAge <: AbstractAge
#    solution::Int
#    residual::Int
#    jacobian::Int
#end
#
#function nonlinear_problem_age()
#    NonlinearProblemAge(0,0,0)
#end
#
#function increment(a::NonlinearProblemAge;solution=0,residual=0,jacobian=0)
#    NonlinearProblemAge(
#                          a.solution + solution,
#                          a.residual + residual,
#                          a.jacobian + jacobian,
#                         )
#end

function nonlinear_problem(args...;uses_mutable_types=Val(true))
    attributes = (;uses_mutable_types)
    NonlinearProblem(args...,attributes)
end

struct NonlinearProblem{A,B,C,D,E,F} <: AbstractNonlinearProblem
    statement::A
    solution::B
    residual::C
    jacobian::D
    workspace::E
    attributes::F
end

#function linear_problem(p::NonlinearProblem)
#    x = p.solution
#    r = p.residual
#    j = p.jacobian
#    attrs = p.attributes
#    dx = similar(x,axes(j,2))
#    linear_problem(dx,j,r;attrs...)
#end
#
#function update(lp::LinearProblem,p::NonlinearProblem)
#    r = p.residual
#    j = p.jacobian
#    update(lp,matrix=j,rhs=j)
#end

function set(p::NonlinearProblem;kwargs...)
    data = (;kwargs...)
    if hasproperty(data,:statement)
        st = data.statement
    else
        st = statement(p)
    end
    if hasproperty(data,:residual)
        b = data.residual
    else
        b = residual(p)
    end
    if hasproperty(data,:jacobian)
        A = data.jacobian
    else
        A = jacobian(p)
    end
    if hasproperty(data,:solution)
        x = data.solution
    else
        x = solution(p)
    end
    if hasproperty(data,:attributes)
        attrs = data.attributes
    else
        attrs = attributes(p)
    end
    if hasproperty(data,:workspace)
        ws = data.workspace
    else
        ws = workspace(p)
    end
    NonlinearProblem(st,x,b,A,ws,attrs)
end

function update(p::NonlinearProblem;kwargs...)
    p = set(p;kwargs...)
    q = set(p,statement=Base.identity)
    q = p.statement(q)
    p = set(q,statement=p.statement)
    p
end

abstract type AbstractNonlinearSolver <: AbstractSolver end

function nonlinear_solver(args...;attributes...)
    NonlinearSolver(args...,attributes)
end

struct NonlinearSolver{A,B,C,D,E} <: AbstractNonlinearSolver
    update::A
    step::B
    problem::C
    workspace::D
    attributes::E
end

function update(s::NonlinearSolver;problem=s.problem,kwargs...)
    p = update(problem;kwargs...)
    workspace = s.update(s.workspace,p)
    NonlinearSolver(s.update,s.step,p,workspace,s.attributes)
end

function step(s::NonlinearSolver;kwargs...)
    next = s.step(s.workspace,s.problem;kwargs...)
    #if next === nothing
    #    return nothing
    #end
    workspace,p,state = next
    s = NonlinearSolver(s.update,s.step,p,workspace,s.attributes)
    s,state
end

function step(s::NonlinearSolver,state)
    next = s.step(s.workspace,s.problem,state)
    #if next === nothing
    #    return nothing
    #end
    workspace,p,state = next
    s = NonlinearSolver(s.update,s.step,p,workspace,s.attributes)
    s,state
end

abstract type AbstractODEProblem <: AbstractProblem end

#function update(p::AbstractODEProblem;kwargs...)
#    function update_ode_problem(x)
#        update(p,x;kwargs...)
#    end
#end

#struct ODEProblemAge <: AbstractAge
#    solution::Int
#    residual::Int
#    jacobian::Int
#end
#
#function ode_problem_age()
#    ODEProblemAge(0,0,0)
#end
#
#function increment(a::ODEProblemAge;solution=0,residual=0,jacobian=0)
#    ODEProblemAge(
#                          a.solution + solution,
#                          a.residual + residual,
#                          a.jacobian + jacobian,
#                         )
#end


function ode_problem(args...;constant_jacobian=Val(false),uses_mutable_types=Val(true))
    attributes = (;constant_jacobian,uses_mutable_types)
    ODEProblem(args...,attributes)
end

struct ODEProblem{A,B,C,D,E,F,G,H} <: AbstractODEProblem
    statement::A
    solution::B
    residual::C
    jacobian::D
    interval::E
    coefficients::F
    workspace::G
    attributes::H
end

function set(p::ODEProblem;kwargs...)
    data = (;kwargs...)
    if hasproperty(data,:statement)
        st = data.statement
    else
        st = statement(p)
    end
    if hasproperty(data,:residual)
        b = data.residual
    else
        b = residual(p)
    end
    if hasproperty(data,:jacobian)
        A = data.jacobian
    else
        A = jacobian(p)
    end
    if hasproperty(data,:solution)
        x = data.solution
    else
        x = solution(p)
    end
    if hasproperty(data,:attributes)
        attrs = data.attributes
    else
        attrs = attributes(p)
    end
    if hasproperty(data,:interval)
        i = data.interval
    else
        i = interval(p)
    end
    if hasproperty(data,:coefficients)
        c = data.coefficients
    else
        c = coefficients(p)
    end
    p = ODEProblem(st,x,b,A,i,c,p.workspace,attrs)
end

function update(p::ODEProblem;kwargs...)
    p = set(p;kwargs...)
    q = set(p,statement=Base.identity)
    q = p.statement(q)
    p = set(q,statement=p.statement)
    p
end

abstract type AbstractODESolver <: AbstractSolver end

function ode_solver(args...;attributes...)
    ODESolver(args...,attributes)
end

struct ODESolver{A,B,C,D,E} <: AbstractODESolver
    update::A
    step::B
    problem::C
    workspace::D
    attributes::E
end

function update(s::ODESolver;problem=s.problem,kwargs...)
    p = update(problem;kwargs...)
    workspace = s.update(s.workspace,p)
    ODESolver(s.update,s.step,p,workspace,s.attributes)
end

function step(s::ODESolver;kwargs...)
    next = s.step(s.workspace,s.problem;kwargs...)
    #if next === nothing
    #    return nothing
    #end
    workspace,p,state = next
    s = ODESolver(s.update,s.step,p,workspace,s.attributes)
    s,state
end

function step(s::ODESolver,state)
    next = s.step(s.workspace,s.problem,state)
    #if next === nothing
    #    return nothing
    #end
    workspace,p,state = next
    s = ODESolver(s.update,s.step,p,workspace,s.attributes)
    s,state
end

#function linear_problem(args...;nullspace=nothing,block_size=1)
#    attributes = (;nullspace,block_size)
#    LinearProblem(args...,attributes)
#end
#
#mutable struct LinearProblem{A,B,C,D}
#    solution::A
#    matrix::B
#    rhs::C
#    attributes::D
#end
#
#solution(a) = a.solution
#matrix(a) = a.matrix
#rhs(a) = a.rhs
#attributes(a) = a.attributes
#
#function update!(p::LinearProblem;kwargs...)
#    @assert issubset(propertynames(kwargs),propertynames(p))
#    if hasproperty(kwargs,:matrix)
#        p.matrix = kwargs.matrix
#    end
#    if hasproperty(kwargs,:rhs)
#        p.rhs = kwargs.rhs
#    end
#    if hasproperty(kwargs,:solution)
#        p.solution = kwargs.solution
#    end
#    if hasproperty(kwargs,:attributes)
#        p.attributes = kwargs.attributes
#    end
#    p
#end
#
#function nonlinear_problem(args...;nullspace=nothing,block_size=1)
#    attributes = (;nullspace,block_size)
#    NonlinearProblem(args...,attributes)
#end
#
#struct NonlinearProblem{A,B,C,D,E}
#    statement::A
#    solution::B
#    residual::C
#    jacobian::D
#    attributes::E
#end
#
#statement(a) = a.statement
#residual(a) = a.residual
#jacobian(a) = a.jacobian
#
##function update!(p::NonlinearProblem;kwargs...)
##    @assert issubset(propertynames(kwargs),propertynames(p))
##    if hasproperty(kwargs,:statement)
##        p.statement = kwargs.statement
##    end
##    if hasproperty(kwargs,:residual)
##        p.residual = kwargs.residual
##    end
##    if hasproperty(kwargs,:jacobian)
##        p.jacobian = kwargs.jacobian
##    end
##    if hasproperty(kwargs,:solution)
##        p.solution = kwargs.solution
##    end
##    if hasproperty(kwargs,:attributes)
##        p.attributes = kwargs.attributes
##    end
##    p
##end
#
#function ode_problem(args...;nullspace=nothing,block_size=1,constant_jacobian=false)
#    attributes = (;nullspace,block_size,constant_jacobian)
#    ODEProblem(args...,attributes)
#end
#
#struct ODEProblem{A,B,C,D,E,F}
#    statement::A
#    interval::B
#    solution::C
#    residual::D
#    jacobian::E
#    attributes::F
#end
#
#interval(a) = a.interval
#
##function update!(p::ODEProblem;kwargs...)
##    @assert issubset(propertynames(kwargs),propertynames(p))
##    if hasproperty(kwargs,:statement)
##        p.statement = kwargs.statement
##    end
##    if hasproperty(kwargs,:residual)
##        p.residual = kwargs.residual
##    end
##    if hasproperty(kwargs,:jacobian)
##        p.jacobian = kwargs.jacobian
##    end
##    if hasproperty(kwargs,:solution)
##        p.solution = kwargs.solution
##    end
##    if hasproperty(kwargs,:attributes)
##        p.attributes = kwargs.attributes
##    end
##    if hasproperty(kwargs,:interval)
##        p.interval = kwargs.interval
##    end
##    p
##end
#
#function solve!(solver;kwargs...)
#    next = step!(solver;kwargs...)
#    while next !== nothing
#        solver,phase = next
#        next = step!(solver,next)
#    end
#    solution(solver)
#end
#
#function history(solver)
#    History(solver)
#end
#
#struct History{A}
#    solver::A
#end
#
#function Base.iterate(a::History)
#    next = step!(a.solver)
#    if next === nothing
#        return nothing
#    end
#    solution(problem(a.solver)), next
#end
#
#function Base.iterate(a::History,next)
#    next = step!(a.solver,next)
#    if next === nothing
#        return nothing
#    end
#    solution(problem(a.solver)), next
#end
#
#struct Solver{A,B,C,D,E}
#    problem::A
#    step!::B
#    update!::C
#    finalize!::D
#    attributes::E
#end
#
#uses_initial_guess(a) = a.uses_initial_guess
#
#function solver(args...;uses_initial_guess=Val(true))
#    attributes = (;uses_initial_guess)
#    Solver(args...,attributes)
#end
#
#function step!(solver::Solver)
#    solver.step!()
#end
#
#function step!(solver::Solver,state)
#    solver.step!(state)
#end
#
#function update!(solver::Solver;kwargs...)
#    kwargs2 = solver.update!(;kwargs...)
#    update!(solver.problem,kwargs2...)
#    solver
#end
#
#function finalize!(solver::Solver)
#    solver.step!()
#end
#
#function LinearAlgebra.ldiv!(x,solver::Solver,b)
#    if uses_initial_guess(solver.attributes)
#        fill!(x,zero(eltype(x)))
#    end
#    smooth!(x,solver,b;zero_guess=true)
#    x
#end
#
#function smooth!(x,solver,b;kwargs...)
#    update!(solver,solution=x,rhs=b)
#    solve!(solver;kwargs...)
#    x
#end
#
#function LinearAlgebra_lu(problem)
#    F = lu(matrix(problem))
#    function lu_step!(phase=:start)
#        if phase === :stop
#            return nothing
#        end
#        x = solution(problem)
#        b = rhs(problem)
#        ldiv!(x,F,b)
#        phase = :stop
#        phase
#    end
#    function lu_update!(;kwargs...)
#        if hasproperty(kwargs,:matrix)
#            lu!(F,kwargs.matrix)
#        end
#        kwargs
#    end
#    function lu_finalize!()
#        nothing
#    end
#    uses_initial_guess = false
#    solver(problem,lu_step!,lu_update!,lu_finalize!;uses_initial_guess)
#end
#
#function identity_solver(problem)
#    function id_step!(phase=:start)
#        if phase === :stop
#            return nothing
#        end
#        x = solution(problem)
#        b = rhs(problem)
#        copyto!(x,b)
#        phase = :stop
#        phase
#    end
#    function id_update!(;kwargs...)
#        kwargs
#    end
#    function id_finalize!()
#        nothing
#    end
#    uses_initial_guess = false
#    solver(problem,id_step!,id_update!,id_finalize!;uses_initial_guess)
#end
#
#function jacobi_correction(problem)
#    Adiag = dense_diag(matrix(problem))
#    function jc_step!(phase=:start)
#        if phase === :stop
#            return nothing
#        end
#        x = solution(problem)
#        b = rhs(problem)
#        x .= Adiag .\ b
#        phase = :stop
#        phase
#    end
#    function jc_update!(;kwargs...)
#        if hasproperty(kwargs,:matrix)
#            dense_diag!(Adiag,kwargs.matrix)
#        end
#        kwargs
#    end
#    function jc_finalize!()
#        nothing
#    end
#    uses_initial_guess = false
#    solver(problem,id_step!,id_update!,id_finalize!;uses_initial_guess)
#end
#
#function richardson(problem;
#    P = identity_solver(problem),
#    iterations = 10,
#    omega = 1,
#    )
#    iteration = 0
#    b = rhs(problem)
#    A = matrix(problem)
#    x = solution(problem)
#    dx = similar(x,axes(A,2))
#    r = similar(b,axes(A,1))
#    function rc_step!(phase=:start;zero_guess=false)
#        @assert phase in (:start,:stop,:advance)
#        if phase === :stop
#            return nothing
#        end
#        if phase === :start
#            iteration = 0
#            phase = :advance
#        end
#        b = rhs(problem)
#        A = matrix(problem)
#        x = solution(problem)
#        dx .= x
#        if zero_guess
#            r .= .- b
#        else
#            mul!(r,A,dx)
#            r .-= b
#        end
#        ldiv!(dx,P,r)
#        x .-= omega .* dx
#        iteration += 1
#        if iteration == iterations
#            phase = :stop
#        end
#        phase
#    end
#    function rc_update!(;kwargs...)
#        kwargs
#    end
#    function rc_finalize!()
#        nothing
#    end
#    solver(problem,id_step!,id_update!,id_finalize!)
#end
#
#function jacobi(problem;iterations=10,omega=1)
#    P = jacobi_correction(problem)
#    R = richardson(problem;P,iterations,omega)
#    function ja_step!(args...)
#        step!(R,args...)
#    end
#    function update!(;kwargs...)
#        update!(P;kwargs...)
#        update!(R;kwargs...)
#        kwargs
#    end
#    function rc_finalize!()
#        finalize!(P)
#        finalize!(R)
#    end
#    solver(problem,ja_step!,ja_update!,ja_finalize!)
#end
#
#function convergence(;kwargs...)
#    convergence(Float64;kwargs...)
#end
#
#function convergence(::Type{T};
#    iterations = 1000,
#    abs_res_tol = typemax(T),
#    rel_res_tol = T(1e-12),
#    abs_sol_tol = zero(T),
#    res_norm = norm,
#    sol_norm = dx -> maximum(abs,dx)
#    ) where T
#    (;iterations,abs_res_tol,rel_res_tol,abs_sol_tol,res_norm,sol_norm)
#end
#
#function verbosity(;
#        level=0,
#        prefix="")
#    (;level,prefix)
#end
#
#function status(params)
#    (;abs_res_tol,abs_sol_tol,rel_res_tol,iterations) = params
#    res_target = zero(abs_res_tol)
#    sol_target = zero(abs_sol_tol)
#    iteration = 0
#    status = Status(iterations,iteration,res_target,sol_target,res_error,sol_error)
#    start!(status,res_error,sol_error)
#    status
#end
#
#mutable struct Status{T}
#    iterations::Int
#    iteration::Int
#    res_target::T
#    sol_target::T
#    res_error::T
#    sol_error::T
#end
#
#function start!(status::Status,res_error,sol_error)
#    res_target = min(abs_res_tol,res_error*rel_res_tol)
#    sol_target = abs_sol_tol
#    iteration = 0
#end
#
#function step!(status::Status,res_error,sol_error)
#    status.iteration += 1
#    status.res_error = res_error
#    status.sol_error = sol_error
#    status
#end
#
#function tired(status)
#    status.iteration >= status.iterations
#end
#
#function converged(status)
#    status.res_error <= status.res_target || status.sol_error <= status.sol_target
#end
#
#function print_progress(verbosity,status)
#    if verbosity.level > 0
#        s = verbosity.prefix
#        @printf "%s%6i %6i %12.3e %12.3e\n" s status.iteration status.iterations status.res_error a.res_target
#    end
#end
#
##struct Usage
##    num_updates::Dict{Symbol,Int}
##end
##
##function usage()
##    num_updates = Dict{Symbol,Int}()
##    Usage(num_updates)
##end
##
##function start!(usage::Usage)
##    for k in keys(usage.num_updates)
##        usage.num_updates[k] = 0
##    end
##    usage
##end
##
##function update!(usage::Usage;kwargs...)
##    for k in propertynames(kwargs)
##        if ! haskey(usage.num_updates,k)
##            usage.num_updates[k] = 0
##        end
##        usage.num_updates[k] += 1
##    end
##    usage
##end
#
#function newton_raphson(problem;
#        solver=lp->LinearAlgebra_lu(lp),
#        convergence = PartitionedSolvers.convergence(eltype(solution(problem))),
#        verbosity = PartitionedSolvers.verbosity(),
#    )
#    x = solution(problem)
#    J = jacobian(problem)
#    r = residual(problem)
#    dx = similar(x,axes(J,2))
#    lp = linear_problem(dx,J,r)
#    S = solver(lp)
#    status = PartitionedSolvers.status(convergence)
#    function nr_step!(phase=:start;kwargs...)
#        @assert phase in (:start,:stop,:advance)
#        if phase === :stop
#            return nothing
#        end
#        x = solution(problem)
#        J = jacobian(problem)
#        r = residual(problem)
#        rj! = statement(problem)
#        if phase === :start
#            rj!(r,J,x)
#            res_error = convergence.res_norm(r)
#            sol_error = typemax(res_error)
#            start!(status,res_error,sol_error)
#            print_progress(verbosity,status)
#            phase = :advance
#        end
#        update!(S,matrix=J)
#        dx = solution(S)
#        ldiv!(dx,S,b)
#        x .-= dx
#        rj!(r,J,x)
#        res_error = convergence.res_norm(r)
#        sol_error = convergence.sol_norm(dx)
#        step!(status,res_error,sol_error)
#        print_progress(verbosity,status)
#        if converged(status) || tired(status)
#            phase = :stop
#        end
#        phase
#    end
#    function nr_update!(;kwargs...)
#        kwargs
#    end
#    function nr_finalize!()
#        nothing
#    end
#    PartitionedSolvers.solver(problem,nr_step!,nr_update!,nr_finalize!)
#end
#
#function print_time_step(verbosity,t,tend)
#    if verbosity.level > 0
#        s = verbosity.prefix
#        @printf "%s%12.3e %12.3e\n" s t tend
#    end
#end
#
#function backward_euler(ode;
#        dt = (interval(ode)[2]-interval(ode)[1])/100,
#        solver = constant_jacobian(ode) ? LinearAlgebra_lu : newton_raphson,
#        verbosity = PartitionedSolvers.verbosity(),
#    )
#
#    (t,u,v) = solution(ode)
#    x = copy(u)
#    J = jacobian(ode)
#    r = residual(ode)
#    attrs = attributes(problem)
#    rj! = statement(problem)
#    if constant_jacobian(ode)
#        rj!(r,j,(t,u,v),(1,1/dt))
#        lp = linear_problem(x,J,r;attrs...)
#        S = solver(lp)
#    else
#        nlp = nonlinear_problem(x,J,r;attrs...) do r,j,x
#            v .= (x .- u) ./ dt
#            rj!(r,j,(t,x,v),(1,1/dt))
#        end
#        S = solver(nlp)
#    end
#    function be_step!(phase=:start;kwargs...)
#        @assert phase in (:start,:stop,:advance)
#        if phase === :stop
#            return nothing
#        end
#        J = jacobian(ode)
#        r = residual(ode)
#        tend = last(interval(ode))
#        if phase === :start
#            t = first(interval(ode))
#            phase = :advance
#            if constant_jacobian
#                rj!(r,J,(t,u,v),(1,1/dt))
#            end
#            print_time_step(verbosity,t,tend)
#        end
#        x = solve!(S)
#        v .= (x .- u) ./ dt
#        u .= x
#        t += dt
#        if constant_jacobian(ode)
#            rj!(r,nothing,(t,u,v),(1,1/dt))
#        end
#        print_time_step(verbosity,t,tend)
#        if t >= tend
#            phase = :stop
#        end
#    end
#    function be_update!(;kwargs...)
#        kwargs
#    end
#    function be_finalize!()
#        nothing
#    end
#    PartitionedSolvers.solver(problem,be_step!,be_update!,be_finalize!)
#end

