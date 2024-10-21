
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

