
struct Status <: AbstractType
    steps::Int
    steps_since_update::Int
    updates::Int
end

function Status()
    Status(0,0,0)
end

function update(status::Status)
    steps = status.steps
    steps_since_update = 0
    updates = status.updates + 1
    Status(steps,steps_since_update,updates)
end

function step(status::Status)
    steps = status.steps + 1
    steps_since_update = status.steps_since_update + 1
    updates = status.updates
    Status(steps,steps_since_update,updates)
end

function Base.show(io::IO,k::MIME"text/plain",data::Status)
    println("Updates since creation: $(data.updates)")
    println("Solver steps since creation: $(data.steps)")
    println("Solver steps since last update: $(data.steps_since_update)")
end

function status(a)
    workspace(a).status
end

struct Convergence{T}
    current::T
    target::T
    iteration::Int
    iterations::Int
end

function Convergence(iterations,target)
    current = zero(typeof(target))
    iteration = 0
    Convergence(current,target,iteration,iterations)
end

function print_progress(a::Convergence,verbosity)
    s =verbosity.indentation
    @printf "%s%6i %12.3e %12.3e\n" s a.iteration a.current a.target
end

function print_progress(a::Convergence{<:Integer},verbosity)
    s =verbosity.indentation
    @printf "%s%6i %6i %6i\n" s a.iteration a.current a.target
end

function converged(a::Convergence)
    a.current <= a.target
end

function tired(a::Convergence)
    a.iteration >= a.iterations
end

function start(a::Convergence,current)
    target = a.target
    iteration = 0
    iterations = a.iterations
    Convergence(current,target,iteration,iterations)
end

function step(a::Convergence,current)
    target = a.target
    iteration = a.iteration + 1
    iterations = a.iterations
    Convergence(current,target,iteration,iterations)
end

function Base.show(io::IO,k::MIME"text/plain",data::Convergence)
    if converged(data)
        println("Converged in $(data.iteration) iterations")
    else
        println("Not converged in $(data.iteration) iterations")
    end
end

function convergence(a)
    workspace(a).convergence
end

function workspace(a)
    a.workspace
end

function timer_output(a)
    workspace(a).timer_output
end

function solve!(x,P,f;zero_guess=false)
    next = step!(x,P,f;zero_guess)
    while next !== nothing
        x,P,state = next
        next = step!(x,P,f,state)
    end
    x,P
end

abstract type AbstractLinearSolver <: AbstractType end

function LinearAlgebra.ldiv!(x,P::AbstractLinearSolver,b)
    if uses_initial_guess(x)
        fill!(x,zero(eltype(x)))
    end
    solve!(x,P,b;zero_guess=true)
    x
end

uses_initial_guess(P) = true

function matrix(a::AbstractMatrix)
    a
end

function verbosity(;indentation="")
    (;indentation)
end

struct LinearAlgebra_LU{A} <: AbstractLinearSolver
    workspace::A
end

function LinearAlgebra_lu(A;
        verbose=false,
        verbosity=PS.verbosity(),
        timer_output=TimerOutput(),
    )
    @timeit timer_output "LinearAlgebra_lu" begin
        factors = lu(A)
        status = Status()
        workspace = (;factors,verbose,verbosity,timer_output,status)
        LinearAlgebra_LU(workspace)
    end
end

function LinearAlgebra_lu(x,p;kwargs...)
    A = matrix(p)
    LinearAlgebra_lu(A;kwargs...)
end

function update!(P::LinearAlgebra_LU,p)
    (;factors,verbose,verbosity,timer_output,status) = P.workspace
    @timeit timer_output "LinearAlgebra_lu update!" begin
        A = matrix(p)
        lu!(factors,A)
        status = update(status)
        workspace = (;factors,verbose,verbosity,timer_output,status)
        LinearAlgebra_LU(workspace)
    end
end

uses_initial_guess(P::LinearAlgebra_LU) = false

function step!(x,P::LinearAlgebra_LU,b,state=:start;kwargs...)
    (;factors,verbose,verbosity,timer_output,status) = P.workspace
    if state === :stop
        return nothing
    end
    @timeit timer_output "LinearAlgebra_lu step!" begin
        ldiv!(x,factors,b)
        status = step(status)
        state = :stop
        workspace = (;factors,verbose,verbosity,timer_output,status)
        P = LinearAlgebra_LU(workspace)
        x,P,state
    end
end

struct JacobiCorrection{A} <: AbstractLinearSolver
    workspace::A
end

function jacobi_correction(A,b;
        verbose=false,
        verbosity=PS.verbosity(),
        timer_output=TimerOutput())
    @timeit timer_output "jacobi_correction" begin
        Adiag = dense_diag(A)
        status = Status()
        workspace = (;Adiag,verbose,verbosity,timer_output,status)
        JacobiCorrection(workspace)
    end
end

function diagonal_solve!(x,Adiag,b)
    x .= Adiag .\ b
    x
end

function update!(P::JacobiCorrection,p)
    (;Adiag,verbose,verbosity,timer_output,status) = P.workspace
    @timeit timer_output "jacobi_correction update!" begin
        A = matrix(p)
        dense_diag!(Adiag,A)
        status = update(status)
        workspace = (;Adiag,verbose,verbosity,timer_output,status)
        JacobiCorrection(workspace)
    end
end

function step!(x,P::JacobiCorrection,b,state=:start;kwargs...)
    (;Adiag,verbose,verbosity,timer_output,status) = P.workspace
    if state === :stop
        return nothing
    end
    @timeit timer_output "jacobi_correction step!" begin
        diagonal_solve!(x,Adiag,b)
        status = step(status)
        state = :stop
        workspace = (;Adiag,verbose,verbosity,timer_output,status)
        P = JacobiCorrection(workspace)
        x,P,state
    end
end

struct IdentitySolver{A} <: AbstractLinearSolver
    workspace::A
end

function identity_solver(;
        verbose=false,
        verbosity=PS.verbosity(),
        timer_output=TimerOutput())
    @timeit timer_output "identity_solver" begin
        status = Status()
        workspace = (;status,verbosity,verbose,timer_output)
        IdentitySolver(workspace)
    end
end

function update!(P::IdentitySolver,p)
    (;status,verbosity,verbose,timer_output) = P.workspace
    @timeit timer_output "identity_solver update!" begin
        status = update(status)
        workspace = (;status,verbosity,verbose,timer_output)
        IdentitySolver(workspace)
    end
end

function step!(x,P::IdentitySolver,b,state=:start;kwargs...)
    (;status,verbosity,verbose,timer_output) = P.workspace
        if state === :stop
            return nothing
        end
    @timeit timer_output "identity_solver step!" begin
        copy!(x,b)
        status = step(status)
        state = :stop
        workspace = (;status,verbosity,verbose,timer_output)
        P = IdentitySolver(workspace)
        x,P,state
    end
end

struct Richardson{A} <: AbstractLinearSolver
    workspace::A
end

function richardson(x,A,b;
        omega = 1,
        iterations = 10,
        preconditioner=identity_solver(),
        verbose=false,
        verbosity=PS.verbosity(),
        timer_output=TimerOutput()
    )
    @timeit timer_output "richardson" begin
        P = preconditioner
        r = similar(b)
        dx = similar(x,axes(A,2))
        status = Status()
        target = 0
        convergence = Convergence(iterations,target)
        workspace = (;A,r,dx,P,status,convergence,omega,verbose,verbosity,timer_output)
        Richardson(workspace)
    end
end

function update!(S::Richardson,p)
    (;A,r,dx,P,status,convergence,omega,verbose,verbosity,timer_output) = S.workspace
    @timeit timer_output "richardson update!" begin
        P = update!(P,p)
        status = update(status)
        workspace = (;A,r,dx,P,status,convergence,omega,verbose,verbosity,timer_output)
        Richardson(workspace)
    end
end

function step!(x,S::Richardson,b,phase=:start;kwargs...)
    (;A,r,dx,P,status,convergence,omega,verbose,verbosity,timer_output) = S.workspace
    @timeit timer_output "richardson step!" begin
        @assert phase in (:start,:stop,:advance)
        if phase === :stop
            verbose && display(convergence)
            return nothing
        end
        if phase === :start
            current = convergence.iterations
            convergence = start(convergence,current)
            verbose && print_progress(convergence,verbosity)
            phase = :advance
        end
        dx .= x
        @timeit timer_output "richardson step! mul!" begin
            mul!(r,A,dx)
        end
        r .-= b
        @timeit timer_output "richardson step! ldiv!" begin
            ldiv!(dx,P,r)
        end
        x .-= omega .* dx
        current = convergence.current - 1
        convergence = step(convergence,current)
        status = step(status)
        verbose && print_progress(convergence,verbosity)
        if converged(convergence)
            phase = :stop
        end
        workspace = (;A,r,dx,P,status,convergence,omega,verbose,verbosity,timer_output)
        S = Richardson(workspace)
        x,S,phase
    end
end

function jacobi(x,A,b;timer_output=TimerOutput(),kwargs...)
    preconditioner = jacobi_correction(A,b;timer_output,kwargs...)
    richardson(x,A,b;timer_output,preconditioner,kwargs...)
end

