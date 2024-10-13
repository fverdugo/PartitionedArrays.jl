
mutable struct Status <: AbstractType
    steps::Int
    steps_since_update::Int
    updates::Int
    function Status()
        new(0,0,0)
    end
end

function update!(status::Status)
    status.updates += 1
    status.steps_since_update = 0
    status
end

function step!(status::Status)
    status.steps_since_update += 1
    status.steps += 1
    status
end

function Base.show(io::IO,k::MIME"text/plain",data::Status)
    println("Updates since creation: $(data.updates)")
    println("Solver steps since creation: $(data.steps)")
    println("Solver steps since last update: $(data.steps_since_update)")
end

function status(a)
    workspace(a).status
end

mutable struct Convergence{T}
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

function start!(a::Convergence,current)
    a.iteration = 0
    a.current = current
    a
end

function step!(a::Convergence,current)
    a.iteration += 1
    a.current = current
    a
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
    factors = lu(A)
    status = Status()
    workspace = (;factors,verbose,verbosity,timer_output,status)
    LinearAlgebra_LU(workspace)
end

function LinearAlgebra_lu(x,p;kwargs...)
    A = matrix(p)
    LinearAlgebra_lu(A;kwargs...)
end

function update!(P::LinearAlgebra_LU,p)
    (;verbose,verbosity,timer_output,status,factors) = P.workspace
    A = matrix(p)
    lu!(factors,A)
    update!(status)
    P
end

uses_initial_guess(P::LinearAlgebra_LU) = false

function step!(x,P::LinearAlgebra_LU,b,state=:start;kwargs...)
    if state === :stop
        return nothing
    end
    (;verbose,verbosity,timer_output,status,factors) = P.workspace
    ldiv!(x,factors,b)
    step!(status)
    state = :stop
    x,P,state
end

struct JacobiCorrection{A} <: AbstractLinearSolver
    workspace::A
end

function jacobi_correction(A,b;
        verbose=false,
        verbosity=PS.verbosity(),
        timer_output=TimerOutput())

    Adiag = dense_diag(A)
    status = Status()
    workspace = (;Adiag,verbose,verbosity,timer_output,status)
    JacobiCorrection(workspace)
end

function diagonal_solve!(x,Adiag,b)
    x .= Adiag .\ b
    x
end

function update!(P::JacobiCorrection,p)
    (;Adiag,verbose,verbosity,timer_output,status) = P.workspace
    A = matrix(p)
    dense_diag!(Adiag,A)
    update!(status)
    P
end

function step!(x,P::JacobiCorrection,b,state=:start;kwargs...)
    if state === :stop
        return nothing
    end
    (;verbose,verbosity,timer_output,status,Adiag) = P.workspace
    diagonal_solve!(x,Adiag,b)
    step!(status)
    state = :stop
    x,P,state
end

struct IdentitySolver{A} <: AbstractLinearSolver
    workspace::A
end

function identity_solver(
        verbose=false,
        verbosity=PS.verbosity(),
        timer_output=TimerOutput())
    status = Status()
    workspace = (;status,verbosity,verbose,timer_output)
    IdentitySolver(workspace)
end

function update!(P::IdentitySolver,p)
    status = P.workspace.status
    update!(status)
    P
end

function step!(x,P::IdentitySolver,b,state=:start;kwargs...)
    if state === :stop
        return nothing
    end
    (;verbose,verbosity,timer_output,status) = P.workspace
    copy!(x,b)
    step!(status)
    state = :stop
    x,P,state
end

struct Richardson{A} <: AbstractLinearSolver
    workspace::A
end

function richardson(x,A,b;
        omega = 1,
        iterations = 10,
        P=identity_solver(),
        verbose=false,
        verbosity=PS.verbosity(),
        timer_output=TimerOutput()
    )
    A_ref = Ref(A)
    P_ref = Ref(P)
    r = similar(b)
    dx = similar(x,axes(A,2))
    status = Status()
    target = 0
    convergence = Convergence(iterations,target)
    workspace = (;A_ref,r,dx,P_ref,status,convergence,omega,verbose,verbosity,timer_output)
    Richardson(workspace)
end

function update!(S::Richardson,p)
    (;A_ref,P_ref,verbose,verbosity,timer_output,status,convergence) = S.workspace
    A = matrix(p)
    A_ref[] = A
    update!(P_ref[],p)
    update!(status)
    S
end

function step!(x,S::Richardson,b,phase=:start;kwargs...)
    @assert phase in (:start,:stop,:advance)
    if phase === :stop
        display(S.workspace.convergence)
        return nothing
    end
    (;verbose,verbosity,timer_output,status,convergence,dx,r,A_ref,P_ref,omega) = S.workspace
    if phase === :start
        current = convergence.iterations
        start!(convergence,current)
        print_progress(convergence,verbosity)
        phase = :advance
    end
    A = A_ref[]
    P = P_ref[]
    dx .= x
    mul!(r,A,dx)
    r .-= b
    ldiv!(dx,P,r)
    x .-= omega .* dx
    current = convergence.current - 1
    step!(convergence,current)
    step!(status)
    print_progress(convergence,verbosity)
    if converged(convergence)
        phase = :stop
    end
    x,S,phase
end

