
function lu_solver()
    setup(x,op,b,options) = lu(op)
    update!(state,op,options) = lu!(state,op)
    function solve!(x,P,b,options)
        ldiv!(x,P,b)
        x,P
    end
    uses_initial_guess = Val(false)
    linear_solver(;setup,solve!,update!,uses_initial_guess)
end

function jacobi_correction()
    setup(x,op,b,options) = dense_diag!(similar(b),op)
    update!(state,op,options) = dense_diag!(state,op)
    function solve!(x,state,b,options)
        x .= state .\ b
        x,state
    end
    uses_initial_guess = Val(false)
    linear_solver(;setup,update!,solve!,uses_initial_guess)
end

function richardson(solver;iters,omega=1)
    function setup(x,A,b,options)
        A_ref = Ref(A)
        r = similar(b)
        dx = similar(x,axes(A,2))
        P = PartitionedSolvers.setup(solver,dx,A,r)
        state = (r,dx,P,A_ref)
    end
    function update!(state,A,options)
        (r,dx,P,A_ref) = state
        A_ref[] = A
        PartitionedSolvers.update!(P,A)
        state
    end
    function solve!(x,state,b,options)
        (r,dx,P,A_ref) = state
        A = A_ref[]
        for iter in 1:iters
            dx .= x
            mul!(r,A,dx)
            r .-= b
            ldiv!(dx,P,r)
            x .-= omega .* dx
        end
        x, state
    end
    function step!(x,state,b,options,step=0)
        if step == iters
            return nothing
        end
        (r,dx,P,A_ref) = state
        A = A_ref[]
        dx .= x
        mul!(r,A,dx)
        r .-= b
        ldiv!(dx,P,r)
        x .-= omega .* dx
        x,state,step+1
    end
    function finalize!(state)
        (r,dx,P,A_ref) = state
        PartitionedSolvers.finalize!(P)
    end
    linear_solver(;setup,update!,solve!,finalize!,step!)
end

function jacobi(;kwargs...)
    solver = jacobi_correction()
    richardson(solver;kwargs...)
end

function gauss_seidel(;iters=1,sweep=:symmetric)
    @assert sweep in (:forward,:backward,:symmetric)
    function setup(x,A,b,options)
        diagA = dense_diag!(similar(b),A)
        A_ref = Ref(A)
        (diagA,A_ref)
    end
    function update!(state,A,options)
        (diagA,A_ref) = state
        dense_diag!(diagA,A)
        A_ref[] = A
        state
    end
    function gauss_seidel_sweep!(x,A::SparseArrays.AbstractSparseMatrixCSC,diagA,b,cols)
        #assumes symmetric matrix
        for col in cols
            s = b[col]
            for p in nzrange(A,col)
                row = A.rowval[p]
                a = A.nzval[p]
                s -= a*x[row]
            end
            d = diagA[col]
            s += d*x[col]
            s = s/d
            x[col] = s
        end
        x
    end
    function solve!(x,state,b,options)
        (diagA,A_ref) = state
        A = A_ref[]
        n = length(b)
        for iter in 1:iters
            if sweep === :symmetric || sweep === :forward
                gauss_seidel_sweep!(x,A,diagA,b,1:n)
            end
            if sweep === :symmetric || sweep === :backward
                gauss_seidel_sweep!(x,A,diagA,b,n:-1:1)
            end
        end
        x,state
    end
    linear_solver(;setup,update!,solve!)
end

function additive_schwarz(local_solver;iters=1)
    richardson(additive_schwarz_correction(local_solver);iters)
end

function local_setup_options(A,options)
    if nullspace(options) !== nothing
        ns = map(i->own_values(i),nullspace(options))
        map(ns) do ns
            setup_options(;nullspace=ns)
        end
    else
        map(partition(A)) do A
            options
        end
    end
end

function local_solver_options(A,options)
    map(partition(A)) do A
        options
    end
end

struct AdditiveSchwarzSetup{A} <: AbstractType
    local_setups::A
end

function additive_schwarz_correction(local_solver)
    # For parallel matrices
    function setup(x,A::PSparseMatrix,b,options)
        map(
            local_solver.setup,
            own_values(x),
            own_own_values(A),
            own_values(b),
            local_setup_options(A,options),
           ) |> AdditiveSchwarzSetup
    end
    function update!(state::AdditiveSchwarzSetup,A,options)
        map(
            local_solver.update!,
            state.local_setups,
            own_own_values(A),
            local_setup_options(A,options),
           )
    end
    function solve!(x,state::AdditiveSchwarzSetup,b,options)
        map(
            local_solver.solve!,
            own_values(x),
            state.local_setups,
            own_values(b),
            local_solver_options(b,options)
           )
        x,state
    end
    function finalize!(state::AdditiveSchwarzSetup)
        map(
            local_solver.finalize!,
            state.local_setups)
        nothing
    end
    # Fall back for sequential matrices
    function setup(x,A,b,options)
        local_solver.setup(x,A,b,options)
    end
    function update!(state,A,options)
        local_solver.update!(state,A,options)
    end
    function solve!(x,state,b,options)
        local_solver.solve!(x,state,b,options)
        x,state
    end
    function finalize!(state)
        local_solver.finalize!(state)
        nothing
    end
    linear_solver(;setup,update!,solve!,finalize!)
end

function identity_preconditioner()
    setup(x,op,b,options) = nothing
    update!(state,op,options) = state
    function solve!(x,P,b,options)
        copy!(x,b)
        x,P
    end
    uses_initial_guess = Val(false)
    linear_solver(;setup,solve!,update!,uses_initial_guess)
end

function print_convergence_step(workspace,verbose,verbose_frequency)
    iteration = workspace.iteration
    current = workspace.current
    target = workspace.target
    if verbose && mod(iteration,verbose_frequency)==0
        @printf "%6i %12.3e %12.3e\n" iteration current target
    end
end

function print_convergence_end(workspace,verbose)
    print_convergence_step(workspace,verbose,1)
    iteration = workspace.iteration
    current = workspace.current
    target = workspace.target
    if verbose
        converged = current <= target
        println("$( converged ? "Converged" : "Not converged" ) after $iteration iterations")
    end
end

function conjugate_gradients(;
        preconditioner = identity_preconditioner(),
        abstol = nothing,
        reltol = nothing,
        maxiters = nothing,
        verbose = false,
        verbose_frequency = 1,
    )
    function setup(x,A,b,options)
        c = similar(x)
        u = similar(x)
        r = similar(x)
        M = PartitionedSolvers.setup(preconditioner,x,A,b)
        current = zero(real(eltype(b)))
        target = current
        iteration=0
        ρ = one(eltype(x))
        workspace = (;c,u,r,A,M,current,target,iteration,maxiters,ρ)
    end
    function update!(workspace,A,options)
        (;c,u,r,M,current,target,iteration,ρ) = workspace
        M = PartitionedSolvers.update!(M,A)
        workspace = (;c,u,r,A,M,current,target,iteration,ρ)
    end
    function step!(x,workspace,b,options,state=:start)
        (;c,u,r,A,M,current,target,iteration,ρ) = workspace
        if state === :start
            fill!(u,zero(eltype(u)))
            if options.zero_guess
                r .= b
            else
                mul!(c,A,x)
                r .= b .- c
            end
            T = real(eltype(b))
            if abstol === nothing
                abstol = zero(T)
            end
            if reltol === nothing
                reltol = sqrt(eps(T))
            end
            ρ = one(eltype(x))
            current = sqrt(dot(r,r))
            target = max(reltol*current,abstol)
            iteration = 0
            state = :advance
        end
        if state == :stop
            print_convergence_end(workspace,verbose)
            return nothing
        end
        print_convergence_step(workspace,verbose,verbose_frequency)
        ldiv!(c,M,r)
        ρ_prev = ρ
        ρ = dot(c,r)
        β = ρ / ρ_prev
        u .= c .+ β .* u
        mul!(c, A, u)
        α = ρ / dot(u,c)
        x .= x .+ α .* u
        r .= r .- α .* c
        current = sqrt(dot(r,r))
        iteration += 1
        converged = current <= target
        if maxiters === nothing
            maxiters = size(A,1)
        end
        tired = iteration == maxiters
        if converged || tired
            state = :stop
        end
        workspace = (;c,u,r,A,M,current,target,iteration,ρ)
        x,workspace,state
    end
    is_iterative = Val(true)
    linear_solver(;setup,step!,update!,is_iterative)
end

#for (x,P,state) in iterations!(x,P,b)
# state.current
# state.step
# state.target
# P.solver_setup.r
#end

#P = setup(solver,x,A,b)
#x,P,state = solve!(x,P,b)
#x,P,state = step!(x,P,b)
#x,P,state = step!(x,P,b,state)



# Wrappers

function linear_solver(::typeof(LinearAlgebra.lu))
    lu_solver()
end

function linear_solver(::typeof(IterativeSolvers.cg);Pl,kwargs...)
    function setup(x,A,b,options)
        Pl_solver = linear_solver(Pl)
        P = PartitionedSolvers.setup(Pl_solver,x,A,b;options...)
        A_ref = Ref(A)
        (;P,A_ref)
    end
    function update!(state,A,options)
        (;P,A_ref) = state
        A_ref[] = A
        P = PartitionedSolvers.update!(P,A;options...)
        (;P,A_ref)
    end
    function solve!(x,state,b,options)
        (;P,A_ref) = state
        A = A_ref[]
        IterativeSolvers.cg!(x,A,b;Pl=P,kwargs...)
        x,state
    end
    function finalize!(state,A,options)
        (;P) = state
        PartitionedSolvers.finalize!(P)
        nothing
    end
    linear_solver(;setup,update!,solve!,finalize!)
end

