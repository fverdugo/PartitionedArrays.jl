
function do_nothing_linear_solver()
    setup(x,A,b) = nothing
    solve!(x,::Nothing,b) = copy!(x,b)
    setup!(::Nothing,A) = nothing
    linear_solver(;setup,setup!,solve!)
end

function lu_solver()
    setup(x,A,b) = lu(A)
    solve! = ldiv!
    setup! = lu!
    linear_solver(;setup,solve!,setup!)
end

function diagonal_solver()
    setup(x,A,b) = diag(A)
    solve!(x,D,b) = x .= D .\ b
    setup! = diag!
    linear_solver(;setup,setup!,solve!)
end

function richardson_solver(solver;niters)
    function setup(x,A,b)
        dx = similar(x,axes(A,2))
        r = similar(b,axes(A,1))
        P = preconditioner(dx,A,r,solver)
        A_ref = Ref(A)
        state = (dx,r,A_ref,P)
    end
    function setup!(state,A)
        (dx,r,A_ref,P) = state
        preconditioner!(P,A)
        A_ref[] = A
        state
    end
    function solve!(x,state,b)
        (dx,r,A_ref,P) = state
        A = A_ref[]
        for iter in 1:niters
            dx .= x
            mul!(r,A,dx)
            r .-= b
            ldiv!(dx,P,r)
            x .-= dx
        end
    end
    function finalize!(state)
        (dx,r,A_ref,P) = state
        PartitionedSolvers.finalize!(P)
    end
    linear_solver(;setup,setup!,solve!,finalize!)
end

function jacobi_solver(;kwargs...)
    solver = diagonal_solver()
    richardson_solver(solver;kwargs...)
end

function additive_schwartz_solver(local_solver)
    function setup(x,A,b)
        f = (x,A,b) -> PartitionedSolvers.setup(local_solver,x,A,b)
        local_setups = map(f,own_values(x),own_own_values(A),own_values(b))
        local_setups
    end
    function setup!(local_setups,A)
        f = (S,A) -> PartitionedSolvers.setup!(local_solver,S,A)
        map(f,local_setups,own_own_values(A))
        local_setups
    end
    function solve!(x,local_setups,b)
        f = (x,S,b) -> PartitionedSolvers.solve!(local_solver,x,S,b)
        map(f,own_values(x),local_setups,own_values(b))
    end
    function finalize!(local_setups)
        f = (S) -> PartitionedSolvers.finalize!(local_solver,S)
        map(f,local_setups)
        nothing
    end
    linear_solver(;setup,setup!,solve!,finalize!)
end

