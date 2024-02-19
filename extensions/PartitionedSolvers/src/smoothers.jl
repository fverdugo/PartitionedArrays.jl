
function lu_solver()
    setup(x,op,b) = lu(matrix(op))
    setup!(state,op) = lu!(state,matrix(op))
    solve! = ldiv!
    linear_solver(;setup,solve!,setup!)
end

function diagonal_solver()
    setup(x,op,b) = diag!(similar(b),matrix(op))
    setup!(state,op) = diag!(state,matrix(op))
    function solve!(x,state,b)
        x .= state .\ b
    end
    linear_solver(;setup,setup!,solve!)
end

function richardson(solver;maxiters,omega=1)
    function setup(x,O,b)
        A = matrix(O)
        A_ref = Ref(A)
        r = similar(b)
        dx = similar(x,axes(A,2))
        P = preconditioner(solver,dx,O,r)
        state = (r,dx,P,A_ref)
    end
    function setup!(state,O)
        (r,dx,P,A_ref) = state
        A_ref[] = matrix(O)
        preconditioner!(P,O)
        state
    end
    function solve!(x,state,b)
        (r,dx,P,A_ref) = state
        A = A_ref[]
        for iter in 1:maxiters
            dx .= x
            mul!(r,A,dx)
            r .-= b
            ldiv!(dx,P,r)
            x .-= omega .* dx
        end
        (;maxiters)
    end
    function finalize!(state)
        (r,dx,P,A_ref) = state
        PartitionedSolvers.finalize!(P)
    end
    linear_solver(;setup,setup!,solve!,finalize!)
end

function jacobi(;kwargs...)
    solver = diagonal_solver()
    richardson(solver;kwargs...)
end

function additive_schwarz(local_solver)
    function build_local_operators(O::MatrixWithNullspace)
        A = matrix(O)
        ns = nullspace(O)
        map(linear_operator,own_own_values(A),ns)
    end
    function build_local_operators(A)
        own_own_values(matrix(A))
    end
    function setup(x,O,b)
        local_O = build_local_operators(O)
        local_setups = map(PartitionedSolvers.setup(local_solver),own_values(x),local_O,own_values(b))
        local_setups
    end
    function setup!(local_setups,O)
        local_O = build_local_operators(O)
        map(PartitionedSolvers.setup!(local_solver),local_setups,local_O)
        local_setups
    end
    function solve!(x,local_setups,b)
        map(PartitionedSolvers.solve!(local_solver),own_values(x),local_setups,own_values(b))
    end
    function finalize!(local_setups)
        map(PartitionedSolvers.finalize!(local_solver),local_setups)
        nothing
    end
    linear_solver(;setup,setup!,solve!,finalize!)
end

