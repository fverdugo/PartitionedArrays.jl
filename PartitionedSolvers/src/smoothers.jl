
function lu_solver()
    setup(x,op,b,options) = lu(op)
    update!(state,op,options) = lu!(state,op)
    solve!(x,P,b,options) = ldiv!(x,P,b)
    linear_solver(;setup,solve!,update!)
end

function jacobi_correction()
    setup(x,op,b,options) = dense_diag!(similar(b),op)
    update!(state,op,options) = dense_diag!(state,op)
    function solve!(x,state,b,options)
        x .= state .\ b
    end
    linear_solver(;setup,update!,solve!)
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
        (;iters)
    end
    function finalize!(state)
        (r,dx,P,A_ref) = state
        PartitionedSolvers.finalize!(P)
    end
    linear_solver(;setup,update!,solve!,finalize!)
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
        x
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

struct AdditiveSchwarzSetup{A}
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
        x
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
        x
    end
    function finalize!(state)
        local_solver.finalize!(state)
        nothing
    end
    linear_solver(;setup,update!,solve!,finalize!)
end

