
function lu_solver()
    setup(x,op,b) = lu(matrix(op))
    setup!(state,op) = lu!(state,matrix(op))
    solve! = ldiv!
    linear_solver(;setup,solve!,setup!)
end

function jacobi_correction()
    setup(x,op,b) = dense_diag!(similar(b),matrix(op))
    setup!(state,op) = dense_diag!(state,matrix(op))
    function solve!(x,state,b)
        x .= state .\ b
    end
    linear_solver(;setup,setup!,solve!)
end

function richardson(solver;iters,omega=1)
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
    linear_solver(;setup,setup!,solve!,finalize!)
end

function jacobi(;kwargs...)
    solver = jacobi_correction()
    richardson(solver;kwargs...)
end

function gauss_seidel(;iters=1,sweep=:symmetric)
    @assert sweep in (:forward,:backward,:symmetric)
    function setup(x,op,b)
        A = matrix(op)
        diagA = dense_diag!(similar(b),A)
        A_ref = Ref(A)
        (diagA,A_ref)
    end
    function setup!(state,op)
        (diagA,A_ref) = state
        A = matrix(op)
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
    function solve!(x,state,b)
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
    linear_solver(;setup,setup!,solve!)
end

function additive_schwarz(local_solver;iters=1)
    richardson(additive_schwarz_correction(local_solver);iters)
end

function additive_schwarz_correction(local_solver)
    function build_local_operators(O::MatrixWithNullspace)
        A = matrix(O)
        ns = map(i->own_values(i),nullspace(O))
        B = map(vcat,ns...)
        map(attach_nullspace,own_own_values(A),B)
    end
    function build_local_operators(A)
        own_own_values(matrix(A))
    end
    is_parallel(A::PSparseMatrix) = Val(true)
    is_parallel(A) = Val(false)
    function setup(x,O,b)
        parallel = is_parallel(matrix(O))
        if PartitionedArrays.val_parameter(parallel)
            local_O = build_local_operators(O)
            local_setups = map(PartitionedSolvers.setup(local_solver),own_values(x),local_O,own_values(b))
        else
            local_setups = PartitionedSolvers.setup(local_solver)(x,O,b)
        end
        (local_setups,parallel)
    end
    function setup!((local_setups,parallel),O)
        if PartitionedArrays.val_parameter(parallel)
            local_O = build_local_operators(O)
            map(PartitionedSolvers.setup!(local_solver),local_setups,local_O)
        else
            PartitionedSolvers.setup!(local_solver)(local_setups,O)
        end
        (local_setups,parallel)
    end
    function solve!(x,(local_setups,parallel),b)
        if PartitionedArrays.val_parameter(parallel)
            map(PartitionedSolvers.solve!(local_solver),own_values(x),local_setups,own_values(b))
        else
            PartitionedSolvers.solve!(local_solver)(x,local_setups,b)
        end
        x
    end
    function finalize!((local_setups,parallel))
        if PartitionedArrays.val_parameter(parallel)
            map(PartitionedSolvers.finalize!(local_solver),local_setups)
        else
            PartitionedSolvers.finalize!(local_solver)(local_setups)
        end
        nothing
    end
    linear_solver(;setup,setup!,solve!,finalize!)
end

