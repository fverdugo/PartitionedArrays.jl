# TODO define functions lu_solver_setup, etc to make solvers
# extensible wrt new matrix properties

function lu_solver()
    setup(x,A,b,props=nothing) = lu(A)
    setup!(state,A,props=nothing) = lu!(state,A)
    solve!(args...;zero_guess=false) = ldiv!(args...)
    linear_solver(;setup,solve!,setup!)
end

function jacobi_correction()
    setup(x,A,b,props=nothing) = dense_diag!(similar(b),A)
    setup!(state,A,props=nothing) = dense_diag!(state,A)
    function solve!(x,state,b;zero_guess=false)
        x .= state .\ b
    end
    linear_solver(;setup,setup!,solve!)
end

function richardson(solver;iters,omega=1)
    function setup(x,A,b,props=nothing)
        A_ref = Ref(A)
        r = similar(b)
        dx = similar(x,axes(A,2))
        P = preconditioner(solver,dx,A,r,props)
        state = (r,dx,P,A_ref)
    end
    function setup!(state,A,props=nothing)
        (r,dx,P,A_ref) = state
        A_ref[] = A
        preconditioner!(P,A,props)
        state
    end
    function solve!(x,state,b;zero_guess=false)
        # TODO if zero_guess == true skip the first SpMV
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
    function setup(x,A,b,props=nothing)
        diagA = dense_diag!(similar(b),A)
        A_ref = Ref(A)
        (diagA,A_ref)
    end
    function setup!(state,A,props=nothing)
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
    function solve!(x,state,b;zero_guess=false)
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

struct AdditiveSchwarzSetup{A}
    local_setups::A
end

function local_matrix_properties(A,props::MatrixProperties)
    ns = map(i->own_values(i),nullspace(props))
    B = map(vcat,ns...)
    map(ns) do ns
        matrix_properties(;nullspace=ns)
    end
end

function local_matrix_properties(A,props::Nothing)
    map(i->nothing,partition(A))
end

function additive_schwarz_correction(local_solver)
    # Fall back for sequential matrices
    function setup(x,A,b,props=nothing)
        PartitionedSolvers.setup(local_solver)(x,A,b,props)
    end
    function setup!(state,A,props=nothing)
        PartitionedSolvers.setup!(local_solver)(state,A,props)
    end
    function solve!(x,state,b;zero_guess=false)
        PartitionedSolvers.solve!(local_solver)(x,state,b)
        x
    end
    function finalize!(state)
        PartitionedSolvers.finalize!(local_solver)(state)
        nothing
    end
    # For parallel matrices
    function setup(x,A::PSparseMatrix,b,props=nothing)
        map(
            PartitionedSolvers.setup(local_solver),
            own_values(x),
            own_own_values(A),
            own_values(b),
            local_matrix_properties(A,props),
           ) |> AdditiveSchwarzSetup
    end
    function setup!(state::AdditiveSchwarzSetup,A,props=nothing)
        map(
            PartitionedSolvers.setup!(local_solver),
            state.local_setups,
            own_own_values(A),
            local_matrix_properties(A,props),
           )
    end
    function solve!(x,state::AdditiveSchwarzSetup,b;zero_guess=false)
        map(
            PartitionedSolvers.solve!(local_solver),
            own_values(x),
            state.local_setups,
            own_values(b))
        x
    end
    function finalize!(state::AdditiveSchwarzSetup)
        map(
            PartitionedSolvers.finalize!(local_solver),
            state.local_setups)
            nothing
    end
    linear_solver(;setup,setup!,solve!,finalize!)
end

