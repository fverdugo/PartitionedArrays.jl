
function identity_solver(p)
    @assert uses_mutable_types(p)
    workspace = nothing
    function update(workspace,A)
        workspace
    end
    function step(x,workspace,b,phase=:start;kwargs...)
        copyto!(x,b)
        phase = :stop
        x,workspace,phase
    end
    uses_initial_guess = Val(false)
    linear_solver(update,step,p,uses_initial_guess)
end

function jacobi_correction(p)
    @assert uses_mutable_types(p)
    Adiag = dense_diag!(similar(rhs(p)),matrix(p))
    function update(Adiag,A)
        dense_diag!(Adiag,A)
        Adiag
    end
    function step(x,Adiag,b,phase=:start;kwargs...)
        x .= Adiag .\ b
        phase = :stop
        x,Adiag,phase
    end
    uses_initial_guess = Val(false)
    linear_solver(update,step,p,Adiag;uses_initial_guess)
end

function richardson(p;
        P=preconditioner(identity_solver,p) ,
        iterations=10,
        omega=1,
        update_P = true,
    )
    @assert uses_mutable_types(p)
    iteration = 0
    A = matrix(p)
    ws = (;iterations,P,iteration,omega,update_P,A)
    linear_solver(richardson_update,richardson_step,p,ws)
end

function richardson_update(ws,A)
    (;iterations,P,iteration,omega,update_P) = ws
    if update_P
        P = update(P,matrix=A)
    end
    iteration = 0
    ws = (;iterations,P,iteration,omega,update_P,A)
end

function richardson_step(x,ws,b,phase=:start;kwargs...)
    (;iterations,P,iteration,omega,update_P,A) = ws
    if phase === :start
        iteration = 0
        phase = :advance
    end
    dx = solution(P)
    r = rhs(P)
    dx .= x
    mul!(r,A,dx)
    r .-= b
    ldiv!(dx,P,r)
    x .-= omega .* dx
    iteration += 1
    if iteration == iterations
        phase = :stop
    end
    ws = (;iterations,P,iteration,omega,update_P,A)
    x,ws,phase
end

function jacobi(p;iterations=10,omega=1)
    P = preconditioner(jacobi_correction,p)
    update_P = true
    richardson(p;P,update_P,iterations,omega)
end

function gauss_seidel(p;iterations=1,sweep=:symmetric)
    @assert uses_mutable_types(p)
    iteration = 0
    A = matrix(p)
    Adiag = dense_diag!(similar(rhs(p)),A)
    ws = (;iterations,sweep,iteration,A,Adiag)
    linear_solver(gauss_seidel_update,gauss_seidel_step,p,ws)
end

function gauss_seidel_update(ws,A)
    (;iterations,sweep,iteration,A,Adiag) = ws
    dense_diag!(Adiag,A)
    iteration = 0
    ws = (;iterations,sweep,iteration,A,Adiag)
end

function gauss_seidel_step(x,ws,b,phase=:start;zero_guess=false,kwargs...)
    (;iterations,sweep,iteration,A,Adiag) = ws
    if phase === :start
        iteration = 0
        phase = :advance
    end
    if (! zero_guess) && isa(x,PVector)
        consistent!(x) |> wait
    end
    # TODO the meaning of :forward and :backward
    # depends on the sparse matrix format
    if sweep === :symmetric || sweep === :forward
        if zero_guess
            gauss_seidel_forward_sweep_zero!(x,A,Adiag,b)
        else
            gauss_seidel_forward_sweep!(x,A,Adiag,b)
        end
    end
    if sweep === :symmetric || sweep === :backward
        gauss_seidel_backward_sweep!(x,A,Adiag,b)
    end
    iteration += 1
    if iteration == iterations
        phase = :stop
    end
    ws = (;iterations,sweep,iteration,A,Adiag)
    x,ws,phase
end

function gauss_seidel_forward_sweep!(x,A,diagA,b)
    n = length(b)
    gauss_seidel_sweep!(x,A,diagA,b,1:n)
end

function gauss_seidel_backward_sweep!(x,A,diagA,b)
    n = length(b)
    gauss_seidel_sweep!(x,A,diagA,b,n:-1:1)
end

function gauss_seidel_forward_sweep!(x,A::PSparseMatrix,diagA,b)
    foreach(gauss_seidel_forward_sweep!,partition(x),partition(A),partition(diagA),own_values(b))
end

function gauss_seidel_backward_sweep!(x,A::PSparseMatrix,diagA,b)
    foreach(gauss_seidel_backward_sweep!,partition(x),partition(A),partition(diagA),own_values(b))
end

function gauss_seidel_sweep!(x,A::SparseArrays.AbstractSparseMatrixCSC,diagA,b,cols)
    # assumes symmetric matrix
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

function gauss_seidel_sweep!(x,A::SparseMatricesCSR.SparseMatrixCSR,diagA,b,rows)
    for row in rows
        s = b[row]
        for p in nzrange(A,row)
            col = A.colval[p]
            a = A.nzval[p]
            s -= a * x[col]
        end
        d = diagA[row]
        s += d * x[row]
        s = s / d
        x[row] = s
    end
    x
end

function gauss_seidel_sweep!(x,A::PartitionedArrays.AbstractSplitMatrix,diagA,b,cols)
    @assert isa(A.row_permutation,UnitRange)
    @assert isa(A.col_permutation,UnitRange)
    Aoo = A.blocks.own_own
    Aoh = A.blocks.own_ghost
    gauss_seidel_sweep_split!(x,Aoo,Aoh,diagA,b,cols)
end

function gauss_seidel_sweep_split!(x,Aoo::SparseMatricesCSR.SparseMatrixCSR,Aoh,diagA,b,rows)
    for row in rows
        s = b[row]
        for p in nzrange(Aoo,row)
            col = Aoo.colval[p]
            a = Aoo.nzval[p]
            s -= a * x[col]
        end
        for p in nzrange(Aoh,row)
            col = Aoh.colval[p]
            a = Aoh.nzval[p]
            s -= a * x[col]
        end
        d = diagA[row]
        s += d * x[row]
        s = s / d
        x[row] = s
    end
    x
end

function gauss_seidel_forward_sweep_zero!(x,A,diagA,b)
    n = length(b)
    gauss_seidel_sweep_zero!(x,A,diagA,b,1:n)
end

# TODO not sure if correct
function gauss_seidel_backward_sweep_zero!(x,A,diagA,b)
    n = length(b)
    gauss_seidel_sweep_zero!(x,A,diagA,b,n:-1:1)
end

function gauss_seidel_forward_sweep_zero!(x,A::PSparseMatrix,diagA,b)
    foreach(gauss_seidel_forward_sweep_zero!,partition(x),partition(A),partition(diagA),own_values(b))
end

function gauss_seidel_backward_sweep_zero!(x,A::PSparseMatrix,diagA,b)
    foreach(gauss_seidel_backward_sweep_zero!,partition(x),partition(A),partition(diagA),own_values(b))
end

# Zero guess: only calculate points below diagonal of sparse matrix in forward sweep.
function gauss_seidel_sweep_zero!(x,A::SparseArrays.AbstractSparseMatrixCSC,diagA,b,cols)
    gauss_seidel_sweep!(x,A,diagA,b,cols)
    ## There is a bug, falling back to nonzero x
    ##assumes symmetric matrix
    #for col in cols
    #    s = b[col]
    #    for p in nzrange(A,col)
    #        row = A.rowval[p]
    #        if col < row
    #            a = A.nzval[p]
    #            s -= a*x[row]
    #        end
    #    end
    #    d = diagA[col]
    #    #s += d*x[col]
    #    s = s/d
    #    x[col] = s
    #end
    #x
end
# Zero guess: only calculate points below diagonal of sparse matrix in forward sweep.
function gauss_seidel_sweep_zero!(x,A::SparseMatricesCSR.SparseMatrixCSR,diagA,b,rows)
    rows
    length(x)
    size(A)
    length(b)
    #assumes symmetric matrix
    for row in rows
        s = b[row]
        for p in nzrange(A,row)
            col = A.colval[p]
            if col < row
                a = A.nzval[p]
                s -= a * x[col]
            end
        end
        d = diagA[row]
        #s += d * x[col]
        s = s / d
        x[row] = s
    end
    x
end

function gauss_seidel_sweep_zero!(x,A::PartitionedArrays.AbstractSplitMatrix,diagA,b,cols)
    @assert isa(A.row_permutation,UnitRange)
    @assert isa(A.col_permutation,UnitRange)
    Aoo = A.blocks.own_own
    Aoh = A.blocks.own_ghost
    gauss_seidel_sweep_zero_split!(x,Aoo,Aoh,diagA,b,cols)
end

function gauss_seidel_sweep_zero_split!(x,Aoo::SparseMatricesCSR.SparseMatrixCSR,Aoh,diagA,b,rows)
    for row in rows
        s = b[row]
        for p in nzrange(Aoo,row)
            col = Aoo.colval[p]
            if col < row
                a = Aoo.nzval[p]
                s -= a * x[col]
            end
        end
        for p in nzrange(Aoh,row)
            col = Aoh.colval[p]
            if col < row
                a = Aoh.nzval[p]
                s -= a * x[col]
            end
        end
        d = diagA[row]
        #s += d * x[col]
        s = s / d
        x[row] = s
    end
    x
end

function additive_schwarz_correction(p;local_solver=LinearAlgebra_lu)
    x = solution(p)
    A = matrix(p)
    b = rhs(p)
    local_s = additive_schwarz_correction_setup(local_solver,x,A,b)
    uses_initial_guess = Val(false)
    linear_solver(
        additive_schwarz_correction_update,
        additive_schwarz_correction_step,
        p,
        local_s;
        uses_initial_guess
       )
end

function additive_schwarz_correction_setup(local_solver,x,A::PSparseMatrix,b)
    local_p = map(linear_problem,own_values(x),own_own_values(A),own_values(b))
    local_s = map(local_solver,local_p)
end

function additive_schwarz_correction_update(local_s,A::PSparseMatrix)
    local_s = map(additive_schwarz_correction_update,local_s,own_own_values(A))
end

function additive_schwarz_correction_step(x::PVector,local_s,b,phase=:start;kwargs...)
    foreach(ldiv!,own_values(x),local_s,own_values(b))
    phase = :stop
    x,local_s,phase
end

function additive_schwarz_correction_setup(local_solver,x,A,b)
    local_p = linear_problem(x,A,b)
    local_s = local_solver(local_p)
end

function additive_schwarz_correction_update(local_s,A)
    local_s = update(local_s,matrix=A)
end

function additive_schwarz_correction_step(x,local_s,b,phase=:start;kwargs...)
    ldiv!(x,local_s,b)
    phase = :stop
    x,local_s,phase
end

function additive_schwarz(p;local_solver=LinearAlgebra_lu,iterations=1)
    P = preconditioner(p) do dp
        additive_schwarz_correction(dp;local_solver)
    end
    update_P = true
    richardson(p;P,update_P,iterations)
end

