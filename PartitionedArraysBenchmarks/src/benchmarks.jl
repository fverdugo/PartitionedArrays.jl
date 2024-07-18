
function benchmark_spmv(distribute,params)
    parts_per_dir = params.parts_per_dir
    cells_per_dir = params.cells_per_dir
    method = params.method
    nruns = params.nruns
    np = prod(parts_per_dir)
    parts = distribute(LinearIndices((np,)))
    Ti = PetscCall.PetscInt
    T = PetscCall.PetscScalar
    psparse_args = coo_scalar_fem(cells_per_dir,parts_per_dir,parts,T,Ti)
    A = psparse(psparse_args...) |> fetch
    cols = axes(A,2)
    rows = axes(A,1)
    x = pones(PetscCall.PetscScalar,partition(cols))
    b = similar(x,rows)
    t = zeros(nruns)
    if method == "PartitionedArrays"
        for irun in 1:nruns
            b .= 0
            t[irun] =  @elapsed mul!(b,A,x)
        end
    elseif method == "Petsc"
        PetscCall.init(finalize_atexit=false)
        mat = Ref{PetscCall.Mat}()
        vec_b = Ref{PetscCall.Vec}()
        vec_x = Ref{PetscCall.Vec}()
        petsc_comm = PetscCall.setup_petsc_comm(parts)
        args_A = PetscCall.MatCreateMPIAIJWithSplitArrays_args(A,petsc_comm)
        args_b = PetscCall.VecCreateMPIWithArray_args(copy(b),petsc_comm)
        args_x = PetscCall.VecCreateMPIWithArray_args(copy(x),petsc_comm)
        ownership = (args_A,args_b,args_x)
        PetscCall.@check_error_code PetscCall.MatCreateMPIAIJWithSplitArrays(args_A...,mat)
        PetscCall.@check_error_code PetscCall.MatAssemblyBegin(mat[],PetscCall.MAT_FINAL_ASSEMBLY)
        PetscCall.@check_error_code PetscCall.MatAssemblyEnd(mat[],PetscCall.MAT_FINAL_ASSEMBLY)
        PetscCall.@check_error_code PetscCall.VecCreateMPIWithArray(args_b...,vec_b)
        PetscCall.@check_error_code PetscCall.VecCreateMPIWithArray(args_x...,vec_x)
        for irun in 1:nruns
            PetscCall.@check_error_code PetscCall.VecZeroEntries(vec_b[])
            t[irun] = @elapsed PetscCall.@check_error_code PetscCall.MatMult(mat[],vec_x[],vec_b[])
        end
        GC.@preserve ownership PetscCall.@check_error_code PetscCall.MatDestroy(mat)
        GC.@preserve ownership PetscCall.@check_error_code PetscCall.VecDestroy(vec_b)
        GC.@preserve ownership PetscCall.@check_error_code PetscCall.VecDestroy(vec_x)
        PetscCall.finalize()
    else
        error("unknown method $method")
    end
    ts_in_main = gather(map(p->t,parts))
    results_in_main = map_main(ts_in_main) do ts
        results = (;spmv=ts,params...)
        results
    end
end

function consistent2!(v::PVector)
    t_sync = @elapsed begin
        vector_partition = partition(v)
        cache_for_assemble = v.cache
        cache = reverse(cache_for_assemble)
        local_indices_snd=cache.local_indices_snd
        local_indices_rcv=cache.local_indices_rcv
        neighbors_snd=cache.neighbors_snd
        neighbors_rcv=cache.neighbors_rcv
        buffer_snd=cache.buffer_snd
        buffer_rcv=cache.buffer_rcv
        exchange_setup=cache.exchange_setup
        function setup_snd!(values,local_indices_snd,buffer_snd)
            for (p,lid) in enumerate(local_indices_snd.data)
                buffer_snd.data[p] = values[lid]
            end
        end
        t_snd = @elapsed foreach(setup_snd!,vector_partition,local_indices_snd,buffer_snd)
        graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
        t_exchange = @elapsed t = exchange!(buffer_rcv,buffer_snd,graph,exchange_setup)
    end
    t_wait_exchange = 0.0
    t_rcv = 0.0
    t_async = @elapsed begin
        tr = PartitionedArrays.@fake_async begin
            t_wait_exchange = @elapsed wait(t)
            function setup_rcv!(values,local_indices_rcv,buffer_rcv)
                for (p,lid) in enumerate(local_indices_rcv.data)
                    values[lid] = buffer_rcv.data[p]
                end
            end
            t_rcv = @elapsed foreach(setup_rcv!,vector_partition,local_indices_rcv,buffer_rcv)
            nothing
        end
    end
    timings = (;t_snd,t_rcv,t_exchange,t_wait_exchange,t_sync,t_async)
    tr, timings
end

muladd!(y,A,x) = mul!(y,A,x,1,1)
function spmv!(b,A,x)
    t_spmv = @elapsed begin
        t_consistent = @elapsed t,timeings_consistent = consistent2!(x)
        t_mul = @elapsed map(mul!,own_values(b),own_own_values(A),own_values(x))
        t_wait_consistent = @elapsed wait(t)
        t_muladd = @elapsed map(muladd!,own_values(b),own_ghost_values(A),ghost_values(x))
    end
    (;t_spmv,t_consistent,t_mul,t_wait_consistent,t_muladd,timeings_consistent...)
end

function spmv_petsc!(b,A,x)
    t1 = @elapsed begin
        mat = Ref{PetscCall.Mat}()
        vec_b = Ref{PetscCall.Vec}()
        vec_x = Ref{PetscCall.Vec}()
        parts = linear_indices(partition(x))
        petsc_comm = PetscCall.setup_petsc_comm(parts)
        args_A = PetscCall.MatCreateMPIAIJWithSplitArrays_args(A,petsc_comm)
        args_b = PetscCall.VecCreateMPIWithArray_args(copy(b),petsc_comm)
        args_x = PetscCall.VecCreateMPIWithArray_args(copy(x),petsc_comm)
        ownership = (args_A,args_b,args_x)
        PetscCall.@check_error_code PetscCall.MatCreateMPIAIJWithSplitArrays(args_A...,mat)
        PetscCall.@check_error_code PetscCall.MatAssemblyBegin(mat[],PetscCall.MAT_FINAL_ASSEMBLY)
        PetscCall.@check_error_code PetscCall.MatAssemblyEnd(mat[],PetscCall.MAT_FINAL_ASSEMBLY)
        PetscCall.@check_error_code PetscCall.VecCreateMPIWithArray(args_b...,vec_b)
        PetscCall.@check_error_code PetscCall.VecCreateMPIWithArray(args_x...,vec_x)
    end
    MPI.Barrier(MPI.COMM_WORLD)
    t5 = @elapsed PetscCall.@check_error_code PetscCall.MatMult(mat[],vec_x[],vec_b[])
    MPI.Barrier(MPI.COMM_WORLD)
    t2 = @elapsed PetscCall.@check_error_code PetscCall.MatMult(mat[],vec_x[],vec_b[])
    t3 = @elapsed PetscCall.VecCreateMPIWithArray_args_reversed!(b,args_b)
    t4 = @elapsed begin
        GC.@preserve ownership PetscCall.@check_error_code PetscCall.MatDestroy(mat)
        GC.@preserve ownership PetscCall.@check_error_code PetscCall.VecDestroy(vec_b)
        GC.@preserve ownership PetscCall.@check_error_code PetscCall.VecDestroy(vec_x)
    end
    (;
     t_julia_to_petsc=t1,
     t_spmv_petsc=t2,
     t_petsc_to_julia=t3,
     t_cleanup=t4,
     t_spmv_petsc_first=t5,
    )
end

function vec_to_nt(ts)
    t = first(ts)
    d = (;)
    data = map(keys(t)) do k
        k=>map(t2->getproperty(t2,k),ts)
    end
    d = (;data...)
    d
end

function benchmark_spmv_2(distribute,params)
    if ! PetscCall.initialized()
        PetscCall.init(finalize_atexit=false)
    end
    parts_per_dir = params.parts_per_dir
    cells_per_dir = params.cells_per_dir
    nruns = params.nruns
    np = prod(parts_per_dir)
    parts = distribute(LinearIndices((np,)))
    Ti = PetscCall.PetscInt
    T = PetscCall.PetscScalar
    psparse_args = coo_scalar_fem(cells_per_dir,parts_per_dir,parts,T,Ti)
    A = psparse(psparse_args...) |> fetch
    cols = axes(A,2)
    rows = axes(A,1)
    x = pones(PetscCall.PetscScalar,partition(cols))
    b1 = similar(x,rows)
    b2 = similar(x,rows)
    spmv!(b1,A,x)
    spmv_petsc!(b2,A,x)
    c = b1-b2
    rel_error = norm(c)/norm(b1)
    ts1 = Vector{Any}(undef,nruns)
    for irun in 1:nruns
        MPI.Barrier(MPI.COMM_WORLD)
        t1 = spmv!(b1,A,x)
        ts1[irun] = t1
    end
    ts2 = Vector{Any}(undef,nruns)
    for irun in 1:nruns
        t2 = spmv_petsc!(b2,A,x)
        ts2[irun] = t2
    end
    ts = map(ts1,ts2) do t1,t2
        (;t1...,t2...)
    end
    results = (;rel_error,vec_to_nt(ts)...)
    results_gather = gather(map(rank->results,parts))
    results_in_main = map_main(results_gather) do results
        (;vec_to_nt(results)...,params...)
    end
    results_in_main
end


function benchmark_spmv_detailed(distribute,params)
    function muladd!(b,A,x)
        T = eltype(A)
        o = one(T)
        mul!(b,A,x,o,o)
    end
    function spmv_sync!(b,A,x)
        t_consistent = @elapsed task = consistent!(x)
        t_wait  = @elapsed wait(task)
        t_mul = @elapsed map(mul!,own_values(b),own_own_values(A),own_values(x))
        t_muladd = @elapsed map(muladd!,own_values(b),own_ghost_values(A),ghost_values(x))
        t_total = t_consistent + t_wait + t_mul + t_muladd
        (t_consistent,t_wait,t_mul,t_muladd,t_total)
    end
    function spmv_async!(b,A,x)
        t_consistent = @elapsed task = consistent!(x)
        t_mul = @elapsed map(mul!,own_values(b),own_own_values(A),own_values(x))
        t_wait  = @elapsed wait(task)
        t_muladd = @elapsed map(muladd!,own_values(b),own_ghost_values(A),ghost_values(x))
        t_total = t_consistent + t_wait + t_mul + t_muladd
        (t_consistent,t_wait,t_mul,t_muladd,t_total)
    end
    function spmv_monolithic!(b,A,x)
        t_consistent = @elapsed task = consistent!(x)
        t_wait  = @elapsed wait(task)
        t_mul = @elapsed map(mul!,partition(b),partition(A),partition(x))
        t_muladd = 0.0
        t_total = t_consistent + t_wait + t_mul + t_muladd
        (t_consistent,t_wait,t_mul,t_muladd,t_total)
    end
    parts_per_dir = params.parts_per_dir
    cells_per_dir = params.cells_per_dir
    method = params.method
    nruns = params.nruns
    np = prod(parts_per_dir)
    parts = distribute(LinearIndices((np,)))
    Ti = params.Ti
    T = params.T
    psparse_args = coo_scalar_fem(cells_per_dir,parts_per_dir,parts,T,Ti)
    @assert params.matrix == "split-csc"
    A = psparse(psparse_args...) |> fetch
    cols = axes(A,2)
    rows = axes(A,1)
    x = pones(T,partition(cols))
    b = similar(x,rows)
    t = Vector{NTuple{5,Float64}}(undef,nruns)
    if method == "sync"
        for irun in 1:nruns
            b .= 0
            t[irun] =  spmv_sync!(b,A,x)
        end
    elseif method == "async"
        for irun in 1:nruns
            b .= 0
            t[irun] =  spmv_async!(b,A,x)
        end
    elseif method == "monolithic"
        for irun in 1:nruns
            b .= 0
            t[irun] =  spmv_monolithic!(b,A,x)
        end
    else
        error("unknown method $method")
    end
    function gettiming(ts,field)
        dict = Dict(
            :t_consistent=>1,
            :t_wait=>2,
            :t_mul=>3,
            :t_muladd=>4,
            :t_total=>5
           )
        map(t->map(ti->ti[dict[field]],t),ts)
    end
    ts_in_main = gather(map(p->t,parts))
    results_in_main = map_main(ts_in_main) do ts
        t_consistent = gettiming(ts,:t_consistent)
        t_wait = gettiming(ts,:t_wait)
        t_mul = gettiming(ts,:t_mul)
        t_muladd = gettiming(ts,:t_muladd)
        t_total = gettiming(ts,:t_total)
        results = (;t_consistent,t_wait,t_mul,t_muladd,t_total,params...)
        results
    end
end

function benchmark_psparse(distribute,params)
    parts_per_dir = params.parts_per_dir
    cells_per_dir = params.cells_per_dir
    method = params.method
    nruns = params.nruns
    np = prod(parts_per_dir)
    parts = distribute(LinearIndices((np,)))
    Ti = PetscCall.PetscInt
    T = PetscCall.PetscScalar
    psparse_args = coo_scalar_fem(cells_per_dir,parts_per_dir,parts,T,Ti)
    V = psparse_args[3]
    t_buildmat = zeros(nruns)
    t_rebuildmat = zeros(nruns)
    petsc_comm = PetscCall.setup_petsc_comm(parts)
    function petsc_setvalues(I,J,V,rows,cols)
        m = own_length(rows)
        n = own_length(cols)
        M = global_length(rows)
        N = global_length(cols)
        I .= I .- 1
        J .= J .- 1
        for irun in 1:nruns
            t_buildmat[irun] = @elapsed begin
                A = Ref{PetscCall.Mat}()
                PetscCall.@check_error_code PetscCall.MatCreate(petsc_comm,A)
                PetscCall.@check_error_code PetscCall.MatSetType(A[],PetscCall.MATMPIAIJ)
                PetscCall.@check_error_code PetscCall.MatSetSizes(A[],m,n,M,N)
                PetscCall.@check_error_code PetscCall.MatMPIAIJSetPreallocation(A[],32,C_NULL,32,C_NULL)
                for p in 1:length(I)
                    PetscCall.@check_error_code PetscCall.MatSetValues(A[],1,view(I,p:p),1,view(J,p:p),view(V,p:p),PetscCall.ADD_VALUES)
                end
                PetscCall.@check_error_code PetscCall.MatAssemblyBegin(A[],PetscCall.MAT_FINAL_ASSEMBLY)
                PetscCall.@check_error_code PetscCall.MatAssemblyEnd(A[],PetscCall.MAT_FINAL_ASSEMBLY)
            end
            PetscCall.@check_error_code PetscCall.MatDestroy(A)
        end
        A = Ref{PetscCall.Mat}()
        PetscCall.@check_error_code PetscCall.MatCreate(petsc_comm,A)
        PetscCall.@check_error_code PetscCall.MatSetType(A[],PetscCall.MATMPIAIJ)
        PetscCall.@check_error_code PetscCall.MatSetSizes(A[],m,n,M,N)
        PetscCall.@check_error_code PetscCall.MatMPIAIJSetPreallocation(A[],32,C_NULL,32,C_NULL)
        for p in 1:length(I)
            PetscCall.@check_error_code PetscCall.MatSetValues(A[],1,view(I,p:p),1,view(J,p:p),view(V,p:p),PetscCall.ADD_VALUES)
        end
        PetscCall.@check_error_code PetscCall.MatAssemblyBegin(A[],PetscCall.MAT_FINAL_ASSEMBLY)
        PetscCall.@check_error_code PetscCall.MatAssemblyEnd(A[],PetscCall.MAT_FINAL_ASSEMBLY)
        for irun in 1:nruns
            t_rebuildmat[irun] = @elapsed begin
                for p in 1:length(I)
                    PetscCall.@check_error_code PetscCall.MatSetValues(A[],1,view(I,p:p),1,view(J,p:p),view(V,p:p),PetscCall.ADD_VALUES)
                end
                PetscCall.@check_error_code PetscCall.MatAssemblyBegin(A[],PetscCall.MAT_FINAL_ASSEMBLY)
                PetscCall.@check_error_code PetscCall.MatAssemblyEnd(A[],PetscCall.MAT_FINAL_ASSEMBLY)
            end
        end
        PetscCall.@check_error_code PetscCall.MatDestroy(A)
    end
    function petsc_coo(I,J,V,rows,cols)
        m = own_length(rows)
        n = own_length(cols)
        M = global_length(rows)
        N = global_length(cols)
        I .= I .- 1
        J .= J .- 1
        ncoo = length(I)
        for irun in 1:nruns
            t_buildmat[irun] = @elapsed begin
                A = Ref{PetscCall.Mat}()
                PetscCall.@check_error_code PetscCall.MatCreate(petsc_comm,A)
                PetscCall.@check_error_code PetscCall.MatSetType(A[],PetscCall.MATMPIAIJ)
                PetscCall.@check_error_code PetscCall.MatSetSizes(A[],m,n,M,N)
                PetscCall.@check_error_code PetscCall.MatSetPreallocationCOO(A[],ncoo,I,J)
                PetscCall.@check_error_code PetscCall.MatSetValuesCOO(A[],V,PetscCall.ADD_VALUES)
                PetscCall.@check_error_code PetscCall.MatAssemblyBegin(A[],PetscCall.MAT_FINAL_ASSEMBLY)
                PetscCall.@check_error_code PetscCall.MatAssemblyEnd(A[],PetscCall.MAT_FINAL_ASSEMBLY)
            end
            PetscCall.@check_error_code PetscCall.MatDestroy(A)
        end
        A = Ref{PetscCall.Mat}()
        PetscCall.@check_error_code PetscCall.MatCreate(petsc_comm,A)
        PetscCall.@check_error_code PetscCall.MatSetType(A[],PetscCall.MATMPIAIJ)
        PetscCall.@check_error_code PetscCall.MatSetSizes(A[],m,n,M,N)
        PetscCall.@check_error_code PetscCall.MatSetPreallocationCOO(A[],ncoo,I,J)
        PetscCall.@check_error_code PetscCall.MatSetValuesCOO(A[],V,PetscCall.ADD_VALUES)
        PetscCall.@check_error_code PetscCall.MatAssemblyBegin(A[],PetscCall.MAT_FINAL_ASSEMBLY)
        PetscCall.@check_error_code PetscCall.MatAssemblyEnd(A[],PetscCall.MAT_FINAL_ASSEMBLY)
        for irun in 1:nruns
            t_rebuildmat[irun] = @elapsed begin
                PetscCall.@check_error_code PetscCall.MatSetValuesCOO(A[],V,PetscCall.ADD_VALUES)
                PetscCall.@check_error_code PetscCall.MatAssemblyBegin(A[],PetscCall.MAT_FINAL_ASSEMBLY)
                PetscCall.@check_error_code PetscCall.MatAssemblyEnd(A[],PetscCall.MAT_FINAL_ASSEMBLY)
            end
        end
        PetscCall.@check_error_code PetscCall.MatDestroy(A)
    end
    if method == "psprse"
        for irun in 1:nruns
            t_buildmat[irun] = @elapsed psparse(psparse_args...;reuse=true) |> fetch
        end
        A, cacheA = psparse(psparse_args...;reuse=true) |> fetch
        for irun in 1:nruns
            t_rebuildmat[irun] = @elapsed psparse!(A,V,cacheA) |> wait
        end
    elseif method == "petsc_setvalues"
        PetscCall.init(finalize_atexit=false)
        map(petsc_setvalues,psparse_args...)
        PetscCall.finalize()
    elseif method == "petsc_coo"
        PetscCall.init(finalize_atexit=false)
        map(petsc_coo,psparse_args...)
        PetscCall.finalize()
    end
    ts_in_main = gather(map(p->(;t_buildmat,t_rebuildmat),parts))
    results_in_main = map_main(ts_in_main) do ts
        buildmat = map(i->i.t_buildmat,ts)
        rebuildmat = map(i->i.t_rebuildmat,ts)
        results = (;buildmat,rebuildmat,params...)
        results
    end
end

function coo_scalar_fem(cells_per_dir,parts_per_dir,parts,::Type{T},::Type{Ti}) where {Ti,T}
    # TODO only symbolic info for the moment
    D = length(cells_per_dir)
    nodes_per_dir = cells_per_dir .+ 1
    ghost_per_dir = ntuple(d->true,Val(D))
    node_partition = uniform_partition(parts,parts_per_dir,nodes_per_dir,ghost_per_dir)
    cell_partition = uniform_partition(parts,parts_per_dir,cells_per_dir)
    isboundary(cartesian_node) = any(map((d,i)->i ∉ 2:(nodes_per_dir[d]-1),1:D,Tuple(cartesian_node)))
    function fill_mask!(local_node_to_mask,nodes)
        node_to_cartesian_node = CartesianIndices(nodes_per_dir)
        n_local_nodes = local_length(nodes)
        local_node_to_node = local_to_global(nodes)
        for local_node in 1:n_local_nodes
            node = local_node_to_node[local_node]
            cartesian_node = node_to_cartesian_node[node]
            mask = ! isboundary(cartesian_node)
            local_node_to_mask[local_node] = mask
        end
    end
    node_to_mask = pfill(false,node_partition)
    map(fill_mask!,partition(node_to_mask),node_partition)
    dof_to_local_node, node_to_local_dof = find_local_indices(node_to_mask)
    dof_partition = partition(axes(dof_to_local_node,1))
    function setup(cells,nodes,dofs,local_node_to_local_dof)
        cell_to_cartesian_cell = CartesianIndices(cells_per_dir)
        cartesian_node_to_node = LinearIndices(nodes_per_dir)
        own_cell_to_cell = own_to_global(cells)
        node_to_local_node = global_to_local(nodes)
        local_dof_to_dof = local_to_global(dofs)
        n_own_cells = length(own_cell_to_cell)
        n_nz = n_own_cells*(2^(2*D))
        myI = zeros(Ti,n_nz)
        myJ = zeros(Ti,n_nz)
        myV = ones(T,n_nz)
        p = 0
        for own_cell in 1:n_own_cells
            cell = own_cell_to_cell[own_cell]
            cartesian_cell = cell_to_cartesian_cell[cell]
            for di in 1:D
                offset_i = ntuple(d->(d==di ? 1 : 0),Val(D))
                cartesian_node_i = CartesianIndex( Tuple(cartesian_cell) .+ offset_i )
                if isboundary(cartesian_node_i)
                    continue
                end
                node_i = cartesian_node_to_node[cartesian_node_i]
                local_node_i = node_to_local_node[node_i]
                local_dof_i = local_node_to_local_dof[local_node_i]
                dof_i = local_dof_to_dof[local_dof_i]
                for dj in 1:D
                    offset_j = ntuple(d->(d==dj ? 1 : 0),Val(D))
                    cartesian_node_j = CartesianIndex( Tuple(cartesian_cell) .+ offset_j )
                    if isboundary(cartesian_node_j)
                        continue
                    end
                    node_j = cartesian_node_to_node[cartesian_node_j]
                    local_node_j = node_to_local_node[node_j]
                    local_dof_j = local_node_to_local_dof[local_node_j]
                    dof_j = local_dof_to_dof[local_dof_j]
                    p += 1
                    myI[p] = dof_i
                    myJ[p] = dof_j
                end
            end
        end
        (myI[1:p],myJ[1:p],myV[1:p])
    end
    I,J,V = map(setup,cell_partition,node_partition,dof_partition,partition(node_to_local_dof)) |> tuple_of_arrays
    row_partition = map(remove_ghost,dof_partition)
    col_partition = row_partition
    I,J,V,row_partition,col_partition
end

