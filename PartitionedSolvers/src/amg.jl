
function aggregate(A,diagA=dense_diag(A);epsilon)
    # TODO It assumes CSC format for the moment

    # This one is algorithm 5.1 from
    # "Algebraic multigrid by smoothed aggregation for second and fourth order elliptic problems"

    epsi = epsilon
    typeof_aggregate = Int32
    typeof_strength = eltype(A.nzval)

    nnodes = size(A,1)
    pending = typeof_aggregate(0)
    isolated = typeof_aggregate(-1)
    
    node_to_aggregate = fill(pending,nnodes)
    node_to_old_aggregate = similar(node_to_aggregate)

    node_to_neigs = jagged_array(A.rowval,A.colptr)
    node_to_vals = jagged_array(A.nzval,A.colptr)
    strongly_connected = (node,ineig) -> begin
        neig = node_to_neigs[node][ineig]
        aii = diagA[node]
        ajj = diagA[neig]
        aij = node_to_vals[node][ineig]
        abs(aij) > epsi*sqrt(aii*ajj)
    end
    coupling_strength = (node,ineig) -> begin
        abs(node_to_vals[node][ineig])
    end

    # Initialization
    for node in 1:nnodes
        neigs = node_to_neigs[node]
        isolated_node = count(i->i!=node,neigs) == 0
        if isolated_node
            node_to_aggregate[node] = isolated
        end
    end

    # Step 1
    aggregate = typeof_aggregate(0)
    for node in 1:nnodes
        if node_to_aggregate[node] != pending
            continue
        end
        neigs = node_to_neigs[node]
        nneigs = length(neigs)
        all_pending = true
        for ineig in 1:nneigs
            neig = neigs[ineig]
            if neig == node || !strongly_connected(node,ineig)
                continue
            end
            all_pending &= (node_to_aggregate[neig] == pending)
        end
        if !all_pending
            continue
        end
        aggregate += typeof_aggregate(1)
        node_to_aggregate[node] = aggregate
        for ineig in 1:nneigs
            neig = neigs[ineig]
            if neig == node || !strongly_connected(node,ineig)
                continue
            end
            node_to_aggregate[neig] = aggregate
        end
    end

    # Step 2
    copy!(node_to_old_aggregate,node_to_aggregate)
    for node in 1:nnodes
        if node_to_aggregate[node] != pending
            continue
        end
        strength = zero(typeof_strength)
        neigs = node_to_neigs[node]
        nneigs = length(neigs)
        for ineig in 1:nneigs
            neig = neigs[ineig]
            if neig == node || !strongly_connected(node,ineig)
                continue
            end
            neig_aggregate = node_to_old_aggregate[neig]
            if neig_aggregate != pending && neig_aggregate != isolated
                neig_strength = coupling_strength(node,ineig)
                if neig_strength > strength
                    strength = neig_strength
                    node_to_aggregate[node] = neig_aggregate
                end
            end
        end
    end

    # Step 3
    for node in 1:nnodes
        if node_to_aggregate[node] != pending
            continue
        end
        aggregate += typeof_aggregate(1)
        node_to_aggregate[node] = aggregate
        neigs = node_to_neigs[node]
        nneigs = length(neigs)
        for ineig in 1:nneigs
            neig = neigs[ineig]
            if neig == node || !strongly_connected(node,ineig)
                continue
            end
            neig_aggregate = node_to_old_aggregate[neig]
            if neig_aggregate == pending || neig_aggregate == isolated
                node_to_aggregate[neig] = aggregate
            end
        end
    end
    naggregates = aggregate

    if nnodes == 1
        node_to_aggregate .= 1
        naggregates = 1
    end
    node_to_aggregate, 1:naggregates
end

function aggregate(A::PSparseMatrix,diagA=dense_diag(A);kwargs...)
    # This is the vanilla "uncoupled" strategy from "Parallel Smoothed Aggregation Multigrid : Aggregation Strategies on Massively Parallel Machines"
    # TODO: implement other more advanced strategies
    @assert A.assembled
    node_to_aggregate_data, local_ranges = map((A,diagA)->aggregate(A,diagA;kwargs...),own_own_values(A),own_values(diagA)) |> tuple_of_arrays
    nown = map(length,local_ranges)
    n_aggregates = sum(nown)
    nparts = length(nown)
    aggregate_partition = variable_partition(nown,n_aggregates)
    node_partition = partition(axes(A,1))
    map(map_own_to_global!,node_to_aggregate_data,aggregate_partition)
    node_to_aggregate = PVector(node_to_aggregate_data,node_partition)
    node_to_aggregate, PRange(aggregate_partition)
end

function constant_prolongator(node_to_aggregate,aggregates,n_nullspace_vecs)
    if n_nullspace_vecs != 1
        error("case not implemented yet")
    end
    typeof_aggregate = eltype(node_to_aggregate)
    nnodes = length(node_to_aggregate)
    pending = typeof_aggregate(0)
    isolated = typeof_aggregate(-1)
    naggregates = length(aggregates)
    aggregate_to_nodes_ptrs = zeros(Int,naggregates+1)
    for node in 1:nnodes
        agg = node_to_aggregate[node]
        if agg == pending
            continue
        end
        aggregate_to_nodes_ptrs[agg+1] += 1
    end
    length_to_ptrs!(aggregate_to_nodes_ptrs)
    ndata = aggregate_to_nodes_ptrs[end]-1
    aggregate_to_nodes_data = zeros(Int,ndata)
    for node in 1:nnodes
        agg = node_to_aggregate[node]
        if agg == pending
            continue
        end
        p = aggregate_to_nodes_ptrs[agg]
        aggregate_to_nodes_data[p] = node
        aggregate_to_nodes_ptrs[agg] += 1
    end
    rewind_ptrs!(aggregate_to_nodes_ptrs)

    P0 = SparseMatrixCSC(
        nnodes,
        naggregates,
        aggregate_to_nodes_ptrs,
        aggregate_to_nodes_data,
        ones(ndata))
    P0
end

function constant_prolongator(node_to_aggregate::PVector,aggregates::PRange,n_nullspace_vecs)
    if n_nullspace_vecs != 1
        error("case not implemented yet")
    end
    function setup_triplets(node_to_aggregate,nodes)
        myI = 1:local_length(nodes)
        myJ = node_to_aggregate
        myV = ones(length(node_to_aggregate))
        (myI,myJ,myV)
    end
    node_partition = partition(axes(node_to_aggregate,1))
    I,J,V = map(setup_triplets,partition(node_to_aggregate),node_partition) |> tuple_of_arrays
    aggregate_partition = partition(aggregates)
    J_owner = find_owner(aggregate_partition,J)
    aggregate_partition = map(union_ghost,aggregate_partition,J,J_owner)
    map(map_global_to_local!,J,aggregate_partition)
    P0 = psparse(I,J,V,node_partition,aggregate_partition;assembled=true,indices=:local) |> fetch
    P0
end

function collect_nodes_in_aggregate(node_to_aggregate,aggregates)
    typeof_aggregate = eltype(node_to_aggregate)
    nnodes = length(node_to_aggregate)
    pending = typeof_aggregate(0)
    isolated = typeof_aggregate(-1)
    nnodes = length(node_to_aggregate)
    naggregates = length(aggregates)
    aggregate_to_nodes_ptrs = zeros(Int,naggregates+1)
    for node in 1:nnodes
        agg = node_to_aggregate[node]
        if agg == pending || agg == isolated
            continue
        end
        aggregate_to_nodes_ptrs[agg+1] += 1
    end 
    length_to_ptrs!(aggregate_to_nodes_ptrs)
    ndata = aggregate_to_nodes_ptrs[end]-1
    aggregate_to_nodes_data = zeros(Int,ndata)
    for node in 1:nnodes
        agg = node_to_aggregate[node]
        if agg == pending || agg == isolated
            continue
        end
        p = aggregate_to_nodes_ptrs[agg]
        aggregate_to_nodes_data[p] = node
        aggregate_to_nodes_ptrs[agg] += 1
    end
    rewind_ptrs!(aggregate_to_nodes_ptrs)
    aggregate_to_nodes = jagged_array(aggregate_to_nodes_data,aggregate_to_nodes_ptrs)
    aggregate_to_nodes
end

function remove_singleton_aggregates(aggregate_to_nodes_old)
    typeof_nodes = eltype(aggregate_to_nodes_old[1])
    n_aggregates_old = length(aggregate_to_nodes_old)
    n_aggregates = 0
    n_data = 0
    for i in 1:n_aggregates_old
        nnodes = aggregate_to_nodes_old.ptrs[i+1] - aggregate_to_nodes_old.ptrs[i]
        if nnodes < 2
            continue
        end
        n_aggregates += 1 
        n_data += nnodes
    end

    # If there are no singleton aggregates, stop and return old array
    if n_aggregates == n_aggregates_old
        return aggregate_to_nodes_old
    end

    # Else copy non-singleton aggregates to new array
    aggregate_to_nodes_ptrs = zeros(Int, n_aggregates+1)
    aggregate_to_nodes_data = zeros(typeof_nodes, n_data)
    aggregate_to_nodes_ptrs[1] = 1
    agg = 1
    for i in 1:n_aggregates_old
        nnodes = aggregate_to_nodes_old.ptrs[i+1] - aggregate_to_nodes_old.ptrs[i]
        if nnodes < 2
            continue 
        end
        aggregate_to_nodes_ptrs[agg+1] = aggregate_to_nodes_ptrs[agg] + nnodes 
        pini = aggregate_to_nodes_ptrs[agg]
        pend = aggregate_to_nodes_ptrs[agg+1]-1
        pini_old = aggregate_to_nodes_old.ptrs[i]
        pend_old = aggregate_to_nodes_old.ptrs[i+1]-1
        aggregate_to_nodes_data[pini:pend] = aggregate_to_nodes_old.data[pini_old:pend_old]
        agg += 1     
    end
    aggregate_to_nodes = jagged_array(aggregate_to_nodes_data,aggregate_to_nodes_ptrs)
    aggregate_to_nodes
end

function tentative_prolongator_for_laplace(P0,B)
    n_nullspace_vecs = length(B)
    if n_nullspace_vecs != 1
        error("Only one nullspace vector allowed")
    end
    Bc = default_nullspace(P0)
    P0,Bc
end

function tentative_prolongator_with_block_size(aggregate_to_nodes::JaggedArray,B, block_size)
    # A draft for the scalar case is commented below
    ## TODO assumes CSC
    # Algorithm 7 in https://mediatum.ub.tum.de/download/1229321/1229321.pdf 

    if length(B) < 1
        error("Null space must contain at least one null vector.")
    end

    n_aggregates = length(aggregate_to_nodes.ptrs)-1
    n_B = length(B)
    n_dofs = length(B[1])
    n_dofs_c = n_aggregates * n_B
    Bc = [Vector{Float64}(undef,n_dofs_c) for _ in 1:n_B]

    # Build P0 colptr
    P0_colptr = zeros(Int, n_dofs_c + 1)
    P0_colptr[1] = 1
    for i_agg in 1:n_aggregates
        pini = aggregate_to_nodes.ptrs[i_agg]
        pend = aggregate_to_nodes.ptrs[i_agg+1]-1
        n_nodes = length(pini:pend)
        for b in 1:n_B 
            col = (i_agg - 1) * n_B + b
            P0_colptr[col+1] += P0_colptr[col] + n_nodes * block_size 
        end
    end
    
    # Build P0 rowvals
    nnz = length(aggregate_to_nodes.data) * block_size * n_B
    P0_rowval = zeros(Int, nnz)
    for i_agg in 1:n_aggregates
        pini = aggregate_to_nodes.ptrs[i_agg]
        pend = aggregate_to_nodes.ptrs[i_agg+1]-1
        i_nodes = aggregate_to_nodes.data[pini:pend]
        for b in 1:n_B
            rval_ini = P0_colptr[(i_agg-1)*n_B+b]
            for i_node in i_nodes
                rval_end = rval_ini+block_size-1
                P0_rowval[rval_ini:rval_end] = node_to_dofs(i_node, block_size)
                rval_ini = rval_end + 1
            end
        end
    end
    
    P0 = SparseMatrixCSC(
        n_dofs,
        n_dofs_c,
        P0_colptr,
        P0_rowval,
        ones(nnz))
    
    # Fill with nullspace vectors 
    for i_agg in 1:n_aggregates
        for b in 1:n_B
            col = (i_agg-1) * n_B + b
            pini = P0.colptr[col]
            pend = P0.colptr[col+1]-1
            rowids = P0.rowval[pini:pend]
            P0.nzval[pini:pend] = B[b][rowids]
        end
    end

    # Compute QR decomposition for nullspace in each aggregate 
    for i_agg in 1:n_aggregates
        # Copy values to Bi 
        pini = aggregate_to_nodes.ptrs[i_agg]
        pend = aggregate_to_nodes.ptrs[i_agg+1]-1
        n_i = length(pini:pend) * block_size
        Bi = zeros(n_i, n_B)
        P0cols = (i_agg-1)*n_B+1 : i_agg*n_B
        for (b, col) in enumerate(P0cols)
            pini = P0.colptr[col]
            pend = P0.colptr[col+1]-1
            Bi[:,b] = P0.nzval[pini:pend]
        end

        # Compute thin QR decomposition 
        # TODO: check if QR decomposition is possible if n_i >= n_B, raise an error
        F= qr(Bi)
        Qi = F.Q * Matrix(I,(n_i, n_B))
        Qi = Qi[:,1:n_B]
        Ri = F.R

        # Build global tentative prolongator 
        for (b, col) in enumerate(P0cols)
            pini = P0.colptr[col]
            pend = P0.colptr[col+1]-1
            P0.nzval[pini:pend] = Qi[:,b]
        end

        # Build coarse null space
        Bcrows = P0cols
        for b in 1:n_B
            Bc[b][Bcrows] = Ri[:, b]
        end
    end

    P0, Bc
end

function generic_tentative_prolongator(P0::SparseArrays.AbstractSparseMatrixCSC,B)
    error("not implemented yet")
    # A draft for the scalar case is commented below
    ## TODO assumes CSC
    ##TODO null space for vector-valued problems
    #tol = 1e-10
    #Tv = eltype(B[1])
    #nf,nc = size(P0)
    #R = zeros(Tv,nc)
    #copy!(P0.nzval,B[1])
    #for j in 1:nc
    #    # For each column, we do its QR factorization.
    #    # This is equivalent to the first step of a Gram-Schimdt process
    #    # (making the column a unit vector)
    #    norm_j = zero(Tv)
    #    for p in nzrange(P0,j)
    #        val = P0.nzval[p]
    #        norm_j += val*val
    #    end
    #    norm_j = sqrt(norm_j)
    #    if norm_j < tol
    #        continue
    #    end
    #    R[j] = norm_j
    #    inv_norm_j = one(Tv)/norm_j
    #    for p in nzrange(P0,j)
    #        P0.nzval[p] *= inv_norm_j
    #    end
    #end
    #Bc = R
    #Bc
end

function generic_tentative_prolongator(P0::PSparseMatrix,B)
    error("not implemented yet")
end

function smoothed_prolongator(A,P0,diagA=dense_diag(A);approximate_omega)
    # TODO the performance of this one can be improved
    invDiagA = 1 ./ diagA
    Dinv = PartitionedArrays.sparse_diag_matrix(invDiagA,(axes(A,1),axes(A,1)))
    omega = approximate_omega(invDiagA,A,Dinv*A)
    P = (I-omega*Dinv*A)*P0
    P
end

function omega_for_1d_laplace(invD,A, DinvA)
    # in general this will be (4/3)/ρ(D^-1*A)
    # with a cheap approximation (e.g., power method) of the spectral radius ρ(D^-1*A)
    # ρ(D^-1*A) == 2 for 1d Laplace problem
    2/3
end

function lambda_generic(invD,A,DinvA)
    ω = 4/3
    # Perform a few iterations of Power method to estimate lambda 
    # (Remark 3.5.2. in https://mediatum.ub.tum.de/download/1229321/1229321.pdf)
    ρ = spectral_radius(DinvA, 20)
    ω/ρ 
end

function spectral_radius(A, x, num_iterations:: Int)
    # Choose diagonal vector as initial guess
    y = zeros(size(A,1))
    # Power iteration
    for i in 1:num_iterations
        mul!(y, A, x)
        y_norm = norm(y)
        x = y/y_norm
    end
    # Compute spectral radius using Rayleigh coefficient
    y = mul!(y, A, x)
    ρ = (y' * x) / (x' * x)
    abs(ρ)
end

function enhance_coarse_partition(A,Ac,Bc,R,P,cache,repartition_threshold)
    Ac,Bc,R,P,cache,repartition_threshold
end

function enhance_coarse_partition(A::PSparseMatrix,Ac,Bc,R,P,cache,repartition_threshold)
    # TODO now we redistribute to a single proc when the threshold is reached
    # We need to redistributed to fewer but several processors
    # TODO a way of avoiding the extra rap ?
    n_coarse = size(Ac,1)
    if n_coarse <= repartition_threshold
        repartition_threshold = 0 # do not repartition again
        parts = linear_indices(partition(Ac))
        coarse_partition = trivial_partition(parts,n_coarse)
        P = repartition(P,partition(axes(P,1)),coarse_partition) |> fetch
        Bc = map(b->repartition(b,partition(axes(P,2)))|>fetch,Bc)
        R = transpose(P)
        Ac,cache = rap(R,A,P;reuse=true)
    end
    Ac,Bc,R,P,cache,repartition_threshold
end

function smoothed_aggregation(;
    epsilon = 0,
    approximate_omega = omega_for_1d_laplace,
    tentative_prolongator = tentative_prolongator_for_laplace,
    repartition_threshold = 2000,
    )
    function coarsen(A,B)
        diagA = dense_diag(A)
        node_to_aggregate, aggregates = aggregate(A,diagA;epsilon)
        n_nullspace_vecs = length(B)
        P0 = constant_prolongator(node_to_aggregate, aggregates,n_nullspace_vecs)
        P0,Bc = tentative_prolongator(P0,B)
        P = smoothed_prolongator(A,P0,diagA;approximate_omega)
        R = transpose(P)
        Ac,cache = rap(R,A,P;reuse=true)
        Ac,Bc,R,P,cache,repartition_threshold = enhance_coarse_partition(A,Ac,Bc,R,P,cache,repartition_threshold)
        Ac,Bc,R,P,cache
    end
    function coarsen!(A,Ac,R,P,cache)
        rap!(Ac,R,A,P,cache)
        Ac,R,P,cache
    end
    (coarsen, coarsen!)
end

function smoothed_aggregation_with_block_size(;
    epsilon = 0,
    approximate_omega = lambda_generic,
    tentative_prolongator = tentative_prolongator_with_block_size,
    repartition_threshold = 2000,
    block_size = 1,
    )
    function coarsen(A,B)
        # build strength graph
        G = strength_graph(A, block_size=block_size, epsilon=epsilon)
        diagG = dense_diag(G)
        node_to_aggregate, node_aggregates = aggregate(G,diagG;epsilon)
        aggregate_to_nodes = collect_nodes_in_aggregate(node_to_aggregate, node_aggregates)
        Pc,Bc,perm = tentative_prolongator(aggregate_to_nodes,B, block_size) 
        P = smoothed_prolongator(A,Pc,diagG;approximate_omega) 
        R = transpose(P)
        Ac,cache = rap(R,A,P;reuse=true)
        Ac,Bc,R,P,cache,repartition_threshold = enhance_coarse_partition(A,Ac,Bc,R,P,cache,repartition_threshold)#maybe changes here
        Ac,Bc,R,P,cache
    end
    function coarsen!(A,Ac,R,P,cache)
        rap!(Ac,R,A,P,cache)
        Ac,R,P,cache
    end
    (coarsen, coarsen!)
end

function getblock!(B,A,ids_i,ids_j)
    for (j,J) in enumerate(ids_j)
        for (i,I) in enumerate(ids_i)
            B[i,j] = A[I,J]
        end
    end
end

function strength_graph(A::AbstractSparseMatrix, block_size::Integer; epsilon = 0)

    if block_size < 1 
        error("Block size must be equal to or larger than 1.")
    end

    if A.n != A.m 
        error("Matrix must be square.")
    end

    if A.n % block_size != 0
        error("Matrix size must be multiple of block size.")
    end

    if block_size == 1
        return A
    end

    if epsilon < 0
        error("Expected a positive epsilon.")
    end

    n_dofs = A.m 
    nnodes = div(n_dofs, block_size)
    B = zeros(block_size,block_size)
    diag_norms = zeros(nnodes)

    # Count nonzero values 
    nnz = 0
    
    # If epsilon <= 1, all diagonal entries are in the graph
    if epsilon <= 1 
        nnz += nnodes
    end

    # Calculate norms of diagonal values first
    for i_node in 1:nnodes
        i_dofs = node_to_dofs(i_node, block_size)
        getblock!(B,A,i_dofs,i_dofs)
        diag_norms[i_node] = norm(B)
    end

    for j_node in 1:nnodes
        for i_node in 1:nnodes
            if j_node != i_node
                i_dofs = node_to_dofs(i_node, block_size) 
                j_dofs = node_to_dofs(j_node, block_size) 
                getblock!(B,A,i_dofs,j_dofs)
                # Calculate strength according to https://github.com/pyamg/pyamg/blob/e1fe54c93be1029c02ddcf84c2338a607b088703/pyamg/strength.py#L275 
                if norm(B) >= epsilon * sqrt(diag_norms[i_node] * diag_norms[j_node])
                    nnz += 1
                end
            end 
        end
    end
    
    # Allocate data structures for sparsematrix
    I = zeros(Int, nnz)
    J = zeros(Int, nnz)
    V = zeros(nnz)

    i_nz = 1

    for j_node in 1:nnodes
        for i_node in 1:nnodes
            if j_node == i_node
                # Diagonal entries are always in graph if epsilon <= 1
                if epsilon <= 1
                    I[i_nz] = i_node 
                    J[i_nz] = j_node
                    V[i_nz] = 1.0
                    i_nz += 1
                end
            else 
                i_dofs = node_to_dofs(i_node, block_size)
                j_dofs = node_to_dofs(j_node, block_size) 
                getblock!(B,A,i_dofs,j_dofs)
                # Calculate strength according to https://github.com/pyamg/pyamg/blob/e1fe54c93be1029c02ddcf84c2338a607b088703/pyamg/strength.py#L275 
                if norm(B) >= epsilon * sqrt(diag_norms[i_node] * diag_norms[j_node])
                    I[i_nz] = i_node 
                    J[i_nz] = j_node
                    V[i_nz] = 1.0
                    i_nz += 1
                end
            end
        end
    end

    G = sparse(I, J, V, nnodes, nnodes)
    G
end 

function node_to_dofs(i_node, block_size)
    # Convert single node in graph to range of dofs
    ((i_node-1)*block_size+1) : i_node*block_size
end

function dofs_to_node(dofs, block_size)
    # Convert range of dofs to single node in graph
    div(dofs[end], block_size)
end

function amg_level_params_linear_elasticity(;
    pre_smoother = additive_schwarz(gauss_seidel(;iters=1);iters=1),
    coarsening = smoothed_aggregation(approximate_omega = lambda_generic,
    tentative_prolongator = tentative_prolongator_with_block_size),
    cycle = v_cycle,
    pos_smoother = pre_smoother,
    )

    level_params = (;pre_smoother,pos_smoother,coarsening,cycle)
    level_params
end

function amg_level_params(;
    pre_smoother = additive_schwarz(gauss_seidel(;iters=1);iters=1),
    coarsening = smoothed_aggregation(;),
    cycle = v_cycle,
    pos_smoother = pre_smoother,
    )

    level_params = (;pre_smoother,pos_smoother,coarsening,cycle)
    level_params
end

function amg_fine_params(;level_params = amg_level_params(),n_fine_levels=6)
    #TODO more resonable defaults?
    fine_params = fill(level_params,n_fine_levels)
    fine_params
end

function amg_coarse_params(;
    #TODO more resonable defaults?
    coarse_solver = lu_solver(),
    coarse_size = 10,
    )
    coarse_params = (;coarse_solver,coarse_size)
    coarse_params
end

function amg(;
        fine_params=amg_fine_params(),
        coarse_params=amg_coarse_params(),)
    amg_params = (;fine_params,coarse_params)
    setup(x,O,b,options) = amg_setup(x,O,b,nullspace(options),amg_params)
    update! = amg_update!
    solve! = amg_solve!
    finalize! = amg_finalize!
    linear_solver(;setup,update!,solve!,finalize!)
end

function amg_setup(x,A,b,::Nothing,amg_params)
    B = default_nullspace(A)
    amg_setup(x,A,b,B,amg_params)
end
function amg_setup(x,A,b,B,amg_params)
    fine_params = amg_params.fine_params
    coarse_params = amg_params.coarse_params
    (;coarse_solver,coarse_size) = coarse_params
    done = false
    fine_levels =  map(fine_params) do fine_level
        if done
            return nothing
        end
        (;pre_smoother,pos_smoother,coarsening,cycle) = fine_level
        pre_setup = setup(pre_smoother,x,A,b)
        pos_setup = setup(pos_smoother,x,A,b)
        coarsen, _ = coarsening
        Ac,Bc,R,P,Ac_setup = coarsen(A,B)
        nc = size(Ac,1)
        if nc <= coarse_size
            done = true
        end
        r = similar(b)
        rc = similar(r,axes(Ac,2)) # we need ghost ids for the mul!(rc,R,r)
        rc2 = similar(r,axes(P,2)) # TODO
        e = similar(x)
        ec = similar(e,axes(Ac,2))
        ec2 = similar(e,axes(P,2)) # TODO
        level_setup = (;R,P,r,rc,rc2,e,ec,ec2,A,B,Ac,Bc,pre_setup,pos_setup,Ac_setup)
        x = ec
        b = rc
        A = Ac
        B = Bc
        level_setup
    end
    n_fine_levels = count(i->i!==nothing,fine_levels)
    nlevels = n_fine_levels+1
    coarse_solver_setup = setup(coarse_solver,x,A,b)
    coarse_level = (;coarse_solver_setup)
    (;nlevels,fine_levels,coarse_level,amg_params)
end

function amg_solve!(x,setup,b,options)
    level=1
    amg_cycle!(x,setup,b,level)
    x
end

function amg_cycle!(x,setup,b,level)
    amg_params = setup.amg_params
    if level == setup.nlevels
        coarse_solver_setup = setup.coarse_level.coarse_solver_setup
        return solve!(x,coarse_solver_setup,b)
    end
    level_params = amg_params.fine_params[level]
    level_setup = setup.fine_levels[level]
    (;cycle) = level_params
    (;R,P,r,rc,rc2,e,ec,ec2,A,Ac,pre_setup,pos_setup) = level_setup
    solve!(x,pre_setup,b)
    mul!(r,A,x)
    r .= b .- r
    mul!(rc2,R,r)
    rc .= rc2
    fill!(ec,zero(eltype(ec)))
    cycle(ec,setup,rc,level+1)
    ec2 .= ec
    mul!(e,P,ec2)
    x .+= e
    solve!(x,pos_setup,b)
    x
end

function amg_statistics(P::Preconditioner)
    # Taken from: An Introduction to Algebraic Multigrid, R. D. Falgout, April 25, 2006
    # Grid complexity is the total number of grid points on all grids divided by the number
    # of grid points on the fine grid. Operator complexity is the total number of nonzeroes in the linear operators
    # on all grids divided by the number of nonzeroes in the fine grid operator
    setup = P.solver_setup
    nlevels = setup.nlevels
    level_rows = zeros(Int,nlevels)
    level_nnz = zeros(Int,nlevels)
    for level in 1:(nlevels-1)
        level_setup = setup.fine_levels[level]
        (;A,) = level_setup
        level_rows[level] = size(A,1)
        level_nnz[level] = nnz(A)
    end
    level_setup = setup.fine_levels[nlevels-1]
    (;Ac) = level_setup
    level_rows[end] = size(Ac,1)
    level_nnz[end] = nnz(Ac)
    nnz_total = sum(level_nnz)
    rows_total = sum(level_rows)
    level_id = collect(1:nlevels)
    op_complexity = fill(nnz_total ./ level_nnz[1],nlevels)
    grid_complexity = fill(rows_total ./ level_rows[1],nlevels)
    Dict([ :unknowns => level_rows,
          :unknowns_rel => level_rows ./ rows_total,
          :nonzeros => level_nnz,
          :nonzers_rel => level_nnz ./ nnz_total,
          :level => level_id,
          :operator_complexity => op_complexity,
          :grid_complexity => grid_complexity,
         ]
        )
end

@inline function v_cycle(args...)
    amg_cycle!(args...)
end

@inline function w_cycle(args...)
    amg_cycle!(args...)
    amg_cycle!(args...)
end

function amg_update!(setup,A,options)
    amg_params = setup.amg_params
    nlevels = setup.nlevels
    for level in 1:(nlevels-1)
        level_params = amg_params.fine_params[level]
        level_setup = setup.fine_levels[level]
        (;coarsening) = level_params
        _, coarsen! = coarsening
        (;R,P,A,Ac,Ac_setup,pre_setup,pos_setup) = level_setup
        update!(pre_setup,A)
        update!(pos_setup,A)
        coarsen!(A,Ac,R,P,Ac_setup)
        A = Ac
    end
    coarse_solver_setup = setup.coarse_level.coarse_solver_setup
    update!(coarse_solver_setup,A)
    setup
end

function amg_finalize!(setup)
    amg_params = setup.amg_params
    nlevels = setup.nlevels
    for level in 1:(nlevels-1)
        level_params = amg_params.fine_params[level]
        level_setup = setup.fine_levels[level]
        (;pre_setup,pos_setup) = level_setup
        finalize!(pre_setup)
        finalize!(pos_setup)
    end
    coarse_solver_setup = setup.coarse_level.coarse_solver_setup
    finalize!(coarse_solver_setup)
    nothing
end

