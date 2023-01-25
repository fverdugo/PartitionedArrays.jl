
function own_ghost_values end

function ghost_own_values end

function allocate_local_values(a,::Type{T},indices_rows,indices_cols) where T
    m = local_length(indices_rows)
    n = local_length(indices_cols)
    similar(a,T,m,n)
end

function allocate_local_values(::Type{V},indices_rows,indices_cols) where V
    m = local_length(indices_rows)
    n = local_length(indices_cols)
    similar(V,m,n)
end

function local_values(values,indices_rows,indices_cols)
    values
end

function own_values(values,indices_rows,indices_cols)
    subindices = (get_own_to_local(indices_rows),get_own_to_local(indices_cols))
    subindices_inv = (get_local_to_own(indices_rows),get_local_to_own(indices_cols))
    SubSparseMatrix(values,subindices,subindices_inv)
end

function ghost_values(values,indices_rows,indices_cols)
    subindices = (get_ghost_to_local(indices_rows),get_ghost_to_local(indices_cols))
    subindices_inv = (get_local_to_ghost(indices_rows),get_local_to_ghost(indices_cols))
    SubSparseMatrix(values,subindices,subindices_inv)
end

function own_ghost_values(values,indices_rows,indices_cols)
    subindices = (get_own_to_local(indices_rows),get_ghost_to_local(indices_cols))
    subindices_inv = (get_local_to_own(indices_rows),get_local_to_ghost(indices_cols))
    SubSparseMatrix(values,subindices,subindices_inv)
end

function ghost_own_values(values,indices_rows,indices_cols)
    subindices = (get_ghost_to_local(indices_rows),get_own_to_local(indices_cols))
    subindices_inv = (get_local_to_ghost(indices_rows),get_local_to_own(indices_cols))
    SubSparseMatrix(values,subindices,subindices_inv)
end

struct PSparseMatrix{V,A,B,C,D,T} <: AbstractMatrix{T}
    matrix_partition::A
    row_partition::B
    col_partition::C
    cache::D
    @doc """
    """
    function PSparseMatrix(
            matrix_partition,
            row_partition,
            col_partition,
            cache=p_sparse_matrix_cache(matrix_partition,row_partition,col_partition))
        V = eltype(matrix_partition)
        T = eltype(V)
        A = typeof(matrix_partition)
        B = typeof(row_partition)
        C = typeof(col_partition)
        D = typeof(cache)
        new{V,A,B,C,D,T}(matrix_partition,row_partition,col_partition,cache)
    end
end

partition(a::PSparseMatrix) = a.matrix_partition
Base.axes(a::PSparseMatrix) = (PRange(a.row_partition),PRange(a.col_partition))

function local_values(a::PSparseMatrix)
    partition(a)
end

function own_values(a::PSparseMatrix)
    map(own_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

function ghost_values(a::PSparseMatrix)
    map(ghost_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

function own_ghost_values(a::PSparseMatrix)
    map(own_ghost_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

function ghost_own_values(a::PSparseMatrix)
    map(ghost_own_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

Base.size(a::PSparseMatrix) = map(length,axes(a))
Base.IndexStyle(::Type{<:PSparseMatrix}) = IndexCartesian()
function Base.getindex(a::PSparseMatrix,gi::Int,gj::Int)
    scalar_indexing_action(a)
end
function Base.setindex!(a::PSparseMatrix,v,gi::Int,gj::Int)
    scalar_indexing_action(a)
end

function Base.show(io::IO,k::MIME"text/plain",data::PSparseMatrix)
    T = eltype(partition(data))
    m,n = size(data)
    np = length(partition(data))
    map_main(partition(data)) do values
        println(io,"$(m)×$(n) PSparseMatrix{$T} partitioned into $np parts")
    end
end

struct SparseMatrixAssemblyCache
    cache::VectorAssemblyCache
end
Base.reverse(a::SparseMatrixAssemblyCache) = SparseMatrixAssemblyCache(reverse(a.cache))
copy_cache(a::SparseMatrixAssemblyCache) = SparseMatrixAssemblyCache(copy_cache(a.cache))

function p_sparse_matrix_cache(matrix_partition,row_partition,col_partition)
    function setup_snd(part,parts_snd,row_indices,col_indices,values)
        local_to_owner = get_local_to_owner(row_indices)
        local_to_global_row = get_local_to_global(row_indices)
        local_to_global_col = get_local_to_global(col_indices)
        owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
        ptrs = zeros(Int32,length(parts_snd)+1)
        for (li,lj,v) in nziterator(values)
            owner = local_to_owner[li]
            if owner != part
                ptrs[owner_to_i[owner]+1] +=1
            end
        end
        length_to_ptrs!(ptrs)
        k_snd_data = zeros(Int32,ptrs[end]-1)
        gi_snd_data = zeros(Int,ptrs[end]-1)
        gj_snd_data = zeros(Int,ptrs[end]-1)
        for (k,(li,lj,v)) in enumerate(nziterator(values))
            owner = local_to_owner[li]
            if owner != part
                p = ptrs[owner_to_i[owner]]
                k_snd_data[p] = k
                gi_snd_data[p] = local_to_global_row[li]
                gj_snd_data[p] = local_to_global_col[lj]
                ptrs[owner_to_i[owner]] += 1
            end
        end
        rewind_ptrs!(ptrs)
        k_snd = JaggedArray(k_snd_data,ptrs)
        gi_snd = JaggedArray(gi_snd_data,ptrs)
        gj_snd = JaggedArray(gj_snd_data,ptrs)
        k_snd, gi_snd, gj_snd
    end
    function setup_rcv(part,row_indices,col_indices,gi_rcv,gj_rcv,values)
        global_to_local_row = get_global_to_local(row_indices)
        global_to_local_col = get_global_to_local(col_indices)
        ptrs = gi_rcv.ptrs
        k_rcv_data = zeros(Int32,ptrs[end]-1)
        for p in 1:length(gi_rcv.data)
            gi = gi_rcv.data[p]
            gj = gj_rcv.data[p]
            li = global_to_local_row[gi]
            lj = global_to_local_col[gj]
            k = nzindex(values,li,lj)
            @boundscheck @assert k > 0 "The sparsity pattern of the ghost layer is inconsistent"
            k_rcv_data[p] = k
        end
        k_rcv = JaggedArray(k_rcv_data,ptrs)
        k_rcv
    end
    part = linear_indices(row_partition)
    parts_snd, parts_rcv = assembly_neighbors(row_partition)
    k_snd, gi_snd, gj_snd = map(setup_snd,part,parts_snd,row_partition,col_partition,matrix_partition) |> tuple_of_arrays
    graph = ExchangeGraph(parts_snd,parts_rcv)
    gi_rcv = exchange_fetch(gi_snd,graph)
    gj_rcv = exchange_fetch(gj_snd,graph)
    k_rcv = map(setup_rcv,part,row_partition,col_partition,gi_rcv,gj_rcv,matrix_partition)
    buffers = map(assembly_buffers,matrix_partition,k_snd,k_rcv) |> tuple_of_arrays
    cache = map(VectorAssemblyCache,parts_snd,parts_rcv,k_snd,k_rcv,buffers...)
    map(SparseMatrixAssemblyCache,cache)
end

function assemble_impl!(f,matrix_partition,cache,::Type{<:SparseMatrixAssemblyCache})
    vcache = map(i->i.cache,cache)
    data = map(nonzeros,matrix_partition)
    assemble!(f,data,vcache)
end

function assemble!(a::PSparseMatrix)
    assemble!(+,a)
end

function assemble!(o,a::PSparseMatrix)
    t = assemble!(o,partition(a),a.cache)
    @async begin
        wait(t)
        map(ghost_values(a)) do a
            LinearAlgebra.fillstored!(a,zero(eltype(a)))
        end
        map(ghost_own_values(a)) do a
            LinearAlgebra.fillstored!(a,zero(eltype(a)))
        end
        a
    end
end

function assemble_coo!(I,J,V,row_partition)
    function setup_snd(part,parts_snd,row_lids,coo_values)
        global_to_local = get_global_to_local(row_lids)
        local_to_owner = get_local_to_owner(row_lids)
        owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
        ptrs = zeros(Int32,length(parts_snd)+1)
        k_gi, k_gj, k_v = coo_values
        for k in 1:length(k_gi)
            gi = k_gi[k]
            li = global_to_local[gi]
            owner = local_to_owner[li]
            if owner != part
                ptrs[owner_to_i[owner]+1] +=1
            end
        end
        length_to_ptrs!(ptrs)
        gi_snd_data = zeros(eltype(k_gi),ptrs[end]-1)
        gj_snd_data = zeros(eltype(k_gj),ptrs[end]-1)
        v_snd_data = zeros(eltype(k_v),ptrs[end]-1)
        for k in 1:length(k_gi)
            gi = k_gi[k]
            li = global_to_local[gi]
            owner = local_to_owner[li]
            if owner != part
                gj = k_gj[k]
                v = k_v[k]
                p = ptrs[owner_to_i[owner]]
                gi_snd_data[p] = gi
                gj_snd_data[p] = gj
                v_snd_data[p] = v
                k_v[k] = zero(v)
                ptrs[owner_to_i[owner]] += 1
            end
        end
        rewind_ptrs!(ptrs)
        gi_snd = JaggedArray(gi_snd_data,ptrs)
        gj_snd = JaggedArray(gj_snd_data,ptrs)
        v_snd = JaggedArray(v_snd_data,ptrs)
        gi_snd, gj_snd, v_snd
    end
    function setup_rcv(gi_rcv,gj_rcv,v_rcv,coo_values)
        k_gi, k_gj, k_v = coo_values
        current_n = length(k_gi)
        new_n = current_n + length(gi_rcv.data)
        resize!(k_gi,new_n)
        resize!(k_gj,new_n)
        resize!(k_v,new_n)
        for p in 1:length(gi_rcv.data)
            k_gi[current_n+p] = gi_rcv.data[p]
            k_gj[current_n+p] = gj_rcv.data[p]
            k_v[current_n+p] = v_rcv.data[p]
        end
    end
    part = linear_indices(row_partition)
    parts_snd, parts_rcv = assembly_neighbors(row_partition)
    coo_values = map(tuple,I,J,V)
    gi_snd, gj_snd, v_snd = map(setup_snd,part,parts_snd,row_partition,coo_values) |> tuple_of_arrays
    graph = ExchangeGraph(parts_snd,parts_rcv)
    t1 = exchange(gi_snd,graph)
    t2 = exchange(gj_snd,graph)
    t3 = exchange(v_snd,graph)
    @async begin
        gi_rcv = fetch(t1)
        gj_rcv = fetch(t2)
        v_rcv = fetch(t3)
        map(setup_rcv,gi_rcv,gj_rcv,v_rcv,coo_values)
        I,J,V
    end
end

function PSparseMatrix{V}(::UndefInitializer,row_partition,col_partition) where V
    matrix_partition = map(row_partition,col_partition) do row_indices, col_indices
        allocate_local_values(V,row_indices,col_indices)
    end
    PSparseMatrix(matrix_partition,row_partition,col_partition)
end

function Base.similar(a::PSparseMatrix,::Type{T},inds::Tuple{<:PRange,<:PRange}) where T
    rows,cols = inds
    matrix_partition = map(partition(a),partition(rows),partition(cols)) do values, row_indices, col_indices
        allocate_local_values(values,T,row_indices,col_indices)
    end
    PSparseMatrix(values,row_partition,col_partition)
end

function Base.similar(::Type{<:PSparseMatrix{V}},inds::Tuple{<:PRange,<:PRange}) where V
    rows,cols = inds
    matrix_partition = map(partition(a),partition(rows),partition(cols)) do values, row_indices, col_indices
        allocate_local_values(V,row_indices,col_indices)
    end
    PSparseMatrix(matrix_partition,row_partition,col_partition)
end

function LinearAlgebra.fillstored!(a::PSparseMatrix,v)
    map(partition(a)) do values
        LinearAlgebra.fillstored!(values,v)
    end
    a
end

function Base.:*(a::Number,b::PSparseMatrix)
    matrix_partition = map(partition(b)) do values
        a*values
    end
    cache = map(copy_cache,b.cache)
    PSparseMatrix(matrix_partition,partition(axes(b,1)),partition(axes(b,2)),cache)
end

function Base.:*(b::PSparseMatrix,a::Number)
    a*b
end

function Base.:*(a::PSparseMatrix,b::PVector)
    Ta = eltype(a)
    Tb = eltype(b)
    T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
    c = PVector{Vector{T}}(undef,partition(axes(a,1)))
    mul!(c,a,b)
    c
end

for op in (:+,:-)
    @eval begin
        function Base.$op(a::PSparseMatrix)
            matrix_partition = map(partition(a)) do a
                $op(a)
            end
            cache = map(copy_cache,a.cache)
            PSparseMatrix(matrix_partition,partition(axes(a,1)),partition(axes(a,2)),cache)
        end
    end
end

function LinearAlgebra.mul!(c::PVector,a::PSparseMatrix,b::PVector,α::Number,β::Number)
    @boundscheck @assert matching_own_indices(axes(c,1),axes(a,1))
    @boundscheck @assert matching_own_indices(axes(a,2),axes(b,1))
    @boundscheck @assert matching_ghost_indices(axes(a,2),axes(b,1))
    # Start the exchange
    t = consistent!(b)
    # Meanwhile, process the owned blocks
    map(own_values(c),own_values(a),own_values(b)) do co,aoo,bo
        if β != 1
            β != 0 ? rmul!(co, β) : fill!(co,zero(eltype(co)))
        end
        mul!(co,aoo,bo,α,1)
    end
    # Wait for the exchange to finish
    wait(t)
    # process the ghost block
    map(own_values(c),own_ghost_values(a),ghost_values(b)) do co,aoh,bh
        mul!(co,aoh,bh,α,1)
    end
    c
end

function psparse!(f,I,J,V,row_partition,col_partition;discover_rows=true)
    if discover_rows
        I_owner = find_owner(row_partition,I)
        row_partition = map(union_ghost,row_partition,I,I_owner)
    end
    t = assemble_coo!(I,J,V,row_partition)
    @async begin
        wait(t)
        J_owner = find_owner(col_partition,J)
        col_partition = map(union_ghost,col_partition,J,J_owner)
        map(to_local!,I,row_partition)
        map(to_local!,J,col_partition)
        matrix_partition = map(f,I,J,V,row_partition,col_partition)
        PSparseMatrix(matrix_partition,row_partition,col_partition)
    end
end

function psparse!(I,J,V,row_partition,col_partition;kwargs...)
    psparse!(default_local_values,I,J,V,row_partition,col_partition;kwargs...)
end

function psparse(f,row_partition,col_partition)
    matrix_partition = map(f,row_partition,col_partition)
    PSparseMatrix(matrix_partition,row_partition,col_partition)
end

function default_local_values(row_indices,col_indices)
    m = local_length(row_indices)
    n = local_length(col_indices)
    sparse(Int32[],Int32[],Float64[],m,n)
end

function default_local_values(I,J,V,row_indices,col_indices)
    m = local_length(row_indices)
    n = local_length(col_indices)
    sparse(I,J,V,m,n)
end

function trivial_partition(row_partition)
    destination = 1
    n_own = map(row_partition) do indices
        owner = get_owner(indices)
        owner == destination ? Int(global_length(indices)) : 0
    end
    partition_in_main = variable_partition(n_own,length(PRange(row_partition)))
    I = map(get_own_to_global,row_partition)
    I_owner = find_owner(partition_in_main,I)
    map(union_ghost,partition_in_main,I,I_owner)
end

function to_trivial_partition(b::PVector,row_partition_in_main)
    destination = 1
    T = eltype(b)
    b_in_main = similar(b,T,PRange(row_partition_in_main))
    map(own_values(b),partition(b_in_main),partition(axes(b,1))) do bown,b_in_main,indices
        part = get_owner(indices)
        if part == destination
            own_to_global = get_own_to_global(indices)
            b_in_main[own_to_global] .= bown
        else
            b_in_main .= bown
        end
    end
    assemble!(b_in_main)
    b_in_main
end

function from_trivial_partition!(c::PVector,c_in_main::PVector)
    destination = 1
    consistent!(c_in_main) |> wait
    map(own_values(c),partition(c_in_main),partition(axes(c,1))) do cown, c_in_main, indices
        part = get_owner(indices)
        if part == destination
            own_to_global = get_own_to_global(indices)
            cown .= view(c_in_main,own_to_global)
        else
            cown .= c_in_main
        end
    end
    c
end

function to_trivial_partition(
        a::PSparseMatrix{M},
        row_partition_in_main=trivial_partition(partition(axes(a,1))),
        col_partition_in_main=trivial_partition(partition(axes(a,2)))) where M
    destination = 1
    Ta = eltype(a)
    I,J,V = map(partition(a),partition(axes(a,1)),partition(axes(a,2))) do a,row_indices,col_indices
        n = 0
        local_to_owner = get_local_to_owner(row_indices)
        owner = get_owner(row_indices)
        local_to_global_row = get_local_to_global(row_indices)
        local_to_global_col = get_local_to_global(col_indices)
        for (i,j,v) in nziterator(a)
            if local_to_owner[i] == owner
                n += 1
            end
        end
        I = zeros(Int,n)
        J = zeros(Int,n)
        V = zeros(Ta,n)
        n = 0
        for (i,j,v) in nziterator(a)
            if local_to_owner[i] == owner
                n += 1
                I[n] = local_to_global_row[i]
                J[n] = local_to_global_col[j]
                V[n] = v
            end
        end
        I,J,V
    end |> tuple_of_arrays
    assemble_coo!(I,J,V,row_partition_in_main) |> wait
    I,J,V = map(partition(axes(a,1)),I,J,V) do row_indices,I,J,V
        owner = get_owner(row_indices)
        if owner == destination
            I,J,V
        else
            similar(I,eltype(I),0),similar(J,eltype(J),0),similar(V,eltype(V),0)
        end
    end |> tuple_of_arrays
    values = map(I,J,V,row_partition_in_main,col_partition_in_main) do I,J,V,row_indices,col_indices
        m = local_length(row_indices)
        n = local_length(col_indices)
        compresscoo(M,I,J,V,m,n)
    end
    PSparseMatrix(values,row_partition_in_main,col_partition_in_main)
end

# Not efficient, just for convenience and debugging purposes
function Base.:\(a::PSparseMatrix,b::PVector)
    Ta = eltype(a)
    Tb = eltype(b)
    T = typeof(one(Ta)\one(Tb)+one(Ta)\one(Tb))
    c = PVector{Vector{T}}(undef,partition(axes(a,2)))
    a_in_main = to_trivial_partition(a)
    b_in_main = to_trivial_partition(b,partition(axes(a_in_main,1)))
    c_in_main = to_trivial_partition(c,partition(axes(a_in_main,2)))
    map_main(partition(c_in_main),partition(a_in_main),partition(b_in_main)) do c, a, b
        c .= a\b
        nothing
    end
    from_trivial_partition!(c,c_in_main)
    c
end

# Not efficient, just for convenience and debugging purposes
struct PLU{A,B,C}
    lu_in_main::A
    rows::B
    cols::C
end
function LinearAlgebra.lu(a::PSparseMatrix)
    a_in_main = to_trivial_partition(a)
    lu_in_main = map_main(lu,partition(a_in_main))
    PLU(lu_in_main,axes(a_in_main,1),axes(a_in_main,2))
end
function LinearAlgebra.lu!(b::PLU,a::PSparseMatrix)
    a_in_main = to_trivial_partition(a,partition(b.rows),partition(b.cols))
    map_main(lu!,b.lu_in_main,partition(a_in_main))
    b
end
function LinearAlgebra.ldiv!(c::PVector,a::PLU,b::PVector)
    b_in_main = to_trivial_partition(b,partition(a.rows))
    c_in_main = to_trivial_partition(c,partition(a.cols))
    map_main(ldiv!,partition(c_in_main),a.lu_in_main,partition(b_in_main))
    from_trivial_partition!(c,c_in_main)
    c
end

# Misc functions that could be removed if IterativeSolvers was implemented in terms
# of axes(A,d) instead of size(A,d)
function IterativeSolvers.zerox(A::PSparseMatrix,b::PVector)
    T = IterativeSolvers.Adivtype(A, b)
    x = similar(b, T, axes(A, 2))
    fill!(x, zero(T))
    return x
end
