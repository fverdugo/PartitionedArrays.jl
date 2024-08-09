
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
    # TODO deprecate this one
    own_own_values(values,indices_rows,indices_cols)
end

function ghost_values(values,indices_rows,indices_cols)
    # TODO deprecate this one
    ghost_ghost_values(values,indices_rows,indices_cols)
end

function own_own_values(values,indices_rows,indices_cols)
    subindices = (own_to_local(indices_rows),own_to_local(indices_cols))
    subindices_inv = (local_to_own(indices_rows),local_to_own(indices_cols))
    SubSparseMatrix(values,subindices,subindices_inv)
end

function own_ghost_values(values,indices_rows,indices_cols)
    subindices = (own_to_local(indices_rows),ghost_to_local(indices_cols))
    subindices_inv = (local_to_own(indices_rows),local_to_ghost(indices_cols))
    SubSparseMatrix(values,subindices,subindices_inv)
end

function ghost_own_values(values,indices_rows,indices_cols)
    subindices = (ghost_to_local(indices_rows),own_to_local(indices_cols))
    subindices_inv = (local_to_ghost(indices_rows),local_to_own(indices_cols))
    SubSparseMatrix(values,subindices,subindices_inv)
end

function ghost_ghost_values(values,indices_rows,indices_cols)
    subindices = (ghost_to_local(indices_rows),ghost_to_local(indices_cols))
    subindices_inv = (local_to_ghost(indices_rows),local_to_ghost(indices_cols))
    SubSparseMatrix(values,subindices,subindices_inv)
end

#struct OldPSparseMatrix{V,A,B,C,D,T} <: AbstractMatrix{T}
#    matrix_partition::A
#    row_partition::B
#    col_partition::C
#    cache::D
#    function OldPSparseMatrix(
#            matrix_partition,
#            row_partition,
#            col_partition,
#            cache=p_sparse_matrix_cache(matrix_partition,row_partition,col_partition))
#        V = eltype(matrix_partition)
#        T = eltype(V)
#        A = typeof(matrix_partition)
#        B = typeof(row_partition)
#        C = typeof(col_partition)
#        D = typeof(cache)
#        new{V,A,B,C,D,T}(matrix_partition,row_partition,col_partition,cache)
#    end
#end
#
#partition(a::OldPSparseMatrix) = a.matrix_partition
#Base.axes(a::OldPSparseMatrix) = (PRange(a.row_partition),PRange(a.col_partition))
#
#function local_values(a::OldPSparseMatrix)
#    partition(a)
#end
#
#function own_values(a::OldPSparseMatrix)
#    map(own_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
#end
#
#function ghost_values(a::OldPSparseMatrix)
#    map(ghost_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
#end
#
#function own_ghost_values(a::OldPSparseMatrix)
#    map(own_ghost_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
#end
#
#function ghost_own_values(a::OldPSparseMatrix)
#    map(ghost_own_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
#end
#
#Base.size(a::OldPSparseMatrix) = map(length,axes(a))
#Base.IndexStyle(::Type{<:OldPSparseMatrix}) = IndexCartesian()
#function Base.getindex(a::OldPSparseMatrix,gi::Int,gj::Int)
#    scalar_indexing_action(a)
#end
#function Base.setindex!(a::OldPSparseMatrix,v,gi::Int,gj::Int)
#    scalar_indexing_action(a)
#end
#
#function Base.show(io::IO,k::MIME"text/plain",data::OldPSparseMatrix)
#    T = eltype(partition(data))
#    m,n = size(data)
#    np = length(partition(data))
#    map_main(partition(data)) do values
#        println(io,"$(m)×$(n) OldPSparseMatrix{$T} partitioned into $np parts")
#    end
#end
#
#struct SparseMatrixAssemblyCache
#    cache::VectorAssemblyCache
#end
#Base.reverse(a::SparseMatrixAssemblyCache) = SparseMatrixAssemblyCache(reverse(a.cache))
#copy_cache(a::SparseMatrixAssemblyCache) = SparseMatrixAssemblyCache(copy_cache(a.cache))
#
#function p_sparse_matrix_cache(matrix_partition,row_partition,col_partition)
#    p_sparse_matrix_cache_impl(eltype(matrix_partition),matrix_partition,row_partition,col_partition)
#end
#
#function p_sparse_matrix_cache_impl(::Type,matrix_partition,row_partition,col_partition)
#    function setup_snd(part,parts_snd,row_indices,col_indices,values)
#        local_row_to_owner = local_to_owner(row_indices)
#        local_to_global_row = local_to_global(row_indices)
#        local_to_global_col = local_to_global(col_indices)
#        owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
#        ptrs = zeros(Int32,length(parts_snd)+1)
#        for (li,lj,v) in nziterator(values)
#            owner = local_row_to_owner[li]
#            if owner != part
#                ptrs[owner_to_i[owner]+1] +=1
#            end
#        end
#        length_to_ptrs!(ptrs)
#        k_snd_data = zeros(Int32,ptrs[end]-1)
#        gi_snd_data = zeros(Int,ptrs[end]-1)
#        gj_snd_data = zeros(Int,ptrs[end]-1)
#        for (k,(li,lj,v)) in enumerate(nziterator(values))
#            owner = local_row_to_owner[li]
#            if owner != part
#                p = ptrs[owner_to_i[owner]]
#                k_snd_data[p] = k
#                gi_snd_data[p] = local_to_global_row[li]
#                gj_snd_data[p] = local_to_global_col[lj]
#                ptrs[owner_to_i[owner]] += 1
#            end
#        end
#        rewind_ptrs!(ptrs)
#        k_snd = JaggedArray(k_snd_data,ptrs)
#        gi_snd = JaggedArray(gi_snd_data,ptrs)
#        gj_snd = JaggedArray(gj_snd_data,ptrs)
#        k_snd, gi_snd, gj_snd
#    end
#    function setup_rcv(part,row_indices,col_indices,gi_rcv,gj_rcv,values)
#        global_to_local_row = global_to_local(row_indices)
#        global_to_local_col = global_to_local(col_indices)
#        ptrs = gi_rcv.ptrs
#        k_rcv_data = zeros(Int32,ptrs[end]-1)
#        for p in 1:length(gi_rcv.data)
#            gi = gi_rcv.data[p]
#            gj = gj_rcv.data[p]
#            li = global_to_local_row[gi]
#            lj = global_to_local_col[gj]
#            k = nzindex(values,li,lj)
#            @boundscheck @assert k > 0 "The sparsity pattern of the ghost layer is inconsistent"
#            k_rcv_data[p] = k
#        end
#        k_rcv = JaggedArray(k_rcv_data,ptrs)
#        k_rcv
#    end
#    part = linear_indices(row_partition)
#    parts_snd, parts_rcv = assembly_neighbors(row_partition)
#    k_snd, gi_snd, gj_snd = map(setup_snd,part,parts_snd,row_partition,col_partition,matrix_partition) |> tuple_of_arrays
#    graph = ExchangeGraph(parts_snd,parts_rcv)
#    gi_rcv = exchange_fetch(gi_snd,graph)
#    gj_rcv = exchange_fetch(gj_snd,graph)
#    k_rcv = map(setup_rcv,part,row_partition,col_partition,gi_rcv,gj_rcv,matrix_partition)
#    buffers = map(assembly_buffers,matrix_partition,k_snd,k_rcv) |> tuple_of_arrays
#    cache = map(VectorAssemblyCache,parts_snd,parts_rcv,k_snd,k_rcv,buffers...)
#    map(SparseMatrixAssemblyCache,cache)
#end
#
#function assemble_impl!(f,matrix_partition,cache,::Type{<:SparseMatrixAssemblyCache})
#    vcache = map(i->i.cache,cache)
#    data = map(nonzeros,matrix_partition)
#    assemble!(f,data,vcache)
#end
#
#function assemble!(a::OldPSparseMatrix)
#    assemble!(+,a)
#end
#
#function assemble!(o,a::OldPSparseMatrix)
#    t = assemble!(o,partition(a),a.cache)
#    @fake_async begin
#        wait(t)
#        map(ghost_values(a)) do a
#            LinearAlgebra.fillstored!(a,zero(eltype(a)))
#        end
#        map(ghost_own_values(a)) do a
#            LinearAlgebra.fillstored!(a,zero(eltype(a)))
#        end
#        a
#    end
#end

function assemble_coo!(I,J,V,row_partition)
    """
      Returns three JaggedArrays with the coo triplets
      to be sent to the corresponding owner parts in parts_snd
    """
    function setup_snd(part,parts_snd,row_lids,coo_values)
        global_to_local_row = global_to_local(row_lids)
        local_row_to_owner = local_to_owner(row_lids)
        owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
        ptrs = zeros(Int32,length(parts_snd)+1)
        k_gi, k_gj, k_v = coo_values
        for k in 1:length(k_gi)
            gi = k_gi[k]
            li = global_to_local_row[gi]
            owner = local_row_to_owner[li]
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
            li = global_to_local_row[gi]
            owner = local_row_to_owner[li]
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
    """
      Pushes to coo_values the triplets gi_rcv,gj_rcv,v_rcv
      received from remote processes
    """
    function setup_rcv!(coo_values,gi_rcv,gj_rcv,v_rcv)
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
    @fake_async begin
        gi_rcv = fetch(t1)
        gj_rcv = fetch(t2)
        v_rcv = fetch(t3)
        map(setup_rcv!,coo_values,gi_rcv,gj_rcv,v_rcv)
        I,J,V
    end
end

#function OldPSparseMatrix{V}(::UndefInitializer,row_partition,col_partition) where V
#    matrix_partition = map(row_partition,col_partition) do row_indices, col_indices
#        allocate_local_values(V,row_indices,col_indices)
#    end
#    OldPSparseMatrix(matrix_partition,row_partition,col_partition)
#end
#
#function Base.similar(a::OldPSparseMatrix,::Type{T},inds::Tuple{<:PRange,<:PRange}) where T
#    rows,cols = inds
#    matrix_partition = map(partition(a),partition(rows),partition(cols)) do values, row_indices, col_indices
#        allocate_local_values(values,T,row_indices,col_indices)
#    end
#    OldPSparseMatrix(matrix_partition,partition(rows),partition(cols))
#end
#
#function Base.similar(::Type{<:OldPSparseMatrix{V}},inds::Tuple{<:PRange,<:PRange}) where V
#    rows,cols = inds
#    matrix_partition = map(partition(rows),partition(cols)) do row_indices, col_indices
#        allocate_local_values(V,row_indices,col_indices)
#    end
#    OldPSparseMatrix(matrix_partition,partition(rows),partition(cols))
#end
#
#function Base.copy(a::OldPSparseMatrix)
#    mats = map(copy,partition(a))
#    cache = map(copy_cache,a.cache)
#    OldPSparseMatrix(mats,partition(axes(a,1)),partition(axes(a,2)),cache)
#end
#
#function Base.copy!(a::OldPSparseMatrix,b::OldPSparseMatrix)
#    @assert size(a) == size(b)
#    copyto!(a,b)
#end
#
#function Base.copyto!(a::OldPSparseMatrix,b::OldPSparseMatrix)
#    if partition(axes(a,1)) === partition(axes(b,1)) && partition(axes(a,2)) === partition(axes(b,2))
#        map(copy!,partition(a),partition(b))
#    elseif matching_own_indices(axes(a,1),axes(b,1)) && matching_own_indices(axes(a,2),axes(b,2))
#        map(copy!,own_values(a),own_values(b))
#    else
#        error("Trying to copy a OldPSparseMatrix into another one with a different data layout. This case is not implemented yet. It would require communications.")
#    end
#    a
#end
#
#function LinearAlgebra.fillstored!(a::OldPSparseMatrix,v)
#    map(partition(a)) do values
#        LinearAlgebra.fillstored!(values,v)
#    end
#    a
#end
#
#function Base.:*(a::Number,b::OldPSparseMatrix)
#    matrix_partition = map(partition(b)) do values
#        a*values
#    end
#    cache = map(copy_cache,b.cache)
#    OldPSparseMatrix(matrix_partition,partition(axes(b,1)),partition(axes(b,2)),cache)
#end
#
#function Base.:*(b::OldPSparseMatrix,a::Number)
#    a*b
#end
#
#function Base.:*(a::OldPSparseMatrix,b::PVector)
#    Ta = eltype(a)
#    Tb = eltype(b)
#    T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
#    c = PVector{Vector{T}}(undef,partition(axes(a,1)))
#    mul!(c,a,b)
#    c
#end
#
#for op in (:+,:-)
#    @eval begin
#        function Base.$op(a::OldPSparseMatrix)
#            matrix_partition = map(partition(a)) do a
#                $op(a)
#            end
#            cache = map(copy_cache,a.cache)
#            OldPSparseMatrix(matrix_partition,partition(axes(a,1)),partition(axes(a,2)),cache)
#        end
#    end
#end
#
#function LinearAlgebra.mul!(c::PVector,a::OldPSparseMatrix,b::PVector,α::Number,β::Number)
#    @boundscheck @assert matching_own_indices(axes(c,1),axes(a,1))
#    @boundscheck @assert matching_own_indices(axes(a,2),axes(b,1))
#    @boundscheck @assert matching_ghost_indices(axes(a,2),axes(b,1))
#    # Start the exchange
#    t = consistent!(b)
#    # Meanwhile, process the owned blocks
#    map(own_values(c),own_values(a),own_values(b)) do co,aoo,bo
#        if β != 1
#            β != 0 ? rmul!(co, β) : fill!(co,zero(eltype(co)))
#        end
#        mul!(co,aoo,bo,α,1)
#    end
#    # Wait for the exchange to finish
#    wait(t)
#    # process the ghost block
#    map(own_values(c),own_ghost_values(a),ghost_values(b)) do co,aoh,bh
#        mul!(co,aoh,bh,α,1)
#    end
#    c
#end
#
#function old_psparse(f,row_partition,col_partition;assembled=false)
#    matrix_partition = map(f,row_partition,col_partition)
#    OldPSparseMatrix(matrix_partition,row_partition,col_partition,assembled)
#end
#
#function old_psparse!(f,I,J,V,row_partition,col_partition;discover_rows=true,discover_cols=true)
#    if discover_rows
#        I_owner = find_owner(row_partition,I)
#        row_partition = map(union_ghost,row_partition,I,I_owner)
#    end
#    t = assemble_coo!(I,J,V,row_partition)
#    @fake_async begin
#        wait(t)
#        if discover_cols
#            J_owner = find_owner(col_partition,J)
#            col_partition = map(union_ghost,col_partition,J,J_owner)
#        end
#        map(to_local!,I,row_partition)
#        map(to_local!,J,col_partition)
#        matrix_partition = map(f,I,J,V,row_partition,col_partition)
#        OldPSparseMatrix(matrix_partition,row_partition,col_partition)
#    end
#end
#
#function old_psparse!(I,J,V,row_partition,col_partition;kwargs...)
#    old_psparse!(default_local_values,I,J,V,row_partition,col_partition;kwargs...)
#end

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

#function old_trivial_partition(row_partition)
#    destination = 1
#    n_own = map(row_partition) do indices
#        owner = part_id(indices)
#        owner == destination ? Int(global_length(indices)) : 0
#    end
#    partition_in_main = variable_partition(n_own,length(PRange(row_partition)))
#    I = map(own_to_global,row_partition)
#    I_owner = find_owner(partition_in_main,I)
#    map(union_ghost,partition_in_main,I,I_owner)
#end
#
#function to_trivial_partition(b::PVector,row_partition_in_main)
#    destination = 1
#    T = eltype(b)
#    b_in_main = similar(b,T,PRange(row_partition_in_main))
#    fill!(b_in_main,zero(T))
#    map(own_values(b),partition(b_in_main),partition(axes(b,1))) do bown,my_b_in_main,indices
#        part = part_id(indices)
#        if part == destination
#            my_b_in_main[own_to_global(indices)] .= bown
#        else
#            my_b_in_main .= bown
#        end
#    end
#    assemble!(b_in_main) |> wait
#    b_in_main
#end
#
#function from_trivial_partition!(c::PVector,c_in_main::PVector)
#    destination = 1
#    consistent!(c_in_main) |> wait
#    map(own_values(c),partition(c_in_main),partition(axes(c,1))) do cown, my_c_in_main, indices
#        part = part_id(indices)
#        if part == destination
#            cown .= view(my_c_in_main,own_to_global(indices))
#        else
#            cown .= my_c_in_main
#        end
#    end
#    c
#end
#
#function to_trivial_partition(
#        a::OldPSparseMatrix{M},
#        row_partition_in_main=old_trivial_partition(partition(axes(a,1))),
#        col_partition_in_main=old_trivial_partition(partition(axes(a,2)))) where M
#    destination = 1
#    Ta = eltype(a)
#    I,J,V = map(partition(a),partition(axes(a,1)),partition(axes(a,2))) do a,row_indices,col_indices
#        n = 0
#        local_row_to_owner = local_to_owner(row_indices)
#        owner = part_id(row_indices)
#        local_to_global_row = local_to_global(row_indices)
#        local_to_global_col = local_to_global(col_indices)
#        for (i,j,v) in nziterator(a)
#            if local_row_to_owner[i] == owner
#                n += 1
#            end
#        end
#        myI = zeros(Int,n)
#        myJ = zeros(Int,n)
#        myV = zeros(Ta,n)
#        n = 0
#        for (i,j,v) in nziterator(a)
#            if local_row_to_owner[i] == owner
#                n += 1
#                myI[n] = local_to_global_row[i]
#                myJ[n] = local_to_global_col[j]
#                myV[n] = v
#            end
#        end
#        myI,myJ,myV
#    end |> tuple_of_arrays
#    assemble_coo!(I,J,V,row_partition_in_main) |> wait
#    I,J,V = map(partition(axes(a,1)),I,J,V) do row_indices,myI,myJ,myV
#        owner = part_id(row_indices)
#        if owner == destination
#            myI,myJ,myV
#        else
#            similar(myI,eltype(myI),0),similar(myJ,eltype(myJ),0),similar(myV,eltype(myV),0)
#        end
#    end |> tuple_of_arrays
#    values = map(I,J,V,row_partition_in_main,col_partition_in_main) do myI,myJ,myV,row_indices,col_indices
#        m = local_length(row_indices)
#        n = local_length(col_indices)
#        compresscoo(M,myI,myJ,myV,m,n)
#    end
#    OldPSparseMatrix(values,row_partition_in_main,col_partition_in_main)
#end
#
## Not efficient, just for convenience and debugging purposes
#function Base.:\(a::OldPSparseMatrix,b::PVector)
#    Ta = eltype(a)
#    Tb = eltype(b)
#    T = typeof(one(Ta)\one(Tb)+one(Ta)\one(Tb))
#    c = PVector{Vector{T}}(undef,partition(axes(a,2)))
#    fill!(c,zero(T))
#    a_in_main = to_trivial_partition(a)
#    b_in_main = to_trivial_partition(b,partition(axes(a_in_main,1)))
#    c_in_main = to_trivial_partition(c,partition(axes(a_in_main,2)))
#    map_main(partition(c_in_main),partition(a_in_main),partition(b_in_main)) do myc, mya, myb
#        myc .= mya\myb
#        nothing
#    end
#    from_trivial_partition!(c,c_in_main)
#    c
#end
#
## Not efficient, just for convenience and debugging purposes
#struct PLU{A,B,C}
#    lu_in_main::A
#    rows::B
#    cols::C
#end
#function LinearAlgebra.lu(a::OldPSparseMatrix)
#    a_in_main = to_trivial_partition(a)
#    lu_in_main = map_main(lu,partition(a_in_main))
#    PLU(lu_in_main,axes(a_in_main,1),axes(a_in_main,2))
#end
#function LinearAlgebra.lu!(b::PLU,a::OldPSparseMatrix)
#    a_in_main = to_trivial_partition(a,partition(b.rows),partition(b.cols))
#    map_main(lu!,b.lu_in_main,partition(a_in_main))
#    b
#end
#function LinearAlgebra.ldiv!(c::PVector,a::PLU,b::PVector)
#    b_in_main = to_trivial_partition(b,partition(a.rows))
#    c_in_main = to_trivial_partition(c,partition(a.cols))
#    map_main(ldiv!,partition(c_in_main),a.lu_in_main,partition(b_in_main))
#    from_trivial_partition!(c,c_in_main)
#    c
#end
#
## Misc functions that could be removed if IterativeSolvers was implemented in terms
## of axes(A,d) instead of size(A,d)
#function IterativeSolvers.zerox(A::OldPSparseMatrix,b::PVector)
#    T = IterativeSolvers.Adivtype(A, b)
#    x = similar(b, T, axes(A, 2))
#    fill!(x, zero(T))
#    return x
#end

# New stuff

struct GenericSplitMatrixBlocks{A,B,C,D}
    own_own::A
    own_ghost::B
    ghost_own::C
    ghost_ghost::D
end
struct SplitMatrixBlocks{A}
    own_own::A
    own_ghost::A
    ghost_own::A
    ghost_ghost::A
end
function split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    GenericSplitMatrixBlocks(own_own,own_ghost,ghost_own,ghost_ghost)
end
function split_matrix_blocks(own_own::A,own_ghost::A,ghost_own::A,ghost_ghost::A) where A
    SplitMatrixBlocks(own_own,own_ghost,ghost_own,ghost_ghost)
end

abstract type AbstractSplitMatrix{T} <: AbstractMatrix{T} end

struct GenericSplitMatrix{A,B,C,T} <: AbstractSplitMatrix{T}
    blocks::A
    row_permutation::B
    col_permutation::C
    function GenericSplitMatrix(blocks,row_permutation,col_permutation)
        T = eltype(blocks.own_own)
        A = typeof(blocks)
        B = typeof(row_permutation)
        C = typeof(col_permutation)
        new{A,B,C,T}(blocks,row_permutation,col_permutation)
    end
end

struct SplitMatrix{A,T} <: AbstractSplitMatrix{T}
    blocks::SplitMatrixBlocks{A}
    row_permutation::UnitRange{Int32}
    col_permutation::UnitRange{Int32}
    function SplitMatrix(
        blocks::SplitMatrixBlocks{A},row_permutation,col_permutation) where A
        T = eltype(blocks.own_own)
        row_perm = convert(UnitRange{Int32},row_permutation)
        col_perm = convert(UnitRange{Int32},col_permutation)
        new{A,T}(blocks,row_perm,col_perm)
    end
end

function split_matrix(blocks,row_permutation,col_permutation)
    GenericSplitMatrix(blocks,row_permutation,col_permutation)
end

function split_matrix(
    blocks::SplitMatrixBlocks,
    row_permutation::UnitRange,
    col_permutation::UnitRange)
    SplitMatrix(blocks,row_permutation,col_permutation)
end

function split_matrix(
    own_own::AbstractMatrix,
    own_ghost::AbstractMatrix,
    ghost_own::AbstractMatrix,
    ghost_ghost::AbstractMatrix,
    row_permutation,
    col_permutation)
    blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    split_matrix(blocks,row_permutation,col_permutation)
end

Base.size(a::AbstractSplitMatrix) = (length(a.row_permutation),length(a.col_permutation))
Base.IndexStyle(::Type{<:AbstractSplitMatrix}) = IndexCartesian()
function Base.getindex(a::AbstractSplitMatrix,i::Int,j::Int)
    n_own_rows, n_own_cols = size(a.blocks.own_own)
    ip = a.row_permutation[i]
    jp = a.col_permutation[j]
    T = eltype(a)
    if ip <= n_own_rows && jp <= n_own_cols
        v = a.blocks.own_own[ip,jp]
    elseif ip <= n_own_rows
        v = a.blocks.own_ghost[ip,jp-n_own_cols]
    elseif jp <= n_own_cols
        v = a.blocks.ghost_own[ip-n_own_rows,jp]
    else
        v = a.blocks.ghost_ghost[ip-n_own_rows,jp-n_own_cols]
    end
    convert(T,v)
end

function own_own_values(values::AbstractSplitMatrix,indices_rows,indices_cols)
    values.blocks.own_own
end
function own_ghost_values(values::AbstractSplitMatrix,indices_rows,indices_cols)
    values.blocks.own_ghost
end
function ghost_own_values(values::AbstractSplitMatrix,indices_rows,indices_cols)
    values.blocks.ghost_own
end
function ghost_ghost_values(values::AbstractSplitMatrix,indices_rows,indices_cols)
    values.blocks.ghost_ghost
end

Base.similar(a::AbstractSplitMatrix) = similar(a,eltype(a))
function Base.similar(a::AbstractSplitMatrix,::Type{T}) where T
    own_own = similar(a.blocks.own_own,T)
    own_ghost = similar(a.blocks.own_ghost,T)
    ghost_own = similar(a.blocks.ghost_own,T)
    ghost_ghost = similar(a.blocks.ghost_ghost,T)
    blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    split_matrix(blocks,a.row_permutation,a.col_permutation)
end

function Base.copy(a::AbstractSplitMatrix)
    own_own = copy(a.blocks.own_own)
    own_ghost = copy(a.blocks.own_ghost)
    ghost_own = copy(a.blocks.ghost_own)
    ghost_ghost = copy(a.blocks.ghost_ghost)
    blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    split_matrix(blocks,a.row_permutation,a.col_permutation)
end

function Base.copy!(a::AbstractSplitMatrix,b::AbstractSplitMatrix)
    copy!(a.blocks.own_own,b.blocks.own_own)
    copy!(a.blocks.own_ghost,b.blocks.own_ghost)
    copy!(a.blocks.ghost_own,b.blocks.ghost_own)
    copy!(a.blocks.ghost_ghost,b.blocks.ghost_ghost)
    a
end
function Base.copyto!(a::AbstractSplitMatrix,b::AbstractSplitMatrix)
    copyto!(a.blocks.own_own,b.blokcs.own_own)
    copyto!(a.blocks.own_ghost,b.blokcs.own_ghost)
    copyto!(a.blocks.ghost_own,b.blokcs.ghost_own)
    copyto!(a.blocks.ghost_ghost,b.blokcs.ghost_ghost)
    a
end

function LinearAlgebra.fillstored!(a::AbstractSplitMatrix,v)
    LinearAlgebra.fillstored!(a.blocks.own_own,v)
    LinearAlgebra.fillstored!(a.blocks.own_ghost,v)
    LinearAlgebra.fillstored!(a.blocks.ghost_own,v)
    LinearAlgebra.fillstored!(a.blocks.ghost_ghost,v)
    a
end

function Base.:*(a::Number,b::AbstractSplitMatrix)
    own_own = a*b.blocks.own_own
    own_ghost = a*b.blocks.own_ghost
    ghost_own = a*b.blocks.ghost_own
    ghost_ghost = a*b.blocks.ghost_ghost
    blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    split_matrix(blocks,b.row_permutation,b.col_permutation)
end

function Base.:*(b::AbstractSplitMatrix,a::Number)
    a*b
end

function Base.:*(A::AbstractSplitMatrix,B::AbstractSplitMatrix)
    own_own = A.blocks.own_own*B.blocks.own_own + A.blocks.own_ghost*B.blocks.ghost_own
    own_ghost = A.blocks.own_own*B.blocks.own_ghost + A.blocks.own_ghost*B.blocks.ghost_ghost
    ghost_own = A.blocks.ghost_own*B.blocks.own_own + A.blocks.ghost_ghost*B.blocks.ghost_own
    ghost_ghost = A.blocks.ghost_own*B.blocks.own_ghost + A.blocks.ghost_ghost*B.blocks.ghost_ghost
    blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    split_matrix(blocks,A.row_permutation,B.col_permutation)
end

function LinearAlgebra.mul!(C::AbstractSplitMatrix,A::AbstractSplitMatrix,B::AbstractSplitMatrix)
    # TODO not sure if there is available an efficient implementation for mul! for SparseMatrixCSC
    mul!(C.blocks.own_own,A.blocks.own_own,B.blocks.own_own,1,0)
    mul!(C.blocks.own_own,A.blocks.own_ghost,B.blocks.ghost_own,1,1)
    mul!(C.blocks.own_ghost,A.blocks.own_own,B.blocks.own_ghost,1,0)
    mul!(C.blocks.own_ghost,A.blocks.own_ghost,B.blocks.ghost_ghost,1,1)
    mul!(C.blocks.ghost_own,A.blocks.ghost_own,B.blocks.own_own,1,0)
    mul!(C.blocks.ghost_own,A.blocks.ghost_ghost,B.blocks.ghost_own,1,1)
    mul!(C.blocks.ghost_ghost,A.blocks.ghost_own,B.blocks.own_ghost,1,0)
    mul!(C.blocks.ghost_ghost,A.blocks.ghost_ghost,B.blocks.ghost_ghost,1,1)
    C
end

function Base.:*(At::Transpose{T,<:AbstractSplitMatrix} where T,B::AbstractSplitMatrix)
    A = At.parent
    A_own_own = transpose(A.blocks.own_own)
    A_own_ghost = transpose(A.blocks.ghost_own)
    A_ghost_own = transpose(A.blocks.own_ghost)
    A_ghost_ghost = transpose(A.blocks.ghost_ghost)
    own_own = A_own_own*B.blocks.own_own + A_own_ghost*B.blocks.ghost_own
    own_ghost = A_own_own*B.blocks.own_ghost + A_own_ghost*B.blocks.ghost_ghost
    ghost_own = A_ghost_own*B.blocks.own_own + A_ghost_ghost*B.blocks.ghost_own
    ghost_ghost = A_ghost_own*B.blocks.own_ghost + A_ghost_ghost*B.blocks.ghost_ghost
    blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    split_matrix(blocks,A.col_permutation,B.col_permutation)
end

function LinearAlgebra.mul!(C::AbstractSplitMatrix,At::Transpose{T,<:AbstractSplitMatrix} where T,B::AbstractSplitMatrix)
    # TODO not sure if there is available an efficient implementation for mul! for SparseMatrixCSC
    A = At.parent
    A_own_own = transpose(A.blocks.own_own)
    A_own_ghost = transpose(A.blocks.ghost_own)
    A_ghost_own = transpose(A.blocks.own_ghost)
    A_ghost_ghost = transpose(A.blocks.ghost_ghost)
    mul!(C.blocks.own_own,A_own_own,B.blocks.own_own,1,0)
    mul!(C.blocks.own_own,A_own_ghost,B.blocks.ghost_own,1,1)
    mul!(C.blocks.own_ghost,A_own_own,B.blocks.own_ghost,1,0)
    mul!(C.blocks.own_ghost,A_own_ghost,B.blocks.ghost_ghost,1,1)
    mul!(C.blocks.ghost_own,A_ghost_own,B.blocks.own_own,1,0)
    mul!(C.blocks.ghost_own,A_ghost_ghost,B.blocks.ghost_own,1,1)
    mul!(C.blocks.ghost_ghost,A_ghost_own,B.blocks.own_ghost,1,0)
    mul!(C.blocks.ghost_ghost,A_ghost_ghost,B.blocks.ghost_ghost,1,1)
end

for op in (:+,:-)
    @eval begin
        function Base.$op(a::AbstractSplitMatrix)
            own_own = $op(a.blocks.own_own)
            own_ghost = $op(a.blocks.own_ghost)
            ghost_own = $op(a.blocks.ghost_own)
            ghost_ghost = $op(a.blocks.ghost_ghost)
            blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
            split_matrix(blocks,a.row_permutation,a.col_permutation)
        end
        function Base.$op(a::AbstractSplitMatrix,b::AbstractSplitMatrix)
            @boundscheck @assert a.row_permutation == b.row_permutation
            @boundscheck @assert a.col_permutation == b.col_permutation
            own_own = $op(a.blocks.own_own,b.blocks.own_own)
            own_ghost = $op(a.blocks.own_ghost,b.blocks.own_ghost)
            ghost_own = $op(a.blocks.ghost_own,b.blocks.ghost_own)
            ghost_ghost = $op(a.blocks.ghost_ghost,b.blocks.ghost_ghost)
            blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
            split_matrix(blocks,b.row_permutation,b.col_permutation)
        end
    end
end

function SparseArrays.nnz(a::SplitMatrix)
    n = 0
    n += nnz(a.blocks.own_own)
    n += nnz(a.blocks.own_ghost)
    n += nnz(a.blocks.ghost_own)
    n += nnz(a.blocks.ghost_ghost)
    n
end

function split_format_locally(A,rows,cols)
    n_own_rows = own_length(rows)
    n_own_cols = own_length(cols)
    n_ghost_rows = ghost_length(rows)
    n_ghost_cols = ghost_length(cols)
    rows_perm = local_permutation(rows)
    cols_perm = local_permutation(cols)
    n_own_own = 0
    n_own_ghost = 0
    n_ghost_own = 0
    n_ghost_ghost = 0
    for (i,j,v) in nziterator(A)
        ip = rows_perm[i]
        jp = cols_perm[j]
        if ip <= n_own_rows && jp <= n_own_cols
            n_own_own += 1
        elseif ip <= n_own_rows
            n_own_ghost += 1
        elseif jp <= n_own_cols
            n_ghost_own += 1
        else
            n_ghost_ghost += 1
        end
    end
    Ti = indextype(A)
    Tv = eltype(A)
    own_own = (I=zeros(Ti,n_own_own),J=zeros(Ti,n_own_own),V=zeros(Tv,n_own_own))
    own_ghost = (I=zeros(Ti,n_own_ghost),J=zeros(Ti,n_own_ghost),V=zeros(Tv,n_own_ghost))
    ghost_own = (I=zeros(Ti,n_ghost_own),J=zeros(Ti,n_ghost_own),V=zeros(Tv,n_ghost_own))
    ghost_ghost = (I=zeros(Ti,n_ghost_ghost),J=zeros(Ti,n_ghost_ghost),V=zeros(Tv,n_ghost_ghost))
    n_own_own = 0
    n_own_ghost = 0
    n_ghost_own = 0
    n_ghost_ghost = 0
    for (i,j,v) in nziterator(A)
        ip = rows_perm[i]
        jp = cols_perm[j]
        if ip <= n_own_rows && jp <= n_own_cols
            n_own_own += 1
            own_own.I[n_own_own] = ip
            own_own.J[n_own_own] = jp
            own_own.V[n_own_own] = v
        elseif ip <= n_own_rows
            n_own_ghost += 1
            own_ghost.I[n_own_ghost] = ip
            own_ghost.J[n_own_ghost] = jp-n_own_cols
            own_ghost.V[n_own_ghost] = v
        elseif jp <= n_own_cols
            n_ghost_own += 1
            ghost_own.I[n_ghost_own] = ip-n_own_cols
            ghost_own.J[n_ghost_own] = jp
            ghost_own.V[n_ghost_own] = v
        else
            n_ghost_ghost += 1
            ghost_ghost.I[n_ghost_ghost] = i-n_own_rows
            ghost_ghost.J[n_ghost_ghost] = j-n_own_cols
            ghost_ghost.V[n_ghost_ghost] = v
        end
    end
    TA = typeof(A) 
    A1 = compresscoo(TA,own_own...,n_own_rows  ,n_own_cols)
    A2 = compresscoo(TA,own_ghost...,n_own_rows  ,n_ghost_cols)
    A3 = compresscoo(TA,ghost_own...,n_ghost_rows,n_own_cols)
    A4 = compresscoo(TA,ghost_ghost...,n_ghost_rows,n_ghost_cols)
    blocks = split_matrix_blocks(A1,A2,A3,A4)
    B = split_matrix(blocks,rows_perm,cols_perm)
    c1 = precompute_nzindex(A1,own_own.I,own_own.J)
    c2 = precompute_nzindex(A2,own_ghost.I,own_ghost.J)
    c3 = precompute_nzindex(A3,ghost_own.I,ghost_own.J)
    c4 = precompute_nzindex(A4,ghost_ghost.I,ghost_ghost.J)
    own_own_V = own_own.V
    own_ghost_V = own_ghost.V
    ghost_own_V = ghost_own.V
    ghost_ghost_V = ghost_ghost.V
    cache = (c1,c2,c3,c4,own_own_V,own_ghost_V,ghost_own_V,ghost_ghost_V)
    B, cache
end

function split_format_locally!(B::AbstractSplitMatrix,A,rows,cols,cache)
    (c1,c2,c3,c4,own_own_V,own_ghost_V,ghost_own_V,ghost_ghost_V) = cache
    n_own_rows = own_length(rows)
    n_own_cols = own_length(cols)
    n_ghost_rows = ghost_length(rows)
    n_ghost_cols = ghost_length(cols)
    rows_perm = local_permutation(rows)
    cols_perm = local_permutation(cols)
    n_own_own = 0
    n_own_ghost = 0
    n_ghost_own = 0
    n_ghost_ghost = 0
    for (i,j,v) in nziterator(A)
        ip = rows_perm[i]
        jp = cols_perm[j]
        if ip <= n_own_rows && jp <= n_own_cols
            n_own_own += 1
            own_own_V[n_own_own] = v
        elseif ip <= n_own_rows
            n_own_ghost += 1
            own_ghost_V[n_own_ghost] = v
        elseif jp <= n_own_cols
            n_ghost_own += 1
            ghost_own_V[n_ghost_own] = v
        else
            n_ghost_ghost += 1
            ghost_ghost_V[n_ghost_ghost] = v
        end
    end
    setcoofast!(B.blocks.own_own,own_own_V,c1)
    setcoofast!(B.blocks.own_ghost,own_ghost_V,c2)
    setcoofast!(B.blocks.ghost_own,ghost_own_V,c3)
    setcoofast!(B.blocks.ghost_ghost,ghost_ghost_V,c4)
    B
end

"""
    struct PSparseMatrix{V,B,C,D,T}

`PSparseMatrix` (partitioned sparse matrix)
is a type representing a matrix whose rows are
distributed (a.k.a. partitioned) over different parts for distributed-memory
parallel computations. Each part stores a subset of the rows of the matrix and their
corresponding non zero columns.

This type overloads numerous array-like operations with corresponding
parallel implementations.

# Properties

- `matrix_partition::A`
- `row_partition::B`
- `col_partition::C`
- `assembled::Bool`

`matrix_partition[i]` contains a (sparse) matrix with the local rows and the
corresponding nonzero columns (the local columns) in the part number `i`.
`eltype(matrix_partition) == V`.
`row_partition[i]` and `col_partition[i]` contain information
about the local, own, and ghost rows and columns respectively in part number `i`.
The types `eltype(row_partition)` and `eltype(col_partition)` implement the
[`AbstractLocalIndices`](@ref) interface. For `assembled==true`, it is assumed that the matrix data
is fully contained in the own rows. 

# Supertype hierarchy

    PSparseMatrix{V,A,B,C,T} <: AbstractMatrix{T}

with `T=eltype(V)`.
"""
struct PSparseMatrix{V,B,C,D,T} <: AbstractMatrix{T}
    matrix_partition::B
    row_partition::C
    col_partition::D
    assembled::Bool
    @doc """
        PSparseMatrix(matrix_partition,row_partition,col_partition,assembled)

    Build an instance for [`PSparseMatrix`](@ref) from the underlying fields
    `matrix_partition`, `row_partition`, `col_partition`, assembled.
    """
    function PSparseMatrix(
        matrix_partition,row_partition,col_partition,assembled)
        V = eltype(matrix_partition)
        T = eltype(V)
        B = typeof(matrix_partition)
        C = typeof(row_partition)
        D = typeof(col_partition)
        new{V,B,C,D,T}(matrix_partition,row_partition,col_partition,assembled)
    end
end
partition(a::PSparseMatrix) = a.matrix_partition
Base.axes(a::PSparseMatrix) = (PRange(a.row_partition),PRange(a.col_partition))
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
        println(io,"$(m)×$(n) PSparseMatrix partitioned into $np parts of type $T")
    end
end
function Base.show(io::IO,data::PSparseMatrix)
    print(io,"PSparseMatrix(…)")
end

"""
    local_values(a::PSparseMatrix)

Get a vector of matrices containing the local rows and columns
in each part of `a`.

The row and column indices of the returned matrices can be mapped to global
indices, own indices, ghost indices, and owner by using
[`local_to_global`](@ref), [`local_to_own`](@ref), [`local_to_ghost`](@ref),
and [`local_to_owner`](@ref), respectively.
"""
function local_values(a::PSparseMatrix)
    partition(a)
end

"""
    own_own_values(a::PSparseMatrix)

Get a vector of matrices containing the own rows and columns
in each part of `a`.

The row and column indices of the returned matrices can be mapped to global
indices, local indices, and owner by using [`own_to_global`](@ref),
[`own_to_local`](@ref), and [`own_to_owner`](@ref), respectively.
"""
function own_own_values(a::PSparseMatrix)
    map(own_own_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

"""
    own_ghost_values(a::PSparseMatrix)

Get a vector of matrices containing the own rows and ghost columns
in each part of `a`.

The *row* indices of the returned matrices can be mapped to global indices,
local indices, and owner by using [`own_to_global`](@ref),
[`own_to_local`](@ref), and [`own_to_owner`](@ref), respectively.

The *column* indices of the returned matrices can be mapped to global indices,
local indices, and owner by using [`ghost_to_global`](@ref),
[`ghost_to_local`](@ref), and [`ghost_to_owner`](@ref), respectively.
"""
function own_ghost_values(a::PSparseMatrix)
    map(own_ghost_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

"""
    ghost_own_values(a::PSparseMatrix)

Get a vector of matrices containing the ghost rows and own columns
in each part of `a`.

The *row* indices of the returned matrices can be mapped to global indices,
local indices, and owner by using [`ghost_to_global`](@ref),
[`ghost_to_local`](@ref), and [`ghost_to_owner`](@ref), respectively.

The *column* indices of the returned matrices can be mapped to global indices,
local indices, and owner by using [`own_to_global`](@ref),
[`own_to_local`](@ref), and [`own_to_owner`](@ref), respectively.
"""
function ghost_own_values(a::PSparseMatrix)
    map(ghost_own_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

"""
    ghost_ghost_values(a::PSparseMatrix)

Get a vector of matrices containing the ghost rows and columns
in each part of `a`.

The row and column indices of the returned matrices can be mapped to global
indices, local indices, and owner by using [`ghost_to_global`](@ref),
[`ghost_to_local`](@ref), and [`ghost_to_owner`](@ref), respectively.
"""
function ghost_ghost_values(a::PSparseMatrix)
    map(ghost_ghost_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

val_parameter(a) = a
val_parameter(::Val{a}) where a = a

function split_format(A::PSparseMatrix;reuse=Val(false))
    rows = partition(axes(A,1))
    cols = partition(axes(A,2))
    values, cache = map(split_format_locally,partition(A),rows,cols) |> tuple_of_arrays
    B = PSparseMatrix(values,rows,cols,A.assembled)
    if val_parameter(reuse) == false
        B
    else
        B, cache
    end
end

function split_format!(B,A::PSparseMatrix,cache)
    rows = partition(axes(A,1))
    cols = partition(axes(A,2))
    map(split_format_locally!,partition(B),partition(A),rows,cols,cache)
    B
end

"""
    psparse(f,row_partition,col_partition;assembled)

Build an instance of [`PSparseMatrix`](@ref) from the initialization function
`f` and the partition for rows and columns `row_partition` and `col_partition`.

Equivalent to

    matrix_partition = map(f,row_partition,col_partition)
    PSparseMatrix(matrix_partition,row_partition,col_partition,assembled)
"""
function psparse(f,row_partition,col_partition;assembled)
    matrix_partition = map(f,row_partition,col_partition)
    PSparseMatrix(matrix_partition,row_partition,col_partition,assembled)
end

function psparse(I,J,V,rows,cols;kwargs...)
    T = SparseMatrixCSC{eltype(eltype(V)),Int}
    psparse(T,I,J,V,rows,cols;kwargs...)
end

function psparse(::Type{T},I,J,V,rows,cols;kwargs...) where T
    f(args...) =  sparse_matrix(T,args...)
    psparse(f,I,J,V,rows,cols;kwargs...)
end

"""
    psparse([f,]I,J,V,row_partition,col_partition;kwargs...) -> Task

Crate an instance of [`PSparseMatrix`](@ref) by setting arbitrary entries
from each of the underlying parts. It returns a task that produces the
instance of [`PSparseMatrix`](@ref) allowing latency hiding while performing
the communications needed in its setup.
"""
function psparse(f,I,J,V,rows,cols;
        split_format=Val(true),
        subassembled=Val(false),
        assembled=Val(false),
        assemble=Val(true),
        indices = :global,
        restore_ids = true,
        assembly_neighbors_options_rows = (;),
        assembly_neighbors_options_cols = (;),
        assembled_rows = nothing,
        reuse=Val(false)
    )

    # TODO for some particular cases
    # this function allocates more
    # intermediate results than needed
    # One can e.g. merge the split_format and assemble steps
    # Even the matrix compression step could be
    # merged with the assembly step

    # Checks
    disassembled = (!val_parameter(subassembled) && ! val_parameter(assembled)) ? true : false

    @assert indices in (:global,:local)
    if count((val_parameter(subassembled),val_parameter(assembled))) == 2
        error("Only one of the folling flags can be set to true: subassembled, assembled")
    end
    if indices === :global
        map(I,J) do I,J
            @assert I !== J
        end
    end

    if disassembled
        # TODO If assemble==true, we can (should) optimize the code
        # to do the conversion from disassembled to (fully) assembled split format
        # in a single shot.
        @assert indices === :global
        I_owner = find_owner(rows,I)
        J_owner = find_owner(cols,J)
        rows_sa = map(union_ghost,rows,I,I_owner)
        cols_sa = map(union_ghost,cols,J,J_owner)
        assembly_neighbors(rows_sa;assembly_neighbors_options_rows...)
        if ! val_parameter(assemble)
            # We only need this if we want a subassembled output.
            # For assembled output, this call will be deleted when optimizing
            # the code to do the conversions in a single shot.
            assembly_neighbors(cols_sa;assembly_neighbors_options_cols...)
        end
        map(map_global_to_local!,I,rows_sa)
        map(map_global_to_local!,J,cols_sa)
        values_sa = map(f,I,J,V,map(local_length,rows_sa),map(local_length,cols_sa))
        if val_parameter(reuse)
            K = map(precompute_nzindex,values_sa,I,J)
        end
        if restore_ids
            map(map_local_to_global!,I,rows_sa)
            map(map_local_to_global!,J,cols_sa)
        end
        A = PSparseMatrix(values_sa,rows_sa,cols_sa,val_parameter(assembled))
        if split_format |> val_parameter
            B,cacheB = PartitionedArrays.split_format(A;reuse=Val{true}())
        else
            B,cacheB = A,nothing
        end
        if val_parameter(assemble)
            t = PartitionedArrays.assemble(B,rows;reuse=Val{true}(),assembly_neighbors_options_cols)
        else
            t = @fake_async B,cacheB
        end
    elseif val_parameter(subassembled)
        rows_sa = rows
        cols_sa = cols
        if assembled_rows == nothing
            assembled_rows = map(remove_ghost,rows_sa)
        end
        if indices === :global
            map(map_global_to_local!,I,rows_sa)
            map(map_global_to_local!,J,cols_sa)
        end
        values_sa = map(f,I,J,V,map(local_length,rows_sa),map(local_length,cols_sa))
        if val_parameter(reuse)
            K = map(precompute_nzindex,values_sa,I,J)
        end
        if indices === :global && restore_ids
            map(map_local_to_global!,I,rows_sa)
            map(map_local_to_global!,J,cols_sa)
        end
        A = PSparseMatrix(values_sa,rows_sa,cols_sa,val_parameter(assembled))
        if split_format |> val_parameter
            B,cacheB = PartitionedArrays.split_format(A;reuse=Val{true}())
        else
            B,cacheB = A,nothing
        end
        if val_parameter(assemble)
            t = PartitionedArrays.assemble(B,assembled_rows;reuse=Val{true}(),assembly_neighbors_options_cols)
        else
            t = @fake_async B,cacheB
        end
    elseif val_parameter(assembled)
        rows_fa = rows
        cols_fa = cols
        if indices === :global
            map(map_global_to_local!,I,rows_fa)
            map(map_global_to_local!,J,cols_fa)
        end
        values_fa = map(f,I,J,V,map(local_length,rows_fa),map(local_length,cols_fa))
        if val_parameter(reuse)
            K = map(precompute_nzindex,values_fa,I,J)
        end
        if indices === :global && restore_ids
            map(map_local_to_global!,I,rows_fa)
            map(map_local_to_global!,J,cols_fa)
        end
        A = PSparseMatrix(values_fa,rows_fa,cols_fa,val_parameter(assembled))
        if split_format |> val_parameter
            B,cacheB = PartitionedArrays.split_format(A;reuse=Val{true}())
        else
            B,cacheB = A,nothing
        end
        t = @fake_async B,cacheB
    else
        error("This line should not be reached")
    end
    if val_parameter(reuse) == false
        return @fake_async begin
            C, cacheC = fetch(t)
            C
        end
    else
        return @fake_async begin
            C, cacheC = fetch(t)
            cache = (A,B,K,cacheB,cacheC,val_parameter(split_format),val_parameter(assembled))
            (C, cache)
        end
    end
end

"""
    psparse!(C::PSparseMatrix,V,cache)
"""
function psparse!(C,V,cache)
    (A,B,K,cacheB,cacheC,split_format,assembled) = cache
    rows_sa = partition(axes(A,1))
    cols_sa = partition(axes(A,2))
    values_sa = partition(A)
    map(sparse_matrix!,values_sa,V,K)
    if split_format
        split_format!(B,A,cacheB)
    end
    if !assembled && C.assembled
        t = PartitionedArrays.assemble!(C,B,cacheC)
    else
        t = @fake_async C
    end
end

function psparse_from_split_blocks(oo,oh,ho,hh,rowp,colp;assembled=false)
    rperms = map(local_permutation,rowp)
    cperms = map(local_permutation,colp)
    values = map(split_matrix,oo,oh,ho,hh,rperms,cperms)
    PSparseMatrix(values,rowp,colp,assembled)
end

function psparse_from_split_blocks(oo,oh,rowp,colp;assembled=true)
    ho = map(oo,rowp,colp) do oo, rows, cols
        T = typeof(oo)
        Tv = eltype(oo)
        Ti = indextype(oo)
        n_ghost_rows = ghost_length(rows)
        n_own_cols = own_length(cols)
        sparse_matrix(T,Ti[],Ti[],Tv[],n_ghost_rows,n_own_cols)
    end
    hh = map(oo,rowp,colp) do oo, rows, cols
        T = typeof(oo)
        Tv = eltype(oo)
        Ti = indextype(oo)
        n_ghost_rows = ghost_length(rows)
        n_ghost_cols = ghost_length(cols)
        sparse_matrix(T,Ti[],Ti[],Tv[],n_ghost_rows,n_ghost_cols)
    end
    psparse_from_split_blocks(oo,oh,ho,hh,rowp,colp;assembled)
end

function assemble(A::PSparseMatrix;kwargs...)
    rows = map(remove_ghost,partition(axes(A,1))) 
    assemble(A,rows;kwargs...)
end

"""
    assemble(A::PSparseMatrix[,rows];kwargs...)
"""
function assemble(A::PSparseMatrix,rows;kwargs...)
    @boundscheck @assert matching_own_indices(axes(A,1),PRange(rows))
    T = eltype(partition(A))
    psparse_assemble_impl(A,T,rows;kwargs...)
end

"""
    assemble!(B::PSparseMatrix,A::PSparseMatrix,cache)
"""
function assemble!(B::PSparseMatrix,A::PSparseMatrix,cache)
    T = eltype(partition(A))
    psparse_assemble_impl!(B,A,T,cache)
end

function psparse_assemble_impl(A,::Type,rows)
    error("Case not implemented yet")
end

function psparse_assemble_impl(
        A,
        ::Type{<:AbstractSplitMatrix},
        rows;
        reuse=Val(false),
        assembly_neighbors_options_cols=(;))

    function setup_cache_snd(A,parts_snd,rows_sa,cols_sa)
        A_ghost_own   = A.blocks.ghost_own
        A_ghost_ghost = A.blocks.ghost_ghost
        gen = ( owner=>i for (i,owner) in enumerate(parts_snd) )
        owner_to_p = Dict(gen)
        ptrs = zeros(Int32,length(parts_snd)+1)
        ghost_to_owner_row = ghost_to_owner(rows_sa)
        ghost_to_global_row = ghost_to_global(rows_sa)
        own_to_global_col = own_to_global(cols_sa)
        ghost_to_global_col = ghost_to_global(cols_sa)
        for (i,_,_) in nziterator(A_ghost_own)
            owner = ghost_to_owner_row[i]
            ptrs[owner_to_p[owner]+1] += 1
        end
        for (i,_,_) in nziterator(A_ghost_ghost)
            owner = ghost_to_owner_row[i]
            ptrs[owner_to_p[owner]+1] += 1
        end
        length_to_ptrs!(ptrs)
        Tv = eltype(A_ghost_own)
        ndata = ptrs[end]-1
        I_snd_data = zeros(Int,ndata)
        J_snd_data = zeros(Int,ndata)
        V_snd_data = zeros(Tv,ndata)
        k_snd_data = zeros(Int32,ndata)
        nnz_ghost_own = 0
        for (k,(i,j,v)) in enumerate(nziterator(A_ghost_own))
            owner = ghost_to_owner_row[i]
            p = ptrs[owner_to_p[owner]]
            I_snd_data[p] = ghost_to_global_row[i]
            J_snd_data[p] = own_to_global_col[j]
            V_snd_data[p] = v
            k_snd_data[p] = k
            ptrs[owner_to_p[owner]] += 1
            nnz_ghost_own += 1
        end
        for (k,(i,j,v)) in enumerate(nziterator(A_ghost_ghost))
            owner = ghost_to_owner_row[i]
            p = ptrs[owner_to_p[owner]]
            I_snd_data[p] = ghost_to_global_row[i]
            J_snd_data[p] = ghost_to_global_col[j]
            V_snd_data[p] = v
            k_snd_data[p] = k+nnz_ghost_own
            ptrs[owner_to_p[owner]] += 1
        end
        rewind_ptrs!(ptrs)
        I_snd = JaggedArray(I_snd_data,ptrs)
        J_snd = JaggedArray(J_snd_data,ptrs)
        V_snd = JaggedArray(V_snd_data,ptrs)
        k_snd = JaggedArray(k_snd_data,ptrs)
        (;I_snd,J_snd,V_snd,k_snd,parts_snd)
    end
    function setup_cache_rcv(I_rcv,J_rcv,V_rcv,parts_rcv)
        k_rcv_data = zeros(Int32,length(I_rcv.data))
        k_rcv = JaggedArray(k_rcv_data,I_rcv.ptrs)
        (;I_rcv,J_rcv,V_rcv,k_rcv,parts_rcv)
    end
    function setup_own_triplets(A,cache_rcv,rows_sa,cols_sa)
        nz_own_own = findnz(A.blocks.own_own)
        nz_own_ghost = findnz(A.blocks.own_ghost)
        I_rcv_data = cache_rcv.I_rcv.data
        J_rcv_data = cache_rcv.J_rcv.data
        V_rcv_data = cache_rcv.V_rcv.data
        k_rcv_data = cache_rcv.k_rcv.data
        global_to_own_col = global_to_own(cols_sa)
        is_ghost = findall(j->global_to_own_col[j]==0,J_rcv_data)
        is_own = findall(j->global_to_own_col[j]!=0,J_rcv_data)
        I_rcv_own = view(I_rcv_data,is_own)
        J_rcv_own = view(J_rcv_data,is_own)
        V_rcv_own = view(V_rcv_data,is_own)
        k_rcv_own = view(k_rcv_data,is_own)
        I_rcv_ghost = view(I_rcv_data,is_ghost)
        J_rcv_ghost = view(J_rcv_data,is_ghost)
        V_rcv_ghost = view(V_rcv_data,is_ghost)
        k_rcv_ghost = view(k_rcv_data,is_ghost)
        # After this col ids in own_ghost triplet remain global
        map_global_to_own!(I_rcv_own,rows_sa)
        map_global_to_own!(J_rcv_own,cols_sa)
        map_global_to_own!(I_rcv_ghost,rows_sa)
        map_ghost_to_global!(nz_own_ghost[2],cols_sa)
        own_own_I = vcat(nz_own_own[1],I_rcv_own)
        own_own_J = vcat(nz_own_own[2],J_rcv_own)
        own_own_V = vcat(nz_own_own[3],V_rcv_own)
        own_own_triplet = (own_own_I,own_own_J,own_own_V)
        own_ghost_I = vcat(nz_own_ghost[1],I_rcv_ghost)
        own_ghost_J = vcat(nz_own_ghost[2],J_rcv_ghost)
        own_ghost_V = vcat(nz_own_ghost[3],V_rcv_ghost)
        map_global_to_ghost!(nz_own_ghost[2],cols_sa)
        own_ghost_triplet = (own_ghost_I,own_ghost_J,own_ghost_V)
        triplets = (own_own_triplet,own_ghost_triplet)
        aux = (I_rcv_own,J_rcv_own,k_rcv_own,I_rcv_ghost,J_rcv_ghost,k_rcv_ghost,nz_own_own,nz_own_ghost)
        triplets, own_ghost_J, aux
    end
    function finalize_values(A,rows_fa,cols_fa,cache_snd,cache_rcv,triplets,aux)
        (own_own_triplet,own_ghost_triplet) = triplets
        (I_rcv_own,J_rcv_own,k_rcv_own,I_rcv_ghost,J_rcv_ghost,k_rcv_ghost,nz_own_own,nz_own_ghost) = aux
        map_global_to_ghost!(own_ghost_triplet[2],cols_fa)
        map_global_to_ghost!(J_rcv_ghost,cols_fa)
        TA = typeof(A.blocks.own_own)
        n_own_rows = own_length(rows_fa)
        n_own_cols = own_length(cols_fa)
        n_ghost_rows = ghost_length(rows_fa)
        n_ghost_cols = ghost_length(cols_fa)
        Ti = indextype(A.blocks.own_own)
        Tv = eltype(A.blocks.own_own)
        own_own = compresscoo(TA,own_own_triplet...,n_own_rows,n_own_cols)
        own_ghost = compresscoo(TA,own_ghost_triplet...,n_own_rows,n_ghost_cols)
        ghost_own = compresscoo(TA,Ti[],Ti[],Tv[],n_ghost_rows,n_own_cols)
        ghost_ghost = compresscoo(TA,Ti[],Ti[],Tv[],n_ghost_rows,n_ghost_cols)
        blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
        values = split_matrix(blocks,local_permutation(rows_fa),local_permutation(cols_fa))
        nnz_own_own = nnz(own_own)
        k_own_sa = precompute_nzindex(own_own,own_own_triplet[1:2]...)
        k_ghost_sa = precompute_nzindex(own_ghost,own_ghost_triplet[1:2]...)
        for p in 1:length(I_rcv_own)
            i = I_rcv_own[p]
            j = J_rcv_own[p]
            k_rcv_own[p] = nzindex(own_own,i,j)
        end
        for p in 1:length(I_rcv_ghost)
            i = I_rcv_ghost[p]
            j = J_rcv_ghost[p]
            k_rcv_ghost[p] = nzindex(own_ghost,i,j) + nnz_own_own
        end
        cache = (;k_own_sa,k_ghost_sa,cache_snd...,cache_rcv...)
        values, cache
    end
    rows_sa = partition(axes(A,1))
    cols_sa = partition(axes(A,2))
    #rows = map(remove_ghost,rows_sa)
    cols = map(remove_ghost,cols_sa)
    parts_snd, parts_rcv = assembly_neighbors(rows_sa)
    cache_snd = map(setup_cache_snd,partition(A),parts_snd,rows_sa,cols_sa)
    I_snd = map(i->i.I_snd,cache_snd)
    J_snd = map(i->i.J_snd,cache_snd)
    V_snd = map(i->i.V_snd,cache_snd)
    graph = ExchangeGraph(parts_snd,parts_rcv)
    t_I = exchange(I_snd,graph)
    t_J = exchange(J_snd,graph)
    t_V = exchange(V_snd,graph)
    @fake_async begin
        I_rcv = fetch(t_I)
        J_rcv = fetch(t_J)
        V_rcv = fetch(t_V)
        cache_rcv = map(setup_cache_rcv,I_rcv,J_rcv,V_rcv,parts_rcv)
        triplets,J,aux = map(setup_own_triplets,partition(A),cache_rcv,rows_sa,cols_sa) |> tuple_of_arrays
        J_owner = find_owner(cols_sa,J)
        rows_fa = rows
        cols_fa = map(union_ghost,cols,J,J_owner)
        assembly_neighbors(cols_fa;assembly_neighbors_options_cols...)
        vals_fa, cache = map(finalize_values,partition(A),rows_fa,cols_fa,cache_snd,cache_rcv,triplets,aux) |> tuple_of_arrays
        assembled = true
        B = PSparseMatrix(vals_fa,rows_fa,cols_fa,assembled)
        if val_parameter(reuse) == false
            B
        else
            B, cache
        end
    end
end

function psparse_assemble_impl!(B,A,::Type,cache)
    error("case not implemented")
end

function psparse_assemble_impl!(B,A,::Type{<:AbstractSplitMatrix},cache)
    function setup_snd(A,cache)
        A_ghost_own   = A.blocks.ghost_own
        A_ghost_ghost = A.blocks.ghost_ghost
        nnz_ghost_own = nnz(A_ghost_own)
        V_snd_data = cache.V_snd.data
        k_snd_data = cache.k_snd.data
        nz_ghost_own = nonzeros(A_ghost_own)
        nz_ghost_ghost = nonzeros(A_ghost_ghost)
        for p in 1:length(k_snd_data)
            k = k_snd_data[p]
            if k <= nnz_ghost_own
                v = nz_ghost_own[k]
            else
                v = nz_ghost_ghost[k-nnz_ghost_own]
            end
            V_snd_data[p] = v
        end
    end
    function setup_sa(B,A,cache)
        setcoofast!(B.blocks.own_own,nonzeros(A.blocks.own_own),cache.k_own_sa)
        setcoofast!(B.blocks.own_ghost,nonzeros(A.blocks.own_ghost),cache.k_ghost_sa)
    end
    function setup_rcv(B,cache)
        B_own_own   = B.blocks.own_own
        B_own_ghost = B.blocks.own_ghost
        nnz_own_own = nnz(B_own_own)
        V_rcv_data = cache.V_rcv.data
        k_rcv_data = cache.k_rcv.data
        nz_own_own = nonzeros(B_own_own)
        nz_own_ghost = nonzeros(B_own_ghost)
        for p in 1:length(k_rcv_data)
            k = k_rcv_data[p]
            v = V_rcv_data[p]
            if k <= nnz_own_own
                nz_own_own[k] += v
            else
                nz_own_ghost[k-nnz_own_own] += v
            end
        end
    end
    map(setup_snd,partition(A),cache)
    parts_snd = map(i->i.parts_snd,cache)
    parts_rcv = map(i->i.parts_rcv,cache)
    V_snd = map(i->i.V_snd,cache)
    V_rcv = map(i->i.V_rcv,cache)
    graph = ExchangeGraph(parts_snd,parts_rcv)
    t = exchange!(V_rcv,V_snd,graph)
    map(setup_sa,partition(B),partition(A),cache)
    @fake_async begin
        wait(t)
        map(setup_rcv,partition(B),cache)
        B
    end
end

"""
    consistent(A::PSparseMatrix,rows;kwargs...)
"""
function consistent(A::PSparseMatrix,rows_co;kwargs...)
    @assert A.assembled
    T = eltype(partition(A))
    psparse_consistent_impl(A,T,rows_co;kwargs...)
end

"""
    consistent!(B::PSparseMatrix,A::PSparseMatrix,cache)
"""
function consistent!(B::PSparseMatrix,A::PSparseMatrix,cache)
    @assert A.assembled
    T = eltype(partition(A))
    psparse_consistent_impl!(B,A,T,cache)
end

function psparse_consistent_impl(
    A,
    ::Type{<:AbstractSplitMatrix},
    rows_co;
    reuse=Val(false))

    function setup_snd(A,parts_snd,lids_snd,rows_co,cols_fa)
        own_to_local_row = own_to_local(rows_co)
        own_to_global_row = own_to_global(rows_co)
        own_to_global_col = own_to_global(cols_fa)
        ghost_to_global_col = ghost_to_global(cols_fa)
        nl = size(A,1)
        li_to_ps_ptrs = zeros(Int32,nl+1)
        for p in 1:length(lids_snd)
            for li in lids_snd[p]
                li_to_ps_ptrs[li+1] += 1
            end
        end
        length_to_ptrs!(li_to_ps_ptrs)
        ndata = li_to_ps_ptrs[end]-1
        li_to_ps_data = zeros(Int32,ndata)
        for p in 1:length(lids_snd)
            for li in lids_snd[p]
                q = li_to_ps_ptrs[li]
                li_to_ps_data[q] = p
                li_to_ps_ptrs[li] = q + 1
            end
        end
        rewind_ptrs!(li_to_ps_ptrs)
        li_to_ps = JaggedArray(li_to_ps_data,li_to_ps_ptrs)
        ptrs = zeros(Int32,length(parts_snd)+1)
        for (i,j,v) in nziterator(A.blocks.own_own)
            li = own_to_local_row[i]
            for p in li_to_ps[li]
                ptrs[p+1] += 1
            end
        end
        for (i,j,v) in nziterator(A.blocks.own_ghost)
            li = own_to_local_row[i]
            for p in li_to_ps[li]
                ptrs[p+1] += 1
            end
        end
        length_to_ptrs!(ptrs)
        ndata = ptrs[end]-1
        T = eltype(A)
        I_snd = JaggedArray(zeros(Int,ndata),ptrs)
        J_snd = JaggedArray(zeros(Int,ndata),ptrs)
        V_snd = JaggedArray(zeros(T,ndata),ptrs)
        k_snd = JaggedArray(zeros(Int32,ndata),ptrs)
        for (k,(i,j,v)) in enumerate(nziterator(A.blocks.own_own))
            li = own_to_local_row[i]
            for p in li_to_ps[li]
                q = ptrs[p]
                I_snd.data[q] = own_to_global_row[i]
                J_snd.data[q] = own_to_global_col[j]
                V_snd.data[q] = v
                k_snd.data[q] = k
                ptrs[p] += 1
            end
        end
        nnz_own_own = nnz(A.blocks.own_own)
        for (k,(i,j,v)) in enumerate(nziterator(A.blocks.own_ghost))
            li = own_to_local_row[i]
            for p in li_to_ps[li]
                q = ptrs[p]
                I_snd.data[q] = own_to_global_row[i]
                J_snd.data[q] = ghost_to_global_col[j]
                V_snd.data[q] = v
                k_snd.data[q] = k+nnz_own_own
                ptrs[p] += 1
            end
        end
        rewind_ptrs!(ptrs)
        cache_snd = (;parts_snd,lids_snd,I_snd,J_snd,V_snd,k_snd)
        cache_snd
    end
    function setup_rcv(parts_rcv,lids_rcv,I_rcv,J_rcv,V_rcv)
        cache_rcv = (;parts_rcv,lids_rcv,I_rcv,J_rcv,V_rcv)
        cache_rcv
    end
    function finalize(A,cache_snd,cache_rcv,rows_co,cols_fa,cols_co)
        I_rcv_data = cache_rcv.I_rcv.data
        J_rcv_data = cache_rcv.J_rcv.data
        V_rcv_data = cache_rcv.V_rcv.data
        global_to_own_col = global_to_own(cols_co)
        global_to_ghost_col = global_to_ghost(cols_co)
        is_own = findall(j->global_to_own_col[j]!=0,J_rcv_data)
        is_ghost = findall(j->global_to_ghost_col[j]!=0,J_rcv_data)
        I_rcv_own = I_rcv_data[is_own]
        J_rcv_own = J_rcv_data[is_own]
        V_rcv_own = V_rcv_data[is_own]
        I_rcv_ghost = I_rcv_data[is_ghost]
        J_rcv_ghost = J_rcv_data[is_ghost]
        V_rcv_ghost = V_rcv_data[is_ghost]
        map_global_to_ghost!(I_rcv_own,rows_co)
        map_global_to_ghost!(I_rcv_ghost,rows_co)
        map_global_to_own!(J_rcv_own,cols_co)
        map_global_to_ghost!(J_rcv_ghost,cols_co)
        I2,J2,V2 = findnz(A.blocks.own_ghost)
        map_ghost_to_global!(J2,cols_fa)
        map_global_to_ghost!(J2,cols_co)
        n_own_rows = own_length(rows_co)
        n_ghost_rows = ghost_length(rows_co)
        n_own_cols = own_length(cols_co)
        n_ghost_cols = ghost_length(cols_co)
        TA = typeof(A.blocks.ghost_own)
        own_own = A.blocks.own_own
        own_ghost = compresscoo(TA,I2,J2,V2,n_own_rows,n_ghost_cols) # TODO this can be improved
        ghost_own = compresscoo(TA,I_rcv_own,J_rcv_own,V_rcv_own,n_ghost_rows,n_own_cols)
        ghost_ghost = compresscoo(TA,I_rcv_ghost,J_rcv_ghost,V_rcv_ghost,n_ghost_rows,n_ghost_cols)
        K_own = precompute_nzindex(ghost_own,I_rcv_own,J_rcv_own)
        K_ghost = precompute_nzindex(ghost_ghost,I_rcv_ghost,J_rcv_ghost)
        blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
        values = split_matrix(blocks,local_permutation(rows_co),local_permutation(cols_co))
        k_snd = cache_snd.k_snd
        V_snd = cache_snd.V_snd
        V_rcv = cache_rcv.V_rcv
        parts_snd = cache_snd.parts_snd
        parts_rcv = cache_rcv.parts_rcv
        cache = (;parts_snd,parts_rcv,k_snd,V_snd,V_rcv,is_ghost,is_own,V_rcv_own,V_rcv_ghost,K_own,K_ghost)
        values,cache
    end
    @assert matching_own_indices(axes(A,1),PRange(rows_co))
    rows_fa = partition(axes(A,1))
    cols_fa = partition(axes(A,2))
    # snd and rcv are swapped on purpose
    parts_rcv,parts_snd = assembly_neighbors(rows_co)
    lids_rcv,lids_snd = assembly_local_indices(rows_co)
    cache_snd = map(setup_snd,partition(A),parts_snd,lids_snd,rows_co,cols_fa)
    I_snd = map(i->i.I_snd,cache_snd)
    J_snd = map(i->i.J_snd,cache_snd)
    V_snd = map(i->i.V_snd,cache_snd)
    graph = ExchangeGraph(parts_snd,parts_rcv)
    t_I = exchange(I_snd,graph)
    t_J = exchange(J_snd,graph)
    t_V = exchange(V_snd,graph)
    @fake_async begin
        I_rcv = fetch(t_I)
        J_rcv = fetch(t_J)
        V_rcv = fetch(t_V)
        J_rcv_data = map(x->x.data,J_rcv)
        J_rcv_owner = find_owner(cols_fa,J_rcv_data)
        cols_co = map(union_ghost,cols_fa,J_rcv_data,J_rcv_owner)
        cache_rcv = map(setup_rcv,parts_rcv,lids_rcv,I_rcv,J_rcv,V_rcv)
        values,cache = map(finalize,partition(A),cache_snd,cache_rcv,rows_co,cols_fa,cols_co) |> tuple_of_arrays
        B = PSparseMatrix(values,rows_co,cols_co,A.assembled)
        if val_parameter(reuse) == false
            B
        else
            B,cache
        end
    end
end

function psparse_consistent_impl!(B,A,::Type{<:AbstractSplitMatrix},cache)
    function setup_snd(A,cache)
        k_snd_data = cache.k_snd.data
        V_snd_data = cache.V_snd.data
        nnz_own_own = nnz(A.blocks.own_own)
        A_own_own = nonzeros(A.blocks.own_own)
        A_own_ghost = nonzeros(A.blocks.own_ghost)
        for (p,k) in enumerate(k_snd_data)
            if k <= nnz_own_own
                v = A_own_own[k]
            else
                v = A_own_ghost[k-nnz_own_own]
            end
            V_snd_data[p] = v
        end
    end
    function setup_rcv(B,cache)
        is_ghost = cache.is_ghost
        is_own = cache.is_own
        V_rcv_data = cache.V_rcv.data
        K_own = cache.K_own
        K_ghost = cache.K_ghost
        V_rcv_own = V_rcv_data[is_own]
        V_rcv_ghost = V_rcv_data[is_ghost]
        setcoofast!(B.blocks.ghost_own,V_rcv_own,K_own)
        setcoofast!(B.blocks.ghost_ghost,V_rcv_ghost,K_ghost)
        B
    end
    map(own_own_values(B),own_own_values(A)) do b,a
        msg = "consistent!(B,A,cache) can only be called if B was obtained as B,cache = consistent(A)|>fetch"
        @assert a === b msg
    end
    map(setup_snd,partition(A),cache)
    parts_snd = map(i->i.parts_snd,cache)
    parts_rcv = map(i->i.parts_rcv,cache)
    V_snd = map(i->i.V_snd,cache)
    V_rcv = map(i->i.V_rcv,cache)
    graph = ExchangeGraph(parts_snd,parts_rcv)
    t = exchange!(V_rcv,V_snd,graph)
    map(own_ghost_values(B),own_ghost_values(A)) do b,a
        if nonzeros(b) !== nonzeros(a)
            copy!(nonzeros(b),nonzeros(a))
        end
    end
    @fake_async begin
        wait(t)
        map(setup_rcv,partition(B),cache)
        B
    end
end

function Base.:*(a::PSparseMatrix,b::PVector)
    Ta = eltype(a)
    Tb = eltype(b)
    T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
    c = PVector{Vector{T}}(undef,partition(axes(a,1)))
    mul!(c,a,b)
    c
end

function Base.:*(a::Number,b::PSparseMatrix)
    matrix_partition = map(partition(b)) do values
        a*values
    end
    rows = partition(axes(b,1))
    cols = partition(axes(b,2))
    PSparseMatrix(matrix_partition,rows,cols,b.assembled)
end

function Base.:*(b::PSparseMatrix,a::Number)
    a*b
end

for op in (:+,:-)
    @eval begin
        function Base.$op(a::PSparseMatrix)
            matrix_partition = map(partition(a)) do a
                $op(a)
            end
            rows = partition(axes(a,1))
            cols = partition(axes(a,2))
            PSparseMatrix(matrix_partition,rows,cols,a.assembled)
        end
        function Base.$op(a::PSparseMatrix,b::PSparseMatrix)
            @boundscheck @assert matching_local_indices(axes(a,1),axes(b,1))
            @boundscheck @assert matching_local_indices(axes(a,2),axes(b,2))
            matrix_partition = map(partition(a),partition(b)) do a,b
                $op(a,b)
            end
            rows = partition(axes(b,1))
            cols = partition(axes(b,2))
            assembled = a.assembled && b.assembled
            PSparseMatrix(matrix_partition,rows,cols,assembled)
        end
    end
end

muladd!(b,A,x) = mul!(b,A,x,one(eltype(b)),one(eltype(b)))

function LinearAlgebra.mul!(c::PVector,a::PSparseMatrix,b::PVector)
    @boundscheck @assert matching_own_indices(axes(c,1),axes(a,1))
    @boundscheck @assert matching_own_indices(axes(a,2),axes(b,1))
    @boundscheck @assert matching_ghost_indices(axes(a,2),axes(b,1))
    if ! a.assembled
        @boundscheck @assert matching_ghost_indices(axes(a,1),axes(c,1))
        return mul!(c,a,b,1,0)
    end
    t = consistent!(b)
    foreach(spmv!,own_values(c),own_own_values(a),own_values(b))
    wait(t)
    foreach(muladd!,own_values(c),own_ghost_values(a),ghost_values(b))
    c
end

function LinearAlgebra.mul!(c::PVector,a::PSparseMatrix,b::PVector,α::Number,β::Number)
    @boundscheck @assert matching_own_indices(axes(c,1),axes(a,1))
    @boundscheck @assert matching_own_indices(axes(a,2),axes(b,1))
    @boundscheck @assert matching_ghost_indices(axes(a,2),axes(b,1))
    if ! a.assembled
        @boundscheck @assert matching_ghost_indices(axes(a,1),axes(c,1))
    end
    # Start the exchange
    t = consistent!(b)
    # Meanwhile, process the owned blocks
    foreach(own_values(c),own_own_values(a),own_values(b)) do co,aoo,bo
        if β != 1
            β != 0 ? rmul!(co, β) : fill!(co,zero(eltype(co)))
        end
        mul!(co,aoo,bo,α,1)
    end
    if ! a.assembled
        foreach(ghost_values(c),ghost_own_values(a),own_values(b)) do ch,aho,bo
            if β != 1
                β != 0 ? rmul!(ch, β) : fill!(ch,zero(eltype(ch)))
            end
            mul!(ch,aho,bo,α,1)
        end
    end
    # Wait for the exchange to finish
    wait(t)
    # process the ghost block
    foreach(own_values(c),own_ghost_values(a),ghost_values(b)) do co,aoh,bh
        mul!(co,aoh,bh,α,1)
    end
    if ! a.assembled
        foreach(ghost_values(c),ghost_ghost_values(a),ghost_values(b)) do ch,ahh,bh
            mul!(ch,ahh,bh,α,1)
        end
        assemble!(c) |> wait
    end
    c
end

function LinearAlgebra.mul!(c::PVector,at::Transpose{T,<:PSparseMatrix} where T,b::PVector,α::Number,β::Number)
    a = at.parent
    @assert a.assembled
    foreach(ghost_values(c),own_ghost_values(a),own_values(b)) do ch,aoh,bo
        fill!(ch,zero(eltype(ch)))
        atoh = transpose(aoh)
        mul!(ch,atoh,bo,α,1)
    end
    t = assemble!(c)
    foreach(own_values(c),own_own_values(a),own_values(b)) do co,aoo,bo
        if β != 1
            β != 0 ? rmul!(co, β) : fill!(co,zero(eltype(co)))
        end
        atoo = transpose(aoo)
        mul!(co,atoo,bo,α,1)
    end
    wait(t)
    c
end

## LinearAlgebra.diag returns a sparse vector for sparse matrices
# this is why we introduce dense_diag
function dense_diag(A)
    d = zeros(eltype(A),size(A,1))
    dense_diag!(d,A)
end

function dense_diag!(d,A)
    d .= diag(A)
    d
end

function dense_diag(A::PSparseMatrix)
    diag(A)
end

function LinearAlgebra.diag(A::PSparseMatrix)
    d = pzeros(eltype(A),partition(axes(A,1)))
    dense_diag!(d,A)
end

function dense_diag!(d::PVector,A::PSparseMatrix)
    map(dense_diag!,own_values(d),own_own_values(A))
    d
end

# TODO SparseArrays.spdiagm already exists
# For the moment we keep it as a private helper function
function sparse_diag_matrix(d,shape)
    n = length(d)
    I = 1:n
    J = 1:n
    V = d
    sparse(I,J,V,map(length,shape)...)
end

function sparse_diag_matrix(d::PVector,shape)
    row_partition,col_partition = map(partition,shape)
    function setup(own_d,rows,cols)
        I = own_to_global(rows) |> collect
        J = own_to_global(cols) |> collect
        V = own_d
        I,J,V
    end
    I,J,V = map(setup,own_values(d),row_partition,col_partition) |> tuple_of_arrays
    psparse(I,J,V,row_partition,col_partition;assembled=true) |> fetch
end

function rap(R,A,P;reuse=Val(false))
    Ac = R*A*P
    if val_parameter(reuse)
        return Ac, nothing
    end
    Ac
end

function rap!(Ac,R,A,P,cache)
    # TODO improve performance
    tmp = R*A*P
    copyto!(Ac,tmp)
    Ac
end

function spmm(A,B;reuse=Val(false))
    C = A*B
    if val_parameter(reuse)
        return C, nothing
    end
    C
end

function spmm!(C,A,B,state)
    mul!(C,A,B)
    C
end

function spmm(A::PSparseMatrix,B::PSparseMatrix;reuse=Val(false))
    # TODO latency hiding
    @assert A.assembled
    @assert B.assembled
    col_partition = partition(axes(A,2))
    C,cacheC = consistent(B,col_partition;reuse=true) |> fetch
    D_partition,cacheD = map((args...)->spmm(args...;reuse=true),partition(A),partition(C)) |> tuple_of_arrays
    assembled = true
    D = PSparseMatrix(D_partition,partition(axes(A,1)),partition(axes(C,2)),assembled)
    if val_parameter(reuse)
        cache = (C,cacheC,cacheD)
        return D,cache
    end
    D
end

function spmm!(D::PSparseMatrix,A::PSparseMatrix,B::PSparseMatrix,cache)
    (C,cacheC,cacheD)= cache
    consistent!(C,B,cacheC) |> wait
    map(spmm!,partition(D),partition(A),partition(C),cacheD)
    D
end

function spmtm(A,B;reuse=Val(false))
    C = transpose(A)*B
    if val_parameter(reuse)
        return C, nothing
    end
    C
end

function spmtm!(C,A,B,cache)
    mul!(C,transpose(A),B)
    C
end

function spmtm(A::PSparseMatrix,B::PSparseMatrix;reuse=Val(false))
    # TODO latency hiding
    @assert A.assembled
    @assert B.assembled
    D_partition,cacheD = map((args...)->spmtm(args...;reuse=true),partition(A),partition(B)) |> tuple_of_arrays
    assembled = false
    D = PSparseMatrix(D_partition,partition(axes(A,2)),partition(axes(B,2)),assembled)
    C,cacheC = assemble(D;reuse=true) |> fetch
    if val_parameter(reuse)
        cache = (D,cacheC,cacheD)
        return C,cache
    end
    C
end

function spmtm!(C::PSparseMatrix,A::PSparseMatrix,B::PSparseMatrix,cache)
    (D,cacheC,cacheD)= cache
    map(spmtm!,partition(D),partition(A),partition(B),cacheD)
    assemble!(C,D,cacheC) |> wait
    C
end

function Base.:*(A::PSparseMatrix,B::PSparseMatrix)
    C = spmm(A,B)
    C
end

function Base.:*(At::Transpose{T,<:PSparseMatrix} where T,B::PSparseMatrix)
    A = At.parent
    C = spmtm(A,B)
    C
end

function Base.:-(I::LinearAlgebra.UniformScaling,A::PSparseMatrix)
    T = eltype(A)
    row_partition = partition(axes(A,1))
    d = pones(T,row_partition)
    D = sparse_diag_matrix(d,axes(A))
    D-A
end

Base.similar(a::PSparseMatrix) = similar(a,eltype(a))
function Base.similar(a::PSparseMatrix,::Type{T}) where T
    matrix_partition = map(partition(a)) do values
        similar(values,T)
    end
    rows, cols = axes(a)
    PSparseMatrix(matrix_partition,partition(rows),partition(cols),a.assembled)
end

function Base.copy(a::PSparseMatrix)
    mats = map(copy,partition(a))
    rows, cols = axes(a)
    PSparseMatrix(mats,partition(rows),partition(cols),a.assembled)
end

function Base.copy!(a::PSparseMatrix,b::PSparseMatrix)
    @assert size(a) == size(b)
    @assert a.assembled == b.assembled
    if partition(axes(a,1)) === partition(axes(b,1)) && partition(axes(a,2)) === partition(axes(b,2))
        copyto!(a,b)
    else
        error("Trying to copy a PSparseMatrix into another one with a different data layout. This case is not implemented yet. It would require communications.")
    end
end

function Base.copyto!(a::PSparseMatrix,b::PSparseMatrix)
    map(copy!,partition(a),partition(b))
    a
end

function LinearAlgebra.fillstored!(a::PSparseMatrix,v)
    map(partition(a)) do values
        LinearAlgebra.fillstored!(values,v)
    end
    a
end

function SparseArrays.nnz(a::PSparseMatrix)
    ns = map(nnz,partition(a))
    sum(ns)
end

# This function could be removed if IterativeSolvers was implemented in terms
# of axes(A,d) instead of size(A,d)
function IterativeSolvers.zerox(A::PSparseMatrix,b::PVector)
    T = IterativeSolvers.Adivtype(A, b)
    x = similar(b, T, axes(A, 2))
    fill!(x, zero(T))
    return x
end

"""
    repartition(A::PSparseMatrix,new_rows,new_cols;reuse=false)
"""
function repartition(A::PSparseMatrix,new_rows,new_cols;reuse=Val(false))
    @assert A.assembled "repartition on a sub-assembled matrix not implemented yet"
    function prepare_triplets(A_own_own,A_own_ghost,A_rows,A_cols)
        I1,J1,V1 = findnz(A_own_own)
        I2,J2,V2 = findnz(A_own_ghost)
        map_own_to_global!(I1,A_rows)
        map_own_to_global!(I2,A_rows)
        map_own_to_global!(J1,A_cols)
        map_ghost_to_global!(J2,A_cols)
        I = vcat(I1,I2)
        J = vcat(J1,J2)
        V = vcat(V1,V2)
        (I,J,V)
    end
    A_own_own = own_own_values(A)
    A_own_ghost = own_ghost_values(A)
    A_rows = partition(axes(A,1))
    A_cols = partition(axes(A,2))
    I,J,V = map(prepare_triplets,A_own_own,A_own_ghost,A_rows,A_cols) |> tuple_of_arrays
    # TODO this one does not preserve the local storage layout of A
    t = psparse(I,J,V,new_rows,new_cols;reuse=true)
    @fake_async begin
        B,cacheB = fetch(t)
        if val_parameter(reuse) == false
            B
        else
            cache = (V,cacheB)
            B, cache
        end
    end
end

"""
    repartition!(B::PSparseMatrix,A::PSparseMatrix,cache)
"""
function repartition!(B::PSparseMatrix,A::PSparseMatrix,cache)
    (V,cacheB) = cache
    function fill_values!(V,A_own_own,A_own_ghost)
        nz_own_own = nonzeros(A_own_own)
        nz_own_ghost = nonzeros(A_own_ghost)
        l1 = length(nz_own_own)
        l2 = length(nz_own_ghost)
        V[1:l1] = nz_own_own
        V[(1:l2).+l1] = nz_own_ghost
    end
    A_own_own = own_own_values(A)
    A_own_ghost = own_ghost_values(A)
    map(fill_values!,V,A_own_own,A_own_ghost)
    psparse!(B,V,cacheB)
end

"""
    repartition(A::PSparseMatrix,b::PVector,new_rows,new_cols;reuse=false)
"""
function repartition(A::PSparseMatrix,b::PVector,new_rows,new_cols;reuse=Val(false))
    # TODO this is just a reference implementation
    # for the moment. It can be optimized.
    t1 = repartition(A,new_rows,new_cols;reuse=true)
    t2 = repartition(b,new_rows;reuse=true)
    @fake_async begin
        B,cacheB = fetch(t1)
        c,cachec = fetch(t2)
        if val_parameter(reuse)
            cache = (cacheB,cachec)
            B,c,cache
        else
            B,c
        end
    end
end

"""
    repartition!(B,c,A,b,cache)

- `B::PSparseMatrix`
- `c::PVector`
- `A::PSparseMatrix`
- `b::PVector`
- `cache`

"""
function repartition!(B::PSparseMatrix,c::PVector,A::PSparseMatrix,b::PVector,cache)
    (cacheB,cachec) = cache
    t1 = repartition!(B,A,cacheB)
    t2 = repartition!(c,b,cachec)
    @fake_async begin
        wait(t1)
        wait(t2)
        B,c
    end
end

function centralize(A::PSparseMatrix)
    m,n = size(A)
    ranks = linear_indices(partition(A))
    rows_trivial = trivial_partition(ranks,m)
    cols_trivial = trivial_partition(ranks,n)
    a_in_main = repartition(A,rows_trivial,cols_trivial) |> fetch
    own_own_values(a_in_main) |> multicast |> getany
end

"""
    psystem(I,J,V,I2,V2,rows,cols;kwargs...)
"""
function psystem(I,J,V,I2,V2,rows,cols;
        subassembled=false,
        assembled=false,
        assemble=true,
        indices = :global,
        restore_ids = true,
        assembly_neighbors_options_rows = (;),
        assembly_neighbors_options_cols = (;),
        assembled_rows = nothing,
        reuse=Val(false)
    )

    # TODO this is just a reference implementation
    # for the moment.
    # It can be optimized to exploit the fact
    # that we want to generate a matrix and a vector

    if assembled_rows === nothing && subassembled
        assembled_rows = map(remove_ghost,rows)
    end

    t1 = psparse(I,J,V,rows,cols;
            subassembled,
            assembled,
            assemble,
            restore_ids,
            assembly_neighbors_options_rows,
            assembly_neighbors_options_cols,
            assembled_rows,
            reuse=true)

    t2 = pvector(I2,V2,rows;
            subassembled,
            assembled,
            assemble,
            restore_ids,
            assembly_neighbors_options_rows,
            assembled_rows,
            reuse=true)

    @fake_async begin
        A,cacheA = fetch(t1)
        b,cacheb = fetch(t2)
        if val_parameter(reuse)
            cache = (cacheA,cacheb)
            A,b,cache
        else
            A,b
        end
    end
end

"""
    psystem!(A,b,V,V2,cache)
"""
function psystem!(A,b,V,V2,cache)
    (cacheA,cacheb) = cache
    t1 = psparse!(A,V,cacheA)
    t2 = pvector!(b,V2,cacheb)
    @fake_async begin
        wait(t1)
        wait(t2)
        (A,b)
    end
end

# Not efficient, just for convenience and debugging purposes
function Base.:\(a::PSparseMatrix,b::PVector)
    m,n = size(a)
    ranks = linear_indices(partition(a))
    rows_trivial = trivial_partition(ranks,m)
    cols_trivial = trivial_partition(ranks,n)
    a_in_main = repartition(a,rows_trivial,cols_trivial) |> fetch
    b_in_main = repartition(b,partition(axes(a_in_main,1))) |> fetch
    @static if VERSION >= v"1.9"
        values = map(\,own_own_values(a_in_main),own_values(b_in_main))
    else
        values = map(\,own_own_values(a_in_main),map(collect,own_values(b_in_main)))
    end
    c_in_main = PVector(values,cols_trivial)
    cols = partition(axes(a,2))
    c = repartition(c_in_main,cols) |> fetch
    c
end

# Not efficient, just for convenience and debugging purposes
struct PLUNew{A,B,C}
    lu_in_main::A
    rows::B
    cols::C
end
function LinearAlgebra.lu(a::PSparseMatrix)
    m,n = size(a)
    ranks = linear_indices(partition(a))
    rows_trivial = trivial_partition(ranks,m)
    cols_trivial = trivial_partition(ranks,n)
    a_in_main = repartition(a,rows_trivial,cols_trivial) |> fetch
    lu_in_main = map_main(lu,own_own_values(a_in_main))
    PLUNew(lu_in_main,axes(a_in_main,1),axes(a_in_main,2))
end
function LinearAlgebra.lu!(b::PLUNew,a::PSparseMatrix)
    rows_trivial = partition(b.rows)
    cols_trivial = partition(b.cols)
    a_in_main = repartition(a,rows_trivial,cols_trivial) |> fetch
    map_main(lu!,b.lu_in_main,own_own_values(a_in_main))
    b
end
function LinearAlgebra.ldiv!(c::PVector,a::PLUNew,b::PVector)
    rows_trivial = partition(a.rows)
    cols_trivial = partition(a.cols)
    b_in_main = repartition(b,rows_trivial) |> fetch
    values = map(partition(c),partition(b_in_main)) do c,b
        similar(c,length(b))
    end
    map_main(ldiv!,values,a.lu_in_main,partition(b_in_main))
    c_in_main = PVector(values,cols_trivial)
    repartition!(c,c_in_main) |> wait
    c
end

function renumber(a::PSparseMatrix;kwargs...)
    row_partition = partition(axes(a,1))
    row_partition_2 = renumber_partition(row_partition;kwargs...)
    col_partition = partition(axes(a,2))
    col_partition_2 = renumber_partition(col_partition;kwargs...)
    renumber(a,row_partition_2,col_partition_2;kwargs...)
end

function renumber(a::PSparseMatrix,row_partition_2,col_partition_2;
    renumber_local_indices=true)
    function setup(oo,oh,ho,hh,rows,cols)
        blocks = split_matrix_blocks(oo,oh,ho,hh)
        row_perm = local_permutation(rows)
        col_perm = local_permutation(cols)
        split_matrix(blocks,row_perm,col_perm)
    end
    if renumber_local_indices
        oo = own_own_values(a)
        oh = own_ghost_values(a)
        ho = ghost_own_values(a)
        hh = ghost_ghost_values(a)
        values = map(setup,oo,oh,ho,hh,row_partition_2,col_partition_2)
        PSparseMatrix(values,row_partition_2,col_partition_2,a.assembled)
    else
        values = partition(a)
        PSparseMatrix(values,row_partition_2,col_partition_2,a.assembled)
    end
end

## Test matrices

# TODO this is deprecated
# Find a replacement in PartitionedSolvers/gallery.jl
function laplace_matrix(nodes_per_dir)
    function is_boundary_node(node_1d,nodes_1d)
        !(node_1d in 1:nodes_1d)
    end
    D = length(nodes_per_dir)
    n = prod(nodes_per_dir)
    node_to_cartesian_node = CartesianIndices(nodes_per_dir)
    cartesian_node_to_node = LinearIndices(nodes_per_dir)
    nnz = (2*D+1)*n
    I = zeros(Int32,nnz)
    J = zeros(Int32,nnz)
    V = zeros(Float64,nnz)
    t = 0
    for node_i in 1:n
        t += 1
        I[t] = node_i
        J[t] = node_i
        V[t] = 2*D
        cartesian_node_i = node_to_cartesian_node[node_i]
        for d in 1:D
            for i in (-1,1)
                inc = ntuple(k->( k==d ? i : 0),Val(D))
                cartesian_node_j = CartesianIndex(Tuple(cartesian_node_i) .+ inc)
                boundary = any(map(is_boundary_node,Tuple(cartesian_node_j),nodes_per_dir))
                if boundary
                    continue
                end
                node_j = cartesian_node_to_node[cartesian_node_j]
                t += 1
                I[t] = node_i
                J[t] = node_j
                V[t] = -1.0
            end
        end
    end
    sparse_matrix(I,J,V,n,n)
end

function laplace_matrix(nodes_per_dir,parts_per_dir,ranks)
    function is_boundary_node(node_1d,nodes_1d)
        !(node_1d in 1:nodes_1d)
    end
    D = length(nodes_per_dir)
    n = prod(nodes_per_dir)
    function setup(nodes)
        node_to_cartesian_node = CartesianIndices(nodes_per_dir)
        cartesian_node_to_node = LinearIndices(nodes_per_dir)
        nnz = (2*D+1)*length(nodes)
        myI = zeros(Int32,nnz)
        myJ = zeros(Int32,nnz)
        myV = zeros(Float64,nnz)
        t = 0
        for node_i in nodes
            t += 1
            myI[t] = node_i
            myJ[t] = node_i
            myV[t] = 2*D
            cartesian_node_i = node_to_cartesian_node[node_i]
            for d in 1:D
                for i in (-1,1)
                    inc = ntuple(k->( k==d ? i : 0),Val(D))
                    cartesian_node_j = CartesianIndex(Tuple(cartesian_node_i) .+ inc)
                    boundary = any(map(is_boundary_node,Tuple(cartesian_node_j),nodes_per_dir))
                    if boundary
                        continue
                    end
                    node_j = cartesian_node_to_node[cartesian_node_j]
                    t += 1
                    myI[t] = node_i
                    myJ[t] = node_j
                    myV[t] = -1.0
                end
            end
        end
        @views myI[1:t],myJ[1:t],myV[1:t]
    end
    node_partition = uniform_partition(ranks,parts_per_dir,nodes_per_dir)
    I,J,V = map(setup,node_partition) |> tuple_of_arrays
    A = psparse(sparse,I,J,V,node_partition,node_partition) |> fetch
end
