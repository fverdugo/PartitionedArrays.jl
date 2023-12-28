
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

"""
    struct PSparseMatrix{V,A,B,C,...}

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

`matrix_partition[i]` contains a (sparse) matrix with the local rows and the
corresponding nonzero columns (the local columns) in the part number `i`.
`eltype(matrix_partition) == V`.
`row_partition[i]` and `col_partition[i]` contain information
about the local, own, and ghost rows and columns respectively in part number `i`.
The types `eltype(row_partition)` and `eltype(col_partition)` implement the
[`AbstractLocalIndices`](@ref) interface.

The rest of fields of this struct and type parameters are private.

# Supertype hierarchy

    PSparseMatrix{V,A,B,C,...} <: AbstractMatrix{T}

with `T=eltype(V)`.
"""
struct PSparseMatrix{V,A,B,C,D,T} <: AbstractMatrix{T}
    matrix_partition::A
    row_partition::B
    col_partition::C
    cache::D
    @doc """
        PSparseMatrix(matrix_partition,row_partition,col_partition)

    Build an instance for [`PSparseMatrix`](@ref) from the underlying fields
    `matrix_partition`, `row_partition`, and `col_partition`.
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
    own_values(a::PSparseMatrix)

Get a vector of matrices containing the own rows and columns
in each part of `a`.

The row and column indices of the returned matrices can be mapped to global
indices, local indices, and owner by using [`own_to_global`](@ref),
[`own_to_local`](@ref), and [`own_to_owner`](@ref), respectively.
"""
function own_values(a::PSparseMatrix)
    map(own_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

"""
    ghost_values(a::PSparseMatrix)

Get a vector of matrices containing the ghost rows and columns
in each part of `a`.

The row and column indices of the returned matrices can be mapped to global
indices, local indices, and owner by using [`ghost_to_global`](@ref),
[`ghost_to_local`](@ref), and [`ghost_to_owner`](@ref), respectively.
"""
function ghost_values(a::PSparseMatrix)
    map(ghost_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
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
    p_sparse_matrix_cache_impl(eltype(matrix_partition),matrix_partition,row_partition,col_partition)
end

function p_sparse_matrix_cache_impl(::Type,matrix_partition,row_partition,col_partition)
    function setup_snd(part,parts_snd,row_indices,col_indices,values)
        local_row_to_owner = local_to_owner(row_indices)
        local_to_global_row = local_to_global(row_indices)
        local_to_global_col = local_to_global(col_indices)
        owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
        ptrs = zeros(Int32,length(parts_snd)+1)
        for (li,lj,v) in nziterator(values)
            owner = local_row_to_owner[li]
            if owner != part
                ptrs[owner_to_i[owner]+1] +=1
            end
        end
        length_to_ptrs!(ptrs)
        k_snd_data = zeros(Int32,ptrs[end]-1)
        gi_snd_data = zeros(Int,ptrs[end]-1)
        gj_snd_data = zeros(Int,ptrs[end]-1)
        for (k,(li,lj,v)) in enumerate(nziterator(values))
            owner = local_row_to_owner[li]
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
        global_to_local_row = global_to_local(row_indices)
        global_to_local_col = global_to_local(col_indices)
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

"""
    assemble!([op,] a::PSparseMatrix) -> Task

Transfer the ghost rows to their owner part
and insert them according with the insertion operation `op` (`+` by default).
It returns a task that produces `a` with updated values. After the transfer,
the source ghost rows are set to zero.
"""
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
    @async begin
        gi_rcv = fetch(t1)
        gj_rcv = fetch(t2)
        v_rcv = fetch(t3)
        map(setup_rcv!,coo_values,gi_rcv,gj_rcv,v_rcv)
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
    PSparseMatrix(matrix_partition,partition(rows),partition(cols))
end

function Base.similar(::Type{<:PSparseMatrix{V}},inds::Tuple{<:PRange,<:PRange}) where V
    rows,cols = inds
    matrix_partition = map(partition(rows),partition(cols)) do row_indices, col_indices
        allocate_local_values(V,row_indices,col_indices)
    end
    PSparseMatrix(matrix_partition,partition(rows),partition(cols))
end

function Base.copy!(a::PSparseMatrix,b::PSparseMatrix)
    @assert size(a) == size(b)
    copyto!(a,b)
end

function Base.copyto!(a::PSparseMatrix,b::PSparseMatrix)
    if partition(axes(a,1)) === partition(axes(b,1)) && partition(axes(a,2)) === partition(axes(b,2))
        map(copy!,partition(a),partition(b))
    elseif matching_own_indices(axes(a,1),axes(b,1)) && matching_own_indices(axes(a,2),axes(b,2))
        map(copy!,own_values(a),own_values(b))
    else
        error("Trying to copy a PSparseMatrix into another one with a different data layout. This case is not implemented yet. It would require communications.")
    end
    a
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

"""
    psparse!([f,]I,J,V,row_partition,col_partition;discover_rows=true,discover_cols=true) -> Task

Crate an instance of [`PSparseMatrix`](@ref) by setting arbitrary entries
from each of the underlying parts. It returns a task that produces the
instance of [`PSparseMatrix`](@ref) allowing latency hiding while performing
the communications needed in its setup.
"""
function psparse!(f,I,J,V,row_partition,col_partition;discover_rows=true,discover_cols=true)
    if discover_rows
        I_owner = find_owner(row_partition,I)
        row_partition = map(union_ghost,row_partition,I,I_owner)
    end
    t = assemble_coo!(I,J,V,row_partition)
    @async begin
        wait(t)
        if discover_cols
            J_owner = find_owner(col_partition,J)
            col_partition = map(union_ghost,col_partition,J,J_owner)
        end
        map(to_local!,I,row_partition)
        map(to_local!,J,col_partition)
        matrix_partition = map(f,I,J,V,row_partition,col_partition)
        PSparseMatrix(matrix_partition,row_partition,col_partition)
    end
end

function psparse!(I,J,V,row_partition,col_partition;kwargs...)
    psparse!(default_local_values,I,J,V,row_partition,col_partition;kwargs...)
end

"""
    psparse(f,row_partition,col_partition)

Build an instance of [`PSparseMatrix`](@ref) from the initialization function
`f` and the partition for rows and columns `row_partition` and `col_partition`.

Equivalent to

    matrix_partition = map(f,row_partition,col_partition)
    PSparseMatrix(matrix_partition,row_partition,col_partition)
"""
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
        owner = part_id(indices)
        owner == destination ? Int(global_length(indices)) : 0
    end
    partition_in_main = variable_partition(n_own,length(PRange(row_partition)))
    I = map(own_to_global,row_partition)
    I_owner = find_owner(partition_in_main,I)
    map(union_ghost,partition_in_main,I,I_owner)
end

function to_trivial_partition(b::PVector,row_partition_in_main)
    destination = 1
    T = eltype(b)
    b_in_main = similar(b,T,PRange(row_partition_in_main))
    fill!(b_in_main,zero(T))
    map(own_values(b),partition(b_in_main),partition(axes(b,1))) do bown,my_b_in_main,indices
        part = part_id(indices)
        if part == destination
            my_b_in_main[own_to_global(indices)] .= bown
        else
            my_b_in_main .= bown
        end
    end
    assemble!(b_in_main) |> wait
    b_in_main
end

function from_trivial_partition!(c::PVector,c_in_main::PVector)
    destination = 1
    consistent!(c_in_main) |> wait
    map(own_values(c),partition(c_in_main),partition(axes(c,1))) do cown, my_c_in_main, indices
        part = part_id(indices)
        if part == destination
            cown .= view(my_c_in_main,own_to_global(indices))
        else
            cown .= my_c_in_main
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
        local_row_to_owner = local_to_owner(row_indices)
        owner = part_id(row_indices)
        local_to_global_row = local_to_global(row_indices)
        local_to_global_col = local_to_global(col_indices)
        for (i,j,v) in nziterator(a)
            if local_row_to_owner[i] == owner
                n += 1
            end
        end
        myI = zeros(Int,n)
        myJ = zeros(Int,n)
        myV = zeros(Ta,n)
        n = 0
        for (i,j,v) in nziterator(a)
            if local_row_to_owner[i] == owner
                n += 1
                myI[n] = local_to_global_row[i]
                myJ[n] = local_to_global_col[j]
                myV[n] = v
            end
        end
        myI,myJ,myV
    end |> tuple_of_arrays
    assemble_coo!(I,J,V,row_partition_in_main) |> wait
    I,J,V = map(partition(axes(a,1)),I,J,V) do row_indices,myI,myJ,myV
        owner = part_id(row_indices)
        if owner == destination
            myI,myJ,myV
        else
            similar(myI,eltype(myI),0),similar(myJ,eltype(myJ),0),similar(myV,eltype(myV),0)
        end
    end |> tuple_of_arrays
    values = map(I,J,V,row_partition_in_main,col_partition_in_main) do myI,myJ,myV,row_indices,col_indices
        m = local_length(row_indices)
        n = local_length(col_indices)
        compresscoo(M,myI,myJ,myV,m,n)
    end
    PSparseMatrix(values,row_partition_in_main,col_partition_in_main)
end

# Not efficient, just for convenience and debugging purposes
function Base.:\(a::PSparseMatrix,b::PVector)
    Ta = eltype(a)
    Tb = eltype(b)
    T = typeof(one(Ta)\one(Tb)+one(Ta)\one(Tb))
    c = PVector{Vector{T}}(undef,partition(axes(a,2)))
    fill!(c,zero(T))
    a_in_main = to_trivial_partition(a)
    b_in_main = to_trivial_partition(b,partition(axes(a_in_main,1)))
    c_in_main = to_trivial_partition(c,partition(axes(a_in_main,2)))
    map_main(partition(c_in_main),partition(a_in_main),partition(b_in_main)) do myc, mya, myb
        myc .= mya\myb
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


# New stuff

struct SplitMatrixBlocks{A,B,C,D}
    own_own::A
    own_ghost::B
    ghost_own::C
    ghost_ghost::D
end
blocktype(::Type{SplitMatrixBlocks{A,B,C,D}}) where {A,B,C,D} = Union{A,B,C,D}
blocktype(::SplitMatrixBlocks{A,B,C,D}) where {A,B,C,D} = Union{A,B,C,D}
struct SplitMatrixUniformBlocks{A}
    own_own::A
    own_ghost::A
    ghost_own::A
    ghost_ghost::A
end
blocktype(::Type{SplitMatrixUniformBlocks{A}}) where A = A
blocktype(::SplitMatrixUniformBlocks{A}) where A = A
function split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    SplitMatrixBlocks(own_own,own_ghost,ghost_own,ghost_ghost)
end
function split_matrix_blocks(own_own::A,own_ghost::A,ghost_own::A,ghost_ghost::A) where A
    SplitMatrixUniformBlocks(own_own,own_ghost,ghost_own,ghost_ghost)
end

struct SplitMatrixGlobal{A,T} <: AbstractMatrix{T}
    blocks::A
    function SplitMatrixGlobal(blocks)
        T = eltype(blocks.own_own)
        A = typeof(blocks)
        new{A,T}(blocks)
    end
end
blocktype(::Type{<:SplitMatrixGlobal{A}}) where A = blocktype(A)
blocktype(::SplitMatrixGlobal{A}) where A = blocktype(A)
Base.size(a::SplitMatrixGlobal) = size(a.blocks.own_own)
Base.IndexStyle(::Type{<:SplitMatrixGlobal}) = IndexCartesian()
# This one is not supposed to be used in practice, maybe for debugging only
function Base.getindex(a::SplitMatrixGlobal,i::Int,j::Int)
    rp = row_predicate(i)
    cp = col_predicate(j)
    T = eltype(a)
    v = zero(T)
    v += a.blocks.own_own[i,j]
    v += a.blocks.own_ghost[i,j]
    v += a.blocks.ghost_own[i,j]
    v += a.blocks.ghost_ghost[i,j]
    v
end
function replace_blocks(A::SplitMatrixGlobal,blocks)
    SplitMatrixGlobal(blocks)
end

function own_own_values(values::SplitMatrixGlobal,indices_rows,indices_cols)
    values.blocks.own_own
end
function own_ghost_values(values::SplitMatrixGlobal,indices_rows,indices_cols)
    values.blocks.own_ghost
end
function ghost_own_values(values::SplitMatrixGlobal,indices_rows,indices_cols)
    values.blocks.ghost_own
end
function ghost_ghost_values(values::SplitMatrixGlobal,indices_rows,indices_cols)
    values.blocks.ghost_ghost
end

Base.similar(a::SplitMatrixGlobal) = similar(a,eltype(a))
function Base.similar(a::SplitMatrixGlobal,::Type{T}) where T
    own_own = similar(a.blocks.own_own,T)
    own_ghost = similar(a.blocks.own_ghost,T)
    ghost_own = similar(a.blocks.ghost_own,T)
    ghost_ghost = similar(a.blocks.ghost_ghost,T)
    blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    SplitMatrixLocal(blocks,a.row_permutation,a.col_permutation)
end

function Base.copy!(a::SplitMatrixGlobal,b::SplitMatrixGlobal)
    copy!(a.blocks.own_own,b.blocks.own_own)
    copy!(a.blocks.own_ghost,b.blocks.own_ghost)
    copy!(a.blocks.ghost_own,b.blocks.ghost_own)
    copy!(a.blocks.ghost_ghost,b.blocks.ghost_ghost)
    a
end
function Base.copyto!(a::SplitMatrixGlobal,b::SplitMatrixGlobal)
    copyto!(a.blocks.own_own,b.blokcs.own_own)
    copyto!(a.blocks.own_ghost,b.blokcs.own_ghost)
    copyto!(a.blocks.ghost_own,b.blokcs.ghost_own)
    copyto!(a.blocks.ghost_ghost,b.blokcs.ghost_ghost)
    a
end

function split_globally(coo::SparseMatrixCOO,rows,cols)
    global_to_own_row = global_to_own(rows)
    global_to_own_col = global_to_own(cols)
    n_own_own = 0
    n_own_ghost = 0
    n_ghost_own = 0
    n_ghost_ghost = 0
    for p in 1:nnz(coo)
        gi = coo.I[p]
        gj = coo.J[p]
        i = global_to_own_row[gi]
        j = global_to_own_row[gj]
        if i != 0 && j != 0
            n_own_own += 1
        elseif i != 0 && j==0
            n_own_ghost += 1
        elseif i == 0 && j!= 0
            n_ghost_own += 1
        else
            n_ghost_ghost += 1
        end
    end
    Tv = eltype(coo)
    m,n = size(coo)
    own_own = similar_coo(coo,size(coo),n_own_own)
    own_ghost = similar_coo(coo,size(coo),n_own_ghost)
    ghost_own = similar_coo(coo,size(coo),n_ghost_own)
    ghost_ghost = similar_coo(coo,size(coo),n_ghost_ghost)
    n_own_own = 0
    n_own_ghost = 0
    n_ghost_own = 0
    n_ghost_ghost = 0
    for p in 1:nnz(coo)
        gi = coo.I[p]
        gj = coo.J[p]
        gv = coo.V[p]
        i = global_to_own_row[gi]
        j = global_to_own_row[gj]
        if i != 0 && j != 0
            n_own_own += 1
            own_own.I[n_own_own] = gi
            own_own.J[n_own_own] = gj
            own_own.V[n_own_own] = gv
        elseif i != 0 && j==0
            n_own_ghost += 1
            own_ghost.I[n_own_ghost] = gi
            own_ghost.J[n_own_ghost] = gj
            own_ghost.V[n_own_ghost] = gv
        elseif i == 0 && j!= 0
            n_ghost_own += 1
            ghost_own.I[n_ghost_own] = gi
            ghost_own.J[n_ghost_own] = gj
            ghost_own.V[n_ghost_own] = gv
        else
            n_ghost_ghost += 1
            ghost_ghost.I[n_ghost_ghost] = gi
            ghost_ghost.J[n_ghost_ghost] = gj
            ghost_ghost.V[n_ghost_ghost] = gv
        end
    end
    blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    B = SplitMatrixGlobal(blocks)
    B
end

function split_globally!(B::SplitMatrixGlobal,coo::SparseMatrixCOO,rows,cols)
    @assert blocktype(B.blocks) <: SparseMatrixCOO
    global_to_own_row = global_to_own(rows)
    global_to_own_col = global_to_own(cols)
    n_own_own = 0
    n_own_ghost = 0
    n_ghost_own = 0
    n_ghost_ghost = 0
    own_own = B.blocks.own_own
    own_ghost = B.blocks.own_ghost
    ghost_own = B.blocks.ghost_own
    ghost_ghost = B.blocks.ghost_ghost
    for p in 1:nnz(coo)
        gi = coo.I[p]
        gj = coo.J[p]
        gv = coo.V[p]
        i = global_to_own_row[gi]
        j = global_to_own_row[gj]
        if i != 0 && j != 0
            n_own_own += 1
            own_own.V[n_own_own] = gv
        elseif i != 0 && j==0
            n_own_ghost += 1
            own_ghost.V[n_own_ghost] = gv
        elseif i == 0 && j!= 0
            n_ghost_own += 1
            ghost_own.V[n_ghost_own] = gv
        else
            n_ghost_ghost += 1
            ghost_ghost.V[n_ghost_ghost] = gv
        end
    end
    B
end

struct SplitMatrixLocal{A,B,C,T} <: AbstractMatrix{T}
    blocks::A
    row_permutation::B
    col_permutation::C
    function SplitMatrixLocal(blocks,row_permutation,col_permutation)
        T = eltype(blocks.own_own)
        A = typeof(blocks)
        B = typeof(row_permutation)
        C = typeof(col_permutation)
        new{A,B,C,T}(blocks,row_permutation,col_permutation)
    end
end
blocktype(::Type{<:SplitMatrixLocal{A}}) where A = blocktype(A)
blocktype(::SplitMatrixLocal{A}) where A = blocktype(A)
Base.size(a::SplitMatrixLocal) = (length(a.row_permutation),length(a.col_permutation))
Base.IndexStyle(::Type{<:SplitMatrixLocal}) = IndexCartesian()
function Base.getindex(a::SplitMatrixLocal,i::Int,j::Int)
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
function replace_blocks(A::SplitMatrixLocal,blocks)
    SplitMatrixLocal(blocks,A.row_permutation,A.col_permutation)
end

function own_own_values(values::SplitMatrixLocal,indices_rows,indices_cols)
    values.blocks.own_own
end
function own_ghost_values(values::SplitMatrixLocal,indices_rows,indices_cols)
    values.blocks.own_ghost
end
function ghost_own_values(values::SplitMatrixLocal,indices_rows,indices_cols)
    values.blocks.ghost_own
end
function ghost_ghost_values(values::SplitMatrixLocal,indices_rows,indices_cols)
    values.blocks.ghost_ghost
end

Base.similar(a::SplitMatrixLocal) = similar(a,eltype(a))
function Base.similar(a::SplitMatrixLocal,::Type{T}) where T
    own_own = similar(a.blocks.own_own,T)
    own_ghost = similar(a.blocks.own_ghost,T)
    ghost_own = similar(a.blocks.ghost_own,T)
    ghost_ghost = similar(a.blocks.ghost_ghost,T)
    blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    SplitMatrixLocal(blocks,a.row_permutation,a.col_permutation)
end

function Base.copy!(a::SplitMatrixLocal,b::SplitMatrixLocal)
    copy!(a.blocks.own_own,b.blocks.own_own)
    copy!(a.blocks.own_ghost,b.blocks.own_ghost)
    copy!(a.blocks.ghost_own,b.blocks.ghost_own)
    copy!(a.blocks.ghost_ghost,b.blocks.ghost_ghost)
    a
end
function Base.copyto!(a::SplitMatrixLocal,b::SplitMatrixLocal)
    copyto!(a.blocks.own_own,b.blokcs.own_own)
    copyto!(a.blocks.own_ghost,b.blokcs.own_ghost)
    copyto!(a.blocks.ghost_own,b.blokcs.ghost_own)
    copyto!(a.blocks.ghost_ghost,b.blokcs.ghost_ghost)
    a
end

function LinearAlgebra.fillstored!(a::SplitMatrixLocal,v)
    LinearAlgebra.fillstored!(a.blocks.own_own,v)
    LinearAlgebra.fillstored!(a.blocks.own_ghost,v)
    LinearAlgebra.fillstored!(a.blocks.ghost_own,v)
    LinearAlgebra.fillstored!(a.blocks.ghost_ghost,v)
    a
end

function to_split_csc(A::SplitMatrixLocal)
    own_own = to_csc(A.blocks.own_own)
    own_ghost = to_csc(A.blocks.own_ghost)
    ghost_own = to_csc(A.blocks.ghost_own)
    ghost_ghost = to_csc(A.blocks.ghost_ghost)
    blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    SplitMatrixLocal(blocks,A.row_permutation,A.col_permutation)
end

function to_split_csc!(B::SplitMatrixLocal,A::SplitMatrixLocal)
    to_csc!(B.blocks.own_own,A.blocks.own_own)
    to_csc!(B.blocks.own_ghost,A.blocks.own_ghost)
    to_csc!(B.blocks.ghost_own,A.blocks.ghost_own)
    to_csc!(B.blocks.ghost_ghost,A.blocks.ghost_ghost)
    B
end

function split_locally(coo::SparseMatrixCOO,rows,cols)
    I,J,V = findnz(coo)
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
    for p in 1:nnz(coo)
        i = I[p]
        j = J[p]
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
    own_own = similar_coo(coo,(n_own_rows,n_own_cols),n_own_own)
    own_ghost = similar_coo(coo,(n_own_rows,n_ghost_cols),n_own_ghost)
    ghost_own = similar_coo(coo,(n_ghost_rows,n_own_cols),n_ghost_own)
    ghost_ghost = similar_coo(coo,(n_ghost_rows,n_ghost_cols),n_ghost_ghost)
    for p in 1:nnz(coo)
        i = I[p]
        j = J[p]
        v = V[p]
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
    blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
    B = SplitMatrixLocal(blocks,local_permutation(rows),local_permutation(cols))
end

function split_locally!(B::SplitMatrixLocal,coo::SparseMatrixCOO,rows,cols)
    @assert blocktype(B) <: SparseMatrixCOO
    I,J,V = findnz(coo)
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
    own_own = B.blocks.own_own
    own_ghost = B.blocks.own_ghost
    ghost_own = B.blocks.ghost_own
    ghost_ghost = B.blocks.ghost_ghost
    for p in 1:nnz(coo)
        i = I[p]
        j = J[p]
        v = V[p]
        ip = rows_perm[i]
        jp = cols_perm[j]
        if ip <= n_own_rows && jp <= n_own_cols
            n_own_own += 1
            own_own.V[n_own_own] = v
        elseif ip <= n_own_rows
            n_own_ghost += 1
            own_ghost.V[n_own_ghost] = v
        elseif jp <= n_own_cols
            n_ghost_own += 1
            ghost_own.V[n_ghost_own] = v
        else
            n_ghost_ghost += 1
            ghost_ghost.V[n_ghost_ghost] = v
        end
    end
    B
end

struct Disassembled end
struct Subassembled end
struct Assembled end
struct Consistent end

struct PSparseMatrixNew{A,V,B,C,D,E,T} <: AbstractMatrix{T}
    style::A
    matrix_partition::B
    row_partition::C
    col_partition::D
    cache::E
    function PSparseMatrixNew(
        style,matrix_partition,row_partition,col_partition,cache=nothing)
        V = eltype(matrix_partition)
        T = eltype(V)
        A = typeof(style)
        B = typeof(matrix_partition)
        C = typeof(row_partition)
        D = typeof(col_partition)
        E = typeof(cache)
        new{A,V,B,C,D,E,T}(style,matrix_partition,row_partition,col_partition,cache)
    end
end
partition(a::PSparseMatrixNew) = a.matrix_partition
Base.axes(a::PSparseMatrixNew) = (PRange(a.row_partition),PRange(a.col_partition))
Base.size(a::PSparseMatrixNew) = map(length,axes(a))
Base.IndexStyle(::Type{<:PSparseMatrixNew}) = IndexCartesian()
function Base.getindex(a::PSparseMatrixNew,gi::Int,gj::Int)
    scalar_indexing_action(a)
end
function Base.setindex!(a::PSparseMatrixNew,v,gi::Int,gj::Int)
    scalar_indexing_action(a)
end

function Base.show(io::IO,k::MIME"text/plain",data::PSparseMatrixNew)
    T = eltype(partition(data))
    m,n = size(data)
    np = length(partition(data))
    style = assembly_style(data)
    style_to_txt(::Disassembled) = "disassembled"
    style_to_txt(::Subassembled) = "subassembled"
    style_to_txt(::Assembled) = "assembled"
    style_txt = style_to_txt(style)
    map_main(partition(data)) do values
        println(io,"$(m)×$(n) $style_txt PSparseMatrixNew partitioned into $np parts of type $T")
    end
end

function assembly_style(A)
    A.style
end

function assembly_style(::Type{<:PSparseMatrixNew{T}}) where T
    T()
end

function replace_matrix_partition(A,values,cache=nothing)
    rows = partition(axes(A,1))
    cols = partition(axes(A,2))
    style = assembly_style(A)
    PSparseMatrixNew(style,values,rows,cols,cache)
end

function replace_cache(A,cache)
    values = partition(A)
    rows = partition(axes(A,1))
    cols = partition(axes(A,2))
    style = assembly_style(A)
    PSparseMatrixNew(style,values,rows,cols,cache)
end

function own_own_values(a::PSparseMatrixNew)
    map(own_own_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end
function own_ghost_values(a::PSparseMatrixNew)
    map(own_ghost_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end
function ghost_own_values(a::PSparseMatrixNew)
    map(ghost_own_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end
function ghost_ghost_values(a::PSparseMatrixNew)
    map(ghost_ghost_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

function split_values(A::PSparseMatrixNew{Disassembled})
    values = map(split_globally,partition(A),partition(axes(A,1)),partition(axes(A,2)))
    replace_matrix_partition(A,values)
end

function split_values!(B,A::PSparseMatrixNew{Disassembled})
    rows = partition(axes(A,1))
    cols = partition(axes(A,2))
    map(split_globally!,partition(B),partition(A),rows,cols)
    B
end

function split_values(A::PSparseMatrixNew)
    values = map(split_locally,partition(A),partition(axes(A,1)),partition(axes(A,2)))
    replace_matrix_partition(A,values)
end

function split_values!(B,A::PSparseMatrixNew)
    rows = partition(axes(A,1))
    cols = partition(axes(A,2))
    map(split_locally!,partition(B),partition(A),rows,cols)
    B
end

"""
"""
function psparse_coo(args...;style=Assembled())
    psparse_coo(style,args...)
end

"""
"""
function psparse_coo! end

function psparse_coo(::Disassembled,I,J,V,rows,cols)
    function local_format(I,J,V,rows,cols)
        m = global_length(rows)
        n = global_length(cols)
        sparse_coo(I,J,V,m,n)
    end
    values = map(local_format,I,J,V,rows,cols)
    rows_da = map(remove_ghost,rows)
    cols_da = map(remove_ghost,cols)
    B = PSparseMatrixNew(Disassembled(),values,rows_da,cols_da)
    @async B
end
function psparse_coo!(B::PSparseMatrixNew{Disassembled},V)
    map(sparse_coo!,partition(B),V)
    @async B
end

function psparse_coo(::Subassembled,args...)
    A_da = psparse_coo(Disassembled(),args...) |> fetch
    t = subassemble(A_da)
    @async begin
        A_sa = fetch(t)
        cache = Ref{Any}((A_da,A_sa))
        B = replace_cache(A_sa,cache)
        B
    end
end
function psparse_coo!(B::PSparseMatrixNew{Subassembled},V)
    (A_da,A_sa) = B.cache[]
    psparse_coo!(A_da,V) |> wait
    t = subassemble!(A_sa,A_da)
    @async begin
        wait(t)
        B
    end
end

function psparse_coo(::Assembled,args...)
    A_sa = psparse_coo(Subassembled(),args...) |> fetch
    t = assemble(A_sa)
    @async begin
        A_fa = fetch(t)
        cache = Ref{Any}((A_sa,A_fa))
        B = replace_cache(A_sa,cache)
        B
    end
end
function psparse_coo!(B::PSparseMatrixNew{Assembled},V)
    (A_sa,A_fa) = B.cache[]
    psparse_coo!(A_sa,V) |> wait
    t = assemble!(A_fa,A_sa)
    @async begin
        wait(t)
        B
    end
end

"""
"""
function psparse_split_coo(args...;style=Assembled())
    psparse_split_coo(style,args...)
end

"""
"""
function psparse_split_coo! end

function psparse_split_coo(::Disassembled,I,J,V,rows,cols)
    A_coo = psparse_coo(Disassembled(),I,J,V,rows,cols) |> fetch
    A_split_coo = split_values(A_coo)
    cache = Ref{Any}((A_coo,A_split_coo))
    B = replace_cache(A_split_coo,cache)
    @async B
end
function psparse_split_coo!(B::PSparseMatrixNew{Disassembled},V)
    (A_coo,A_split_coo) = B.cache[]
    psparse_coo!(A_coo,V) |> wait
    split_values!(A_split_coo,A_coo)
    @async B
end

function psparse_split_coo(::Subassembled,I,J,V,rows,cols)
    A_da = psparse_split_coo(Disassembled(),I,J,V,rows,cols) |> fetch
    t = subassemble(A_da)
    @async begin
        A_sa = fetch(t)
        cache = Ref{Any}((A_da,A_sa))
        B = replace_cache(A_sa,cache)
        B
    end
end
function psparse_split_coo!(B::PSparseMatrixNew{Subassembled},V)
    (A_da,A_sa) = B.cache[]
    psparse_split_coo!(A_da,V) |> wait
    t = subassemble!(A_sa,A_da)
    @async begin
        wait(t)
        B
    end
end

function psparse_split_coo(::Assembled,I,J,V,rows,cols)
    A_sa = psparse_split_coo(Subassembled(),I,J,V,rows,cols) |> fetch
    t = assemble(A_sa)
    @async begin
        A_fa = fetch(t)
        cache = Ref{Any}((A_sa,A_fa))
        B = replace_cache(A_fa,cache)
        B
    end
end
function psparse_split_coo!(B::PSparseMatrixNew{Assembled},V)
    (A_sa,A_fa) = B.cache[]
    psparse_split_coo!(A_sa,V) |> wait
    t = assemble!(A_fa,A_sa)
    @async begin
        wait(t)
        B
    end
end

"""
"""
function psparse_csc(args...;style=Assembled())
    psparse_csc(style,args...)
end

"""
"""
function psparse_csc! end

function psparse_csc(::Disassembled,I,J,V,rows,cols)
    error("Not implemented since it makes little sense in practice")
end
function psparse_csc!(B::PSparseMatrixNew{Disassembled},V)
    error("Not implemented since it makes little sense in practice")
end

function psparse_csc(::Subassembled,I,J,V,rows,cols)
    t = psparse_coo(Subassembled(),I,J,V,rows,cols)
    @async begin
        A_sa = fetch(t)
        values = map(to_csc,partition(A_sa))
        cache = Ref{Any}(A_sa)
        B = replace_matrix_partition(A_sa,values,cache)
        B
    end
end
function psparse_csc!(B::PSparseMatrixNew{Subassembled},V)
    A_sa = B.cache[]
    t = psparse_coo!(A_sa,V)
    @async begin
        wait(t)
        map(to_csc!,partition(B),partition(A_sa))
        B
    end
end

function psparse_csc(::Assembled,I,J,V,rows,cols)
    A_sa = psparse_coo(Subassembled(),I,J,V,rows,cols) |> fetch
    # TODO we can save some communication by compressing
    # ghost rows before assembling
    t = assemble(A_sa)
    @async begin
        A_fa = fetch(t)
        values = map(to_csc,partition(A_fa))
        cache = Ref{Any}((A_sa,A_fa))
        B = replace_matrix_partition(A_fa,values,cache)
        B
    end
end
function psparse_csc!(B::PSparseMatrixNew{Assembled},V)
    (A_sa,A_fa) = B.cache[]
    psparse_coo!(A_sa,V) |> wait
    t = assemble!(A_fa,A_sa)
    @async begin
        wait(t)
        map(to_csc!,partition(B),partition(A_fa))
        B
    end
end

"""
"""
function psparse_split_csc(args...;style=Assembled())
    psparse_split_csc(style,args...)
end

"""
"""
function psparse_split_csc! end

function psparse_split_csc(::Disassembled,I,J,V,rows,cols)
    error("Not implemented since it makes little sense in practice")
end
function psparse_split_csc!(B::PSparseMatrixNew{Disassembled},V)
    error("Not implemented since it makes little sense in practice")
end

function psparse_split_csc(::Subassembled,I,J,V,rows,cols)
    t = psparse_split_coo(Subassembled(),I,J,V,rows,cols)
    @async begin
        A_sa = fetch(t)
        values = map(to_split_csc,partition(A_sa))
        cache = Ref{Any}(A_sa)
        B = replace_matrix_partition(A_sa,values,cache)
        B
    end
end
function psparse_split_csc!(B::PSparseMatrixNew{Subassembled},V)
    A_sa = B.cache[]
    t = psparse_split_coo!(A_sa,V)
    @async begin
        wait(t)
        map(to_split_csc!,partition(B),partition(A_sa))
        B
    end
end

function psparse_split_csc(::Assembled,I,J,V,rows,cols)
    A_sa = psparse_split_coo(Subassembled(),I,J,V,rows,cols) |> fetch
    # TODO we can save some communication by compressing
    # ghost rows before assembling
    t = assemble(A_sa)
    @async begin
        A_fa = fetch(t)
        values = map(to_split_csc,partition(A_fa))
        cache = Ref{Any}((A_sa,A_fa))
        B = replace_matrix_partition(A_fa,values,cache)
        B
    end
end
function psparse_split_csc!(B::PSparseMatrixNew{Assembled},V)
    (A_sa,A_fa) = B.cache[]
    psparse_split_coo!(A_sa,V) |> wait
    t = assemble!(A_fa,A_sa)
    @async begin
        wait(t)
        map(to_split_csc!,partition(B),partition(A_fa))
        B
    end
end

function subassemble(A::PSparseMatrixNew;exchange_graph_options=(;))
    @assert assembly_style(A) == Disassembled()
    subassemble_impl(A,eltype(partition(A)),exchange_graph_options)
end

function subassemble!(B,A::PSparseMatrixNew;exchange_graph_options=(;))
    @assert assembly_style(A) == Disassembled()
    subassemble_impl!(B,A,eltype(partition(A)),exchange_graph_options)
end

function subassemble_impl(A::PSparseMatrixNew,::Type{<:SparseMatrixCOO},exchange_graph_options)
    rows_da = partition(axes(A,1))
    cols_da = partition(axes(A,2))
    I = map(i->i.I,partition(A))
    J = map(i->i.J,partition(A))
    V = map(i->i.V,partition(A))
    I_owner = find_owner(rows_da,I)
    J_owner = find_owner(cols_da,J)
    rows_sa = map(union_ghost,rows_da,I,I_owner)
    cols_sa = map(union_ghost,cols_da,J,J_owner)
    function setup_matrix_partition(A,rows_sa,cols_sa)
        map_global_to_local!(A.I,rows_sa)
        map_global_to_local!(A.J,cols_sa)
        m = local_length(rows_sa)
        n = local_length(cols_sa)
        SparseMatrixCOO(A.I,A.J,A.V,m,n)
    end
    values_sa = map(setup_matrix_partition,partition(A),rows_sa,cols_sa)
    assembly_neighbors(rows_sa;exchange_graph_options...)
    @async PSparseMatrixNew(Subassembled(),values_sa,rows_sa,cols_sa)
end

function subassemble_impl!(B,A::PSparseMatrixNew,::Type{<:SparseMatrixCOO},exchange_graph_options)
    function setup_matrix_partition(A,B)
        copy!(B.V,A.V)
    end
    map(setup_matrix_partition,partition(B),partition(A))
    @async B
end

function subassemble_impl(A::PSparseMatrixNew,::Type{<:SplitMatrixGlobal},exchange_graph_options)
    function setup_matrix_partition(A,rows_sa,cols_sa)
        map_global_to_own!(A.blocks.own_own.I,rows_sa)
        map_global_to_own!(A.blocks.own_own.J,cols_sa)
        map_global_to_own!(A.blocks.own_ghost.I,rows_sa)
        map_global_to_ghost!(A.blocks.own_ghost.J,cols_sa)
        map_global_to_ghost!(A.blocks.ghost_own.I,rows_sa)
        map_global_to_own!(A.blocks.ghost_own.J,cols_sa)
        map_global_to_ghost!(A.blocks.ghost_ghost.I,rows_sa)
        map_global_to_ghost!(A.blocks.ghost_ghost.J,cols_sa)
        n_own_rows = own_length(rows_sa)
        n_own_cols = own_length(cols_sa)
        n_ghost_rows = ghost_length(rows_sa)
        n_ghost_cols = ghost_length(cols_sa)
        own_own = sparse_coo(findnz(A.blocks.own_own)...,n_own_rows,n_own_cols)
        own_ghost = sparse_coo(findnz(A.blocks.own_ghost)...,n_own_rows,n_ghost_cols)
        ghost_own = sparse_coo(findnz(A.blocks.ghost_own)...,n_ghost_rows,n_own_cols)
        ghost_ghost = sparse_coo(findnz(A.blocks.ghost_ghost)...,n_ghost_rows,n_ghost_cols)
        blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
        SplitMatrixLocal(blocks,local_permutation(rows_sa),local_permutation(cols_sa))
    end
    @assert blocktype(eltype(partition(A))) <: SparseMatrixCOO
    rows_da = partition(axes(A,1))
    cols_da = partition(axes(A,2))
    # TODO vcat can be avoided
    I = map(i->vcat(i.blocks.ghost_own.I,i.blocks.ghost_ghost.I),partition(A))
    J = map(i->vcat(i.blocks.own_ghost.J,i.blocks.ghost_ghost.J),partition(A))
    I_owner = find_owner(rows_da,I)
    J_owner = find_owner(cols_da,J)
    rows_sa = map(union_ghost,rows_da,I,I_owner)
    cols_sa = map(union_ghost,cols_da,J,J_owner)
    values_sa = map(setup_matrix_partition,partition(A),rows_sa,cols_sa)
    assembly_neighbors(rows_sa;exchange_graph_options...)
    @async PSparseMatrixNew(Subassembled(),values_sa,rows_sa,cols_sa)
end

function subassemble_impl!(B,A::PSparseMatrixNew,::Type{<:SplitMatrixGlobal},exchange_graph_options)
    @assert blocktype(eltype(partition(A))) <: SparseMatrixCOO
    function setup_matrix_partition(A,B)
        copy!(B.blocks.own_own.V,A.blocks.own_own.V)
        copy!(B.blocks.own_ghost.V,A.blocks.own_ghost.V)
        copy!(B.blocks.ghost_own.V,A.blocks.ghost_own.V)
        copy!(B.blocks.ghost_ghost.V,A.blocks.ghost_ghost.V)
    end
    map(setup_matrix_partition,partition(B),partition(A))
    @async B
end

function assemble(A::PSparseMatrixNew;exchange_graph_options=(;))
    @assert assembly_style(A) == Subassembled()
    psparse_assemble_impl(A,eltype(partition(A)),exchange_graph_options)
end

function assemble!(B::PSparseMatrixNew,A::PSparseMatrixNew;exchange_graph_options=(;))
    @assert assembly_style(A) == Subassembled()
    psparse_assemble_impl!(B,A,eltype(partition(A)),exchange_graph_options)
end

function psparse_assemble_impl(A,::Type{<:SplitMatrixLocal},exchange_graph_options)
    @assert blocktype(eltype(partition(A))) <: SparseMatrixCOO
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
        I_rcv_data = cache_rcv.I_rcv.data
        J_rcv_data = cache_rcv.J_rcv.data
        V_rcv_data = cache_rcv.V_rcv.data
        k_rcv_data = cache_rcv.k_rcv.data
        global_to_own_col = global_to_own(cols_sa)
        is_ghost = findall(j->global_to_own_col[j]==0,J_rcv_data)
        is_own = findall(j->global_to_own_col[j]!=0,J_rcv_data)
        I_rcv_own = I_rcv_data[is_own]
        J_rcv_own = J_rcv_data[is_own]
        V_rcv_own = V_rcv_data[is_own]
        I_rcv_ghost = I_rcv_data[is_ghost]
        J_rcv_ghost = J_rcv_data[is_ghost]
        V_rcv_ghost = V_rcv_data[is_ghost]
        # After this col ids in own_ghost triplet remain global
        map_global_to_own!(I_rcv_own,rows_sa)
        map_global_to_own!(J_rcv_own,cols_sa)
        map_global_to_own!(I_rcv_ghost,rows_sa)
        map_ghost_to_global!(A.blocks.own_ghost.J,cols_sa)
        kini = nnz(A.blocks.own_own) + 1
        kend = kini + length(V_rcv_own) - 1
        own_own_I = vcat(A.blocks.own_own.I,I_rcv_own)
        own_own_J = vcat(A.blocks.own_own.J,J_rcv_own)
        own_own_V = vcat(A.blocks.own_own.V,V_rcv_own)
        own_own_triplet = (own_own_I,own_own_J,own_own_V)
        k_rcv_data[is_own] = kini:kend
        kini = kend + 1
        kend = kini + length(V_rcv_ghost) - 1
        own_ghost_I = vcat(A.blocks.own_ghost.I,I_rcv_ghost)
        own_ghost_J = vcat(A.blocks.own_ghost.J,J_rcv_ghost)
        own_ghost_V = vcat(A.blocks.own_ghost.V,V_rcv_ghost)
        own_ghost_triplet = (own_ghost_I,own_ghost_J,own_ghost_V)
        k_rcv_data[is_ghost] = kini:kend
        map_global_to_ghost!(A.blocks.own_ghost.J,cols_sa)
        triplets = (own_own_triplet,own_ghost_triplet)
        triplets, own_ghost_J
    end
    function finalize_values(A,rows_fa,cols_fa,cache_snd,cache_rcv,triplets)
        (own_own_triplet,own_ghost_triplet) = triplets
        n_own_rows = own_length(rows_fa)
        n_own_cols = own_length(cols_fa)
        n_ghost_rows = ghost_length(rows_fa)
        n_ghost_cols = ghost_length(cols_fa)
        own_own = sparse_coo(own_own_triplet...,n_own_rows,n_own_cols)
        map_global_to_ghost!(own_ghost_triplet[2],cols_fa)
        own_ghost = sparse_coo(own_ghost_triplet...,n_own_rows,n_ghost_cols)
        ghost_own = similar_coo(own_own,(0,own_length(cols_fa)),0)
        ghost_ghost = similar_coo(own_own,(0,own_length(cols_fa)),0)
        blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
        values = SplitMatrixLocal(blocks,local_permutation(rows_fa),local_permutation(rows_fa))
        cache = (;cache_snd...,cache_rcv...)
        values, cache
    end
    rows_sa = partition(axes(A,1))
    cols_sa = partition(axes(A,2))
    rows = map(remove_ghost,rows_sa)
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
    @async begin
        I_rcv = fetch(t_I)
        J_rcv = fetch(t_J)
        V_rcv = fetch(t_V)
        cache_rcv = map(setup_cache_rcv,I_rcv,J_rcv,V_rcv,parts_rcv)
        triplets,J = map(setup_own_triplets,partition(A),cache_rcv,rows_sa,cols_sa) |> tuple_of_arrays
        J_owner = find_owner(cols_sa,J)
        rows_fa = rows
        cols_fa = map(union_ghost,cols,J,J_owner)
        assembly_neighbors(cols_fa;exchange_graph_options...)
        vals_fa, cache = map(finalize_values,partition(A),rows_fa,cols_fa,cache_snd,cache_rcv,triplets) |> tuple_of_arrays
        PSparseMatrixNew(Assembled(),vals_fa,rows_fa,cols_fa,cache)
    end
end

function psparse_assemble_impl!(B,A,::Type{<:SplitMatrixLocal},exchange_graph_options)
    @assert blocktype(eltype(partition(A))) <: SparseMatrixCOO
    function setup_snd(A,cache)
        A_ghost_own   = A.blocks.ghost_own
        A_ghost_ghost = A.blocks.ghost_ghost
        nnz_ghost_own = nnz(A_ghost_own)
        V_snd_data = cache.V_snd.data
        k_snd_data = cache.k_snd.data
        for p in 1:length(k_snd_data)
            k = k_snd_data[p]
            if k <= nnz_ghost_own
                v = A_ghost_own.V[k]
            else
                v = A_ghost_ghost.V[k-nnz_ghost_own]
            end
            V_snd_data[p] = v
        end
    end
    function setup_rcv(B,cache)
        B_own_own   = B.blocks.own_own
        B_own_ghost = B.blocks.own_ghost
        nnz_own_own = nnz(B_own_own)
        V_rcv_data = cache.V_rcv.data
        k_rcv_data = cache.k_rcv.data
        for p in 1:length(k_rcv_data)
            k = k_rcv_data[p]
            v = V_rcv_data[p]
            if k <= nnz_own_own
                B_own_own.V[k]
            else
                B_own_ghost.V[k-nnz_own_own]
            end
        end
    end
    cache = B.cache
    map(setup_snd,partition(A),cache)
    parts_snd = map(i->i.parts_snd,cache)
    parts_rcv = map(i->i.parts_rcv,cache)
    V_snd = map(i->i.V_snd,cache)
    V_rcv = map(i->i.V_rcv,cache)
    graph = ExchangeGraph(parts_snd,parts_rcv)
    t = exchange!(V_rcv,V_snd,graph)
    @async begin
        wait(t)
        map(setup_rcv,partition(B),cache)
        B
    end
end

function consistent(A::PSparseMatrixNew,rows_co)
    @assert assembly_style(A) == Assembled()
    psparse_consitent_impl(A,eltype(partition(A)),rows_co)
end

function consistent!(B::PSparseMatrixNew,A::PSparseMatrixNew)
    @assert assembly_style(A) == Assembled()
    @assert assembly_style(B) == Consistent()
    psparse_consitent_impl!(B,A,eltype(partition(A)))
end

function psparse_consitent_impl(A::PSparseMatrixNew,::Type{<:SplitMatrixLocal},rows_co)
    function setup_snd(A,parts_snd,lids_snd,rows_co,cols_fa)
        own_to_local_row = own_to_local(rows_co)
        own_to_global_row = own_to_global(rows_co)
        own_to_global_col = own_to_global(cols_fa)
        ghost_to_global_col = ghost_to_global(cols_fa)
        li_to_p = zeros(Int32,size(A,1))
        for p in 1:length(lids_snd)
            li_to_p[lids_snd[p]] .= p
        end
        ptrs = zeros(Int32,length(parts_snd)+1)
        for (i,j,v) in nziterator(A.blocks.own_own)
            li = own_to_local_row[i]
            p = li_to_p[li]
            if p == 0
                continue
            end
            ptrs[p+1] += 1
        end
        for (i,j,v) in nziterator(A.blocks.own_ghost)
            li = own_to_local_row[i]
            p = li_to_p[li]
            if p == 0
                continue
            end
            ptrs[p+1] += 1
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
            p = li_to_p[li]
            if p == 0
                continue
            end
            q = ptrs[p]
            I_snd.data[q] = own_to_global_row[i]
            J_snd.data[q] = own_to_global_col[j]
            V_snd.data[q] = v
            k_snd.data[q] = k
            ptrs[p] += 1
        end
        nnz_own_own = nnz(A.blocks.own_own)
        for (k,(i,j,v)) in enumerate(nziterator(A.blocks.own_ghost))
            li = own_to_local_row[i]
            p = li_to_p[li]
            if p == 0
                continue
            end
            q = ptrs[p]
            I_snd.data[q] = own_to_global_row[i]
            J_snd.data[q] = ghost_to_global_col[j]
            V_snd.data[q] = v
            k_snd.data[q] = k+nnz_own_own
            ptrs[p] += 1
        end
        rewind_ptrs!(ptrs)
        cache_snd = (;parts_snd,lids_snd,I_snd,J_snd,V_snd,k_snd)
        cache_snd
    end
    function setup_rcv(parts_rcv,lids_rcv,I_rcv,J_rcv,V_rcv)
        cache_rcv = (;parts_rcv,lids_rcv,I_rcv,J_rcv,V_rcv)
        cache_rcv
    end
    function finalize(A,cache_snd,cache_rcv,rows_co,cols_fa)
        I_rcv_data = cache_rcv.I_rcv.data
        J_rcv_data = cache_rcv.J_rcv.data
        V_rcv_data = cache_rcv.V_rcv.data
        global_to_own_col = global_to_own(cols_fa)
        global_to_ghost_col = global_to_ghost(cols_fa)
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
        map_global_to_own!(J_rcv_own,cols_fa)
        map_global_to_ghost!(J_rcv_ghost,cols_fa)
        own_own = A.blocks.own_own
        own_ghost = A.blocks.own_ghost
        @assert blocktype(A) <: SparseMatrixCSC
        n_ghost_rows = ghost_length(rows_co)
        n_own_cols = own_length(cols_fa)
        n_ghost_cols = ghost_length(cols_fa)
        ghost_own,K_own = sparse_csc(I_rcv_own,J_rcv_own,V_rcv_own,n_ghost_rows,n_own_cols)
        ghost_ghost,K_ghost = sparse_csc(I_rcv_ghost,J_rcv_ghost,V_rcv_ghost,n_ghost_rows,n_ghost_cols)
        blocks = split_matrix_blocks(own_own,own_ghost,ghost_own,ghost_ghost)
        values = SplitMatrixLocal(blocks,local_permutation(rows_co),local_permutation(cols_fa))
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
    @async begin
        I_rcv = fetch(t_I)
        J_rcv = fetch(t_J)
        V_rcv = fetch(t_V)
        cache_rcv = map(setup_rcv,parts_rcv,lids_rcv,I_rcv,J_rcv,V_rcv)
        values,cache = map(finalize,partition(A),cache_snd,cache_rcv,rows_co,cols_fa) |> tuple_of_arrays
        PSparseMatrixNew(Consistent(),values,rows_co,cols_fa,cache)
    end
end

function psparse_consitent_impl!(B,A::PSparseMatrixNew,::Type{<:SplitMatrixLocal})
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
        sparse_csc!(B.blocks.ghost_own,K_own,V_rcv_own)
        sparse_csc!(B.blocks.ghost_ghost,K_ghost,V_rcv_ghost)
        B
    end
    cache = B.cache
    map(setup_snd,partition(A),cache)
    parts_snd = map(i->i.parts_snd,cache)
    parts_rcv = map(i->i.parts_rcv,cache)
    V_snd = map(i->i.V_snd,cache)
    V_rcv = map(i->i.V_rcv,cache)
    graph = ExchangeGraph(parts_snd,parts_rcv)
    t = exchange!(V_rcv,V_snd,graph)
    @async begin
        wait(t)
        map(setup_rcv,partition(B),cache)
        B
    end
end

function LinearAlgebra.mul!(c::PVector,a::PSparseMatrixNew{Assembled},b::PVector,α::Number,β::Number)
    @boundscheck @assert matching_own_indices(axes(c,1),axes(a,1))
    @boundscheck @assert matching_own_indices(axes(a,2),axes(b,1))
    @boundscheck @assert matching_ghost_indices(axes(a,2),axes(b,1))
    # Start the exchange
    t = consistent!(b)
    # Meanwhile, process the owned blocks
    map(own_values(c),own_own_values(a),own_values(b)) do co,aoo,bo
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

function LinearAlgebra.mul!(c::PVector,a::PSparseMatrixNew{Subassembled},b::PVector,α::Number,β::Number)
    @boundscheck @assert matching_own_indices(axes(c,1),axes(a,1))
    @boundscheck @assert matching_ghost_indices(axes(c,1),axes(a,1))
    @boundscheck @assert matching_own_indices(axes(a,2),axes(b,1))
    @boundscheck @assert matching_ghost_indices(axes(a,2),axes(b,1))
    error("Not implemented yet")
end

Base.similar(a::PSparseMatrixNew) = similar(a,eltype(a))
function Base.similar(a::PSparseMatrixNew,::Type{T}) where T
    matrix_partition = map(partition(a)) do values
        similar(values,T)
    end
    style = assembly_style(a)
    rows, cols = axes(a)
    PSparseMatrixNew(style,matrix_partition,partition(rows),partition(cols))
end

function Base.copy!(a::PSparseMatrixNew,b::PSparseMatrixNew)
    @assert size(a) == size(b)
    copyto!(a,b)
end

function Base.copyto!(a::PSparseMatrixNew,b::PSparseMatrixNew)
    if partition(axes(a,1)) === partition(axes(b,1)) && partition(axes(a,2)) === partition(axes(b,2))
        map(copy!,partition(a),partition(b))
    else
        error("Trying to copy a PSparseMatrix into another one with a different data layout. This case is not implemented yet. It would require communications.")
    end
    a
end

function LinearAlgebra.fillstored!(a::PSparseMatrixNew,v)
    map(partition(a)) do values
        LinearAlgebra.fillstored!(values,v)
    end
    a
end

# Misc functions that could be removed if IterativeSolvers was implemented in terms
# of axes(A,d) instead of size(A,d)
function IterativeSolvers.zerox(A::PSparseMatrixNew,b::PVector)
    T = IterativeSolvers.Adivtype(A, b)
    x = similar(b, T, axes(A, 2))
    fill!(x, zero(T))
    return x
end
