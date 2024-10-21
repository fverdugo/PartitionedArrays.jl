
function local_values end

function own_values end

function ghost_values end

function allocate_local_values(a,::Type{T},indices) where T
    similar(a,T,local_length(indices))
end

function allocate_local_values(::Type{V},indices) where V
    similar(V,local_length(indices))
end

function local_values(values,indices)
    values
end

function own_values(values,indices)
    view(values,own_to_local(indices))
end

function ghost_values(values,indices)
    view(values,ghost_to_local(indices))
end

# OwnAndGhostVectors is deprecated in favor of SplitVector (for consistency with SplitMatrix)

"""
    struct OwnAndGhostVectors{A,C,T}

Vector type that stores the local values of a [`PVector`](@ref) instance
using a vector of own values, a vector of ghost values, and a permutation.

# Properties

- `own_values::A`: The vector of own values.
- `ghost_values::A`: The vector of ghost values.
- `permumation::C`: A permutation vector such that `vcat(own_values,ghost_values)[permutation]` corresponds to the local values.

# Supertype hierarchy

    OwnAndGhostVectors{A,C,T} <: AbstractVector{T}
"""
struct OwnAndGhostVectors{A,C,T} <: AbstractVector{T}
    own_values::A
    ghost_values::A
    permutation::C
    function OwnAndGhostVectors{A,C}(own_values,ghost_values,perm) where {A,C}
        T = eltype(A)
        new{A,C,T}(
          convert(A,own_values),
          convert(A,ghost_values),
          convert(C,perm))
    end
end
function OwnAndGhostVectors{A}(own_values,ghost_values,perm) where A
    C = typeof(perm)
    OwnAndGhostVectors{A,C}(own_values,ghost_values,perm)
end

"""
    OwnAndGhostVectors(own_values,ghost_values,permutation)

Build an instance of [`OwnAndGhostVectors`](@ref) from the underlying fields.
"""
function OwnAndGhostVectors(own_values,ghost_values,perm)
    A = typeof(own_values)
    OwnAndGhostVectors{A}(own_values,ghost_values,perm)
end
Base.IndexStyle(::Type{<:OwnAndGhostVectors}) = IndexLinear()
Base.size(a::OwnAndGhostVectors) = (length(a.own_values)+length(a.ghost_values),)
function Base.getindex(a::OwnAndGhostVectors,local_id::Int)
    n_own = length(a.own_values)
    j = a.permutation[local_id]
    if j > n_own
        a.ghost_values[j-n_own]
    else
        a.own_values[j]
    end
end
function Base.setindex!(a::OwnAndGhostVectors,v,local_id::Int)
    n_own = length(a.own_values)
    j = a.permutation[local_id]
    if j > n_own
        a.ghost_values[j-n_own] = v
    else
        a.own_values[j] = v
    end
    v
end

function own_values(values::OwnAndGhostVectors,indices)
    values.own_values
end

function ghost_values(values::OwnAndGhostVectors,indices)
    values.ghost_values
end

function allocate_local_values(values::OwnAndGhostVectors,::Type{T},indices) where T
    n_own = own_length(indices)
    n_ghost = ghost_length(indices)
    own_values = similar(values.own_values,T,n_own)
    ghost_values = similar(values.ghost_values,T,n_ghost)
    perm = local_permutation(indices)
    OwnAndGhostVectors(own_values,ghost_values,perm)
end

function allocate_local_values(::Type{<:OwnAndGhostVectors{A}},indices) where {A}
    n_own = own_length(indices)
    n_ghost = ghost_length(indices)
    own_values = similar(A,n_own)
    ghost_values = similar(A,n_ghost)
    perm = local_permutation(indices)
    OwnAndGhostVectors{A}(own_values,ghost_values,perm)
end

struct SplitVectorBlocks{A}
    own::A
    ghost::A
end
function split_vector_blocks(own,ghost)
    T = typeof(own)
    SplitVectorBlocks(own,convert(T,ghost))
end
function split_vector_blocks(own::A,ghost::A) where A
    SplitVectorBlocks(own,ghost)
end

struct SplitVector{A,B,T} <: AbstractVector{T}
    blocks::SplitVectorBlocks{A}
    permutation::B
    function SplitVector(
        blocks::SplitVectorBlocks{A},permutation) where A
        T = eltype(blocks.own)
        B = typeof(permutation)
        new{A,B,T}(blocks,permutation)
    end
end

function split_vector(blocks::SplitVectorBlocks,permutation)
    SplitVector(blocks,permutation)
end

function split_vector(
    own::AbstractVector,
    ghost::AbstractVector,
    permutation)
    blocks = split_vector_blocks(own,ghost)
    split_vector(blocks,permutation)
end

Base.IndexStyle(::Type{<:SplitVector}) = IndexLinear()
Base.size(a::SplitVector) = (length(a.blocks.own)+length(a.blocks.ghost),)
function Base.getindex(a::SplitVector,local_id::Int)
    T = eltype(a)
    n_own = length(a.blocks.own)
    j = a.permutation[local_id]
    v = if j > n_own
        a.blocks.ghost[j-n_own]
    else
        a.blocks.own[j]
    end
    convert(T,v)
end
function Base.setindex!(a::SplitVector,v,local_id::Int)
    n_own = length(a.blocks.own)
    j = a.permutation[local_id]
    if j > n_own
        a.blocks.ghost[j-n_own] = v
    else
        a.blocks.own[j] = v
    end
    v
end

function own_values(values::SplitVector,indices)
    values.blocks.own
end

function ghost_values(values::SplitVector,indices)
    values.blocks.ghost
end

function allocate_local_values(values::SplitVector,::Type{T},indices) where T
    n_own = own_length(indices)
    n_ghost = ghost_length(indices)
    own_values = similar(values.blocks.own,T,n_own)
    ghost_values = similar(values.blocks.ghost,T,n_ghost)
    perm = local_permutation(indices)
    blocks = split_vector_blocks(own_values,ghost_values)
    split_vector(blocks,perm)
end

function allocate_local_values(::Type{<:SplitVector{A}},indices) where {A}
    n_own = own_length(indices)
    n_ghost = ghost_length(indices)
    own_values = similar(A,n_own)
    ghost_values = similar(A,n_ghost)
    blocks = split_vector_blocks(own_values,ghost_values)
    perm = local_permutation(indices)
    split_vector(blocks,perm)
end

Base.similar(a::SplitVector) = similar(a,eltype(a))
function Base.similar(a::SplitVector,::Type{T}) where T
    own = similar(a.blocks.own,T)
    ghost = similar(a.blocks.ghost,T)
    blocks = split_vector_blocks(own,ghost)
    split_vector(blocks,a.permutation)
end

function Base.copy(a::SplitVector)
    own = copy(a.blocks.own)
    ghost = copy(a.blocks.ghost)
    blocks = split_vector_blocks(own,ghost)
    split_vector(blocks,a.permutation)
end

function Base.copy!(a::SplitVector,b::SplitVector)
    copy!(a.blocks.own,b.blocks.own)
    copy!(a.blocks.ghost,b.blocks.ghost)
    a
end
function Base.copyto!(a::SplitVector,b::SplitVector)
    copyto!(a.blocks.own,b.blocks.own)
    copyto!(a.blocks.ghost,b.blocks.ghost)
    a
end

function Base.fill!(a::SplitVector,v)
    LinearAlgebra.fill!(a.blocks.own,v)
    LinearAlgebra.fill!(a.blocks.ghost,v)
    a
end

function Base.:*(a::Number,b::SplitVector)
    own = a*b.blocks.own
    ghost = a*b.blocks.ghost
    blocks = split_vector_blocks(own,ghost)
    split_vector(blocks,b.permutation)
end

function Base.:*(b::SplitVector,a::Number)
    a*b
end

for op in (:+,:-)
    @eval begin
        function Base.$op(a::SplitVector)
            own = $op(a.blocks.own)
            ghost = $op(a.blocks.ghost)
            blocks = split_vector_blocks(own,ghost)
            split_vector(blocks,a.permutation)
        end
        function Base.$op(a::SplitVector,b::SplitVector)
            @boundscheck @assert a.permutation == b.permutation
            own = $op(a.blocks.own,b.blocks.own)
            ghost = $op(a.blocks.ghost,b.blocks.ghost)
            blocks = split_vector_blocks(own,ghost)
            split_vector(blocks,b.permutation)
        end
    end
end

function split_format_locally(a::SplitVector,rows)
    a
end

function split_format_locally(a::AbstractVector,rows)
    n_own = own_length(rows)
    n_ghost = ghost_length(rows)
    perm = local_permutation(rows)
    own = similar(a,n_own)
    ghost = similar(a,n_ghost)
    blocks = split_vector_blocks(own,ghost)
    b = split_vector(blocks,perm)
    split_format_locally!(b,a,rows)
    b
end

function split_format_locally!(b::SplitVector,a::AbstractVector,rows)
    b.blocks.own .= view(a,own_to_local(rows))
    b.blocks.ghost .= view(a,ghost_to_local(rows))
    b
end

function split_format_locally!(b::SplitVector,a::SplitVector,rows)
    if b !== a
        b.blocks.own .= a.blocks.own
        b.blocks.ghost .= a.blocks.ghost
    end
    b
end

"""
    struct PVector{V,A,B,...}

`PVector` (partitioned vector) is a type representing a vector whose entries are
distributed (a.k.a. partitioned) over different parts for distributed-memory
parallel computations.

This type overloads numerous array-like operations with corresponding
parallel implementations.

# Properties

- `vector_partition::A`
- `index_partition::B`

`vector_partition[i]` contains the vector of local values of the `i`-th part in the data distribution. The first type parameter `V` corresponds to `typeof(values[i])` i.e. the vector type used to store the local values. The item `index_partition[i]` implements the [`AbstractLocalIndices`](@ref) interface providing information about the
local, own, and ghost indices in the `i`-th part.

The rest of fields of this struct and type parameters are private.

# Supertype hierarchy

    PVector{V,A,B,...} <: AbstractVector{T}

with `T=eltype(V)`.
"""
struct PVector{V,A,B,C,T} <: AbstractVector{T}
    vector_partition::A
    index_partition::B
    cache::C
    @doc """
        PVector(vector_partition,index_partition)

    Create an instance of [`PVector`](@ref) from the underlying properties
    `vector_partition` and `index_partition`.
    """
    function PVector(
            vector_partition,
            index_partition,
            cache=p_vector_cache(vector_partition,index_partition))
        T = eltype(eltype(vector_partition))
        V = eltype(vector_partition)
        A = typeof(vector_partition)
        B = typeof(index_partition)
        C = typeof(cache)
        new{V,A,B,C,T}(vector_partition,index_partition,cache)
    end
end

partition(a::PVector) = a.vector_partition
Base.axes(a::PVector) = (PRange(a.index_partition),)

"""
    local_values(a::PVector)

Get a vector of vectors containing the local values
in each part of `a`.

The indices of the returned vectors can be mapped to global indices, own
indices, ghost indices, and owner by using [`local_to_global`](@ref),
[`local_to_own`](@ref), [`local_to_ghost`](@ref), and [`local_to_owner`](@ref),
respectively.
"""
function local_values(a::PVector)
    partition(a)
end

"""
    own_values(a::PVector)

Get a vector of vectors containing the own values
in each part of `a`.

The indices of the returned vectors can be mapped to global indices, local
indices, and owner by using [`own_to_global`](@ref), [`own_to_local`](@ref),
and [`own_to_owner`](@ref), respectively.
"""
function own_values(a::PVector)
    map(own_values,partition(a),partition(axes(a,1)))
end

"""
    ghost_values(a::PVector)

Get a vector of vectors containing the ghost values
in each part of `a`.

The indices of the returned matrices can be mapped to global indices, local
indices, and owner by using [`ghost_to_global`](@ref),
[`ghost_to_local`](@ref), and [`ghost_to_owner`](@ref), respectively.
"""
function ghost_values(a::PVector)
    map(ghost_values,partition(a),partition(axes(a,1)))
end

Base.size(a::PVector) = (length(axes(a,1)),)
Base.IndexStyle(::Type{<:PVector}) = IndexLinear()
function Base.getindex(a::PVector,gid::Int)
    scalar_indexing_action(a)
end
function Base.setindex(a::PVector,v,gid::Int)
    scalar_indexing_action(a)
end

function Base.show(io::IO,k::MIME"text/plain",data::PVector)
    T = eltype(partition(data))
    n = length(data)
    np = length(partition(data))
    map_main(partition(data)) do values
        println(io,"$n-element PVector partitioned into $np parts of type $T")
    end
end
function Base.show(io::IO,data::PVector)
    print(io,"PVector(…)")
end

function p_vector_cache(vector_partition,index_partition)
    p_vector_cache_impl(eltype(vector_partition),vector_partition,index_partition)
end

struct VectorAssemblyCache{A,B,C,D}
    neighbors_snd::A
    neighbors_rcv::A
    local_indices_snd::B
    local_indices_rcv::B
    buffer_snd::C
    buffer_rcv::C
    exchange_setup::D
end
function Base.reverse(a::VectorAssemblyCache)
    VectorAssemblyCache(
                    a.neighbors_rcv,
                    a.neighbors_snd,
                    a.local_indices_rcv,
                    a.local_indices_snd,
                    a.buffer_rcv,
                    a.buffer_snd,
                    a.exchange_setup,
                   )
end
function copy_cache(a::VectorAssemblyCache)
    buffer_snd = deepcopy(a.buffer_snd) # TODO ugly
    buffer_rcv = deepcopy(a.buffer_rcv)
    VectorAssemblyCache(
                    a.neighbors_snd,
                    a.neighbors_rcv,
                    a.local_indices_snd,
                    a.local_indices_rcv,
                    buffer_snd,
                    buffer_rcv,
                    a.exchange_setup
                   )
end
function p_vector_cache_impl(::Type,vector_partition,index_partition)
    neighbors_snd,neighbors_rcv= assembly_neighbors(index_partition)
    indices_snd,indices_rcv = assembly_local_indices(index_partition,neighbors_snd,neighbors_rcv)
    buffers_snd,buffers_rcv = map(assembly_buffers,vector_partition,indices_snd,indices_rcv) |> tuple_of_arrays
    graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
    exchange_setup = setup_exchange(buffers_rcv,buffers_snd,graph)
    VectorAssemblyCache(neighbors_snd,neighbors_rcv,indices_snd,indices_rcv,buffers_snd,buffers_rcv,exchange_setup)
end
function assembly_buffers(values,local_indices_snd,local_indices_rcv)
    T = eltype(values)
    ptrs = local_indices_snd.ptrs
    data = zeros(T,ptrs[end]-1)
    buffer_snd = JaggedArray(data,ptrs)
    ptrs = local_indices_rcv.ptrs
    data = zeros(T,ptrs[end]-1)
    buffer_rcv = JaggedArray(data,ptrs)
    buffer_snd, buffer_rcv
end

struct JaggedArrayAssemblyCache{T<:VectorAssemblyCache}
    cache::T
end
Base.reverse(a::JaggedArrayAssemblyCache) = JaggedArrayAssemblyCache(reverse(a.cache))
copy_cache(a::JaggedArrayAssemblyCache) = JaggedArrayAssemblyCache(copy_cache(a.cache))
function p_vector_cache_impl(::Type{<:JaggedArray},vector_partition,index_partition)
    function data_index_snd(lids_snd,values)
        tptrs = values.ptrs
        ptrs = similar(lids_snd.ptrs)
        fill!(ptrs,zero(eltype(ptrs)))
        np = length(ptrs)-1
        for p in 1:np
            iini = lids_snd.ptrs[p]
            iend = lids_snd.ptrs[p+1]-1
            for i in iini:iend
                d = lids_snd.data[i]
                ptrs[p+1] += tptrs[d+1]-tptrs[d]
            end
        end
        length_to_ptrs!(ptrs)
        ndata = ptrs[end]-1
        data = similar(lids_snd.data,eltype(lids_snd.data),ndata)
        for p in 1:np
            iini = lids_snd.ptrs[p]
            iend = lids_snd.ptrs[p+1]-1
            for i in iini:iend
                d = lids_snd.data[i]
                jini = tptrs[d]
                jend = tptrs[d+1]-1
                for j in jini:jend
                    data[ptrs[p]] = j
                    ptrs[p] += 1
                end
            end
        end
        rewind_ptrs!(ptrs)
        JaggedArray(data,ptrs)
    end
    neighbors_snd,neighbors_rcv = assembly_neighbors(index_partition)
    local_indices_snd, local_indices_rcv = assembly_local_indices(index_partition,neighbors_snd,neighbors_rcv)
    p_snd = map(data_index_snd,local_indices_snd,vector_partition)
    p_rcv = map(data_index_snd,local_indices_rcv,vector_partition)
    data = map(getdata,vector_partition)
    buffer_snd, buffer_rcv = map(assembly_buffers,data,p_snd,p_rcv) |> tuple_of_arrays
    graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
    exchange_setup = setup_exchange(buffer_rcv,buffer_snd,graph)
    cache = VectorAssemblyCache(neighbors_snd,neighbors_rcv,p_snd,p_rcv,buffer_snd,buffer_rcv,exchange_setup)
    JaggedArrayAssemblyCache(cache)
end

# NB these fields could be removed if the ghost
# are sorted according to their owner
struct SplitVectorAssemblyCache{A,B,C,D}
    neighbors_snd::A
    neighbors_rcv::A
    ghost_indices_snd::B # NB
    own_indices_rcv::B
    buffer_snd::C # NB
    buffer_rcv::C
    exchange_setup::D
    reversed::Bool
end
function Base.reverse(a::SplitVectorAssemblyCache)
    SplitVectorAssemblyCache(
                    a.neighbors_rcv,
                    a.neighbors_snd,
                    a.own_indices_rcv,
                    a.ghost_indices_snd,
                    a.buffer_rcv,
                    a.buffer_snd,
                    a.exchange_setup,
                    !(a.reversed),
                   )
end
function copy_cache(a::SplitVectorAssemblyCache)
    buffer_snd = deepcopy(a.buffer_snd) # TODO ugly
    buffer_rcv = deepcopy(a.buffer_rcv) # TODO ugly
    VectorAssemblyCache(
                    a.neighbors_snd,
                    a.neighbors_rcv,
                    a.ghost_indices_snd,
                    a.own_indices_rcv,
                    buffer_snd,
                    buffer_rcv,
                    a.exchange_setup,
                    a.reversed,
                   )
end

function p_vector_cache_impl(::Type{<:SplitVector},vector_partition,index_partition)
    neighbors_snd,neighbors_rcv= assembly_neighbors(index_partition)
    indices_snd,indices_rcv = assembly_local_indices(index_partition,neighbors_snd,neighbors_rcv)
    ghost_indices_snd = map(indices_snd) do ids
        JaggedArray(copy(ids.data),ids.ptrs)
    end
    own_indices_rcv = map(indices_rcv) do ids
        JaggedArray(copy(ids.data),ids.ptrs)
    end
    foreach(ghost_indices_snd,own_indices_rcv,index_partition) do ids_snd,ids_rcv,myids
        map_local_to_ghost!(ids_snd.data,myids)
        map_local_to_own!(ids_rcv.data,myids)
    end
    buffers_snd,buffers_rcv = map(assembly_buffers,vector_partition,ghost_indices_snd,own_indices_rcv) |> tuple_of_arrays
    graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
    exchange_setup = setup_exchange(buffers_rcv,buffers_snd,graph)
    reversed = false
    SplitVectorAssemblyCache(neighbors_snd,neighbors_rcv,ghost_indices_snd,own_indices_rcv,buffers_snd,buffers_rcv,exchange_setup,reversed)
end

function p_vector_cache_impl(::Type{<:SplitVector{<:JaggedArray}},vector_partition,index_partition)
    error("Case not implemented yet")
end

function assemble!(f,vector_partition,cache)
    assemble_impl!(f,vector_partition,cache)
end

function assemble_impl!(f,vector_partition,cache::VectorAssemblyCache)
    local_indices_snd=cache.local_indices_snd
    local_indices_rcv=cache.local_indices_rcv
    neighbors_snd=cache.neighbors_snd
    neighbors_rcv=cache.neighbors_rcv
    buffer_snd=cache.buffer_snd
    buffer_rcv=cache.buffer_rcv
    exchange_setup=cache.exchange_setup
    foreach(vector_partition,local_indices_snd,buffer_snd) do values,local_indices_snd,buffer_snd
        for (p,lid) in enumerate(local_indices_snd.data)
            buffer_snd.data[p] = values[lid]
        end
    end
    graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
    t = exchange!(buffer_rcv,buffer_snd,graph,exchange_setup)
    # Fill values from rcv buffer fake_asynchronously
    @fake_async begin
        wait(t)
        foreach(vector_partition,local_indices_rcv,buffer_rcv) do values,local_indices_rcv,buffer_rcv
            for (p,lid) in enumerate(local_indices_rcv.data)
                values[lid] = f(values[lid],buffer_rcv.data[p])
            end
        end
        nothing
    end
end

function assemble_impl!(f,vector_partition,cache::JaggedArrayAssemblyCache)
    vcache = cache.cache
    data = map(getdata,vector_partition)
    assemble!(f,data,vcache)
end

function assemble_impl!(f,vector_partition,cache::SplitVectorAssemblyCache)
    reversed = cache.reversed
    ghost_indices_snd=cache.ghost_indices_snd
    own_indices_rcv=cache.own_indices_rcv
    neighbors_snd=cache.neighbors_snd
    neighbors_rcv=cache.neighbors_rcv
    buffer_snd=cache.buffer_snd
    buffer_rcv=cache.buffer_rcv
    exchange_setup=cache.exchange_setup
    foreach(vector_partition,ghost_indices_snd,buffer_snd) do values,ghost_indices_snd,buffer_snd
        if reversed
            ghost_vals = values.blocks.own
        else
            ghost_vals = values.blocks.ghost
        end
        for (p,hid) in enumerate(ghost_indices_snd.data)
            buffer_snd.data[p] = ghost_vals[hid]
        end
    end
    graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
    t = exchange!(buffer_rcv,buffer_snd,graph,exchange_setup)
    # Fill values from rcv buffer fake_asynchronously
    @fake_async begin
        wait(t)
        foreach(vector_partition,own_indices_rcv,buffer_rcv) do values,own_indices_rcv,buffer_rcv
            if reversed
                own_vals = values.blocks.ghost
            else
                own_vals = values.blocks.own
            end
            for (p,oid) in enumerate(own_indices_rcv.data)
                own_vals[oid] = f(own_vals[oid],buffer_rcv.data[p])
            end
        end
        nothing
    end
end

"""
    assemble!([op,] a::PVector) -> Task

Transfer the ghost values to their owner part
and insert them according with the insertion operation `op` (`+` by default).
It returns a task that produces `a` with updated values. After the transfer,
the source ghost values are set to zero.

# Examples
```jldoctest
julia> using PartitionedArrays

julia> rank = LinearIndices((2,));

julia> row_partition = uniform_partition(rank,6,true);

julia> map(local_to_global,row_partition)
2-element Vector{PartitionedArrays.BlockPartitionLocalToGlobal{1, Vector{Int32}}}:
 [1, 2, 3, 4]
 [3, 4, 5, 6]

julia> a = pones(row_partition)
6-element PVector partitioned into 2 parts of type Vector{Float64}

julia> local_values(a)
2-element Vector{Vector{Float64}}:
 [1.0, 1.0, 1.0, 1.0]
 [1.0, 1.0, 1.0, 1.0]

julia> assemble!(a) |> wait

julia> local_values(a)
2-element Vector{Vector{Float64}}:
 [1.0, 1.0, 2.0, 0.0]
 [0.0, 2.0, 1.0, 1.0]
```
"""
function assemble!(a::PVector)
    assemble!(+,a)
end

function assemble!(o,a::PVector)
    t = assemble!(o,partition(a),a.cache)
    @fake_async begin
        wait(t)
        map(ghost_values(a)) do a
            fill!(a,zero(eltype(a)))
        end
        a
    end
end

"""
    consistent!(a::PVector) -> Task

Make the local values of `a` globally consistent. I.e., the
ghost values are updated with the corresponding own value in the
part that owns the associated global global id.

# Examples

```jldoctest
julia> using PartitionedArrays

julia> rank = LinearIndices((2,));

julia> row_partition = uniform_partition(rank,6,true);

julia> map(local_to_global,row_partition)
2-element Vector{PartitionedArrays.BlockPartitionLocalToGlobal{1, Vector{Int32}}}:
 [1, 2, 3, 4]
 [3, 4, 5, 6]

julia> a = pvector(inds->fill(part_id(inds),length(inds)),row_partition)
6-element PVector partitioned into 2 parts of type Vector{Int32}

julia> local_values(a)
2-element Vector{Vector{Int32}}:
 [1, 1, 1, 1]
 [2, 2, 2, 2]

julia> consistent!(a) |> wait

julia> local_values(a)
2-element Vector{Vector{Int32}}:
 [1, 1, 1, 2]
 [1, 2, 2, 2]
```
"""
function consistent!(a::PVector)
    cache = reverse(a.cache)
    t = assemble!(insert,partition(a),cache)
    @fake_async begin
        wait(t)
        a
    end
end
insert(a,b) = b


function Base.similar(a::PVector,::Type{T},inds::Tuple{<:PRange}) where T
    rows = inds[1]
    values = map(partition(a),partition(rows)) do values, indices
        allocate_local_values(values,T,indices)
    end
    PVector(values,partition(rows))
end

function Base.similar(::Type{<:PVector{V}},inds::Tuple{<:PRange}) where V
    rows = inds[1]
    values = map(partition(rows)) do indices
        allocate_local_values(V,indices)
    end
    PVector(values,partition(rows))
end

function PVector(::UndefInitializer,index_partition)
    PVector{Vector{Float64}}(undef,index_partition)
end

function PVector(::UndefInitializer,r::PRange)
    PVector(undef,partition(r))
end

"""
    PVector{V}(undef,index_partition)
    PVector(undef,index_partition)

Create an instance of [`PVector`](@ref) with local uninitialized values
stored in a vector of type `V` (which defaults to `V=Vector{Float64}`).
"""
function PVector{V}(::UndefInitializer,index_partition) where V
    vector_partition = map(index_partition) do indices
        allocate_local_values(V,indices)
    end
    PVector(vector_partition,index_partition)
end

function PVector{V}(::UndefInitializer,r::PRange) where V
    PVector{V}(undef,partition(r))
end

function Base.copy!(a::PVector,b::PVector)
    @assert length(a) == length(b)
    copyto!(a,b)
end

function Base.copyto!(a::PVector,b::PVector)
    if partition(axes(a,1)) === partition(axes(b,1))
        map(copy!,partition(a),partition(b))
    elseif matching_own_indices(axes(a,1),axes(b,1))
        map(copy!,own_values(a),own_values(b))
    else
        error("Trying to copy a PVector into another one with a different data layout. This case is not implemented yet. It would require communications.")
    end
    a
end

function Base.fill!(a::PVector,v)
    map(partition(a)) do values
        fill!(values,v)
    end
    a
end

"""
    pvector(f,index_partition)

Equivalent to 

    vector_partition = map(f,index_partition)
    PVector(vector_partition,index_partition)

"""
@inline function pvector(f,index_partition;split_format=Val(false))
    vector_partition = map(f,index_partition)
    b = PVector(vector_partition,index_partition)
    if !(val_parameter(split_format))
        return b
    end
    PartitionedArrays.split_format(b)
end
pvector(f,r::PRange;kwargs...) = pvector(f,partition(r);kwargs...)

function old_pvector!(f,I,V,index_partition;discover_rows=true)
    if discover_rows
        I_owner = find_owner(index_partition,I)
        index_partition = map(union_ghost,index_partition,I,I_owner)
    end
    map(to_local!,I,index_partition)
    vector_partition = map(f,I,V,index_partition)
    v = PVector(vector_partition,index_partition)
    assemble!(v)
end

function dense_vector(I,V,n)
    T = eltype(V)
    a = zeros(T,n)
    for (i,v) in zip(I,V)
        if i < 1
            continue
        end
        a[i] += v
    end
    a
end

function dense_vector!(A,K,V)
    fill!(A,0)
    for (k,v) in zip(K,V)
        if k < 1
            continue
        end
        A[k] += v
    end
end

function pvector(I,V,rows;kwargs...)
    pvector(dense_vector,I,V,rows;kwargs...)
end

"""
    pvector([f,]I,V,index_partition;kwargs...) -> Task

Crate an instance of [`PVector`](@ref) by setting arbitrary entries
from each of the underlying parts. It returns a task that produces the
instance of [`PVector`](@ref) allowing latency hiding while performing
the communications needed in its setup.
"""
function pvector(f,I,V,rows;
        subassembled=false,
        assembled=false,
        assemble=true,
        split_format = Val(false),
        restore_ids = true,
        indices = :global,
        reuse=Val(false),
        assembled_rows = nothing,
        assembly_neighbors_options_rows = (;)
    )

    # Checks
    disassembled = (!subassembled && ! assembled) ? true : false
    @assert indices in (:global,:local)
    if count((subassembled,assembled)) == 2
        error("Only one of the folling flags can be set to true: subassembled, assembled")
    end

    if disassembled
        @assert indices === :global
        I_owner = find_owner(rows,I)
        rows_sa = map(union_ghost,rows,I,I_owner)
        assembly_neighbors(rows_sa;assembly_neighbors_options_rows...)
        map(map_global_to_local!,I,rows_sa)
        values_sa = map(f,I,V,map(local_length,rows_sa))
        if val_parameter(reuse)
            K = map(copy,I)
        end
        if restore_ids
            map(map_local_to_global!,I,rows_sa)
        end
        A = PVector(values_sa,rows_sa)
        if assemble
            t = PartitionedArrays.assemble(A,rows;reuse=true)
        else
            t = @fake_async A, nothing
        end
    elseif subassembled
        if assembled_rows === nothing
            assembled_rows = map(remove_ghost,rows)
        end
        rows_sa = rows
        if indices === :global
            map(map_global_to_local!,I,rows_sa)
        end
        values_sa = map(f,I,V,map(local_length,rows_sa))
        if val_parameter(reuse)
            K = map(copy,I)
        end
        if indices === :global && restore_ids
            map(map_local_to_global!,I,rows_sa)
        end
        A = PVector(values_sa,rows_sa)
        if assemble
            t = PartitionedArrays.assemble(A,assembled_rows;reuse=true)
        else
            t = @fake_async A, nothing
        end
    elseif assembled
        rows_fa = rows
        if indices === :global
            map(map_global_to_local!,I,rows_fa)
        end
        values_fa = map(f,I,V,map(local_length,rows_fa))
        if val_parameter(reuse)
            K = map(copy,I)
        end
        if indices === :global && restore_ids
            map(map_local_to_global!,I,rows_fa)
        end
        A = PVector(values_fa,rows_fa)
        t = @fake_async A, nothing
    else
        error("This line should not be reached")
    end
    if val_parameter(reuse) == false
        return @fake_async begin
            B, cacheB = fetch(t)
            C = if val_parameter(split_format)
                PartitionedArrays.split_format(B)
            else
                B
            end
            C
        end
    else
        return @fake_async begin
            B, cacheB = fetch(t)
            C = if val_parameter(split_format)
                PartitionedArrays.split_format(B)
            else
                B
            end
            cache = (A,B,cacheB,assemble,assembled,K,split_format) 
            (C, cache)
        end
    end
end

"""
    pvector!(B::PVector,V,cache)
"""
function pvector!(C,V,cache)
    (A,B,cacheB,assemble,assembled,K,split_format) = cache
    rows_sa = partition(axes(A,1))
    values_sa = partition(A)
    map(dense_vector!,values_sa,K,V)
    t = if !assembled && assemble
        PartitionedArrays.assemble!(B,A,cacheB)
    else
        @fake_async B
    end
    if !val_parameter(split_format)
        return t
    end
    @fake_async begin
        wait(t)
        split_format!(C,B)
        C
    end
end

function pvector_from_split_blocks(own,ghost,row_partition)
    perms = map(local_permutation,row_partition)
    values = map(split_vector,own,ghost,perms)
    PVector(values,row_partition)
end

function old_pvector!(I,V,index_partition;kwargs...)
    old_pvector!(default_local_values,I,V,index_partition;kwargs...)
end

function default_local_values(indices)
    Vector{Float64}(undef,local_length(indices))
end

function default_local_values(I,V,indices)
    values = Vector{eltype(V)}(undef,local_length(indices))
    fill!(values,zero(eltype(values)))
    for k in 1:length(I)
        li = I[k]
        values[li] += V[k]
    end
    values
end

function split_format(A::PVector)
    rows = partition(axes(A,1))
    values = map(split_format_locally,partition(A),rows)
    b = PVector(values,rows)
    b
end

function split_format!(B,A::PVector)
    rows = partition(axes(A,1))
    map(split_format_locally!,partition(B),partition(A),rows)
    B
end

"""
    pfill(v,index_partition)
"""
pfill(v,index_partition;kwargs...) = pvector(indices->fill(v,local_length(indices)),index_partition;kwargs...)

"""
    pzeros([T,]index_partition)

Equivalent to

    pfill(zero(T),index_partition)
"""
pzeros(index_partition;kwargs...) = pzeros(Float64,index_partition;kwargs...)
pzeros(::Type{T},index_partition;kwargs...) where T = pvector(indices->zeros(T,local_length(indices)),index_partition;kwargs...)

"""
    pones([T,]index_partition)

Equivalent to

    pfill(one(T),index_partition)
"""
pones(index_partition;kwargs...) = pones(Float64,index_partition;kwargs...)
pones(::Type{T},index_partition;kwargs...) where T = pvector(indices->ones(T,local_length(indices)),index_partition;kwargs...)

"""
    prand([rng,][s,]index_partition)

Create a [`PVector`](@ref) object with uniform random values and the data partition in `index_partition`.
The optional arguments have the same meaning and default values as in `rand`.
"""
prand(index_partition;kwargs...) = pvector(indices->rand(local_length(indices)),index_partition;kwargs...)
prand(s,index_partition;kwargs...) = pvector(indices->rand(s,local_length(indices)),index_partition;kwargs...)
prand(rng,s,index_partition;kwargs...) = pvector(indices->rand(rng,s,local_length(indices)),index_partition;kwargs...)

"""
    prandn([rng,][s,]index_partition)

Create a [`PVector`](@ref) object with normally distributed random values and the data partition in `index_partition`.
The optional arguments have the same meaning and default values as in `randn`.
"""
prandn(index_partition;kwargs...) = pvector(indices->randn(local_length(indices)),index_partition;kwargs...)
prandn(s,index_partition;kwargs...) = pvector(indices->randn(s,local_length(indices)),index_partition;kwargs...)
prandn(rng,s,index_partition;kwargs...) = pvector(indices->randn(rng,s,local_length(indices)),index_partition;kwargs...)

function Base.:(==)(a::PVector,b::PVector)
    @boundscheck @assert matching_own_indices(axes(a,1),axes(b,1))
    length(a) == length(b) &&
    reduce(&,map(==,own_values(a),own_values(b)),init=true)
end

function Base.any(f::Function,x::PVector)
    partials = map(own_values(x)) do o
        any(f,o)
    end
    reduce(|,partials,init=false)
end

function Base.all(f::Function,x::PVector)
    partials = map(own_values(x)) do o
        all(f,o)
    end
    reduce(&,partials,init=true)
end

Base.maximum(x::PVector) = maximum(identity,x)
function Base.maximum(f::Function,x::PVector)
    partials = map(own_values(x)) do o
        maximum(f,o,init=typemin(eltype(x)))
    end
    reduce(max,partials,init=typemin(eltype(x)))
end

Base.minimum(x::PVector) = minimum(identity,x)
function Base.minimum(f::Function,x::PVector)
    partials = map(own_values(x)) do o
        minimum(f,o,init=typemax(eltype(x)))
    end
    reduce(min,partials,init=typemax(eltype(x)))
end

function Base.collect(v::PVector)
    own_values_v = own_values(v)
    own_to_global_v = map(own_to_global,partition(axes(v,1)))
    vals = gather(own_values_v,destination=:all)
    ids = gather(own_to_global_v,destination=:all)
    n = length(v)
    T = eltype(v)
    map(vals,ids) do myvals,myids
        u = Vector{T}(undef,n)
        for (a,b) in zip(myvals,myids)
            u[b] = a
        end
        u
    end |> getany
end

function Base.:*(a::Number,b::PVector)
    values = map(partition(b)) do values
        a*values
    end
    PVector(values,partition(axes(b,1)))
end

function Base.:*(b::PVector,a::Number)
    a*b
end

function Base.:/(b::PVector,a::Number)
    (1/a)*b
end

for op in (:+,:-)
    @eval begin
        function Base.$op(a::PVector)
            values = map($op,partition(a))
            PVector(values,partition(axes(a,1)))
        end
        function Base.$op(a::PVector,b::PVector)
            $op.(a,b)
        end
    end
end

function neutral_element end
neutral_element(::typeof(+),::Type{T}) where T = zero(T)
neutral_element(::typeof(&),::Type) = true
neutral_element(::typeof(|),::Type) = false
neutral_element(::typeof(min),::Type{T}) where T = typemax(T)
neutral_element(::typeof(max),::Type{T}) where T = typemin(T)

function Base.reduce(op,a::PVector;neutral=neutral_element(op,eltype(a)),kwargs...)
    b = map(own_values(a)) do a
        reduce(op,a,init=neutral)
    end
    reduce(op,b;kwargs...)
end

function Base.sum(a::PVector)
  reduce(+,a,init=zero(eltype(a)))
end

function LinearAlgebra.dot(a::PVector,b::PVector)
    c = map(dot,own_values(a),own_values(b))
    sum(c)
end

function LinearAlgebra.rmul!(a::PVector,v::Number)
    map(partition(a)) do l
        rmul!(l,v)
    end
    a
end

function LinearAlgebra.norm(a::PVector,p::Real=2)
    contibs = map(own_values(a)) do oid_to_value
        norm(oid_to_value,p)^p
    end
    reduce(+,contibs;init=zero(eltype(contibs)))^(1/p)
end

struct BroadcastedPVector{A,B,C}
    own_values::A
    ghost_values::B
    index_partition::C
end
own_values(a::BroadcastedPVector) = a.own_values
ghost_values(a::BroadcastedPVector) = a.ghost_values

function Base.broadcasted(f, args::Union{PVector,BroadcastedPVector}...)
    a1 = first(args)
    @boundscheck @assert all(ai->matching_own_indices(PRange(ai.index_partition),PRange(a1.index_partition)),args)
    own_values_in = map(own_values,args)
    own_values_out = map((largs...)->Base.broadcasted(f,largs...),own_values_in...)
    if all(ai->ai.index_partition===a1.index_partition,args) && !any(ai->ghost_values(ai)===nothing,args)
        ghost_values_in = map(ghost_values,args)
        ghost_values_out = map((largs...)->Base.broadcasted(f,largs...),ghost_values_in...)
    else
        ghost_values_out = nothing
    end
    BroadcastedPVector(own_values_out,ghost_values_out,a1.index_partition)
end

function Base.broadcasted( f, a::Number, b::Union{PVector,BroadcastedPVector})
    own_values_out = map(b->Base.broadcasted(f,a,b),own_values(b))
    if ghost_values(b) !== nothing
        ghost_values_out = map(b->Base.broadcasted(f,a,b),ghost_values(b))
    else
        ghost_values_out = nothing
    end
    BroadcastedPVector(own_values_out,ghost_values_out,b.index_partition)
end

function Base.broadcasted( f, a::Union{PVector,BroadcastedPVector}, b::Number)
    own_values_out = map(a->Base.broadcasted(f,a,b),own_values(a))
    if ghost_values(a) !== nothing
        ghost_values_out = map(a->Base.broadcasted(f,a,b),ghost_values(a))
    else
        ghost_values_out = nothing
    end
    BroadcastedPVector(own_values_out,ghost_values_out,a.index_partition)
end

function Base.broadcasted(f,
                          a::Union{PVector,BroadcastedPVector},
                          b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
    Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
    f,
    a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
    b::Union{PVector,BroadcastedPVector})
    Base.broadcasted(f,Base.materialize(a),b)
 end

function Base.materialize(b::BroadcastedPVector)
    own_values_out = map(Base.materialize,b.own_values)
    T = eltype(eltype(own_values_out))
    a = PVector{Vector{T}}(undef,b.index_partition)
    Base.materialize!(a,b)
    a
end

function Base.materialize!(a::PVector,b::BroadcastedPVector)
    map(Base.materialize!,own_values(a),own_values(b))
    if b.ghost_values !== nothing && a.index_partition === b.index_partition
        map(Base.materialize!,ghost_values(a),ghost_values(b))
    end
    a
end

for M in Distances.metrics
    @eval begin
        function (d::$M)(a::PVector,b::PVector)
            s = distance_eval_body(d,a,b)
            Distances.eval_end(d,s)
        end
    end
end

function distance_eval_body(d,a::PVector,b::PVector)
    if Distances.parameters(d) !== nothing
        error("Only distances without parameters are implemented at this moment")
    end
    partials = map(own_values(a),own_values(b)) do a,b
        @boundscheck if length(a) != length(b)
            throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
        end
        if length(a) == 0
            return zero(Distances.result_type(d, a, b))
        end
        @inbounds begin
            s = Distances.eval_start(d, a, b)
            if (IndexStyle(a, b) === IndexLinear() && eachindex(a) == eachindex(b)) || axes(a) == axes(b)
                for I in eachindex(a, b)
                    ai = a[I]
                    bi = b[I]
                    s = Distances.eval_reduce(d, s, Distances.eval_op(d, ai, bi))
                end
            else
                for (ai, bi) in zip(a, b)
                    s = Distances.eval_reduce(d, s, Distances.eval_op(d, ai, bi))
                end
            end
            return s
        end
    end
    s = reduce((i,j)->Distances.eval_reduce(d,i,j),
               partials,
               init=Distances.eval_start(d, a, b))
    s
end

# New stuff

function assemble(v::PVector;kwargs...)
    rows = map(remove_ghost,partition(axes(v,1)))
    assemble(v,rows;kwargs...)
end

"""
    assemble(v::PVector[,rows];reuse=false)
"""
function assemble(v::PVector,rows;reuse=Val(false))
    @boundscheck @assert matching_own_indices(axes(v,1),PRange(rows))
    # TODO this is just a reference implementation
    # for the moment.
    # The construction of v2 can (should) be avoided
    w = similar(v,PRange(rows))
    v2 = copy(v)
    t = assemble!(v2)
    @fake_async begin
        wait(t)
        w .= v2
        if val_parameter(reuse)
            cache = v2
            w,cache
        else
            w
        end
    end
end

"""
    assemble!(w::PVector,v::PVector,cache)
"""
function assemble!(w::PVector,v::PVector,cache)
    # TODO this is just a reference implementation
    # for the moment.
    # The construction of v2 can (should) be avoided
    v2 = cache
    copy!(v2,v)
    t = assemble!(v2)
    @fake_async begin
        wait(t)
        w .= v2
        w
    end
end

"""
    consistent(v::PVector,rows;reuse=false)
"""
function consistent(v::PVector,rows;reuse=Val(false))
    # TODO this is just a reference implementation
    # for the moment. It can be optimized
    @boundscheck @assert matching_own_indices(axes(v,1),PRange(rows))
    w = similar(v,PRange(rows))
    w .= v
    t = consistent!(w)
    @fake_async begin
        wait(t)
        if val_parameter(reuse)
            w,nothing
        else
            w
        end
    end
end

"""
    consistent!(w::PVector,v::PVector,cache)
"""
function consistent!(w::PVector,v::PVector,cache)
    w .= v
    t = consistent!(w)
    @fake_async begin
        wait(t)
        w
    end
end


function repartition_cache(v::PVector,new_partition)
    rows_da = map(remove_ghost,new_partition)
    row_partition = partition(axes(v,1))
    I = map(collect∘own_to_global,row_partition)
    V = own_values(v)
    I_owner = find_owner(rows_da,I)
    rows_sa = map(union_ghost,rows_da,I,I_owner)
    map(map_global_to_local!,I,rows_sa)
    v_sa = similar(v,PRange(rows_sa))
    (;I,v_sa)
end

"""
    repartition(v::PVector,new_partition;reuse=false)
"""
function repartition(v::PVector,new_partition;reuse=Val(false))
    w = similar(v,PRange(new_partition))
    cache = repartition_cache(v,new_partition)
    t = repartition!(w,v,cache)
    @fake_async begin
        wait(t)
        if val_parameter(reuse) == true
            w, cache
        else
            w
        end
    end
end

function repartition!(w::PVector,v::PVector;kwargs...)
    cache=repartition_cache(v,partition(axes(w,1)))
    repartition!(w,v,cache;kwargs...)
end

"""
    repartition!(w::PVector,v::PVector[,cache];reversed=false)
"""
function repartition!(w::PVector,v::PVector,cache;reversed=false)
    new_partition = partition(axes(w,1))
    old_partition = partition(axes(v,1))
    I = cache.I
    v_sa = cache.v_sa
    if ! reversed
        V = own_values(v)
        fill!(v_sa,0)
        map(setindex!,partition(v_sa),V,I)
        t = assemble!(v_sa)
        return @fake_async begin
            wait(t)
            w .= v_sa
            w
        end
    else
        v_sa .= v
        t = consistent!(v_sa)
        return @fake_async begin
            wait(t)
            map(partition(v_sa),partition(w),I) do v_sa,w,I
                for k in 1:length(I)
                    w[k] = v_sa[I[k]]
                end
            end
            w
        end
    end
end

function find_local_indices(node_to_mask::PVector)
    n_own_dofs = map(count,own_values(node_to_mask))
    n_dofs = sum(n_own_dofs)
    dof_partition = variable_partition(n_own_dofs,n_dofs)
    node_partition = partition(axes(node_to_mask,1))
    node_to_global_dof = pzeros(Int,node_partition)
    function fill_own_dofs!(own_node_to_global_dof,own_node_to_boundary,dofs)
        own_to_global_dof = own_to_global(dofs)
        own_node_to_global_dof[own_node_to_boundary] = own_to_global_dof
    end
    map(fill_own_dofs!,own_values(node_to_global_dof),own_values(node_to_mask),dof_partition)
    consistent!(node_to_global_dof) |> wait
    function add_ghost_dofs(ghost_node_to_global_dof,nodes,dofs)
        ghost_node_to_owner = ghost_to_owner(nodes)
        free_ghost_nodes = findall(global_dof->global_dof!=0,ghost_node_to_global_dof)
        owners = view(ghost_node_to_owner,free_ghost_nodes)
        ghost_dofs = view(ghost_node_to_global_dof,free_ghost_nodes)
        union_ghost(dofs,ghost_dofs,owners)
    end
    dof_partition = map(add_ghost_dofs,ghost_values(node_to_global_dof),node_partition,dof_partition)
    neighbors = assembly_graph(node_partition)
    assembly_neighbors(dof_partition;neighbors)
    node_to_local_dof = pzeros(Int32,node_partition)
    dof_to_local_node = pzeros(Int32,dof_partition)
    function finalize!(local_node_to_global_dof,local_node_to_local_dof,local_dof_to_local_node,dofs)
        global_to_local_dof = global_to_local(dofs)
        n_local_nodes = length(local_node_to_global_dof)
        for local_node in 1:n_local_nodes
            global_dof = local_node_to_global_dof[local_node]
            if global_dof == 0
                continue
            end
            local_dof = global_to_local_dof[global_dof]
            local_node_to_local_dof[local_node] = local_dof
            local_dof_to_local_node[local_dof] = local_node
        end
    end
    map(finalize!,partition(node_to_global_dof),partition(node_to_local_dof),partition(dof_to_local_node),dof_partition)
    dof_to_local_node, node_to_local_dof
end

function renumber(a::PVector;kwargs...)
    row_partition = partition(axes(a,1))
    row_partition_2 = renumber_partition(row_partition;kwargs...)
    renumber(a,row_partition_2;kwargs...)
end

function renumber(a::PVector,row_partition_2;renumber_local_indices=Val(true))
    if val_parameter(renumber_local_indices)
        perms = map(row_partition_2) do myrows
            Int32(1):Int32(local_length(myrows))
        end
        values = map(split_vector,own_values(a),ghost_values(a),perms)
    else
        values = local_values(a)
    end
    PVector(values,row_partition_2)
end

