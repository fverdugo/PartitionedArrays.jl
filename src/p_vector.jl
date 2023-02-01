
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

"""
    struct OwnAndGhostVectors{A,C,T}

Vector type that stores the local values of a [`PVector`](@ref) instance
using a vector of own values, a vector of ghost values, and a permutation.
This is not the default data layout of [`PVector`](@ref) (which is a plain vector),
but corresponds to the layout of distributed vectors in other packages, such as PETSc.
Use this type to avoid duplicating memory when passing data to these other packages.

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
"""
function local_values(a::PVector)
    partition(a)
end

"""
    own_values(a::PVector)

Get a vector of vectors containing the own values
in each part of `a`.
"""
function own_values(a::PVector)
    map(own_values,partition(a),partition(axes(a,1)))
end

"""
    ghost_values(a::PVector)

Get a vector of vectors containing the ghost values
in each part of `a`.
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
        println(io,"$n-element PVector{$T} partitioned into $np parts")
    end
end

function p_vector_cache(vector_partition,index_partition)
    p_vector_cache_impl(eltype(vector_partition),vector_partition,index_partition)
end

struct VectorAssemblyCache{T}
    neighbors_snd::Vector{Int32}
    neighbors_rcv::Vector{Int32}
    local_indices_snd::JaggedArray{Int32,Int32}
    local_indices_rcv::JaggedArray{Int32,Int32}
    buffer_snd::JaggedArray{T,Int32}
    buffer_rcv::JaggedArray{T,Int32}
end
function Base.reverse(a::VectorAssemblyCache)
    VectorAssemblyCache(
                    a.neighbors_rcv,
                    a.neighbors_snd,
                    a.local_indices_rcv,
                    a.local_indices_snd,
                    a.buffer_rcv,
                    a.buffer_snd)
end
function copy_cache(a::VectorAssemblyCache)
    buffer_snd = JaggedArray(copy(a.buffer_snd.data),a.buffer_snd.ptrs)
    buffer_rcv = JaggedArray(copy(a.buffer_rcv.data),a.buffer_rcv.ptrs)
    VectorAssemblyCache(
                    a.neighbors_snd,
                    a.neighbors_rcv,
                    a.local_indices_snd,
                    a.local_indices_rcv,
                    buffer_snd,
                    buffer_rcv)
end
function p_vector_cache_impl(::Type,vector_partition,index_partition)
    neighbors = assembly_neighbors(index_partition)
    indices = assembly_local_indices(index_partition,neighbors...)
    buffers = map(assembly_buffers,vector_partition,indices...) |> tuple_of_arrays
    map(VectorAssemblyCache,neighbors...,indices...,buffers...)
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

struct JaggedArrayAssemblyCache
    cache::VectorAssemblyCache
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
    neighbors = assembly_neighbors(index_partition)
    local_indices_snd, local_indices_rcv = assembly_local_indices(index_partition,neighbors...)
    p_snd = map(data_index_snd,local_indices_snd,vector_partition)
    p_rcv = map(data_index_snd,local_indices_rcv,vector_partition)
    data = map(getdata,vector_partition)
    buffers = map(assembly_buffers,data,p_snd,p_rcv) |> tuple_of_arrays
    cache = map(VectorAssemblyCache,neighbors...,p_snd,p_rcv,buffers...)
    map(JaggedArrayAssemblyCache,cache)
end


function assemble!(f,vector_partition,cache)
    assemble_impl!(f,vector_partition,cache,eltype(cache))
end

function assemble_impl!(f,vector_partition,cache,::Type{<:VectorAssemblyCache})
    buffer_snd = map(vector_partition,cache) do values,cache
        local_indices_snd = cache.local_indices_snd
        buffer_snd = cache.buffer_snd
        for (p,lid) in enumerate(local_indices_snd.data)
            buffer_snd.data[p] = values[lid]
        end
        buffer_snd
    end
    neighbors_snd, neighbors_rcv, buffer_rcv = map(cache) do cache
        cache.neighbors_snd, cache.neighbors_rcv, cache.buffer_rcv
    end |> tuple_of_arrays
    graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
    t = exchange!(buffer_rcv,buffer_snd,graph)
    # Fill values from rcv buffer asynchronously
    @async begin
        wait(t)
        map(vector_partition,cache) do values,cache
            local_indices_rcv = cache.local_indices_rcv
            buffer_rcv = cache.buffer_rcv
            for (p,lid) in enumerate(local_indices_rcv.data)
                values[lid] = f(values[lid],buffer_rcv.data[p])
            end
        end
        nothing
    end
end

function assemble_impl!(f,vector_partition,cache,::Type{<:JaggedArrayAssemblyCache})
    vcache = map(i->i.cache,cache)
    data = map(getdata,vector_partition)
    assemble!(f,data,vcache)
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
6-element PVector{Vector{Float64}} partitioned into 2 parts

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
    @async begin
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
6-element PVector{Vector{Int32}} partitioned into 2 parts

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
    insert(a,b) = b
    cache = map(reverse,a.cache)
    t = assemble!(insert,partition(a),cache)
    @async begin
        wait(t)
        a
    end
end


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
function pvector(f,index_partition)
    vector_partition = map(f,index_partition)
    PVector(vector_partition,index_partition)
end

"""
    pvector!([f,]I,V,index_partition;discover_rows=true) -> Task

Crate an instance of [`PVector`](@ref) by setting arbitrary entries
from each of the underlying parts. It returns a task that produces the
instance of [`PVector`](@ref) allowing latency hiding while performing
the communications needed in its setup.
"""
function pvector!(f,I,V,index_partition;discover_rows=true)
    if discover_rows
        I_owner = find_owner(index_partition,I)
        index_partition = map(union_ghost,index_partition,I,I_owner)
    end
    map(to_local!,I,index_partition)
    vector_partition = map(f,I,V,index_partition)
    v = PVector(vector_partition,index_partition)
    assemble!(v)
end

function pvector!(I,V,index_partition;kwargs...)
    pvector!(default_local_values,I,V,index_partition;kwargs...)
end

function default_local_values(indices)
    Vector{Float64}(undef,local_length(indices))
end

function default_local_values(I,V,indices)
    values = Vector{Float64}(undef,local_length(indices))
    fill!(values,zero(eltype(values)))
    for k in 1:length(I)
        li = I[k]
        values[li] += V[k]
    end
    values
end

"""
    pfill(v,index_partition)
"""
pfill(v,index_partition) = pvector(indices->fill(v,local_length(indices)),index_partition)

"""
    pzeros([T,]index_partition)

Equivalent to

    pfill(zero(T),index_partition)
"""
pzeros(index_partition) = pzeros(Float64,index_partition)
pzeros(::Type{T},index_partition) where T = pvector(indices->zeros(T,local_length(indices)),index_partition)

"""
    pones([T,]index_partition)

Equivalent to

    pfill(one(T),index_partition)
"""
pones(index_partition) = pones(Float64,index_partition)
pones(::Type{T},index_partition) where T = pvector(indices->ones(T,local_length(indices)),index_partition)

"""
    prand([rng,][s,]index_partition)

Create a [`PVector`](@ref) object with uniform random values and the data partition in `index_partition`.
The optional arguments have the same meaning and default values as in `rand`.
"""
prand(index_partition) = pvector(indices->rand(local_length(indices)),index_partition)
prand(s,index_partition) = pvector(indices->rand(s,local_length(indices)),index_partition)
prand(rng,s,index_partition) = pvector(indices->rand(rng,s,local_length(indices)),index_partition)

"""
    prandn([rng,][s,]index_partition)

Create a [`PVector`](@ref) object with normally distributed random values and the data partition in `index_partition`.
The optional arguments have the same meaning and default values as in `randn`.
"""
prandn(index_partition) = pvector(indices->randn(local_length(indices)),index_partition)
prandn(s,index_partition) = pvector(indices->randn(s,local_length(indices)),index_partition)
prandn(rng,s,index_partition) = pvector(indices->randn(rng,s,local_length(indices)),index_partition)

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

struct PBroadcasted{A,B,C}
    own_values::A
    ghost_values::B
    index_partition::C
end
own_values(a::PBroadcasted) = a.own_values
ghost_values(a::PBroadcasted) = a.ghost_values

function Base.broadcasted(f, args::Union{PVector,PBroadcasted}...)
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
    PBroadcasted(own_values_out,ghost_values_out,a1.index_partition)
end

function Base.broadcasted( f, a::Number, b::Union{PVector,PBroadcasted})
    own_values_out = map(b->Base.broadcasted(f,a,b),own_values(b))
    if ghost_values(b) !== nothing
        ghost_values_out = map(b->Base.broadcasted(f,a,b),ghost_values(b))
    else
        ghost_values_out = nothing
    end
    PBroadcasted(own_values_out,ghost_values_out,b.index_partition)
end

function Base.broadcasted( f, a::Union{PVector,PBroadcasted}, b::Number)
    own_values_out = map(a->Base.broadcasted(f,a,b),own_values(a))
    if ghost_values(a) !== nothing
        ghost_values_out = map(a->Base.broadcasted(f,a,b),ghost_values(a))
    else
        ghost_values_out = nothing
    end
    PBroadcasted(own_values_out,ghost_values_out,a.index_partition)
end

function Base.broadcasted(f,
                          a::Union{PVector,PBroadcasted},
                          b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
    Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
    f,
    a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
    b::Union{PVector,PBroadcasted})
    Base.broadcasted(f,Base.materialize(a),b)
 end

function Base.materialize(b::PBroadcasted)
    own_values_out = map(Base.materialize,b.own_values)
    T = eltype(eltype(own_values_out))
    a = PVector{Vector{T}}(undef,b.index_partition)
    Base.materialize!(a,b)
    a
end

function Base.materialize!(a::PVector,b::PBroadcasted)
    map(Base.materialize!,own_values(a),own_values(b))
    if b.ghost_values !== nothing && a.index_partition === b.index_partition
        map(Base.materialize!,ghost_values(a),ghost_values(b))
    end
    a
end

for M in Distances.metrics
    @eval begin
        function (d::$M)(a::PVector,b::PVector)
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
                        @simd for I in eachindex(a, b)
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
            Distances.eval_end(d,s)
        end
    end
end

