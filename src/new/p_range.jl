"""
    abstract type AbstractLocalIndices

Abstract type representing the *local*, *own*, and *ghost* indices in
a part of an instance of [`PRange`](@ref).

The following functions form the `AbstractLocalIndices` interface:

- [`get_n_local`](@ref)
- [`get_n_own`](@ref)
- [`get_n_ghost`](@ref)
- [`get_n_global`](@ref)
- [`get_owner`](@ref)
- [`get_local_to_global`](@ref)
- [`get_own_to_global`](@ref)
- [`get_ghost_to_global`](@ref)
- [`get_local_to_owner`](@ref)
- [`get_own_to_owner`](@ref)
- [`get_ghost_to_owner`](@ref)
- [`get_global_to_local`](@ref)
- [`get_global_to_own`](@ref)
- [`get_global_to_ghost`](@ref)
- [`get_own_to_local`](@ref)
- [`get_ghost_to_local`](@ref)
- [`get_local_to_own`](@ref)
- [`get_local_to_ghost`](@ref)
- [`replace_ghost`](@ref)
- [`union_ghost`](@ref)

# Supertype hierarchy

    AbstractLocalIndices <: AbstractVector{Int}

"""
abstract type AbstractLocalIndices <: AbstractVector{Int} end
Base.size(a::AbstractLocalIndices) = (get_n_local(a),)
Base.IndexStyle(::Type{<:AbstractLocalIndices}) = IndexLinear()
@inline Base.getindex(a::AbstractLocalIndices,i::Int) = get_local_to_global(a)[i]

"""
    get_n_local(indices)

Get number of local ids in `indices`.
"""
get_n_local(a) = get_n_own(a) + get_n_ghost(a)

"""
    get_n_own(indices)

Get number of own ids in `indices`.
"""
get_n_own(a) = length(get_own_to_owner(a))

"""
    get_n_ghost(indices)

Get number of ghost ids in `indices`.
"""
get_n_ghost(a) = length(get_ghost_to_global(a))

"""
    get_n_global(indices)

Get number of global ids associated with `indices`.
"""
get_n_global(a) = length(get_global_to_own(a))

"""
    get_owner(indices)

Return the id of the part that is storing `indices`.
"""
function get_owner end

"""
    get_local_to_global(indices)

Return an array with the global indices of the local indices in `indices`.
"""
function get_local_to_global end

"""
    get_own_to_global(indices)

Return an array with the global indices of the own indices in `indices`.
"""
function get_own_to_global end

"""
    get_ghost_to_global(indices)

Return an array with the global indices of the ghost indices in `indices`.
"""
function get_ghost_to_global end

"""
    get_local_to_owner(indices)

Return an array with the owners of the local indices in `indices`.
"""
function get_local_to_owner end

"""
    get_own_to_owner(indices)

Return an array with the owners of the own indices in `indices`.
"""
function get_own_to_owner end

"""
    get_ghost_to_owner(indices)

Return an array with the owners of the ghost indices in `indices`.
"""
function get_ghost_to_owner end

"""
    get_global_to_local(indices)

Return an array with the inverse index map of `get_local_to_global(indices)`.
"""
function get_global_to_local end

"""
    get_global_to_own(indices)

Return an array with the inverse index map of `get_own_to_global(indices)`.
"""
function get_global_to_own end

"""
    get_global_to_ghost(indices)

Return an array with the inverse index map of `get_ghost_to_global(indices)`.
"""
function get_global_to_ghost end

"""
    get_own_to_local(indices)

Return an array with the local ids of the own indices in `indices`.
"""
function get_own_to_local end

"""
    get_ghost_to_local(indices)

Return an array with the local ids of the ghost indices in `indices`.
"""
function get_ghost_to_local end

"""
    get_local_to_own(indices)

Return an array with the inverse index map of `get_own_to_local(indices)`.
"""
function get_local_to_own end

"""
    get_local_to_ghost(indices)
Return an array with the inverse index map of `get_ghost_to_local(indices)`.
"""
function get_local_to_ghost end

function get_permutation(indices)
    n_local = get_n_local(indices)
    n_own = get_n_own(indices)
    n_ghost = get_n_ghost(indices)
    own_to_local = get_own_to_local(indices)
    ghost_to_local = get_ghost_to_local(indices)
    perm = zeros(Int32,n_local)
    perm[own_to_local] = 1:n_own
    perm[ghost_to_local] = (1:n_ghost) .+ n_own
    perm
end

function matching_local_indices(a,b)
    a === b && return true
    get_local_to_global(a) == get_local_to_global(b) &&
    get_local_to_owner(a) == get_local_to_owner(b)
end

function matching_own_indices(a,b)
    a === b && return true
    get_own_to_global(a) == get_own_to_global(b) &&
    get_owner(a) == get_owner(b)
end

function matching_ghost_indices(a,b)
    a === b && return true
    get_ghost_to_global(a) == get_ghost_to_global(b) &&
    get_ghost_to_owner(a) == get_ghost_to_owner(b)
end

"""
    replace_ghost(indices,gids,owners)

Replaces the ghost indices in `indices` with global ids in `gids` and owners in 
 `owners`. Returned object takes ownership of `gids`  and `owners`. This method 
only makes sense if `indices` stores ghost ids in separate vectors like in
[`OwnAndGhostIndices`](@ref). `gids` should be unique and not being owned by
 `indices`.
"""
function replace_ghost end

function filter_ghost(indices,gids,owners)
    set = Set{Int}()
    part_owner = get_owner(indices)
    n_new_ghost = 0
    global_to_ghost = get_global_to_ghost(indices)
    for (global_i,owner) in zip(gids,owners)
        if owner != part_owner
            ghost_i = global_to_ghost[global_i]
            if ghost_i == 0 && !(global_i in set)
                n_new_ghost += 1
                push!(set,global_i)
            end
        end
    end
    new_ghost_to_global = zeros(Int,n_new_ghost)
    new_ghost_to_owner = zeros(Int32,n_new_ghost)
    new_ghost_i = 0
    set = Set{Int}()
    for (global_i,owner) in zip(gids,owners)
        if owner != part_owner
            ghost_i = global_to_ghost[global_i]
            if ghost_i == 0 && !(global_i in set)
                new_ghost_i += 1
                new_ghost_to_global[new_ghost_i] = global_i
                new_ghost_to_owner[new_ghost_i] = owner
                push!(set,global_i)
            end
        end
    end
    new_ghost_to_global, new_ghost_to_owner
end

"""
    union_ghost(indices,gids,owners)

Make the union of the ghost indices in `indices` with 
 the global indices `gids` and owners `owners`.
 Return an object  of the same type as `indices` with the new ghost indices and the same
 own indices as in `indices`.
 The result does not take ownership of `gids`  and `owners`. 
"""
function union_ghost(indices,gids,owners)
    extra_gids, extra_owners = filter_ghost(indices,gids,owners)
    ghost_to_global = get_ghost_to_global(indices)
    ghost_to_owner = get_ghost_to_owner(indices)
    new_gids = vcat(ghost_to_global,extra_gids)
    new_owners = vcat(ghost_to_owner,extra_owners)
    n_global = get_n_global(indices)
    ghost = GhostIndices(n_global,new_gids,new_owners)
    replace_ghost(indices,ghost)
end

function to_local!(I,indices)
    global_to_local = get_global_to_local(indices)
    for k in 1:length(I)
        I[k] = global_to_local[I[k]]
    end
    I
end

function to_global!(I,indices)
    local_to_global = get_local_to_global(indices)
    for k in 1:length(I)
        I[k] = local_to_global[I[k]]
    end
    I
end

"""
    find_owner(indices,global_ids)

Find the owners of the global ids in `global_ids`. The input `global_ids` is
a vector of vectors distributed over the same parts as `pr`. Each part will
look for the owners in parallel, when using a parallel back-end.

# Example


    julia> using PartitionedArrays
    
    julia> rank = LinearIndices((4,));
    
    julia> pr = PRange(ConstantBlockSize(),rank,4,10)
    1:1:10
    
    julia> gids = [[3],[4,5],[7,2],[9,10,1]];
    
    julia> find_owner(pr,gids)
    4-element Vector{Vector{Int32}}:
     [2]
     [2, 3]
     [3, 1]
     [4, 4, 1]
"""
function find_owner(indices,global_ids)
    find_owner(indices,global_ids,eltype(indices))
end

struct AssemblyCache
    neighbors_snd::Base.RefValue{Vector{Int32}}
    neighbors_rcv::Base.RefValue{Vector{Int32}}
    local_indices_snd::Base.RefValue{JaggedArray{Int32,Int32}}
    local_indices_rcv::Base.RefValue{JaggedArray{Int32,Int32}}
end

function Base.copy!(a::AssemblyCache,b::AssemblyCache)
    a.neighbors_snd[] = b.neighbors_snd[]
    a.neighbors_rcv[] = b.neighbors_rcv[]
    a.local_indices_snd[] = b.local_indices_snd[]
    a.local_indices_rcv[] = b.local_indices_rcv[]
    a
end

function AssemblyCache()
    AssemblyCache(
                  Ref{Vector{Int32}}(),
                  Ref{Vector{Int32}}(),
                  Ref{JaggedArray{Int32,Int32}}(),
                  Ref{JaggedArray{Int32,Int32}}()
                 )
end

assembly_cache(a) = AssemblyCache()

function empty_assembly_cache()
    AssemblyCache(
                  Ref(Int32[]),
                  Ref(Int32[]),
                  Ref(JaggedArray(Int32[],Int32[1])),
                  Ref(JaggedArray(Int32[],Int32[1])),
                 )
end

function assembly_neighbors(indices;kwargs...)
    cache = map(assembly_cache,indices)
    mask =  map(cache) do cache
        isassigned(cache.neighbors_snd) && isassigned(cache.neighbors_rcv)
    end
    if ! getany(mask)
        neighbors_snd, neighbors_rcv = compute_assembly_neighbors(indices;kwargs...)
        map(cache,neighbors_snd,neighbors_rcv) do cache, neigs_snd, neigs_rcv
            cache.neighbors_snd[] = neigs_snd
            cache.neighbors_rcv[] = neigs_rcv
        end
        return neighbors_snd, neighbors_rcv
    end
    neigs_snd, neigs_rcv = map(cache) do cache
        cache.neighbors_snd[], cache.neighbors_rcv[]
    end |> tuple_of_arrays
    neigs_snd, neigs_rcv
end

function compute_assembly_neighbors(indices;kwargs...)
    parts_snd = map(indices) do indices
        rank = get_owner(indices)
        local_to_owner = get_local_to_owner(indices)
        set = Set{Int32}()
        for owner in local_to_owner
            if owner != rank
                push!(set,owner)
            end
        end
        sort(collect(set))
    end
    graph = ExchangeGraph(parts_snd;kwargs...)
    graph.snd, graph.rcv
end

function assembly_local_indices(indices,neighbors_snd,neighbors_rcv)
    cache = map(assembly_cache,indices)
    mask =  map(cache) do cache
        isassigned(cache.local_indices_snd) && isassigned(cache.local_indices_rcv)
    end
    if ! getany(mask)
        local_indices_snd, local_indices_rcv = compute_assembly_local_indices(indices,neighbors_snd,neighbors_rcv)
        map(cache,local_indices_snd,local_indices_rcv) do cache, local_indices_snd, local_indices_rcv
            cache.local_indices_snd[] = local_indices_snd
            cache.local_indices_rcv[] = local_indices_rcv
        end
        return local_indices_snd, local_indices_rcv
    end
    local_indices_snd, local_indices_rcv = map(cache) do cache
        cache.local_indices_snd[], cache.local_indices_rcv[]
    end |> tuple_of_arrays
    local_indices_snd, local_indices_rcv
end

function compute_assembly_local_indices(indices,neighbors_snd,neighbors_rcv)
    parts_snd = neighbors_snd
    parts_rcv = neighbors_rcv
    local_indices_snd, global_indices_snd = map(indices,parts_snd) do indices,parts_snd
        rank = get_owner(indices)
        local_to_owner = get_local_to_owner(indices)
        owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
        ptrs = zeros(Int32,length(parts_snd)+1)
        for owner in local_to_owner
            if owner != rank
                ptrs[owner_to_i[owner]+1] +=1
            end
        end
        length_to_ptrs!(ptrs)
        data_lids = zeros(Int32,ptrs[end]-1)
        data_gids = zeros(Int,ptrs[end]-1)
        local_to_global = get_local_to_global(indices)
        for (lid,owner) in enumerate(local_to_owner)
            if owner != rank
                p = ptrs[owner_to_i[owner]]
                data_lids[p]=lid
                data_gids[p]=local_to_global[lid]
                ptrs[owner_to_i[owner]] += 1
            end
        end
        rewind_ptrs!(ptrs)
        local_indices_snd = JaggedArray(data_lids,ptrs)
        global_indices_snd = JaggedArray(data_gids,ptrs)
        local_indices_snd, global_indices_snd
    end |>  tuple_of_arrays
    graph = ExchangeGraph(parts_snd,parts_rcv)
    global_indices_rcv = exchange_fetch(global_indices_snd,graph)
    local_indices_rcv = map(global_indices_rcv,indices) do global_indices_rcv,indices
        ptrs = global_indices_rcv.ptrs
        data_lids = zeros(Int32,ptrs[end]-1)
        global_to_local = get_global_to_local(indices)
        for (k,gid) in enumerate(global_indices_rcv.data)
            data_lids[k] = global_to_local[gid]
        end
        local_indices_rcv = JaggedArray(data_lids,ptrs)
    end
    local_indices_snd,local_indices_rcv
end

"""
    uniform_partition(ranks,np,n[,ghost[,periodic]])

Generate an `N` dimensional
block partition with a (roughly) constant block size.

# Arguments
- `ranks`: Array containing the distribution of ranks.
-  `np::NTuple{N}`: Number of parts per direction.
-  `n::NTuple{N}`: Number of global indices per direction.
-  `ghost::NTuple{N}=ntuple(i->false,N)`: Use or not ghost indices per direction.
-  `periodic::NTuple{N}=ntuple(i->false,N)`: Use or not periodic boundaries per direction.

For convenience, one can also provide scalar inputs instead tuples
to create 1D block partitions.

# Examples

2D partition of 4x4 indices into 2x2 parts with ghost

    julia> using PartitionedArrays
    
    julia> rank = LinearIndices((4,));
    
    julia> pr = uniform_partition(rank,(2,2),(4,4),(true,true))
    1:1:16
    
    julia> get_local_to_global(pr)
    4-element Vector{PartitionedArrays.BlockPartitionLocalToGlobal{2, Vector{Int32}}}:
     [1, 2, 3, 5, 6, 7, 9, 10, 11]
     [2, 3, 4, 6, 7, 8, 10, 11, 12]
     [5, 6, 7, 9, 10, 11, 13, 14, 15]
     [6, 7, 8, 10, 11, 12, 14, 15, 16]

"""
function uniform_partition(rank,np,n,args...)
    @assert prod(np) == length(rank)
    indices = map(rank) do rank
        block_with_constant_size(rank,np,n,args...)
    end
    if length(args) == 0
        map(indices) do indices
            cache = assembly_cache(indices)
            copy!(cache,empty_assembly_cache())
        end
    else
        assembly_neighbors(indices;symmetric=true)
    end
    indices
end

"""
    uniform_partition(ranks,n::Integer[,ghost::Bool[,periodic::Bool]])

Generate an  1d dimensional
block partition with a (roughly) constant block size by inferring the number of parts to use from `ranks`.

# Arguments
- `ranks`: Array containing the distribution of ranks. The number of parts is taken as `length(ranks)`.
-  `n`: Number of global indices.
-  `ghost`: Use or not ghost indices.
-  `periodic`: Use or not periodic boundaries.
"""
function uniform_partition(rank,n::Integer)
    uniform_partition(rank,length(rank),n)
end

function uniform_partition(rank,n::Integer,ghost::Bool,periodic::Bool=false)
    uniform_partition(rank,length(rank),n,ghost,periodic)
end

function uniform_partition(rank,np::Integer,n::Integer)
    uniform_partition(rank,(np,),(n,))
end

function uniform_partition(rank,np::Integer,n::Integer,ghost::Bool,periodic::Bool=false)
    uniform_partition(rank,(np,),(n,),(ghost,),(periodic,))
end

function block_with_constant_size(rank,np,n)
    N = length(n)
    p = CartesianIndices(np)[rank]
    ghost = GhostIndices(prod(n))
    LocalIndicesWithConstantBlockSize(p,np,n,ghost)
end

function block_with_constant_size(rank,np,n,ghost,periodic=map(i->false,ghost))
    N = length(n)
    p = CartesianIndices(np)[rank]
    own_ranges = map(local_range,Tuple(p),np,n)
    local_ranges = map(local_range,Tuple(p),np,n,ghost,periodic)
    owners = map(Tuple(p),own_ranges,local_ranges) do p,or,lr
        owners = zeros(Int32,length(lr))
        for i in 1:length(lr)
            if lr[i] in or
                owners[i] = p
            end
        end
        if owners[1] == 0
            owners[1] = p-1
        end
        if owners[end] == 0
            owners[end] = p+1
        end
        owners
    end
    n_ghost = 0
    cis = CartesianIndices(map(length,local_ranges))
    predicate(p,i,owners) = owners[i] == p
    for ci in cis
        flags = map(predicate,Tuple(p),Tuple(ci),owners)
        if !all(flags)
            n_ghost += 1
        end
    end
    ghost_to_global = zeros(Int,n_ghost)
    ghost_to_owner = zeros(Int32,n_ghost)
    n_local = prod(map(length,local_ranges))
    perm = zeros(Int32,n_local)
    i_ghost = 0
    i_own = 0
    n_own = prod(map(length,own_ranges))
    lis = CircularArray(LinearIndices(n))
    local_cis = CartesianIndices(local_ranges)
    owner_lis = CircularArray(LinearIndices(np))
    for (i,ci) in enumerate(cis)
        flags = map(predicate,Tuple(p),Tuple(ci),owners)
        if !all(flags)
            i_ghost += 1
            ghost_to_global[i_ghost] = lis[local_cis[i]]
            o = map(getindex,owners,Tuple(ci))
            o_ci = CartesianIndex(o)
            ghost_to_owner[i_ghost] = owner_lis[o_ci]
            perm[i] = i_ghost + n_own
        else
            i_own += 1
            perm[i] = i_own
        end
    end
    ghostids = GhostIndices(prod(n),ghost_to_global,ghost_to_owner)
    ids = LocalIndicesWithConstantBlockSize(p,np,n,ghostids)
    PermutedLocalIndices(ids,perm)
end

"""
    variable_partition(n_own,n_global[;start])

Build a 1D variable-size block partition.

# Arguments

-  `n_own::AbstractArray{<:Integer}`: Array containing the block size for each part.
-  `n_global::Integer`: Number of global indices. It should be equal to `sum(n_own)`.
-  `start::AbstractArray{Int}=scan(+,n_own,type=:exclusive,init=1)`: First global index in each part.

We ask the user to provide `n_global` and (optionally) `start` since discovering them requires communications.

# Examples

    julia> using PartitionedArrays
    
    julia> rank = LinearIndices((4,));
    
    julia> n_own = [3,2,2,3];
    
    julia> pr = variable_partition(n_own,sum(n_own))
    1:1:10
    
    julia> get_own_to_global(pr)
    4-element Vector{PartitionedArrays.BlockPartitionOwnToGlobal{1}}:
     [1, 2, 3]
     [4, 5]
     [6, 7]
     [8, 9, 10]

"""
function variable_partition(
    n_own,
    n_global,
    ghost=false,
    periodic=false;
    start=scan(+,n_own,type=:exclusive,init=one(eltype(n_own))))
    rank = linear_indices(n_own)
    if ghost == true || periodic == true
        error("This case is not yet implemented.")
    end
    n_parts = length(n_own)
    indices = map(rank,n_own,start) do rank,n_own,start
        p = CartesianIndex((rank,))
        np = (n_parts,)
        n = (n_global,)
        ranges = ((1:n_own).+(start-1),)
        ghost = GhostIndices(n_global)
        indices = LocalIndicesWithVariableBlockSize(p,np,n,ranges,ghost)
        # This should be changed when including ghost
        cache = assembly_cache(indices)
        copy!(cache,empty_assembly_cache())
        indices
    end
    indices
end

struct VectorFromDict{Tk,Tv} <: AbstractVector{Tv}
    dict::Dict{Tk,Tv}
    length::Int
end
Base.IndexStyle(::Type{<:VectorFromDict}) = IndexLinear()
Base.size(a::VectorFromDict) = (Int(a.length),)
function Base.getindex(a::VectorFromDict,i::Int)
    Tv = eltype(a)
    haskey(a.dict,i) || return zero(Tv)
    a.dict[i]
end
function Base.setindex!(a::VectorFromDict,v,i::Int)
    a.dict[i] = v
    v
end

function VectorFromDict(ids,vals,n)
    Tk = eltype(ids)
    Tv = eltype(vals)
    dict = Dict{Tk,Tv}()
    @assert length(ids) == length(vals)
    for i in 1:length(ids)
        dict[ids[i]] = vals[i]
    end
    VectorFromDict(dict,n)
end

"""
    struct OwnIndices

Container for own indices.

# Properties

- `n_global::Int`: Number of global indices
- `owner::Int32`: Id of the part that owns these indices
- `own_to_global::Vector{Int}`: Global ids of the indices owned by this part. `own_to_global[i_own]` is the global id corresponding to the own index number `i_own`. 

# Supertype hierarchy

    OwnIndices <: Any

"""
struct OwnIndices
    n_global::Int
    owner::Int32
    own_to_global::Vector{Int}
    global_to_own::VectorFromDict{Int,Int32}
end

"""
    OwnIndices(n_global,owner,own_to_global)

Build an instance of [`OwnIndices`](@ref) from the underlying properties `n_global`,
`owner`, and `own_to_global`. The types of these variables need to match
the type of the properties in [`OwnIndices`](@ref).
"""
function OwnIndices(n_global::Int,owner::Integer,own_to_global::Vector{Int})
    n_own = length(own_to_global)
    global_to_own = VectorFromDict(
      own_to_global,Int32.(1:n_own),n_global)
    OwnIndices(n_global,Int32(owner),own_to_global,global_to_own)
end

"""
    struct GhostIndices

Container for ghost indices.

# Properties

- `n_global::Int`: Number of global indices
- `ghost_to_global::Vector{Int}`: Global ids of the ghost indices in this part. `ghost_to_global[i_ghost]` is the global id corresponding to the ghost index number `i_ghost`. 
- `ghost_to_owner::Vector{Int32}`: Owners of the ghost ids. `ghost_to_owner[i_ghost]`is the id of the owner of the ghost index number `i_ghost`.

# Supertype hierarchy

    GhostIndices <: Any
"""
struct GhostIndices
    n_global::Int
    ghost_to_global::Vector{Int}
    ghost_to_owner::Vector{Int32}
    global_to_ghost::VectorFromDict{Int,Int32}
end

"""
    GhostIndices(n_global,ghost_to_global,ghost_to_owner)

Build an instance of [`GhostIndices`](@ref) from the underlying fields `n_global`,
`ghost_to_global`, and `ghost_to_owner`.
The types of these variables need to match
the type of the properties in [`GhostIndices`](@ref).
"""
function GhostIndices(n_global,ghost_to_global,ghost_to_owner)
    n_ghost = length(ghost_to_global)
    @assert length(ghost_to_owner) == n_ghost
    global_to_ghost = VectorFromDict(
      ghost_to_global,Int32.(1:n_ghost),n_global)
    GhostIndices(
      n_global,ghost_to_global,ghost_to_owner,global_to_ghost)
end

"""
    GhostIndices(n_global)

Build an empty instance of [`GhostIndices`](@ref) for a range of `n_global` indices.
"""
function GhostIndices(n_global)
    ghost_to_global = Int[]
    ghost_to_owner = Int32[]
    GhostIndices(n_global,ghost_to_global,ghost_to_owner)
end

function replace_ghost(indices,gids,owners)
    n_global = get_n_global(indices)
    ghost = GhostIndices(n_global,gids,owners)
    replace_ghost(indices,ghost)
end

# This is essentially a FillArray
# but we add this to improve stack trace
struct OwnToOwner <: AbstractVector{Int32}
    owner::Int32
    n_own::Int
end
Base.IndexStyle(::Type{<:OwnToOwner}) = IndexLinear()
Base.size(a::OwnToOwner) = (Int(a.n_own),)
function Base.getindex(a::OwnToOwner,own_id::Int)
    a.owner
end

struct GlobalToLocal{A,B,C} <: AbstractVector{Int32}
    global_to_own::A
    global_to_ghost::VectorFromDict{Int,Int32}
    own_to_local::B
    ghost_to_local::C
end
Base.size(a::GlobalToLocal) = size(a.global_to_own)
Base.IndexStyle(::Type{<:GlobalToLocal}) = IndexLinear()
function Base.getindex(a::GlobalToLocal,global_id::Int)
    own_id = a.global_to_own[global_id]
    z = Int32(0)
    if own_id != z
        return a.own_to_local[own_id]
    end
    ghost_id = a.global_to_ghost[global_id]
    if ghost_id != z
        return a.ghost_to_local[ghost_id]
    end
    return z
end

struct LocalToOwn{A} <: AbstractVector{Int32}
    n_own::Int
    perm::A
end
Base.size(a::LocalToOwn) = (length(a.perm),)
Base.IndexStyle(::Type{<:LocalToOwn}) = IndexLinear()
function Base.getindex(a::LocalToOwn,local_id::Int)
    i = a.perm[local_id]
    if i > a.n_own
        Int32(0)
    else
        Int32(i)
    end
end

struct LocalToGhost{A} <: AbstractVector{Int32}
    n_own::Int
    perm::A
end
Base.size(a::LocalToGhost) = (length(a.perm),)
Base.IndexStyle(::Type{<:LocalToGhost}) = IndexLinear()
function Base.getindex(a::LocalToGhost,local_id::Int)
    i = a.perm[local_id]
    if i > a.n_own
        Int32(i-a.n_own)
    else
        Int32(0)
    end
end

struct LocalToGlobal{A,C} <: AbstractVector{Int}
    own_to_global::A
    ghost_to_global::Vector{Int}
    perm::C
end
Base.IndexStyle(::Type{<:LocalToGlobal}) = IndexLinear()
Base.size(a::LocalToGlobal) = (length(a.own_to_global)+length(a.ghost_to_global),)
function Base.getindex(a::LocalToGlobal,local_id::Int)
    n_own = length(a.own_to_global)
    j = a.perm[local_id]
    if j > n_own
        a.ghost_to_global[j-n_own]
    else
        a.own_to_global[j]
    end
end

struct LocalToOwner{C} <: AbstractVector{Int32}
    own_to_owner::OwnToOwner
    ghost_to_owner::Vector{Int32}
    perm::C
end
Base.IndexStyle(::Type{<:LocalToOwner}) = IndexLinear()
Base.size(a::LocalToOwner) = (length(a.own_to_owner)+length(a.ghost_to_owner),)
function Base.getindex(a::LocalToOwner,local_id::Int)
    n_own = length(a.own_to_owner)
    j = a.perm[local_id]
    if j > n_own
        a.ghost_to_owner[j-n_own]
    else
        a.own_to_owner[j]
    end
end

struct GlobalToOwn{A} <: AbstractVector{Int32}
    n_own::Int32
    global_to_local::VectorFromDict{Int,Int32}
    perm::A
end
Base.IndexStyle(::Type{<:GlobalToOwn}) = IndexLinear()
Base.size(a::GlobalToOwn) = size(a.global_to_local)
function Base.getindex(a::GlobalToOwn,global_i::Int)
    local_i = a.global_to_local[global_i]
    z = Int32(0)
    local_i == z && return z
    i = a.perm[local_i]
    i > a.n_own && return z
    return Int32(i)
end

struct GlobalToGhost{A} <: AbstractVector{Int32}
    n_own::Int
    global_to_local::VectorFromDict{Int,Int32}
    perm::A
end
Base.IndexStyle(::Type{<:GlobalToGhost}) = IndexLinear()
Base.size(a::GlobalToGhost) = size(a.global_to_local)
function Base.getindex(a::GlobalToGhost,global_i::Int)
    local_i = a.global_to_local[global_i]
    z = Int32(0)
    local_i == z && return z
    i = a.perm[local_i]
    i <= a.n_own && return z
    return Int32(i-a.n_own)
end

"""
    struct LocalIndices

Container for local indices.

# Properties

- `n_global::Int`: Number of global indices.
- `owner::Int32`: Id of the part that stores the local indices
- `local_to_global::Vector{Int}`:  Global ids of the local indices in this part.  `local_to_global[i_local]` is the global id corresponding to the local index number `i_local`.
- `local_to_owner::Vector{Int32}`: Owners of the local ids. `local_to_owner[i_local]`is the id of the owner of the local index number `i_local`.

# Supertype hierarchy

    LocalIndices <: AbstractLocalIndices

"""
struct LocalIndices <: AbstractLocalIndices
    n_global::Int
    owner::Int32
    local_to_global::Vector{Int}
    local_to_owner::Vector{Int32}
    perm::Vector{Int32}
    own_to_local::Vector{Int32}
    ghost_to_local::Vector{Int32}
    global_to_local::VectorFromDict{Int,Int32}
    assembly_cache::AssemblyCache
end

assembly_cache(a::LocalIndices) = a.assembly_cache
get_permutation(a::LocalIndices) = a.perm

"""
    LocalIndices(n_global,owner,local_to_global,local_to_owner)

Build an instance of [`LocalIndices`](@ref) from the underlying properties
`n_global`, `owner`, `local_to_global`, and `local_to_owner`.
 The types of these variables need to match
the type of the properties in [`LocalIndices`](@ref).
"""
function LocalIndices(
    n_global::Integer,
    owner::Integer,
    local_to_global::Vector{Int},
    local_to_owner::Vector{Int32})

    own_to_local = findall(i->i==owner,local_to_owner)
    ghost_to_local = findall(i->i!=owner,local_to_owner)
    n_local = length(local_to_global)
    n_own = length(own_to_local)
    n_ghost = length(ghost_to_local)
    perm = zeros(Int32,n_local)
    perm[own_to_local] .= (1:n_own)
    perm[ghost_to_local] .= (1:n_ghost) .+ n_own
    global_to_local = VectorFromDict(local_to_global,Int32.(1:n_local),n_global)

    LocalIndices(
        Int(n_global),
        Int32(owner),
        local_to_global,
        local_to_owner,
        perm,
        Int32.(own_to_local),
        Int32.(ghost_to_local),
        global_to_local,
        AssemblyCache())
end

function replace_ghost(a::LocalIndices,ghost::GhostIndices)
    error("replace_ghost only makes sense for un-permuted local indices.")
end

get_owner(a::LocalIndices) = a.owner

get_n_local(a::LocalIndices) = length(a.local_to_global)

function get_own_to_global(a::LocalIndices)
    view(a.local_to_global,a.own_to_local)
end

function get_own_to_owner(a::LocalIndices)
    n_own = length(a.own_to_local)
    OwnToOwner(a.owner,n_own)
end

function get_global_to_own(a::LocalIndices)
    n_own = Int32(length(a.own_to_local))
    GlobalToOwn(n_own,a.global_to_local,a.perm)
end

function get_ghost_to_global(a::LocalIndices)
    view(a.local_to_global,a.ghost_to_local)
end

function get_ghost_to_owner(a::LocalIndices)
    view(a.local_to_owner,a.ghost_to_local)
end

function get_global_to_ghost(a::LocalIndices)
    n_own = length(a.own_to_local)
    GlobalToGhost(n_own,a.global_to_local,a.perm)
end

function get_own_to_local(a::LocalIndices)
    a.own_to_local
end

function get_ghost_to_local(a::LocalIndices)
    a.ghost_to_local
end

function get_local_to_own(a::LocalIndices)
    n_own = length(a.own_to_local)
    LocalToOwn(n_own,a.perm)
end

function get_local_to_ghost(a::LocalIndices)
    n_own = length(a.own_to_local)
    LocalToGhost(n_own,a.perm)
end

function get_global_to_local(a::LocalIndices)
    a.global_to_local
end

function get_local_to_global(a::LocalIndices)
    a.local_to_global
end

function get_local_to_owner(a::LocalIndices)
    a.local_to_owner
end

"""
    OwnAndGhostIndices

Container for local indices stored as own and ghost indices separately.
Local indices are defined by concatenating own and ghost ones.

# Properties

- `own::OwnIndices`: Container for the own indices.
- `ghost::GhostIndices`: Container for the ghost indices.

# Supertype hierarchy

    OwnAndGhostIndices <: AbstractLocalIndices

"""
struct OwnAndGhostIndices <: AbstractLocalIndices
    own::OwnIndices
    ghost::GhostIndices
    assembly_cache::AssemblyCache
    @doc """
        OwnAndGhostIndices(own::OwnIndices,ghost::GhostIndices)

    Build an instance of [`OwnAndGhostIndices`](@ref) from the underlying properties `own` and `ghost`.
    """
    function OwnAndGhostIndices(own::OwnIndices,ghost::GhostIndices)
        new(own,ghost,AssemblyCache())
    end
end
assembly_cache(a::OwnAndGhostIndices) = a.assembly_cache

get_permutation(a::OwnAndGhostIndices) = Int32(1):Int32(get_n_local(a))

function replace_ghost(a::OwnAndGhostIndices,ghost::GhostIndices)
    OwnAndGhostIndices(a.own,ghost)
end

get_owner(a::OwnAndGhostIndices) = a.own.owner

function get_own_to_global(a::OwnAndGhostIndices)
    a.own.own_to_global
end

function get_own_to_owner(a::OwnAndGhostIndices)
    owner = Int32(a.own.owner)
    n_own = length(a.own.own_to_global)
    OwnToOwner(owner,n_own)
end

function get_global_to_own(a::OwnAndGhostIndices)
    a.own.global_to_own
end

function get_ghost_to_global(a::OwnAndGhostIndices)
    a.ghost.ghost_to_global
end

function get_ghost_to_owner(a::OwnAndGhostIndices)
    a.ghost.ghost_to_owner
end

function get_global_to_ghost(a::OwnAndGhostIndices)
    a.ghost.global_to_ghost
end

function get_own_to_local(a::OwnAndGhostIndices)
    n_own = length(a.own.own_to_global)
    Int32.(1:n_own)
end

function get_ghost_to_local(a::OwnAndGhostIndices)
    n_own = length(a.own.own_to_global)
    n_ghost = length(a.ghost.ghost_to_global)
    Int32.((1:n_ghost).+n_own)
end

function get_local_to_own(a::OwnAndGhostIndices)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    n_own = length(own_to_local)
    n_ghost = length(ghost_to_local)
    n_local = n_own + n_ghost
    LocalToOwn(n_own,1:n_local)
end

function get_local_to_ghost(a::OwnAndGhostIndices)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    n_own = length(own_to_local)
    n_ghost = length(ghost_to_local)
    n_local = n_own + n_ghost
    LocalToGhost(n_own,1:n_local)
end

function get_global_to_local(a::OwnAndGhostIndices)
    global_to_own = get_global_to_own(a)
    global_to_ghost = get_global_to_ghost(a)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    GlobalToLocal(global_to_own,global_to_ghost,own_to_local,ghost_to_local)
end

function get_local_to_global(a::OwnAndGhostIndices)
    own_to_global = get_own_to_global(a)
    ghost_to_global = get_ghost_to_global(a)
    n_own = length(own_to_global)
    n_ghost = length(ghost_to_global)
    n_local = n_own + n_ghost
    perm = 1:n_local
    LocalToGlobal(own_to_global,ghost_to_global,perm)
end

function get_local_to_owner(a::OwnAndGhostIndices)
    own_to_owner = get_own_to_owner(a)
    ghost_to_owner = get_ghost_to_owner(a)
    n_own = length(own_to_owner)
    n_ghost = length(ghost_to_owner)
    n_local = n_own + n_ghost
    perm = 1:n_local
    LocalToOwner(own_to_owner,ghost_to_owner,perm)
end

permute_indices(a,b) = PermutedLocalIndices(a,b)

"""
    PermutedLocalIndices{A}

Type representing local indices subjected to a permutation.

# Properties

- `indices::A`: Local indices before permutation. `typeof(indices)` is a type implementing the `AbstractLocalIndices` interface.
- `perm::Vector{Int32}`: Permutation vector. `perm[local_i]` contains the local indexid in `indices` corresponding with the new local index id `local_i`.

# Supertype hierarchy

    PermutedLocalIndices{A} <: AbstractLocalIndices

"""
struct PermutedLocalIndices{A} <: AbstractLocalIndices
    indices::A
    perm::Vector{Int32}
    own_to_local::Vector{Int32}
    ghost_to_local::Vector{Int32}
    assembly_cache::AssemblyCache
end
assembly_cache(a::PermutedLocalIndices) = a.assembly_cache

"""
    PermutedLocalIndices(indices,perm)

Build an instance of [`PermutedLocalIndices`](@ref) from the underlying properties `indices` and `perm`.
 The types of these variables need to match
the type of the properties in [`PermutedLocalIndices`](@ref).
"""
function PermutedLocalIndices(indices,perm)
    n_own = length(get_own_to_owner(indices))
    n_local = length(perm)
    n_ghost = n_local - n_own
    own_to_local = zeros(Int32,n_own)
    ghost_to_local = zeros(Int32,n_ghost)
    for i_local in 1:n_local
        k = perm[i_local]
        if k > n_own
            i_ghost = k - n_own
            ghost_to_local[i_ghost] = i_local
        else
            i_own = k
            own_to_local[i_own] = i_local
        end
    end
    _perm = convert(Vector{Int32},perm)
    PermutedLocalIndices(indices,_perm,own_to_local,ghost_to_local,AssemblyCache())
end

function replace_ghost(a::PermutedLocalIndices,::GhostIndices)
    error("replace_ghost only makes sense for un-permuted local indices.")
end

get_owner(a::PermutedLocalIndices) = get_owner(a.indices)

function get_own_to_global(a::PermutedLocalIndices)
    get_own_to_global(a.indices)
end

function get_own_to_owner(a::PermutedLocalIndices)
    get_own_to_owner(a.indices)
end

function get_global_to_own(a::PermutedLocalIndices)
    get_global_to_own(a.indices)
end

function get_ghost_to_global(a::PermutedLocalIndices)
    get_ghost_to_global(a.indices)
end

function get_ghost_to_owner(a::PermutedLocalIndices)
    get_ghost_to_owner(a.indices)
end

function get_global_to_ghost(a::PermutedLocalIndices)
    get_global_to_ghost(a.indices)
end

function get_own_to_local(a::PermutedLocalIndices)
    a.own_to_local
end

function get_ghost_to_local(a::PermutedLocalIndices)
    a.ghost_to_local
end

function get_local_to_own(a::PermutedLocalIndices)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    n_own = length(own_to_local)
    LocalToOwn(n_own,a.perm)
end

function get_local_to_ghost(a::PermutedLocalIndices)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    n_own = length(own_to_local)
    LocalToGhost(n_own,a.perm)
end

function get_global_to_local(a::PermutedLocalIndices)
    global_to_own = get_global_to_own(a)
    global_to_ghost = get_global_to_ghost(a)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    GlobalToLocal(global_to_own,global_to_ghost,own_to_local,ghost_to_local)
end

function get_local_to_global(a::PermutedLocalIndices)
    own_to_global = get_own_to_global(a)
    ghost_to_global = get_ghost_to_global(a)
    LocalToGlobal(own_to_global,ghost_to_global,a.perm)
end

function get_local_to_owner(a::PermutedLocalIndices)
    own_to_owner = get_own_to_owner(a)
    ghost_to_owner = get_ghost_to_owner(a)
    LocalToOwner(own_to_owner,ghost_to_owner,a.perm)
end

function find_owner(indices,global_ids,::Type{<:PermutedLocalIndices})
    inner_parts = map(i->i.indices,indices)
    find_owner(inner_parts,global_ids)
end

struct BlockPartitionOwnToGlobal{N} <: AbstractVector{Int}
    n::NTuple{N,Int}
    ranges::NTuple{N,UnitRange{Int}}
end
Base.size(a::BlockPartitionOwnToGlobal) = (prod(length,a.ranges),)
Base.IndexStyle(::Type{<:BlockPartitionOwnToGlobal}) = IndexLinear()
function Base.getindex(a::BlockPartitionOwnToGlobal,own_id::Int)
    global_ci = CartesianIndices(a.ranges)[own_id]
    global_id = LinearIndices(a.n)[global_ci]
    global_id
end

struct BlockPartitionGlobalToOwn{N} <: AbstractVector{Int32}
    n::NTuple{N,Int}
    ranges::NTuple{N,UnitRange{Int}}
end
Base.size(a::BlockPartitionGlobalToOwn) = (prod(a.n),)
Base.IndexStyle(::Type{<:BlockPartitionGlobalToOwn}) = IndexLinear()
function Base.getindex(a::BlockPartitionGlobalToOwn,global_id::Int)
    global_ci = CartesianIndices(a.n)[global_id]
    if all(map(in,Tuple(global_ci),a.ranges))
        j = map(Tuple(global_ci),a.ranges) do i,r
            i-first(r)+1
        end
        own_ci = CartesianIndex(j)
        own_id = LinearIndices(map(length,a.ranges))[own_ci]
        return Int32(own_id)
    end
    return Int32(0)
end

struct BlockPartitionGlobalToOwner{N} <: AbstractVector{Int32}
    start::NTuple{N,Vector{Int}}
end
Base.size(a::BlockPartitionGlobalToOwner) = (prod(map(i->i[end]-1,a.start)),)
Base.IndexStyle(::Type{<:BlockPartitionGlobalToOwner}) = IndexLinear()
function Base.getindex(a::BlockPartitionGlobalToOwner,i::Int)
    n = map(i->i[end]-1,a.start)
    np = map(i->length(i)-1,a.start)
    i_ci = CartesianIndices(n)[i]
    j = map(searchsortedlast,a.start,Tuple(i_ci))
    LinearIndices(np)[CartesianIndex(j)]
end

# This one is just to improve the display of the type LocalToGlobal
struct BlockPartitionLocalToGlobal{N,C} <: AbstractVector{Int}
    own_to_global::BlockPartitionOwnToGlobal{N}
    ghost_to_global::Vector{Int}
    perm::C
end
Base.IndexStyle(::Type{<:BlockPartitionLocalToGlobal}) = IndexLinear()
Base.size(a::BlockPartitionLocalToGlobal) = (length(a.own_to_global)+length(a.ghost_to_global),)
function Base.getindex(a::BlockPartitionLocalToGlobal,local_id::Int)
    n_own = length(a.own_to_global)
    j = a.perm[local_id]
    if j > n_own
        a.ghost_to_global[j-n_own]
    else
        a.own_to_global[j]
    end
end
function LocalToGlobal(
    own_to_global::BlockPartitionOwnToGlobal,
    ghost_to_global::Vector{Int},
    perm)
    BlockPartitionLocalToGlobal(
        own_to_global,
        ghost_to_global,
        perm)
end

# This one is just to improve the display of the type GlobalToLocal
struct BlockPartitionGlobalToLocal{N,V} <: AbstractVector{Int32}
    global_to_own::BlockPartitionGlobalToOwn{N}
    global_to_ghost::VectorFromDict{Int,Int32}
    own_to_local::V
    ghost_to_local::V
end
Base.size(a::BlockPartitionGlobalToLocal) = size(a.global_to_own)
Base.IndexStyle(::Type{<:BlockPartitionGlobalToLocal}) = IndexLinear()
function Base.getindex(a::BlockPartitionGlobalToLocal,global_id::Int)
    own_id = a.global_to_own[global_id]
    z = Int32(0)
    if own_id != z
        return a.own_to_local[own_id]
    end
    ghost_id = a.global_to_ghost[global_id]
    if ghost_id != z
        return a.ghost_to_local[ghost_id]
    end
    return z
end
function GlobalToLocal(
    global_to_own::BlockPartitionGlobalToOwn,
    global_to_ghost::VectorFromDict{Int,Int32},
    own_to_local,
    ghost_to_local)
    BlockPartitionGlobalToLocal(
        global_to_own,
        global_to_ghost,
        own_to_local,
        ghost_to_local)
end

struct LocalIndicesWithConstantBlockSize{N} <: AbstractLocalIndices
    p::CartesianIndex{N}
    np::NTuple{N,Int}
    n::NTuple{N,Int}
    ghost::GhostIndices
    assembly_cache::AssemblyCache
    function LocalIndicesWithConstantBlockSize(
            p::CartesianIndex{N},
            np::NTuple{N,Int},
            n::NTuple{N,Int},
            ghost::GhostIndices) where N
        new{N}(p, np, n, ghost, AssemblyCache())
    end
end
assembly_cache(a::LocalIndicesWithConstantBlockSize) = a.assembly_cache

function Base.getproperty(a::LocalIndicesWithConstantBlockSize, sym::Symbol)
    if sym === :ranges
        map(local_range,Tuple(a.p),a.np,a.n)
    else
        getfield(a,sym)
    end
end

function Base.propertynames(x::LocalIndicesWithConstantBlockSize, private::Bool=false)
  (fieldnames(typeof(x))...,:ranges)
end

function replace_ghost(a::LocalIndicesWithConstantBlockSize,ghost::GhostIndices)
    LocalIndicesWithConstantBlockSize(a.p,a.np,a.n,ghost)
end

function find_owner(indices,global_ids,::Type{<:LocalIndicesWithConstantBlockSize})
    map(indices,global_ids) do indices,global_ids
        start = map(indices.np,indices.n) do np,n
            start = [ first(local_range(p,np,n)) for p in 1:np ]
            push!(start,n+1)
            start
        end
        global_to_owner = BlockPartitionGlobalToOwner(start)
        global_to_owner[global_ids]
    end
end

struct LocalIndicesWithVariableBlockSize{N} <: AbstractLocalIndices
    p::CartesianIndex{N}
    np::NTuple{N,Int}
    n::NTuple{N,Int}
    ranges::NTuple{N,UnitRange{Int}}
    ghost::GhostIndices
    assembly_cache::AssemblyCache
    function LocalIndicesWithVariableBlockSize(
        p::CartesianIndex{N},
        np::NTuple{N,Int},
        n::NTuple{N,Int},
        ranges::NTuple{N,UnitRange{Int}},
        ghost::GhostIndices) where N
        new{N}(p,np,n,ranges,ghost,AssemblyCache())
    end
end
assembly_cache(a::LocalIndicesWithVariableBlockSize) = a.assembly_cache

function replace_ghost(a::LocalIndicesWithVariableBlockSize,ghost::GhostIndices)
    LocalIndicesWithVariableBlockSize(a.p,a.np,a.n,a.ranges,ghost)
end

function find_owner(indices,global_ids,::Type{<:LocalIndicesWithVariableBlockSize})
    initial = map(indices->map(first,indices.ranges),indices) |> collect |> tuple_of_arrays
    map(indices,global_ids) do indices,global_ids
        start = map(indices.n,initial) do n,initial
            start = vcat(initial,[n+1])
        end
        global_to_owner = BlockPartitionGlobalToOwner(start)
        global_to_owner[global_ids]
    end
end

const LocalIndicesInBlockPartition = Union{LocalIndicesWithConstantBlockSize,LocalIndicesWithVariableBlockSize}

get_permutation(a::LocalIndicesInBlockPartition) = Int32(1):Int32(get_n_local(a))

function get_owner(a::LocalIndicesInBlockPartition)
    owner = LinearIndices(a.np)[a.p]
    Int32(owner)
end

function get_own_to_global(a::LocalIndicesInBlockPartition)
    BlockPartitionOwnToGlobal(a.n,a.ranges)
end

function get_own_to_owner(a::LocalIndicesInBlockPartition)
    lis = LinearIndices(a.np)
    owner = Int32(lis[a.p])
    n_own = prod(map(length,a.ranges))
    OwnToOwner(owner,n_own)
end

function get_global_to_own(a::LocalIndicesInBlockPartition)
    BlockPartitionGlobalToOwn(a.n,a.ranges)
end

function get_ghost_to_global(a::LocalIndicesInBlockPartition)
    a.ghost.ghost_to_global
end

function get_ghost_to_owner(a::LocalIndicesInBlockPartition)
    a.ghost.ghost_to_owner
end

function get_global_to_ghost(a::LocalIndicesInBlockPartition)
    a.ghost.global_to_ghost
end

function get_own_to_local(a::LocalIndicesInBlockPartition)
    n_own = prod(map(length,a.ranges))
    Int32(1):Int32(n_own)
end

function get_ghost_to_local(a::LocalIndicesInBlockPartition)
    n_own = prod(map(length,a.ranges))
    n_ghost = length(a.ghost.ghost_to_global)
    ((Int32(1):Int32(n_ghost)).+Int32(n_own))
end

function get_local_to_own(a::LocalIndicesInBlockPartition)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    n_own = length(own_to_local)
    n_ghost = length(ghost_to_local)
    n_local = n_own + n_ghost
    LocalToOwn(n_own,1:n_local)
end

function get_local_to_ghost(a::LocalIndicesInBlockPartition)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    n_own = length(own_to_local)
    n_ghost = length(ghost_to_local)
    n_local = n_own + n_ghost
    LocalToGhost(n_own,1:n_local)
end

function get_global_to_local(a::LocalIndicesInBlockPartition)
    global_to_own = get_global_to_own(a)
    global_to_ghost = get_global_to_ghost(a)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    GlobalToLocal(global_to_own,global_to_ghost,own_to_local,ghost_to_local)
end

function get_local_to_global(a::LocalIndicesInBlockPartition)
    own_to_global = get_own_to_global(a)
    ghost_to_global = get_ghost_to_global(a)
    n_own = length(own_to_global)
    n_ghost = length(ghost_to_global)
    n_local = n_own + n_ghost
    perm = 1:n_local
    LocalToGlobal(own_to_global,ghost_to_global,perm)
end

function get_local_to_owner(a::LocalIndicesInBlockPartition)
    own_to_owner = get_own_to_owner(a)
    ghost_to_owner = get_ghost_to_owner(a)
    n_own = length(own_to_owner)
    n_ghost = length(ghost_to_owner)
    n_local = n_own + n_ghost
    perm = 1:n_local
    LocalToOwner(own_to_owner,ghost_to_owner,perm)
end

"""
    local_range(p, np, n, ghost=false, periodic=false)

Return the local range of indices in the component number `p`
of a uniform partition of indices `1:n` into `np` parts.
If `ghost==true` then include a layer of
"ghost" entries. If `periodic == true` the ghost layer is created assuming
periodic boundaries in the range  `1:n`. In this case, the first ghost
index is `0` for `p==1` and the last ghost index is `n+1`  for `p==np`

# Examples

Without ghost entries

    julia> using PartitionedArrays
    
    julia> local_range(1,3,10)
    1:3

    julia> local_range(2,3,10)
    4:6

    julia> local_range(3,3,10)
    7:10

With ghost entries

    julia> using PartitionedArrays
    
    julia> local_range(1,3,10,true)
    1:4

    julia> local_range(2,3,10,true)
    3:7

    julia> local_range(3,3,10,true)
    6:10

With periodic boundaries

    julia> using PartitionedArrays
    
    julia> local_range(1,3,10,true,true)
    0:4

    julia> local_range(2,3,10,true,true)
    3:7

    julia> local_range(3,3,10,true,true)
    6:11
"""
function local_range(p,np,n,ghost=false,periodic=false)
    l = n ÷ np
    offset = l * (p-1)
    rem = n % np
    if rem >= (np-p+1)
        l = l + 1
        offset = offset + p - (np-rem) - 1
    end
    start = 1+offset
    stop = l+offset
    if ghost && np != 1
        if periodic || p!=1
            start -= 1
        end
        if periodic || p!=np
            stop += 1
        end
    end
    start:stop
end

"""
    boundary_owner(p,np,n,ghost=false,periodic=false)

The object `o=bounday_owner(args...)` is such that `first(o)` is
the owner of `first(r)` and `last(o)` is the owner of `last(r)` for
 `r=local_range(args...)`.
"""
function boundary_owner(p,np,n,ghost=false,periodic=false)
    start = p
    stop = p
    if ghost && np != 1
        if periodic || p!=1
            start -= 1
        end
        if periodic || p!=np
            stop += 1
        end
    end
    (start,p,stop)
end


"""
    struct PRange{A,B}

`PRange` (partitioned range) is a type representing a range of indices `1:n_global`
distributed into several parts. The indices in the range `1:n_global` are called the
*global* indices. Each global index is *owned* by one part and only one part.
The set of indices owned by a part are called the *own* indices of this part.
Each part contains a second set of indices called the *ghost* indices. 
The set of ghost indices in a given part is an arbitrary subset
of the global indices that are owned by other parts. The union of the own and ghost
indices is referred to as the *local* indices of this part.
The sets of own, ghost, and local indices are stored using vector-like containers,
which equips them with a certain order. Thus, the `i`-th own index
in a part is the one being stored at index `i` in the array that contains
the own indices in this part.
The same rationale applies for ghost and local indices.

# Properties
- `indices::A`: Array-like object with `length(indices)` equal to the number of parts in the partitioned range.


The item `indices[i]` is an object that contains information about the own, ghost, and local indices of part number `i`.
`typeof(indices[i])` is a type that
implements the methods of the [`AbstractLocalIndices`](@ref) interface. Use this
interface to access the underlying information about own, ghost, and local indices.

# Supertype hierarchy

    PRange{A} <: AbstractUnitRange{Int}

"""
struct PRange{A} <: AbstractUnitRange{Int}
    partition::A
    @doc """
        PRange(n_global,indices)

    Build an instance of [`Prange`](@ref) from the underlying properties
    `n_global` and `indices`.

    # Examples
   
        julia> using PartitionedArrays
        
        julia> rank = LinearIndices((2,));
        
        julia> indices = map(rank) do rank
                   if rank == 1
                       LocalIndices(8,1,[1,2,3,4,5],Int32[1,1,1,1,2])
                   else
                       LocalIndices(8,2,[4,5,6,7,8],Int32[1,2,2,2,2])
                   end
               end;
        
        julia> pr = PRange(8,indices)
        1:1:8
        
        julia> get_local_to_global(pr)
        2-element Vector{Vector{Int64}}:
         [1, 2, 3, 4, 5]
         [4, 5, 6, 7, 8]
    """
    function PRange(indices)
        A = typeof(indices)
        new{A}(indices)
    end
end
partition(a::PRange) = a.partition
Base.first(a::PRange) = 1
Base.last(a::PRange) = getany(map(get_n_global,partition(a)))
function Base.show(io::IO,k::MIME"text/plain",data::PRange)
    np = length(partition(data))
    map_main(partition(data)) do indices
        println(io,"1:$(get_n_global(indices)) partitioned into $(np) parts")
    end
end

function matching_local_indices(a::PRange,b::PRange)
    partition(a) === partition(b) && return true
    c = map(matching_local_indices,partition(a),partition(b))
    reduce(&,c,init=true)
end

function matching_own_indices(a::PRange,b::PRange)
    partition(a) === partition(b) && return true
    c = map(matching_own_indices,partition(a),partition(b))
    reduce(&,c,init=true)
end

function matching_ghost_indices(a::PRange,b::PRange)
    partition(a) === partition(b) && return true
    c = map(matching_ghost_indices,partition(a),partition(b))
    reduce(&,c,init=true)
end

##prange(f,args...) = PRange(f(args...))
#
#"""
#    get_n_global(pr::PRange)
#
#Equivalent to `map(get_n_global,pr.indices)`.
#"""
#get_n_global(pr::PRange) = map(get_n_local,partition(pr))
#
#"""
#    get_n_local(pr::PRange)
#
#Equivalent to `map(get_n_local,pr.indices)`.
#"""
#get_n_local(pr::PRange) = map(get_n_local,partition(pr))
#
#"""
#    get_n_own(pr::PRange)
#
#Equivalent to `map(get_n_own,pr.indices)`.
#"""
#get_n_own(pr::PRange) = map(get_n_own,partition(pr))
#
#"""
#    get_local_to_global(pr::PRange)
#
#Equivalent to `map(get_local_to_global,pr.indices)`.
#"""
#get_local_to_global(pr::PRange) = map(get_local_to_global,partition(pr))
#
#"""
#    get_own_to_global(pr::PRange)
#
#Equivalent to `map(get_own_to_global,pr.indices)`.
#"""
#get_own_to_global(pr::PRange) = map(get_own_to_global,partition(pr))
#
#"""
#    get_ghost_to_global(pr::PRange)
#
#Equivalent to `map(get_ghost_to_global,pr.indices)`.
#"""
#get_ghost_to_global(pr::PRange) = map(get_ghost_to_global,partition(pr))
#
#"""
#    get_local_to_owner(pr::PRange)
#
#Equivalent to `map(get_local_to_owner,pr.indices)`.
#"""
#get_local_to_owner(pr::PRange) = map(get_local_to_owner,partition(pr))
#
#"""
#    get_own_to_owner(pr::PRange)
#
#Equivalent to `map(get_own_to_owner,pr.indices)`.
#"""
#get_own_to_owner(pr::PRange) = map(get_own_to_owner,partition(pr))
#
#"""
#    get_ghost_to_owner(pr::PRange)
#
#Equivalent to `map(get_ghost_to_owner,pr.indices)`.
#"""
#get_ghost_to_owner(pr::PRange) = map(get_ghost_to_owner,partition(pr))
#
#"""
#    get_global_to_local(pr::PRange)
#
#Equivalent to `map(get_global_to_local,pr.indices)`.
#"""
#get_global_to_local(pr::PRange) = map(get_global_to_local,partition(pr))
#
#"""
#    get_global_to_own(pr::PRange)
#
#Equivalent to `map(get_global_to_own,pr.indices)`.
#"""
#get_global_to_own(pr::PRange) = map(get_global_to_own,partition(pr))
#
#"""
#    get_global_to_ghost(pr::PRange)
#
#Equivalent to `map(get_global_to_ghost,pr.indices)`.
#"""
#get_global_to_ghost(pr::PRange) = map(get_global_to_ghost,partition(pr))
#
#"""
#    get_own_to_local(pr::PRange)
#
#Equivalent to `map(get_own_to_local,pr.indices)`.
#"""
#get_own_to_local(pr::PRange) = map(get_own_to_local,partition(pr))
#
#"""
#    get_ghost_to_local(pr::PRange)
#
#Equivalent to `map(get_ghost_to_local,pr.indices)`.
#"""
#get_ghost_to_local(pr::PRange) = map(get_ghost_to_local,partition(pr))
#
#"""
#    get_local_to_own(pr::PRange)
#
#Equivalent to `map(get_local_to_own,pr.indices)`.
#"""
#get_local_to_own(pr::PRange) = map(get_local_to_own,partition(pr))
#
#"""
#    get_local_to_ghost(pr::PRange)
#
#Equivalent to `map(get_local_to_ghost,pr.indices)`.
#"""
#get_local_to_ghost(pr::PRange) = map(get_local_to_ghost,pr.indices)
#
#find_owner(pr::PRange,global_ids) = find_owner(pr.indices,global_ids)
#
#assembly_neighbors(pr::PRange) = pr.assembler.neighbors
#
#"""
#    replace_ghost(pr::PRange,gids,owners=find_owner(pr,gids))
#
#Return an object of the same type as `pr` obtained by replacing the ghost
#ids in `pr` by the global ids in `gids`.
#
#Equivalent to
#
#    indices = map(replace_ghost,pr.indices,gids,owners)
#    PRange(indices)
#"""
#function replace_ghost(pr::PRange,gids,owners=find_owner(pr,gids);kwargs...)
#    indices = map(replace_ghost,pr.indices,gids,owners)
#    assembler = vector_assembler(indices;kwargs...)
#    PRange(indices,assembler)
#end
#
#"""
#    union_ghost(pr::PRange,gids,owners=find_owner(pr,gids))
#
#Return an object of the same type as `pr` that contains the union of the ghost
#ids in `pr` and the global ids in `gids`. 
#
#Equivalent to
#
#    indices = map(union_ghost,pr.indices,gids,owners)
#    PRange(indices)
#"""
#function union_ghost(pr::PRange,gids,owners=find_owner(pr,gids);kwargs...)
#    indices = map(union_ghost,pr.indices,gids,owners)
#    assembler = vector_assembler(indices;kwargs...)
#    PRange(indices,assembler)
#end
#
#function to_local!(I,rows::PRange)
#    map(to_local!,I,rows.indices)
#end
#
#function to_global!(I,rows::PRange)
#    map(to_global!,I,rows.indices)
#end



#struct Assembler{A,B}
#    neighbors::A
#    local_indices::B
#end
#function Base.show(io::IO,k::MIME"text/plain",data::Assembler)
#    println(io,nameof(typeof(data))," partitioned in $(length(data.neighbors.snd)) parts")
#end
#Base.reverse(a::Assembler) = Assembler(reverse(a.neighbors),reverse(a.local_indices))
#
#struct AssemblyLocalIndices{A,B}
#    snd::A
#    rcv::B
#end
#function Base.show(io::IO,k::MIME"text/plain",data::AssemblyLocalIndices)
#    println(io,nameof(typeof(data))," partitioned in $(length(data.snd)) parts")
#end
#Base.reverse(a::AssemblyLocalIndices) = AssemblyLocalIndices(a.rcv,a.snd)
#
#function vector_assembler(indices;kwargs...)
#    neighbors = assembly_neighbors(indices;kwargs...)
#    local_indices = assembly_local_indices(indices,neighbors)
#    Assembler(neighbors,local_indices)
#end
#
#function empty_assembler(indices)
#    neigs_snd = map(i->Int32[],indices)
#    neighbors = ExchangeGraph(neigs_snd,neigs_snd)
#    local_indices_snd = map(i->JaggedArray{Int32,Int32}([Int32[]]),indices)
#    local_indices = AssemblyLocalIndices(local_indices_snd,local_indices_snd)
#    Assembler(neighbors,local_indices)
#end





