"""
    abstract type AbstractLocalIndices

Abstract type representing the *local*, *own*, and *ghost* indices in
a part of a partition of a range `1:n` with length `n`.

# Notation

Let `1:n` be an integer range with length `n`. We denote the indices in `1:n` as the 
*global* indices. Let us consider a partition of `1:n`. The indices in a part
 in the partition are called the *own* indices of this part.
I.e., each part *owns* a subset of `1:n`. All these subsets are disjoint.
Let us assume that each part is equipped with a second set of indices called the *ghost* indices. 
The set of ghost indices in a given part is an arbitrary subset
of the global indices `1:n` that are owned by other parts. The union of the own and ghost
indices is referred to as the *local* indices of this part. The sets of local indices might overlap
between the different parts.

The sets of own, ghost, and local indices are stored using vector-like containers
in concrete implementations of `AbstractLocalIndices`. This
equips them with a certain order. The `i`-th own index
in a part is defined as the one being stored at index `i` in the array that contains
the own indices in this part (idem for ghost and local indices).
The map between indices in these ordered index sets are given by functions such as [`local_to_global`](@ref),
[`own_to_local`](@ref) etc.

# Supertype hierarchy

    AbstractLocalIndices <: AbstractVector{Int}

"""
abstract type AbstractLocalIndices <: AbstractVector{Int} end
Base.size(a::AbstractLocalIndices) = (local_length(a),)
Base.IndexStyle(::Type{<:AbstractLocalIndices}) = IndexLinear()
@inline Base.getindex(a::AbstractLocalIndices,i::Int) = local_to_global(a)[i]

"""
    local_length(indices)

Get number of local ids in `indices`.
"""
local_length(a) = own_length(a) + ghost_length(a)

"""
    own_length(indices)

Get number of own ids in `indices`.
"""
own_length(a) = length(own_to_owner(a))

"""
    ghost_length(indices)

Get number of ghost ids in `indices`.
"""
ghost_length(a) = length(ghost_to_global(a))

"""
    global_length(indices)

Get number of global ids associated with `indices`.
"""
global_length(a) = length(global_to_own(a))

"""
    part_id(indices)

Return the id of the part that is storing `indices`.
"""
function part_id end

"""
    local_to_global(indices)

Return an array with the global indices of the local indices in `indices`.
"""
function local_to_global end

"""
    own_to_global(indices)

Return an array with the global indices of the own indices in `indices`.
"""
function own_to_global end

"""
    ghost_to_global(indices)

Return an array with the global indices of the ghost indices in `indices`.
"""
function ghost_to_global end

"""
    local_to_owner(indices)

Return an array with the owners of the local indices in `indices`.
"""
function local_to_owner end

"""
    own_to_owner(indices)

Return an array with the owners of the own indices in `indices`.
"""
function own_to_owner end

"""
    ghost_to_owner(indices)

Return an array with the owners of the ghost indices in `indices`.
"""
function ghost_to_owner end

"""
    global_to_local(indices)

Return an array with the inverse index map of `local_to_global(indices)`.
"""
function global_to_local end

"""
    global_to_own(indices)

Return an array with the inverse index map of `own_to_global(indices)`.
"""
function global_to_own end

"""
    global_to_ghost(indices)

Return an array with the inverse index map of `ghost_to_global(indices)`.
"""
function global_to_ghost end

"""
    own_to_local(indices)

Return an array with the local ids of the own indices in `indices`.
"""
function own_to_local end

"""
    ghost_to_local(indices)

Return an array with the local ids of the ghost indices in `indices`.
"""
function ghost_to_local end

"""
    local_to_own(indices)

Return an array with the inverse index map of `own_to_local(indices)`.
"""
function local_to_own end

"""
    local_to_ghost(indices)
Return an array with the inverse index map of `ghost_to_local(indices)`.
"""
function local_to_ghost end

function local_permutation(indices)
    n_local = local_length(indices)
    n_own = own_length(indices)
    n_ghost = ghost_length(indices)
    perm = zeros(Int32,n_local)
    perm[own_to_local(indices)] = 1:n_own
    perm[ghost_to_local(indices)] = (1:n_ghost) .+ n_own
    perm
end

function matching_local_indices(a,b)
    a === b && return true
    local_to_global(a) == local_to_global(b) &&
    local_to_owner(a) == local_to_owner(b)
end

function matching_own_indices(a,b)
    a === b && return true
    own_to_global(a) == own_to_global(b) &&
    part_id(a) == part_id(b)
end

function matching_ghost_indices(a,b)
    a === b && return true
    ghost_to_global(a) == ghost_to_global(b) &&
    ghost_to_owner(a) == ghost_to_owner(b)
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

function remove_ghost(indices)
    replace_ghost(indices,Int[],Int32[])
end

function filter_ghost(indices,gids,owners)
    set = Set{Int}()
    part_owner = part_id(indices)
    n_new_ghost = 0
    global_to_ghost_indices = global_to_ghost(indices)
    for (global_i,owner) in zip(gids,owners)
        if global_i < 1
            continue
        end
        if owner != part_owner
            ghost_i = global_to_ghost_indices[global_i]
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
        if global_i < 1
            continue
        end
        if owner != part_owner
            ghost_i = global_to_ghost_indices[global_i]
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
    new_gids = vcat(ghost_to_global(indices),extra_gids)
    new_owners = vcat(ghost_to_owner(indices),extra_owners)
    n_global = global_length(indices)
    ghost = GhostIndices(n_global,new_gids,new_owners)
    replace_ghost(indices,ghost)
end

"""
    to_local!(I,indices)

Transform the global indices in `I` into local ids according to `indices`.
"""
function to_local!(I,indices)
    global_to_local_indices = global_to_local(indices)
    for k in 1:length(I)
        I[k] = global_to_local_indices[I[k]]
    end
    I
end

"""
    to_global!(I,indices)

Transform the local indices in `I` into global ids according to `indices`.
"""
function to_global!(I,indices)
    local_to_global_indices = local_to_global(indices)
    for k in 1:length(I)
        I[k] = local_to_global_indices[I[k]]
    end
    I
end

map_global_to_local!(I,indices) = map_x_to_y!(global_to_local,I,indices)
map_global_to_ghost!(I,indices) = map_x_to_y!(global_to_ghost,I,indices)
map_global_to_own!(I,indices) = map_x_to_y!(global_to_own,I,indices)
map_local_to_global!(I,indices) = map_x_to_y!(local_to_global,I,indices)
map_local_to_ghost!(I,indices) = map_x_to_y!(local_to_ghost,I,indices)
map_local_to_own!(I,indices) = map_x_to_y!(local_to_own,I,indices)
map_own_to_global!(I,indices) = map_x_to_y!(own_to_global,I,indices)
map_own_to_local!(I,indices) = map_x_to_y!(own_to_local,I,indices)
map_ghost_to_global!(I,indices) = map_x_to_y!(ghost_to_global,I,indices)
map_ghost_to_local!(I,indices) = map_x_to_y!(ghost_to_local,I,indices)

function map_x_to_y!(x_to_y,I,indices)
    local_to_global_indices = x_to_y(indices)
    for k in 1:length(I)
        Ik = I[k]
        if Ik < 1
            continue
        end
        I[k] = local_to_global_indices[Ik]
    end
    I
end

"""
    find_owner(index_partition,global_ids)

Find the owners of the global ids in `global_ids`. The input `global_ids` is
a vector of vectors distributed over the same parts as `index_partition`. Each part will
look for the owners in parallel, when using a parallel back-end.

# Example

```jldoctest
julia> using PartitionedArrays

julia> rank = LinearIndices((4,));

julia> index_partition = uniform_partition(rank,10)
4-element Vector{PartitionedArrays.LocalIndicesWithConstantBlockSize{1}}:
 [1, 2]
 [3, 4]
 [5, 6, 7]
 [8, 9, 10]

julia> gids = [[3],[4,5],[7,2],[9,10,1]]
4-element Vector{Vector{Int64}}:
 [3]
 [4, 5]
 [7, 2]
 [9, 10, 1]

julia> find_owner(index_partition,gids)
4-element Vector{Vector{Int32}}:
 [2]
 [2, 3]
 [3, 1]
 [4, 4, 1]
```
"""
function find_owner(indices,global_ids)
    find_owner(indices,global_ids,eltype(indices))
end

function global_to_owner(indices)
    global_to_owner(indices,eltype(indices))
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

"""
    assembly_graph(index_partition;kwargs...)

Return an instance of [`ExchangeGraph`](@ref) representing the communication
graph needed to perform assembly of distributed vectors defined on the index
partition `index_partition`. `kwargs` are delegated to [`ExchangeGraph`](@ref)
in order to find the receiving neighbors from the sending ones.

Equivalent to

    neighbors = assembly_neighbors(index_partition;kwargs...)
    ExchangeGraph(neighbors...)

"""
function assembly_graph(index_partition;kwargs...)
    neighbors_snd,neighbors_rcv = assembly_neighbors(index_partition;kwargs...)
    ExchangeGraph(neighbors_snd,neighbors_rcv)
end

"""
    neigs_snd, neigs_rcv = assembly_neighbors(index_partition;kwargs...)

Return the ids of the neighbor parts from which we send and receive data respectively
in the assembly of distributed vectors defined on the index
partition `index_partition`.
partition `index_partition`. `kwargs` are delegated to [`ExchangeGraph`](@ref)
in order to find the receiving neighbors from the sending ones.
"""
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
        rank = part_id(indices)
        local_index_to_owner = local_to_owner(indices)
        set = Set{Int32}()
        for owner in local_index_to_owner
            if owner != rank
                push!(set,owner)
            end
        end
        sort(collect(set))
    end
    graph = ExchangeGraph(parts_snd;kwargs...)
    graph.snd, graph.rcv
end

"""
    ids_snd, ids_rcv = assembly_local_indices(index_partition)

Return the local ids to be sent and received  
in the assembly of distributed vectors defined on the index
partition `index_partition`.

Local values corresponding to the local
indices in `ids_snd[i]` (respectively `ids_rcv[i]`)
are sent to part `neigs_snd[i]` (respectively `neigs_rcv[i]`),
where `neigs_snd, neigs_rcv = assembly_neighbors(index_partition)`.


"""
function assembly_local_indices(index_partition)
    neigs = assembly_neighbors(index_partition)
    assembly_local_indices(index_partition,neigs...)
end

function assembly_local_indices(indices,neighbors_snd,neighbors_rcv)
    cache = map(assembly_cache,indices)
    mask = map(cache) do mycache
        isassigned(mycache.local_indices_snd) && isassigned(mycache.local_indices_rcv)
    end
    if ! getany(mask)
        new_local_indices_snd, new_local_indices_rcv = compute_assembly_local_indices(indices,neighbors_snd,neighbors_rcv)
        map(cache,new_local_indices_snd,new_local_indices_rcv) do mycache, mylocal_indices_snd, mylocal_indices_rcv
            mycache.local_indices_snd[] = mylocal_indices_snd
            mycache.local_indices_rcv[] = mylocal_indices_rcv
        end
    end
    local_indices_snd, local_indices_rcv = map(cache) do mycache
        mycache.local_indices_snd[], mycache.local_indices_rcv[]
    end |> tuple_of_arrays
    local_indices_snd, local_indices_rcv
end

function compute_assembly_local_indices(indices,neighbors_snd,neighbors_rcv)
    parts_snd = neighbors_snd
    parts_rcv = neighbors_rcv
    local_indices_snd, global_indices_snd = map(indices,parts_snd) do indices,parts_snd
        rank = part_id(indices)
        local_index_to_owner = local_to_owner(indices)
        owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
        ptrs = zeros(Int32,length(parts_snd)+1)
        for owner in local_index_to_owner
            if owner != rank
                ptrs[owner_to_i[owner]+1] +=1
            end
        end
        length_to_ptrs!(ptrs)
        data_lids = zeros(Int32,ptrs[end]-1)
        data_gids = zeros(Int,ptrs[end]-1)
        local_to_global_indices = local_to_global(indices)
        for (lid,owner) in enumerate(local_index_to_owner)
            if owner != rank
                p = ptrs[owner_to_i[owner]]
                data_lids[p]=lid
                data_gids[p]=local_to_global_indices[lid]
                ptrs[owner_to_i[owner]] += 1
            end
        end
        rewind_ptrs!(ptrs)
        my_local_indices_snd = JaggedArray(data_lids,ptrs)
        my_global_indices_snd = JaggedArray(data_gids,ptrs)
        my_local_indices_snd, my_global_indices_snd
    end |>  tuple_of_arrays
    graph = ExchangeGraph(parts_snd,parts_rcv)
    global_indices_rcv = exchange_fetch(global_indices_snd,graph)
    local_indices_rcv = map(global_indices_rcv,indices) do myglobal_indices_rcv,myindices
        ptrs = myglobal_indices_rcv.ptrs
        data_lids = zeros(Int32,ptrs[end]-1)
        global_to_local_indices = global_to_local(myindices)
        for (k,gid) in enumerate(myglobal_indices_rcv.data)
            data_lids[k] = global_to_local_indices[gid]
        end
        my_local_indices_rcv = JaggedArray(data_lids,ptrs)
    end
    local_indices_snd,local_indices_rcv
end

"""
    permute_indices(indices,perm)
"""
permute_indices(a,b) = PermutedLocalIndices(a,b)

"""
    uniform_partition(ranks,np,n[,ghost[,periodic]])

Generate an `N` dimensional
block partition of the indices in `LinearIndices(np)` with a (roughly) constant block size.
The output is a vector of vectors containing the indices in each component of
the partition. The `eltype` of the result implements the [`AbstractLocalIndices`](@ref)
interface.

# Arguments
- `ranks`: Array containing the distribution of ranks.
-  `np::NTuple{N}`: Number of parts per direction.
-  `n::NTuple{N}`: Number of global indices per direction.
-  `ghost::NTuple{N}=ntuple(i->false,N)`: Number of ghost indices per direction.
-  `periodic::NTuple{N}=ntuple(i->false,N)`: Use or not periodic boundaries per direction.

For convenience, one can also provide scalar inputs instead tuples
to create 1D block partitions. In this case, the argument `np` can be omitted
and it will be computed as `np=length(ranks)`. At the moment, it's only possible
to use this syntax for zero (with `ghost=false`) or one (with `ghost=true`) layer(s)
of ghost indices. If you wish to have more ghost indices, use tuples instead.

# Examples

2D partition of 4x4 indices into 2x2 parts without ghost

```jldoctest
julia> using PartitionedArrays

julia> rank = LinearIndices((4,));

julia> uniform_partition(rank,10)
4-element Vector{PartitionedArrays.LocalIndicesWithConstantBlockSize{1}}:
 [1, 2]
 [3, 4]
 [5, 6, 7]
 [8, 9, 10]

julia> uniform_partition(rank,(2,2),(4,4))
4-element Vector{PartitionedArrays.LocalIndicesWithConstantBlockSize{2}}:
 [1, 2, 5, 6]
 [3, 4, 7, 8]
 [9, 10, 13, 14]
 [11, 12, 15, 16]
```

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

function uniform_partition(rank,n::Integer)
    uniform_partition(rank,length(rank),n)
end

function uniform_partition(rank,n::Integer,ghost::Bool,periodic::Bool=false)
    uniform_partition(rank,length(rank),n,ghost,periodic)
end

function uniform_partition(rank,np::Integer,n::Integer) uniform_partition(rank,(np,),(n,)) end

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
    owners = map(Tuple(p), np, n, local_ranges) do p, np, n, lr
        myowners = zeros(Int32,length(lr))
        i = 1
        for p in Iterators.cycle(1:np)
            plr = local_range(p, np, n)
            while mod(lr[i]-1, n)+1 in plr
                myowners[i] = p
                (i += 1) > length(myowners) && return myowners
            end
        end
    end
    n_local = prod(map(length, local_ranges))
    n_own = prod(map(length, own_ranges))
    n_ghost = n_local - n_own

    ghost_to_global = zeros(Int,n_ghost)
    ghost_to_owner = zeros(Int32,n_ghost)
    perm = zeros(Int32,n_local)
    i_ghost = 0
    i_own = 0

    cis = CartesianIndices(map(length,local_ranges))
    lis = CircularArray(LinearIndices(n))
    local_cis = CartesianIndices(local_ranges)
    owner_lis = LinearIndices(np)
    for (i,ci) in enumerate(cis)
        flags = map(Tuple(ci), own_ranges, local_ranges) do i, or, lr
            i in (or .- first(lr) .+ 1)
        end
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

Build a 1D variable-size block partition of the range `1:n`.
The output is a vector of vectors containing the indices in each component of
the partition. The `eltype` of the result implements the [`AbstractLocalIndices`](@ref)
interface.

# Arguments

-  `n_own::AbstractArray{<:Integer}`: Array containing the block size for each part.
-  `n_global::Integer`: Number of global indices. It should be equal to `sum(n_own)`.
-  `start::AbstractArray{Int}=scan(+,n_own,type=:exclusive,init=1)`: First global index in each part.

We ask the user to provide `n_global` and (optionally) `start` since discovering them requires communications.

# Examples
```jldoctest
julia> using PartitionedArrays

julia> rank = LinearIndices((4,));

julia> n_own = [3,2,2,3];

julia> variable_partition(n_own,sum(n_own))
4-element Vector{PartitionedArrays.LocalIndicesWithVariableBlockSize{1}}:
 [1, 2, 3]
 [4, 5]
 [6, 7]
 [8, 9, 10]
```
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

"""
    partition_from_color(ranks,global_to_color;multicast=false,source=MAIN)

Build an arbitrary 1d partition by defining the parts via the argument `global_to_color` (see below).
The output is a vector of vectors containing the indices in each component of
the partition. The `eltype` of the result implements the [`AbstractLocalIndices`](@ref)
interface.

# Arguments

- `ranks`: Array containing the distribution of ranks.
- `global_to_color`: If `multicast==false`,  `global_to_color[gid]` contains the part id that owns the global id `gid`. If `multicast==true`, then   `global_to_color[source][gid]` contains the part id that owns the global id `gid`.

# Key-word arguments
- `multicast=false`
- `source=MAIN`

This function is useful when generating a partition using a graph partitioner such as METIS.
The argument `global_to_color` is the usual output of such tools.
"""
function partition_from_color(ranks,global_to_color;multicast=false,source=MAIN)
    if multicast == true
        global_to_owner = getany(PartitionedArrays.multicast(global_to_color;source))
    else
        global_to_owner = global_to_color
    end
    map(ranks) do rank
        nglobal = length(global_to_owner)
        own_to_global = findall(owner->owner==rank,global_to_owner)
            ghost_to_global = Int[]
        ghost_to_owner = Int32[]
        own = OwnIndices(nglobal,rank,own_to_global)
        ghost = GhostIndices(nglobal,ghost_to_global,ghost_to_owner)
        OwnAndGhostIndices(own,ghost,global_to_owner)
    end
end

"""
    trivial_partition(ranks,n;destination=MAIN)

!!! warning
    Document me!
"""
function trivial_partition(ranks,n;destination=MAIN)
    n_own = map(ranks) do rank
        rank == destination ? Int(n) : 0
    end
    partition_in_main = variable_partition(n_own,n)
    partition_in_main
end

function renumber_partition(partition_in;renumber_local_indices=true)
    own_ids = map(own_to_global,partition_in)
    if eltype(own_ids) <: BlockPartitionOwnToGlobal{1}
        return partition_in
    end
    n_global = PartitionedArrays.getany(map(global_length,partition_in))
    n_own = map(own_length,partition_in)
    new_gids = variable_partition(n_own,n_global)
    v = PVector{Vector{Int}}(undef,partition_in)
    map(own_values(v),new_gids) do own_v, new_gids
        own_v .= own_to_global(new_gids)
    end
    consistent!(v) |> wait
    I = ghost_values(v)
    I_owner = map(ghost_to_owner,partition_in)
    new_ids2 = map(union_ghost,new_gids,I,I_owner)
    if renumber_local_indices
        return new_ids2
    end
    perm = map(PartitionedArrays.local_permutation,partition_in)
    partition_out = map(permute_indices,new_ids2,perm)
    partition_out
end

function local_range(p,np,n,ghost=false,periodic=false)
    l, rem = divrem(n, np)
    offset = l * (p-1)
    if rem >= (np-p+1)
        l += 1
        offset += p - (np-rem) - 1
    end
    start = 1+offset-ghost
    stop = l+offset+ghost

    periodic && return start:stop
    return max(1, start):min(n,stop)
end

## unused
# function boundary_owner(p,np,n,ghost=false,periodic=false)
#     start = p
#     stop = p

#     if periodic || p!=1
#         start -= ghost
#     end
#     if periodic || p!=np
#         stop += ghost
#     end
#     (start,p,stop)
# end

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
function OwnIndices(n_global,owner,own_to_global)
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
    n_global = global_length(indices)
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

Container for arbitrary local indices.

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
local_permutation(a::LocalIndices) = a.perm

"""
    LocalIndices(n_global,owner,local_to_global,local_to_owner)

Build an instance of [`LocalIndices`](@ref) from the underlying properties
`n_global`, `owner`, `local_to_global`, and `local_to_owner`.
 The types of these variables need to match
the type of the properties in [`LocalIndices`](@ref).
"""
function LocalIndices(n_global,owner,local_to_global,local_to_owner)

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

part_id(a::LocalIndices) = a.owner

local_length(a::LocalIndices) = length(a.local_to_global)

function own_to_global(a::LocalIndices)
    view(a.local_to_global,a.own_to_local)
end

function own_to_owner(a::LocalIndices)
    n_own = length(a.own_to_local)
    OwnToOwner(a.owner,n_own)
end

function global_to_own(a::LocalIndices)
    n_own = Int32(length(a.own_to_local))
    GlobalToOwn(n_own,a.global_to_local,a.perm)
end

function ghost_to_global(a::LocalIndices)
    view(a.local_to_global,a.ghost_to_local)
end

function ghost_to_owner(a::LocalIndices)
    view(a.local_to_owner,a.ghost_to_local)
end

function global_to_ghost(a::LocalIndices)
    n_own = length(a.own_to_local)
    GlobalToGhost(n_own,a.global_to_local,a.perm)
end

function own_to_local(a::LocalIndices)
    a.own_to_local
end

function ghost_to_local(a::LocalIndices)
    a.ghost_to_local
end

function local_to_own(a::LocalIndices)
    n_own = length(a.own_to_local)
    LocalToOwn(n_own,a.perm)
end

function local_to_ghost(a::LocalIndices)
    n_own = length(a.own_to_local)
    LocalToGhost(n_own,a.perm)
end

function global_to_local(a::LocalIndices)
    a.global_to_local
end

function local_to_global(a::LocalIndices)
    a.local_to_global
end

function local_to_owner(a::LocalIndices)
    a.local_to_owner
end

"""
    OwnAndGhostIndices

Container for local indices stored as own and ghost indices separately.
Local indices are defined by concatenating own and ghost ones.

# Properties

- `own::OwnIndices`: Container for the own indices.
- `ghost::GhostIndices`: Container for the ghost indices.
- `global_to_owner`: [optional: it can be `nothing`] Vector containing the owner of each global id.

# Supertype hierarchy

    OwnAndGhostIndices{A} <: AbstractLocalIndices

where `A=typeof(global_to_owner)`.

"""
struct OwnAndGhostIndices{A} <: AbstractLocalIndices
    own::OwnIndices
    ghost::GhostIndices
    global_to_owner::A
    assembly_cache::AssemblyCache
    @doc """
        OwnAndGhostIndices(own::OwnIndices,ghost::GhostIndices,global_to_owner=nothing)

    Build an instance of [`OwnAndGhostIndices`](@ref) from the underlying properties `own`, `ghost`, and `global_to_owner`.
    """
    function OwnAndGhostIndices(own::OwnIndices,ghost::GhostIndices,global_to_owner=nothing)
        A = typeof(global_to_owner)
        new{A}(own,ghost,global_to_owner,AssemblyCache())
    end
end
assembly_cache(a::OwnAndGhostIndices) = a.assembly_cache

local_permutation(a::OwnAndGhostIndices) = Int32(1):Int32(local_length(a))

function replace_ghost(a::OwnAndGhostIndices,ghost::GhostIndices)
    OwnAndGhostIndices(a.own,ghost,a.global_to_owner)
end

function find_owner(indices,global_ids,::Type{<:OwnAndGhostIndices{T}}) where T
    if T == Nothing
        error("Not enough data to perform this operation without communciation")
    end
    map(global_ids,indices) do global_ids, indices
        map_global_to_owner(global_ids,indices.global_to_owner)
    end
end

function global_to_owner(a::OwnAndGhostIndices)
    a.global_to_owner
end

function global_to_owner(indices,::Type{<:OwnAndGhostIndices{T}}) where T
    map(global_to_owner,indices) |> getany
end

function map_global_to_owner(I,global_to_owner::AbstractArray)
    Ti = eltype(global_to_owner)
    owners = Vector{Ti}(undef,length(I))
    for k in 1:length(I)
        i = I[k]
        if i<1
            owners[k] = zero(Ti)
            continue
        end
        owners[k] = global_to_owner[i]
    end
    owners
end

function map_global_to_owner(I,global_to_owner::Function)
    Ti = Int32
    owners = Vector{Ti}(undef,length(I))
    for k in 1:length(I)
        i = I[k]
        if i<1
            owners[k] = zero(Ti)
            continue
        end
        owners[k] = global_to_owner(i)
    end
    owners
end

part_id(a::OwnAndGhostIndices) = a.own.owner

function own_to_global(a::OwnAndGhostIndices)
    a.own.own_to_global
end

function own_to_owner(a::OwnAndGhostIndices)
    owner = Int32(a.own.owner)
    n_own = length(a.own.own_to_global)
    OwnToOwner(owner,n_own)
end

function global_to_own(a::OwnAndGhostIndices)
    a.own.global_to_own
end

function ghost_to_global(a::OwnAndGhostIndices)
    a.ghost.ghost_to_global
end

function ghost_to_owner(a::OwnAndGhostIndices)
    a.ghost.ghost_to_owner
end

function global_to_ghost(a::OwnAndGhostIndices)
    a.ghost.global_to_ghost
end

function own_to_local(a::OwnAndGhostIndices)
    n_own = length(a.own.own_to_global)
    Int32.(1:n_own)
end

function ghost_to_local(a::OwnAndGhostIndices)
    n_own = length(a.own.own_to_global)
    n_ghost = length(a.ghost.ghost_to_global)
    Int32.((1:n_ghost).+n_own)
end

function local_to_own(a::OwnAndGhostIndices)
    n_own = own_length(a)
    n_ghost = ghost_length(a)
    n_local = n_own + n_ghost
    LocalToOwn(n_own,1:n_local)
end

function local_to_ghost(a::OwnAndGhostIndices)
    n_own = own_length(a)
    n_ghost = ghost_length(a)
    n_local = n_own + n_ghost
    LocalToGhost(n_own,1:n_local)
end

function global_to_local(a::OwnAndGhostIndices)
    GlobalToLocal(global_to_own(a),global_to_ghost(a),own_to_local(a),ghost_to_local(a))
end

function local_to_global(a::OwnAndGhostIndices)
    n_own = own_length(a)
    n_ghost = ghost_length(a)
    n_local = n_own + n_ghost
    perm = 1:n_local
    LocalToGlobal(own_to_global(a),ghost_to_global(a),perm)
end

function local_to_owner(a::OwnAndGhostIndices)
    n_own = own_length(a)
    n_ghost = ghost_length(a)
    n_local = n_own + n_ghost
    perm = 1:n_local
    LocalToOwner(own_to_owner(a),ghost_to_owner(a),perm)
end

struct PermutedLocalIndices{A} <: AbstractLocalIndices
    indices::A
    perm::Vector{Int32}
    own_to_local::Vector{Int32}
    ghost_to_local::Vector{Int32}
    assembly_cache::AssemblyCache
end
assembly_cache(a::PermutedLocalIndices) = a.assembly_cache

function PermutedLocalIndices(indices,perm)
    n_own = length(own_to_owner(indices))
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

part_id(a::PermutedLocalIndices) = part_id(a.indices)

function own_to_global(a::PermutedLocalIndices)
    own_to_global(a.indices)
end

function own_to_owner(a::PermutedLocalIndices)
    own_to_owner(a.indices)
end

function global_to_own(a::PermutedLocalIndices)
    global_to_own(a.indices)
end

function ghost_to_global(a::PermutedLocalIndices)
    ghost_to_global(a.indices)
end

function ghost_to_owner(a::PermutedLocalIndices)
    ghost_to_owner(a.indices)
end

function global_to_ghost(a::PermutedLocalIndices)
    global_to_ghost(a.indices)
end

function own_to_local(a::PermutedLocalIndices)
    a.own_to_local
end

function ghost_to_local(a::PermutedLocalIndices)
    a.ghost_to_local
end

function local_to_own(a::PermutedLocalIndices)
    n_own = own_length(a)
    LocalToOwn(n_own,a.perm)
end

function local_to_ghost(a::PermutedLocalIndices)
    n_own = own_length(a)
    LocalToGhost(n_own,a.perm)
end

function global_to_local(a::PermutedLocalIndices)
    GlobalToLocal(global_to_own(a),global_to_ghost(a),own_to_local(a),ghost_to_local(a))
end

function local_to_global(a::PermutedLocalIndices)
    LocalToGlobal(own_to_global(a),ghost_to_global(a),a.perm)
end

function local_to_owner(a::PermutedLocalIndices)
    LocalToOwner(own_to_owner(a),ghost_to_owner(a),a.perm)
end

function find_owner(indices,global_ids,::Type{<:PermutedLocalIndices})
    inner_parts = map(i->i.indices,indices)
    find_owner(inner_parts,global_ids)
end

function global_to_owner(indices,::Type{<:PermutedLocalIndices})
    inner_parts = map(i->i.indices,indices)
    global_to_owner(inner_parts)
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
    ranges::NTuple{N,UnitRange{Int}}
    function LocalIndicesWithConstantBlockSize(
            p::CartesianIndex{N},
            np::NTuple{N,Int},
            n::NTuple{N,Int},
            ghost::GhostIndices) where N
            ranges = map(local_range,Tuple(p),np,n)
        new{N}(p, np, n, ghost, AssemblyCache(),ranges)
    end
end
assembly_cache(a::LocalIndicesWithConstantBlockSize) = a.assembly_cache

#function Base.getproperty(a::LocalIndicesWithConstantBlockSize, sym::Symbol)
#    if sym === :ranges
#        map(local_range,Tuple(a.p),a.np,a.n)
#    else
#        getfield(a,sym)
#    end
#end
#
#function Base.propertynames(x::LocalIndicesWithConstantBlockSize, private::Bool=false)
#  (fieldnames(typeof(x))...,:ranges)
#end

function replace_ghost(a::LocalIndicesWithConstantBlockSize,ghost::GhostIndices)
    LocalIndicesWithConstantBlockSize(a.p,a.np,a.n,ghost)
end

function find_owner(indices,global_ids,::Type{<:LocalIndicesWithConstantBlockSize})
    map(indices,global_ids) do indices,global_ids
        start2 = map(indices.np,indices.n) do np,n
            start = [ first(local_range(p,np,n)) for p in 1:np ]
            push!(start,n+1)
            start
        end
        global_to_owner = BlockPartitionGlobalToOwner(start2)
        map_global_to_owner(global_ids,global_to_owner)
    end
end

function global_to_owner(indices,::Type{<:LocalIndicesWithConstantBlockSize})
    map(indices) do indices
        start2 = map(indices.np,indices.n) do np,n
            start = [ first(local_range(p,np,n)) for p in 1:np ]
            push!(start,n+1)
            start
        end
        global_to_owner = BlockPartitionGlobalToOwner(start2)
    end |> getany
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
        map_global_to_owner(global_ids,global_to_owner)
    end
end

function global_to_owner(indices,::Type{<:LocalIndicesWithVariableBlockSize})
    initial = map(indices->map(first,indices.ranges),indices) |> collect |> tuple_of_arrays
    map(indices) do indices
        start = map(indices.n,initial) do n,initial
            start = vcat(initial,[n+1])
        end
        global_to_owner = BlockPartitionGlobalToOwner(start)
    end |> getany
end

const LocalIndicesInBlockPartition = Union{LocalIndicesWithConstantBlockSize,LocalIndicesWithVariableBlockSize}

local_permutation(a::LocalIndicesInBlockPartition) = Int32(1):Int32(local_length(a))

function part_id(a::LocalIndicesInBlockPartition)
    owner = LinearIndices(a.np)[a.p]
    Int32(owner)
end

function own_to_global(a::LocalIndicesInBlockPartition)
    BlockPartitionOwnToGlobal(a.n,a.ranges)
end

function own_to_owner(a::LocalIndicesInBlockPartition)
    lis = LinearIndices(a.np)
    owner = Int32(lis[a.p])
    n_own = prod(map(length,a.ranges))
    OwnToOwner(owner,n_own)
end

function global_to_own(a::LocalIndicesInBlockPartition)
    BlockPartitionGlobalToOwn(a.n,a.ranges)
end

function ghost_to_global(a::LocalIndicesInBlockPartition)
    a.ghost.ghost_to_global
end

function ghost_to_owner(a::LocalIndicesInBlockPartition)
    a.ghost.ghost_to_owner
end

function global_to_ghost(a::LocalIndicesInBlockPartition)
    a.ghost.global_to_ghost
end

function own_to_local(a::LocalIndicesInBlockPartition)
    n_own = prod(map(length,a.ranges))
    Int32(1):Int32(n_own)
end

function ghost_to_local(a::LocalIndicesInBlockPartition)
    n_own = prod(map(length,a.ranges))
    n_ghost = length(a.ghost.ghost_to_global)
    ((Int32(1):Int32(n_ghost)).+Int32(n_own))
end

function local_to_own(a::LocalIndicesInBlockPartition)
    n_own = own_length(a)
    n_ghost = ghost_length(a)
    n_local = n_own + n_ghost
    LocalToOwn(n_own,1:n_local)
end

function local_to_ghost(a::LocalIndicesInBlockPartition)
    n_own = own_length(a)
    n_ghost = ghost_length(a)
    n_local = n_own + n_ghost
    LocalToGhost(n_own,1:n_local)
end

function global_to_local(a::LocalIndicesInBlockPartition)
    GlobalToLocal(global_to_own(a),global_to_ghost(a),own_to_local(a),ghost_to_local(a))
end

function local_to_global(a::LocalIndicesInBlockPartition)
    n_own = own_length(a)
    n_ghost = ghost_length(a)
    n_local = n_own + n_ghost
    perm = 1:n_local
    LocalToGlobal(own_to_global(a),ghost_to_global(a),perm)
end

function local_to_owner(a::LocalIndicesInBlockPartition)
    n_own = own_length(a)
    n_ghost = ghost_length(a)
    n_local = n_own + n_ghost
    perm = 1:n_local
    LocalToOwner(own_to_owner(a),ghost_to_owner(a),perm)
end

"""
    struct PRange{A}

`PRange` (partitioned range) is a type representing a range of indices `1:n`
partitioned into several parts. This type is used to represent the axes of instances
of [`PVector`](@ref) and [`PSparseMatrix`](@ref).

# Properties
- `partition::A`

The item `partition[i]` is an object that contains information about the own, ghost, and local indices of part number `i`.
`typeof(partition[i])` is a type that
implements the methods of the [`AbstractLocalIndices`](@ref) interface. Use this
interface to access the underlying information about own, ghost, and local indices.

# Supertype hierarchy

    PRange{A} <: AbstractUnitRange{Int}

"""
struct PRange{A} <: AbstractUnitRange{Int}
    partition::A
    @doc """
        PRange(partition)

    Build an instance of [`PRange`](@ref) from the underlying `partition`.
    """
    function PRange(partition)
        A = typeof(partition)
        new{A}(partition)
    end
end
"""
    partition(a::PRange)

Get `a.partition`.
"""
partition(a::PRange) = a.partition
Base.first(a::PRange) = 1
Base.last(a::PRange) = getany(map(global_length,partition(a)))
function Base.show(io::IO,k::MIME"text/plain",data::PRange)
    np = length(partition(data))
    map_main(partition(data)) do indices
        println(io,"PRange 1:$(global_length(indices)) partitioned into $(np) parts")
    end
end

function Base.show(io::IO,data::PRange)
    print(io,"PRange(…)")
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

global_length(pr::PRange) = map(global_length,partition(pr))
local_length(pr::PRange) = map(local_length,partition(pr))
own_length(pr::PRange) = map(own_length,partition(pr))
local_to_global(pr::PRange) = map(local_to_global,partition(pr))
own_to_global(pr::PRange) = map(own_to_global,partition(pr))
ghost_to_global(pr::PRange) = map(ghost_to_global,partition(pr))
local_to_owner(pr::PRange) = map(local_to_owner,partition(pr))
own_to_owner(pr::PRange) = map(own_to_owner,partition(pr))
ghost_to_owner(pr::PRange) = map(ghost_to_owner,partition(pr))
global_to_local(pr::PRange) = map(global_to_local,partition(pr))
global_to_own(pr::PRange) = map(global_to_own,partition(pr))
global_to_ghost(pr::PRange) = map(global_to_ghost,partition(pr))
own_to_local(pr::PRange) = map(own_to_local,partition(pr))
ghost_to_local(pr::PRange) = map(ghost_to_local,partition(pr))
local_to_own(pr::PRange) = map(local_to_own,partition(pr))
local_to_ghost(pr::PRange) = map(local_to_ghost,partition(pr))

