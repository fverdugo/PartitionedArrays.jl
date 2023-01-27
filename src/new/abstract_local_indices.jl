"""
    abstract type AbstractLocalIndices

Abstract type representing the *local*, *own*, and *ghost* indices in
a part of an instance of [`PRange`](@ref).

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

function filter_ghost(indices,gids,owners)
    set = Set{Int}()
    part_owner = part_id(indices)
    n_new_ghost = 0
    global_to_ghost_indices = global_to_ghost(indices)
    for (global_i,owner) in zip(gids,owners)
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

function to_local!(I,indices)
    global_to_local_indices = global_to_local(indices)
    for k in 1:length(I)
        I[k] = global_to_local_indices[I[k]]
    end
    I
end

function to_global!(I,indices)
    local_to_global_indices = local_to_global(indices)
    for k in 1:length(I)
        I[k] = local_to_global_indices[I[k]]
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

function assembly_graph(index_partition;kwargs...)
    neighbors_snd,neighbors_rcv = assembly_neighbors(index_partition;kwargs...)
    ExchangeGraph(neighbors_snd,neighbors_rcv)
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
        local_indices_snd = JaggedArray(data_lids,ptrs)
        global_indices_snd = JaggedArray(data_gids,ptrs)
        local_indices_snd, global_indices_snd
    end |>  tuple_of_arrays
    graph = ExchangeGraph(parts_snd,parts_rcv)
    global_indices_rcv = exchange_fetch(global_indices_snd,graph)
    local_indices_rcv = map(global_indices_rcv,indices) do global_indices_rcv,indices
        ptrs = global_indices_rcv.ptrs
        data_lids = zeros(Int32,ptrs[end]-1)
        global_to_local_indices = global_to_local(indices)
        for (k,gid) in enumerate(global_indices_rcv.data)
            data_lids[k] = global_to_local_indices[gid]
        end
        local_indices_rcv = JaggedArray(data_lids,ptrs)
    end
    local_indices_snd,local_indices_rcv
end

"""
    permute_indices(indices,perm)
"""
permute_indices(a,b) = PermutedLocalIndices(a,b)

