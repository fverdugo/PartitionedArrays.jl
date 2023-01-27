
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
local_permutation(a::LocalIndices) = a.perm

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

local_permutation(a::OwnAndGhostIndices) = Int32(1):Int32(local_length(a))

function replace_ghost(a::OwnAndGhostIndices,ghost::GhostIndices)
    OwnAndGhostIndices(a.own,ghost)
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

