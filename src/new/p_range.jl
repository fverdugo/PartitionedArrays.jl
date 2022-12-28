"""
    abstract type AbstractPartInPRange end

Abstract type representing a part in the `PRange` type.

Sub-types must implement the following interface:

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
- [`append_ghost!`](@ref)
- [`union_ghost!`](@ref)

"""
abstract type AbstractPartInPRange end

"""
    get_local_to_global(part)
"""
function get_local_to_global end

"""
    get_own_to_global(part)
"""
function get_own_to_global end

"""
    get_ghost_to_global(part)
"""
function get_ghost_to_global end

"""
    get_local_to_owner(part)
"""
function get_local_to_owner end

"""
    get_own_to_owner(part)
"""
function get_own_to_owner end

"""
    get_ghost_to_owner(part)
"""
function get_ghost_to_owner end

"""
    get_global_to_local(part)
"""
function get_global_to_local end

"""
    get_global_to_own(part)
"""
function get_global_to_own end

"""
    get_global_to_ghost(part)
"""
function get_global_to_ghost end

"""
    get_own_to_local(part)
"""
function get_own_to_local end

"""
    get_ghost_to_local(part)
"""
function get_ghost_to_local end

"""
    get_local_to_own(part)
"""
function get_local_to_own end

"""
    get_local_to_ghost(part)
"""
function get_local_to_ghost end


function union_ghost!(part,gids,owners)
    part_owner = get_owner(part)
    n_new_ghost = 0
    global_to_ghost = get_global_to_ghost(part)
    for (global_i,owner) in zip(gids,owners)
        if owner != part_owner
            ghost_i = global_to_ghost[global_i]
            if ghost_i == 0
                n_new_ghost += 1
            end
        end
    end
    new_ghost_to_global = zeros(Int,n_new_ghost)
    new_ghost_to_owner = zeros(Int32,n_new_ghost)
    new_ghost_i = 0
    for (global_i,owner) in zip(gids,owners)
        if owner != part_owner
            ghost_i = global_to_ghost[global_i]
            if ghost_i == 0
                new_ghost_i += 1
                new_ghost_to_global[new_ghost_i] = global_i
                new_ghost_to_owner[new_ghost_i] = owner
            end
        end
    end
    append_ghost!(part,new_ghost_to_global,new_ghost_to_owner)
end

function find_owner(parts,global_ids)
    find_owner(parts,global_ids,eltype(parts))
end

"""
    struct PRange{A}

`PRange` (partitioned range) is a type representing a range of indices `1:n_global`
distributed into several parts. A part in a `PRange` contains three subsets of
 `1:n_global` referred to as *local*, *own*, and *ghost* indices
respectively. A part is represented by some type implementing the 
[`AbstractPartInPRange`](@ref) interface.

# Properties
- `n_global::Int`: Number of *global* indices in the range.
- `parts::A`: Array-like object containing the parts in the `PRange`. `eltype(parts) <: AbstractPartInPRange`

# Remarks

For an object `pr::PRange`, the indices in `1:length(pr)` are referred to as the
*global* indices. In particular, `pr.n_global == length(pr)`. For `i`
in `1:length(pr.parts)`, the array `get_own_to_global(pr.parts[i])`
contains the subset of `1:length(pr)` *owned* by  the `i`-th part.
These subsets are disjoint, meaning that a global index is owned by only one part.
The array `get_local_to_global(pr.parts[i])` contains the *local* indices in the `i`-th
part. The indices in `get_local_to_global(pr.parts[i])` are a super set of the ones
in `get_own_to_global(pr.parts[i])` and they can overlap between parts.
Finally, the array
`get_ghost_to_global(pr.parts[i])` contains the *ghost*
indices in part number `i`. `get_ghost_to_global(pr.parts[i])` contains the indices
that are in `get_local_to_global(pr.parts[i])`, but not in
`get_own_to_global(pr.parts[i])`. I.e., ghost indices the ones that are local by not
owned by a given part.

# Supertype hierarchy

    PRange{A} <: AbstractUnitRange{Int}

"""
struct PRange{A} <: AbstractUnitRange{Int}
    n_global::Int
    parts::A
    @doc """
        PRange(n_global::Integer,parts::AbstractArray)

    Build an instance of `PRange` from the underlying fields
    (inner constructor).
    """
    function PRange(n_global::Integer,parts::AbstractArray)
        A = typeof(parts)
        new{A}(Int(n_global),parts)
    end
end
Base.first(a::PRange) = 1
Base.last(a::PRange) = a.n_global

get_local_to_global(pr::PRange) = map(get_local_to_global,pr.parts)
get_own_to_global(pr::PRange) = map(get_own_to_global,pr.parts)
get_ghost_to_global(pr::PRange) = map(get_ghost_to_global,pr.parts)
get_local_to_owner(pr::PRange) = map(get_local_to_owner,pr.parts)
get_own_to_owner(pr::PRange) = map(get_own_to_owner,pr.parts)
get_ghost_to_owner(pr::PRange) = map(get_ghost_to_owner,pr.parts)
get_global_to_local(pr::PRange) = map(get_global_to_local,pr.parts)
get_global_to_own(pr::PRange) = map(get_global_to_own,pr.parts)
get_global_to_ghost(pr::PRange) = map(get_global_to_ghost,pr.parts)
get_own_to_local(pr::PRange) = map(get_own_to_local,pr.parts)
get_ghost_to_local(pr::PRange) = map(get_ghost_to_local,pr.parts)
get_local_to_own(pr::PRange) = map(get_local_to_own,pr.parts)
get_local_to_ghost(pr::PRange) = map(get_local_to_ghost,pr.parts)

find_owner(pr::PRange,global_ids) = find_owner(pr.parts,global_ids)

function append_ghost!(pr::PRange,gids,owners=find_owner(pr,gids))
    map(append_ghost!,pr.parts,gids,owners)
end

function union_ghost!(pr::PRange,gids,owners=find_owner(pr,gids))
    map(union_ghost!,pr.parts,gids,owners)
end

struct ConstantBlockSize end

"""
    PRange(ConstantBlockSize(),ranks,np,n[,ghost[,periodic]])

Generate an instance of `PRange` by using an `N` dimensional
block partition with (roughly) constant block size.

# Arguments
- `ranks::AbstractArray{Int}` 
-  np::NTuple{N,Int}
-  n::NTuple{N,Int}
-  ghost::NTuple{N,Bool}=ntuple(i->false,N)
-  periodic::NTuple{N,Bool}=ntuple(i->false,N)

For convenience, one can also provide scalar inputs instead tuples
to create 1D block partitions.
"""
function PRange(::ConstantBlockSize,ranks,np,n,args...)
    parts = map(ranks) do rank
        block_with_constant_size(rank,np,n,args...)
    end
    PRange(prod(n),parts)
end

function block_with_constant_size(rank::Int,np::Int,n::Int)
    block_with_constant_size(rank,(np,),(n,))
end

function block_with_constant_size(
    rank::Int,np::Int,n::Int,ghost::Bool,periodic::Bool=false)
    block_with_constant_size(rank,(np,),(n,),(ghost,),(periodic,))
end

function block_with_constant_size(
    rank::Int,np::Dims{N},n::Dims{N}) where N
    p = CartesianIndices(np)[rank]
    ghost = GhostIndices(prod(n))
    PartWithConstantBlockSize(p,np,n,ghost)
end

function block_with_constant_size(
    rank::Int,np::Dims{N},n::Dims{N},
    ghost::NTuple{N,Bool},
    periodic::NTuple{N,Bool}=ntuple(i->false,N)) where N

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
    ids = PartWithConstantBlockSize(p,np,n,ghostids)
    PermutedPart(ids,perm)
end

struct VariableBlockSize end

"""
    PRange(VariableBlockSize(),ranges[,n[,ghost]])

"""
function PRange(::VariableBlockSize,
    ranges,
    n_global=sum(map(r->1+last(r)-first(r),ranges)),
    ghost=map(GhostIndices(n_global),ranges))
    ranks = linear_indices(ranges)
    n_parts = length(ranges)
    parts = map(ranks,ranges,ghost) do rank,r,ghost
        p = CartesianIndex((rank,))
        np = (n_parts,)
        n = (n_global,)
        ranges = (r,)
        PartWithVariableBlockSize(p,np,n,ranges,ghost)
    end
    PRange(n_global,parts)
end

struct OwnIndices
    n_global::Int
    owner::Int32
    own_to_global::Vector{Int}
    global_to_own::SparseVector{Int32,Int}
end

function OwnIndices(n_global::Int,owner::Integer,own_to_global::Vector{Int})
    n_own = length(own_to_global)
    global_to_own = sparsevec(
      own_to_global,Int32.(1:n_own),n_global)
    OwnIndices(n_global,Int32(owner),own_to_global,global_to_own)
end

struct GhostIndices
    n_global::Int
    ghost_to_global::Vector{Int}
    ghost_to_owner::Vector{Int32}
    global_to_ghost::SparseVector{Int32,Int}
end

function GhostIndices(n_global,ghost_to_global,ghost_to_owner)
    n_ghost = length(ghost_to_global)
    @assert length(ghost_to_owner) == n_ghost
    global_to_ghost = sparsevec(
      ghost_to_global,Int32.(1:n_ghost),n_global)
    GhostIndices(
      n_global,ghost_to_global,ghost_to_owner,global_to_ghost)
end

function GhostIndices(n_global)
    ghost_to_global = Int[]
    ghost_to_owner = Int32[]
    GhostIndices(n_global,ghost_to_global,ghost_to_owner)
end

function append_ghost!(a::GhostIndices,new_ghost_to_global,new_ghost_to_owner)
    n_ghost = length(a.ghost_to_global)
    n_new_ghost = length(new_ghost_to_global)
    append!(a.ghost_to_global,new_ghost_to_global)
    append!(a.ghost_to_owner,new_ghost_to_owner)
    a.global_to_ghost[new_ghost_to_global] = (1:n_new_ghost) .+ n_ghost
    a
end

# This is essentially a FillArray
# but we add this to improve stack trace
struct OwnToOwner <: AbstractVector{Int32}
    owner::Int32
    n_own::Int
end
Base.IndexStyle(::Type{<:OwnToOwner}) = IndexLinear()
Base.size(a::OwnToOwner) = (a.n_own,)
function Base.getindex(a::OwnToOwner,own_id::Int)
    a.owner
end

struct GlobalToLocal{A,B,C} <: AbstractVector{Int32}
    global_to_own::A
    global_to_ghost::SparseVector{Int32,Int}
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
    global_to_local::SparseVector{Int32,Int}
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
    global_to_local::SparseVector{Int32,Int}
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

struct LocalIndices{A} <: AbstractPartInPRange
    n_global::Int
    owner::Int32
    local_to_global::Vector{Int}
    local_to_owner::Vector{Int32}
    perm::Vector{Int32}
    own_to_local::Vector{Int32}
    ghost_to_local::Vector{Int32}
    global_to_local::SparseVector{Int32,Int}
end

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
    global_to_local = sparsevec(local_to_global,Int32.(1:n_local),n_global)

    LocalIndices(
        n_global,
        owner,
        local_to_global,
        local_to_owner,
        perm,
        own_to_local,
        ghost_to_local
        global_to_local)
end

function append_ghost!(a::LocalIndices,new_ghost_to_global,new_ghost_to_owner)
    n_local = length(a.local_to_global)
    n_new_ghost = length(new_ghost_to_global)
    r = (1:n_new_ghost).+n_local
    append!(a.local_to_global,new_ghost_to_global)
    append!(a.local_to_owner,new_ghost_to_owner)
    append!(a.perm,r)
    append!(a.ghost_to_local,r)
    a.global_to_local[new_ghost_to_global] = r
    a
end

get_owner(a::LocalIndices) = a.owner

function get_own_to_global(a::LocalIndices)
    view(a.local_to_global,a.own_to_local)
end

function get_own_to_owner(a::LocalIndices)
    n_own = length(a.own_to_local)
    OwnToOwner(a.owner,n_own)
end

function get_global_to_own(a::LocalIndices)
    n_own = legnth(a.own_to_local)
    GlobalToOwn(n_own,a.global_to_local,a.perm)
end

function get_ghost_to_global(a::LocalIndices)
    view(a.local_to_global,a.ghost_to_local)
end

function get_ghost_to_owner(a::LocalIndices)
    view(a.local_to_owner,a.ghost_to_local)
end

function get_global_to_ghost(a::LocalIndices)
    n_own = legnth(a.own_to_local)
    GlobalToGhost(n_own,a.global_to_local,a.perm)
end

function get_own_to_local(a::LocalIndices)
    a.own_to_local
end

function get_ghost_to_local(a::LocalIndices)
    a.ghost_to_local
end

function get_local_to_own(a::LocalIndices)
    n_own = legnth(a.own_to_local)
    LocalToOwn(n_own,a.perm)
end

function get_local_to_ghost(a::LocalIndices)
    n_own = legnth(a.own_to_local)
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

struct OwnAndGhostIndices{A} <: AbstractPartInPRange
    own::OwnIndices
    ghost::GhostIndices
end

function OwnAndGhostIndices(own::OwnIndices,ghost::GhostIndices)
    OwnAndGhostIndices(own,ghost)
end

function append_ghost!(a::OwnAndGhostIndices,new_ghost_to_global,new_ghost_to_owner)
    append_ghost!(a.ghost,new_ghost_to_global,new_ghost_to_owner)
    a
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

struct PermutedPart{A} <: AbstractPartInPRange
    part::A
    perm::Vector{Int32}
    own_to_local::Vector{Int32}
    ghost_to_local::Vector{Int32}
end

function append_ghost!(a::PermutedPart,new_ghost_to_global,new_ghost_to_owner)
    n_local = length(a.perm)
    n_new_ghost = length(new_ghost_to_global)
    r = (1:n_new_ghost).+n_local
    append_ghost!(a.part,new_ghost_to_global,new_ghost_to_owner)
    append!(a.perm,r)
    append!(a.ghost_to_local,r)
    a
end

function PermutedPart(part,perm)
    n_own = length(get_own_to_owner(part))
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
    PermutedPart(part,perm,own_to_local,ghost_to_local)
end

get_owner(a::PermutedPart) = get_owner(a.part)

function get_own_to_global(a::PermutedPart)
    get_own_to_global(a.part)
end

function get_own_to_owner(a::PermutedPart)
    get_own_to_owner(a.part)
end

function get_global_to_own(a::PermutedPart)
    get_global_to_own(a.part)
end

function get_ghost_to_global(a::PermutedPart)
    get_ghost_to_global(a.part)
end

function get_ghost_to_owner(a::PermutedPart)
    get_ghost_to_owner(a.part)
end

function get_global_to_ghost(a::PermutedPart)
    get_global_to_ghost(a.part)
end

function get_own_to_local(a::PermutedPart)
    a.own_to_local
end

function get_ghost_to_local(a::PermutedPart)
    a.ghost_to_local
end

function get_local_to_own(a::PermutedPart)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    n_own = length(own_to_local)
    LocalToOwn(n_own,a.perm)
end

function get_local_to_ghost(a::PermutedPart)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    n_own = length(own_to_local)
    LocalToGhost(n_own,a.perm)
end

function get_global_to_local(a::PermutedPart)
    global_to_own = get_global_to_own(a)
    global_to_ghost = get_global_to_ghost(a)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    GlobalToLocal(global_to_own,global_to_ghost,own_to_local,ghost_to_local)
end

function get_local_to_global(a::PermutedPart)
    own_to_global = get_own_to_global(a)
    ghost_to_global = get_ghost_to_global(a)
    LocalToGlobal(own_to_global,ghost_to_global,a.perm)
end

function get_local_to_owner(a::PermutedPart)
    own_to_owner = get_own_to_owner(a)
    ghost_to_owner = get_ghost_to_owner(a)
    LocalToOwner(own_to_owner,ghost_to_owner,a.perm)
end

function find_owner(parts,global_ids,::Type{<:PermutedPart})
    inner_parts = map(i->i.part,parts)
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

struct PartWithConstantBlockSize{N} <: AbstractPartInPRange
    p::CartesianIndex{N}
    np::NTuple{N,Int}
    n::NTuple{N,Int}
    ghost::GhostIndices
end

function Base.getproperty(a::PartWithConstantBlockSize, sym::Symbol)
    if sym === :ranges
        map(local_range,Tuple(a.p),a.np,a.n)
    else
        getfield(a,sym)
    end
end

function Base.propertynames(x::PartWithConstantBlockSize, private::Bool=false)
  (fieldnames(typeof(x))...,:ranges)
end

function find_owner(parts,global_ids,::Type{<:PartWithConstantBlockSize})
    map(parts,global_ids) do part,global_ids
        start = map(part.np,part.n) do np,n
            start = [ local_range(p,np,n) for p in 1:np ]
            push!(start,n+1)
            start
        end
        global_to_owner = BlockPartitionGlobalToOwner(start)
        global_to_owner[global_ids]
    end
end

struct PartWithVariableBlockSize{N} <: AbstractPartInPRange
    p::CartesianIndex{N}
    np::NTuple{N,Int}
    n::NTuple{N,Int}
    ranges::NTuple{N,Int}
    ghost::GhostIndices
end

function find_owner(parts,global_ids,::Type{<:PartWithConstantBlockSize})
    initial = map(part->map(first,part.ranges),parts) |> collect |> unpack
    map(parts,global_ids) do part,global_ids
        start = map(part.n,initial) do n,initial
            start = vcat(initial,[n+1])
        end
        global_to_owner = BlockPartitionGlobalToOwner(start)
        global_to_owner[global_ids]
    end
end

const PartInBlockParition = Union{PartWithConstantBlockSize,PartWithVariableBlockSize}

function append_ghost!(a::PartInBlockParition,new_ghost_to_global,new_ghost_to_owner)
    append_ghost!(a.ghost,new_ghost_to_global,new_ghost_to_owner)
    a
end

function get_owner(a::PartInBlockParition)
    owner = LinearIndices(a.np)[a.p]
    Int32(owner)
end

function get_own_to_global(a::PartInBlockParition)
    BlockPartitionOwnToGlobal(a.n,a.ranges)
end

function get_own_to_owner(a::PartInBlockParition)
    lis = LinearIndices(a.np)
    owner = Int32(lis[a.p])
    n_own = prod(map(length,a.ranges))
    OwnToOwner(owner,n_own)
end

function get_global_to_own(a::PartInBlockParition)
    BlockPartitionGlobalToOwn(a.n,a.ranges)
end

function get_ghost_to_global(a::PartInBlockParition)
    a.ghost.ghost_to_global
end

function get_ghost_to_owner(a::PartInBlockParition)
    a.ghost.ghost_to_owner
end

function get_global_to_ghost(a::PartInBlockParition)
    a.ghost.global_to_ghost
end

function get_own_to_local(a::PartInBlockParition)
    n_own = prod(map(length,a.ranges))
    Int32.(1:n_own)
end

function get_ghost_to_local(a::PartInBlockParition)
    n_own = prod(map(length,a.ranges))
    n_ghost = length(a.ghost.ghost_to_global)
    Int32.((1:n_ghost).+n_own)
end

function get_local_to_own(a::PartInBlockParition)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    n_own = length(own_to_local)
    n_ghost = length(ghost_to_local)
    n_local = n_own + n_ghost
    LocalToOwn(n_own,1:n_local)
end

function get_local_to_ghost(a::PartInBlockParition)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    n_own = length(own_to_local)
    n_ghost = length(ghost_to_local)
    n_local = n_own + n_ghost
    LocalToGhost(n_own,1:n_local)
end

function get_global_to_local(a::PartInBlockParition)
    global_to_own = get_global_to_own(a)
    global_to_ghost = get_global_to_ghost(a)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    GlobalToLocal(global_to_own,global_to_ghost,own_to_local,ghost_to_local)
end

function get_local_to_global(a::PartInBlockParition)
    own_to_global = get_own_to_global(a)
    ghost_to_global = get_ghost_to_global(a)
    n_own = length(own_to_global)
    n_ghost = length(ghost_to_global)
    n_local = n_own + n_ghost
    perm = 1:n_local
    LocalToGlobal(own_to_global,ghost_to_global,perm)
end

function get_local_to_owner(a::PartInBlockParition)
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

## Without ghost entries

    julia> using PartitionedArrays
    
    julia> local_range(1,3,10)
    1:3

    julia> local_range(2,3,10)
    4:6

    julia> local_range(3,3,10)
    7:10

## With ghost entries

    julia> using PartitionedArrays
    
    julia> local_range(1,3,10,true)
    1:4

    julia> local_range(2,3,10,true)
    3:7

    julia> local_range(3,3,10,true)
    6:10

## With periodic boundaries

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

