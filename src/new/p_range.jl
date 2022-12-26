
"""
    struct PRange{A}

`PRange` (partitioned range) is a type representing a range of indices `1:n_global`
distributed into several parts. A part in a `PRange` contains three subsets of
 `1:n_global` referred to as *local*, *own*, and *ghost* indices
respectively. A part is represented by some type implementing the 
[`PRangePart`](@ref) interface.

# Properties
- `n_global::Int`: Number of *global* indices in the range.
- `parts::A`: Array-like object containing the parts in the `PRange`.

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
    ranges = map(local_range,Tuple(p),np,n)
    ghost = GhostIndices(prod(n))
    BlockPartitionPart(p,np,n,ranges,ghost)
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
    ids = BlockPartitionPart(p,np,n,own_ranges,ghostids)
    PermutedPart(ids,perm)
end

struct VariableBlockSize end

"""
    PRange(VariableBlockSize(),n_own[,n[,offset]])

"""
function PRange(::VariableBlockSize,
    n_own,n_global=sum(n_own),
    offset=scan(+,n_own,type=:exclusive,init=zero(eltype(n_own))))
    ranks = linear_indices(n_own)
    n_parts = length(n_own)
    parts = map(ranks,n_own,offset) do rank,n_own,offset
        r = (1:n_own) .+ offset
        p = CartesianIndex((rank,))
        np = (n_parts,)
        ranges = (r,)
        ghost = GhostIndices(n_global)
        n = (n_global,)
        BlockPartitionPart(p,np,n,ranges,ghost)
    end
    PRange(n_global,parts)
end

struct GhostIndices
    n_global::Int
    ghost_to_global::Vector{Int}
    ghost_to_owner::Vector{Int32}
    global_to_ghost::SparseVector{Int32,Int32}
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
    global_to_ghost::SparseVector{Int32,Int32}
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

struct PermutedPart{A}
    part::A
    perm::Vector{Int32}
    own_to_local::Vector{Int32}
    ghost_to_local::Vector{Int32}
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

struct BlockPartitionPart{N}
    p::CartesianIndex{N}
    np::NTuple{N,Int}
    n::NTuple{N,Int}
    ranges::NTuple{N,UnitRange{Int}}
    ghost::GhostIndices
end

function get_own_to_global(a::BlockPartitionPart)
    BlockPartitionOwnToGlobal(a.n,a.ranges)
end

function get_own_to_owner(a::BlockPartitionPart)
    lis = LinearIndices(a.np)
    owner = Int32(lis[a.p])
    n_own = prod(map(length,a.ranges))
    OwnToOwner(owner,n_own)
end

function get_global_to_own(a::BlockPartitionPart)
    BlockPartitionGlobalToOwn(a.n,a.ranges)
end

function get_ghost_to_global(a::BlockPartitionPart)
    a.ghost.ghost_to_global
end

function get_ghost_to_owner(a::BlockPartitionPart)
    a.ghost.ghost_to_owner
end

function get_global_to_ghost(a::BlockPartitionPart)
    a.ghost.global_to_ghost
end

function get_own_to_local(a::BlockPartitionPart)
    n_own = prod(map(length,a.ranges))
    Int32.(1:n_own)
end

function get_ghost_to_local(a::BlockPartitionPart)
    n_own = prod(map(length,a.ranges))
    n_ghost = length(a.ghost.ghost_to_global)
    Int32.((1:n_ghost).+n_own)
end

function get_local_to_own(a::BlockPartitionPart)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    n_own = length(own_to_local)
    n_ghost = length(ghost_to_local)
    n_local = n_own + n_ghost
    LocalToOwn(n_own,1:n_local)
end

function get_local_to_ghost(a::BlockPartitionPart)
    own_to_local = get_own_to_local(a)
    ghost_to_local = get_ghost_to_local(a)
    n_own = length(own_to_local)
    n_ghost = length(ghost_to_local)
    n_local = n_own + n_ghost
    LocalToGhost(n_own,1:n_local)
end

function get_global_to_local(a::BlockPartitionPart)
    global_to_own = get_global_to_own(a)
    global_to_ghost = get_global_to_ghost(a)
    own_to_local = get_own_to_local(a,OWN)
    ghost_to_local = get_ghost_to_local(a)
    GlobalToLocal(global_to_own,global_to_ghost,own_to_local,ghost_to_local)
end

function get_local_to_global(a::BlockPartitionPart)
    own_to_global = get_own_to_global(a)
    ghost_to_global = get_ghost_to_global(a)
    n_own = length(own_to_global)
    n_ghost = length(ghost_to_global)
    n_local = n_own + n_ghost
    perm = 1:n_local
    LocalToGlobal(own_to_global,ghost_to_global,perm)
end

function get_local_to_owner(a::BlockPartitionPart)
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

