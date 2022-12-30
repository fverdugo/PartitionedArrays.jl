"""
    abstract type AbstractLocalIndices

Abstract type representing a part in the `PRange` type.

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
- [`set_ghost!`](@ref)
- [`append_ghost!`](@ref)
- [`union_ghost!`](@ref)

"""
abstract type AbstractLocalIndices end

get_n_local(a) = get_n_own(a) + get_n_ghost(a)
get_n_own(a) = length(get_own_to_owner(a))
get_n_ghost(a) = length(get_ghost_to_global(a))
get_n_global(a) = length(get_global_to_own(a))

"""
    get_local_to_global(local_indices)
"""
function get_local_to_global end

"""
    get_own_to_global(local_indices)
"""
function get_own_to_global end

"""
    get_ghost_to_global(local_indices)
"""
function get_ghost_to_global end

"""
    get_local_to_owner(local_indices)
"""
function get_local_to_owner end

"""
    get_own_to_owner(local_indices)
"""
function get_own_to_owner end

"""
    get_ghost_to_owner(local_indices)
"""
function get_ghost_to_owner end

"""
    get_global_to_local(local_indices)
"""
function get_global_to_local end

"""
    get_global_to_own(local_indices)
"""
function get_global_to_own end

"""
    get_global_to_ghost(local_indices)
"""
function get_global_to_ghost end

"""
    get_own_to_local(local_indices)
"""
function get_own_to_local end

"""
    get_ghost_to_local(local_indices)
"""
function get_ghost_to_local end

"""
    get_local_to_own(local_indices)
"""
function get_local_to_own end

"""
    get_local_to_ghost(local_indices)
"""
function get_local_to_ghost end

function set_ghost! end

function append_ghost! end

function union_ghost!(local_indices,gids,owners)
    part_owner = get_owner(local_indices)
    n_new_ghost = 0
    global_to_ghost = get_global_to_ghost(local_indices)
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
    append_ghost!(local_indices,new_ghost_to_global,new_ghost_to_owner)
end

function find_owner(local_indices,global_ids)
    find_owner(local_indices,global_ids,eltype(local_indices))
end

"""
    struct PRange{A}

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

# Fields
- `n_global::Int`: Number of global indices.
- `local_indices::A`: Array-like object with `length(local_indices)` equal to the number of parts in the partitioned range.

The item `local_indices[i]` is an object that contains information about the own, ghost, and local indices of part number `i`. `typeof(local_indices[i])` is a type that
implements the methods of the [`AbstractLocalIndices`](@ref) interface. Use this
interface to access the underlying information about own, ghost, and local indices.

# Supertype hierarchy

    PRange{A} <: AbstractUnitRange{Int}

"""
struct PRange{A} <: AbstractUnitRange{Int}
    n_global::Int
    local_indices::A
    @doc """
        PRange(n_global,local_indices)

    Build an instance of [`Prange`](@ref) from the underlying fields
    `n_global` and `local_indices`.

    # Examples
   
        julia> using PartitionedArrays
        
        julia> rank = LinearIndices((2,));
        
        julia> local_indices = map(rank) do rank
                   if rank == 1
                       LocalIndices(8,1,[1,2,3,4,5],Int32[1,1,1,1,2])
                   else
                       LocalIndices(8,2,[4,5,6,7,8],Int32[1,2,2,2,2])
                   end
               end;
        
        julia> pr = PRange(8,local_indices)
        1:1:8
        
        julia> get_local_to_global(pr)
        2-element Vector{Vector{Int64}}:
         [1, 2, 3, 4, 5]
         [4, 5, 6, 7, 8]
    """
    function PRange(n_global,local_indices)
        A = typeof(local_indices)
        new{A}(Int(n_global),local_indices)
    end
end
Base.first(a::PRange) = 1
Base.last(a::PRange) = a.n_global

get_n_global(pr::PRange) = a.n_global
get_n_local(pr::PRange) = map(get_n_local,pr.local_indices)
get_n_own(pr::PRange) = map(get_n_own,pr.local_indices)
get_local_to_global(pr::PRange) = map(get_local_to_global,pr.local_indices)
get_own_to_global(pr::PRange) = map(get_own_to_global,pr.local_indices)
get_ghost_to_global(pr::PRange) = map(get_ghost_to_global,pr.local_indices)
get_local_to_owner(pr::PRange) = map(get_local_to_owner,pr.local_indices)
get_own_to_owner(pr::PRange) = map(get_own_to_owner,pr.local_indices)
get_ghost_to_owner(pr::PRange) = map(get_ghost_to_owner,pr.local_indices)
get_global_to_local(pr::PRange) = map(get_global_to_local,pr.local_indices)
get_global_to_own(pr::PRange) = map(get_global_to_own,pr.local_indices)
get_global_to_ghost(pr::PRange) = map(get_global_to_ghost,pr.local_indices)
get_own_to_local(pr::PRange) = map(get_own_to_local,pr.local_indices)
get_ghost_to_local(pr::PRange) = map(get_ghost_to_local,pr.local_indices)
get_local_to_own(pr::PRange) = map(get_local_to_own,pr.local_indices)
get_local_to_ghost(pr::PRange) = map(get_local_to_ghost,pr.local_indices)

find_owner(pr::PRange,global_ids) = find_owner(pr.local_indices,global_ids)

function set_ghost!(pr::PRange,gids,owners=find_owner(pr,gids))
    map(set_ghost!,pr.local_indices,gids,owners)
end

function append_ghost!(pr::PRange,gids,owners=find_owner(pr,gids))
    map(append_ghost!,pr.local_indices,gids,owners)
end

function union_ghost!(pr::PRange,gids,owners=find_owner(pr,gids))
    map(union_ghost!,pr.local_indices,gids,owners)
end

struct ConstantBlockSize end

"""
    PRange(ConstantBlockSize(),ranks,np,n[,ghost[,periodic]])

Generate an instance of `PRange` by using an `N` dimensional
block partition with a (roughly) constant block size.

# Arguments
- `ranks::AbstractArray{<:Integer}`: Array containing the distribution of ranks.
-  `np::NTuple{N,Int}`: Number of parts per direction.
-  `n::NTuple{N,Int}`: Number of global indices per direction.
-  `ghost::NTuple{N,Bool}=ntuple(i->false,N)`: Use or not ghost indices per direction.
-  `periodic::NTuple{N,Bool}=ntuple(i->false,N)`: Use or not periodic boundaries per direction.

For convenience, one can also provide scalar inputs instead tuples
to create 1D block partitions.

# Examples

1D partition of 10 indices into 4 parts

    julia> using PartitionedArrays
    
    julia> rank = LinearIndices((4,));
    
    julia> pr = PRange(ConstantBlockSize(),rank,4,10)
    1:1:10
    
    julia> get_local_to_global(pr)
    4-element Vector{PartitionedArrays.LocalToGlobal{PartitionedArrays.BlockPartitionOwnToGlobal{1}, UnitRange{Int64}}}:
     [1, 2]
     [3, 4]
     [5, 6, 7]
     [8, 9, 10]

2D partition of 4x4 indices into 2x2 parts with ghost

    julia> pr = PRange(ConstantBlockSize(),rank,(2,2),(4,4),(true,true))
    1:1:16
    
    julia> get_local_to_global(pr)
    4-element Vector{PartitionedArrays.LocalToGlobal{PartitionedArrays.BlockPartitionOwnToGlobal{2}, Vector{Int32}}}:
     [1, 2, 3, 5, 6, 7, 9, 10, 11]
     [2, 3, 4, 6, 7, 8, 10, 11, 12]
     [5, 6, 7, 9, 10, 11, 13, 14, 15]
     [6, 7, 8, 10, 11, 12, 14, 15, 16]

"""
function PRange(::ConstantBlockSize,ranks,np,n,args...)
    @assert prod(np) == length(ranks)
    local_indices = map(ranks) do rank
        block_with_constant_size(rank,np,n,args...)
    end
    PRange(prod(n),local_indices)
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
    LocalIndicesWithConstantBlockSize(p,np,n,ghost)
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
    ids = LocalIndicesWithConstantBlockSize(p,np,n,ghostids)
    PermutedLocalIndices(ids,perm)
end

struct VariableBlockSize end

"""
    PRange(VariableBlockSize(),rank,n_own,n_global[;start])

Build an instance of [`PRange`](@ref) using a 1D variable-size block partition.

# Arguments

- `ranks::AbstractArray{<:Integer}`: Array containing the distribution of ranks.
-  `n_own::AbstractArray{<:Integer}`: Array containing the block size for each part.
-  `n_global::Integer`: Number of global indices. It should be equal to `sum(n_own)`.
-  `start::AbstractArray{Int}=scan(+,n_own,type=:exclusive,init=1)`: First global index in each part.

# Examples

    julia> using PartitionedArrays
    
    julia> rank = LinearIndices((4,));
    
    julia> n_own = [3,2,2,3];
    
    julia> pr = PRange(VariableBlockSize(),rank,n_own,sum(n_own))
    1:1:10
    
    julia> get_own_to_global(pr)
    4-element Vector{PartitionedArrays.BlockPartitionOwnToGlobal{1}}:
     [1, 2, 3]
     [4, 5]
     [6, 7]
     [8, 9, 10]

"""
function PRange(::VariableBlockSize,
    rank,
    n_own,
    n_global,
    ghost=false,
    periodic=false;
    start=scan(+,n_own,type=:exclusive,init=one(eltype(n_own))))

    @assert length(rank) == length(n_own)

    if ghost == true || periodic == true
        error("This case is not yet implemented.")
    end
    n_parts = length(n_own)
    local_indices = map(rank,n_own,start) do rank,n_own,start
        p = CartesianIndex((rank,))
        np = (n_parts,)
        n = (n_global,)
        ranges = ((1:n_own).+(start-1),)
        ghost = GhostIndices(n_global)
        LocalIndicesWithVariableBlockSize(p,np,n,ranges,ghost)
    end
    PRange(n_global,local_indices)
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

struct OwnIndices
    n_global::Int
    owner::Int32
    own_to_global::Vector{Int}
    global_to_own::VectorFromDict{Int,Int32}
end

function OwnIndices(n_global::Int,owner::Integer,own_to_global::Vector{Int})
    n_own = length(own_to_global)
    global_to_own = VectorFromDict(
      own_to_global,Int32.(1:n_own),n_global)
    OwnIndices(n_global,Int32(owner),own_to_global,global_to_own)
end

mutable struct GhostIndices
    n_global::Int
    ghost_to_global::Vector{Int}
    ghost_to_owner::Vector{Int32}
    global_to_ghost::VectorFromDict{Int,Int32}
end

function copy!(a::GhostIndices,b::GhostIndices)
    a.n_global = b.n_global
    a.ghost_to_global = b.ghost_to_global
    a.ghost_to_owner = b.ghost_to_owner
    a.global_to_ghost = b.global_to_ghost
    a
end

function GhostIndices(n_global,ghost_to_global,ghost_to_owner)
    n_ghost = length(ghost_to_global)
    @assert length(ghost_to_owner) == n_ghost
    global_to_ghost = VectorFromDict(
      ghost_to_global,Int32.(1:n_ghost),n_global)
    GhostIndices(
      n_global,ghost_to_global,ghost_to_owner,global_to_ghost)
end

function GhostIndices(n_global)
    ghost_to_global = Int[]
    ghost_to_owner = Int32[]
    GhostIndices(n_global,ghost_to_global,ghost_to_owner)
end

function set_ghost!(a::GhostIndices,new_ghost_to_global,new_ghost_to_owner)
    new_ghost = GhostIndices(a.n_global,new_ghost_to_global,new_ghost_to_owner)
    copy!(a,new_ghost)
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

struct LocalIndices <: AbstractLocalIndices
    n_global::Int
    owner::Int32
    local_to_global::Vector{Int}
    local_to_owner::Vector{Int32}
    perm::Vector{Int32}
    own_to_local::Vector{Int32}
    ghost_to_local::Vector{Int32}
    global_to_local::VectorFromDict{Int,Int32}
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
    global_to_local = VectorFromDict(local_to_global,Int32.(1:n_local),n_global)

    LocalIndices(
        Int(n_global),
        Int32(owner),
        local_to_global,
        local_to_owner,
        perm,
        Int32.(own_to_local),
        Int32.(ghost_to_local),
        global_to_local)
end

function set_ghost!(a::LocalIndices,new_ghost_to_global,new_ghost_to_owner)
    error("set_ghost! only makes sense for un-permuted local indices.")
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

struct OwnAndGhostIndices <: AbstractLocalIndices
    own::OwnIndices
    ghost::GhostIndices
end

function append_ghost!(a::OwnAndGhostIndices,new_ghost_to_global,new_ghost_to_owner)
    append_ghost!(a.ghost,new_ghost_to_global,new_ghost_to_owner)
    a
end

function set_ghost!(a::OwnAndGhostIndices,new_ghost_to_global,new_ghost_to_owner)
    set_ghost!(a.ghost,new_ghost_to_global,new_ghost_to_owner)
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

struct PermutedLocalIndices{A} <: AbstractLocalIndices
    local_indices::A
    perm::Vector{Int32}
    own_to_local::Vector{Int32}
    ghost_to_local::Vector{Int32}
end

function PermutedLocalIndices(local_indices,perm)
    n_own = length(get_own_to_owner(local_indices))
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
    PermutedLocalIndices(local_indices,_perm,own_to_local,ghost_to_local)
end

function append_ghost!(a::PermutedLocalIndices,new_ghost_to_global,new_ghost_to_owner)
    n_local = length(a.perm)
    n_new_ghost = length(new_ghost_to_global)
    r = (1:n_new_ghost).+n_local
    append_ghost!(a.local_indices,new_ghost_to_global,new_ghost_to_owner)
    append!(a.perm,r)
    append!(a.ghost_to_local,r)
    a
end

function set_ghost!(a::PermutedLocalIndices,new_ghost_to_global,new_ghost_to_owner)
    error("set_ghost! only makes sense for un-permuted local indices.")
end

get_owner(a::PermutedLocalIndices) = get_owner(a.local_indices)

function get_own_to_global(a::PermutedLocalIndices)
    get_own_to_global(a.local_indices)
end

function get_own_to_owner(a::PermutedLocalIndices)
    get_own_to_owner(a.local_indices)
end

function get_global_to_own(a::PermutedLocalIndices)
    get_global_to_own(a.local_indices)
end

function get_ghost_to_global(a::PermutedLocalIndices)
    get_ghost_to_global(a.local_indices)
end

function get_ghost_to_owner(a::PermutedLocalIndices)
    get_ghost_to_owner(a.local_indices)
end

function get_global_to_ghost(a::PermutedLocalIndices)
    get_global_to_ghost(a.local_indices)
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

function find_owner(local_indices,global_ids,::Type{<:PermutedLocalIndices})
    inner_parts = map(i->i.local_indices,local_indices)
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

struct LocalIndicesWithConstantBlockSize{N} <: AbstractLocalIndices
    p::CartesianIndex{N}
    np::NTuple{N,Int}
    n::NTuple{N,Int}
    ghost::GhostIndices
end

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

function find_owner(local_indices,global_ids,::Type{<:LocalIndicesWithConstantBlockSize})
    map(local_indices,global_ids) do local_indices,global_ids
        start = map(local_indices.np,local_indices.n) do np,n
            start = [ local_range(p,np,n) for p in 1:np ]
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
end

function find_owner(local_indices,global_ids,::Type{<:LocalIndicesWithVariableBlockSize})
    initial = map(local_indices->map(first,local_indices.ranges),local_indices) |> collect |> unpack
    map(local_indices,global_ids) do local_indices,global_ids
        start = map(local_indices.n,initial) do n,initial
            start = vcat(initial,[n+1])
        end
        global_to_owner = BlockPartitionGlobalToOwner(start)
        global_to_owner[global_ids]
    end
end

const LocalIndicesInBlockPartition = Union{LocalIndicesWithConstantBlockSize,LocalIndicesWithVariableBlockSize}

function append_ghost!(a::LocalIndicesInBlockPartition,new_ghost_to_global,new_ghost_to_owner)
    append_ghost!(a.ghost,new_ghost_to_global,new_ghost_to_owner)
    a
end

function set_ghost!(a::LocalIndicesInBlockPartition,new_ghost_to_global,new_ghost_to_owner)
    set_ghost!(a.ghost,new_ghost_to_global,new_ghost_to_owner)
    a
end

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
    Int32.(1:n_own)
end

function get_ghost_to_local(a::LocalIndicesInBlockPartition)
    n_own = prod(map(length,a.ranges))
    n_ghost = length(a.ghost.ghost_to_global)
    Int32.((1:n_ghost).+n_own)
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

function bounday_owner(p,np,n,ghost=false,periodic=false)
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

