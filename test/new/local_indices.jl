module LocalIndicesTest

using PartitionedArrays

using SparseArrays

struct Global end
const GLOBAL = Global()

struct Own end
const OWN = Own()

struct Ghost end
const GHOST = Ghost()

struct Local end
const LOCAL = Local()

struct Owner end
const OWNER = Owner()

function indices end

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

struct GlobalToLocal{A} <: AbstractVector{Int32}
    n_own::Int32
    global_to_own::A
    global_to_ghost::SparseVector{Int32,Int32}
end
Base.size(a::GlobalToLocal) = size(a.global_to_own)
Base.IndexStyle(::Type{<:GlobalToLocal}) = IndexLinear()
function Base.getindex(a::GlobalToLocal,global_id::Int)
    own_id = a.global_to_own[global_id]
    z = Int32(0)
    if own_id != z
        return own_id
    end
    ghost_id = a.global_to_ghost[global_id]
    if ghost_id != z
        return ghost_id + a.n_own
    end
    return z
end


struct LocalToOwn <: AbstractVector{Int32}
    n_own::Int
    n_local::Int
end
Base.size(a::LocalToOwn) = (a.n_local,)
Base.IndexStyle(::Type{<:LocalToOwn}) = IndexLinear()
function Base.getindex(a::LocalToOwn,local_id::Int)
    if local_id > a.n_own
        Int32(0)
    else
        Int32(local_id)
    end
end

struct LocalToGhost <: AbstractVector{Int32}
    n_own::Int
    n_local::Int
end
Base.size(a::LocalToGhost) = (a.n_local,)
Base.IndexStyle(::Type{<:LocalToGhost}) = IndexLinear()
function Base.getindex(a::LocalToGhost,local_id::Int)
    if local_id > a.n_own
        Int32(local_id-a.n_own)
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
    offset = length(a.own_to_global)
    j = a.perm[local_id]
    if j > offset
        a.ghost_to_global[j-offset]
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
    offset = length(a.own_to_owner)
    j = a.perm[local_id]
    if j > offset
        a.ghost_to_owner[j-offset]
    else
        a.own_to_owner[j]
    end
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

struct BlockPartitionLocalIndices{N}
    p::CartesianIndex{N}
    np::NTuple{N,Int}
    n::NTuple{N,Int}
    ranges::NTuple{N,UnitRange{Int}}
    ghost::GhostIndices
end

function indices(a::BlockPartitionLocalIndices,::Own,::Global)
    BlockPartitionOwnToGlobal(a.n,a.ranges)
end

function indices(a::BlockPartitionLocalIndices,::Own,::Owner)
    lis = LinearIndices(a.n)
    owner = Int32(lis[a.p])
    OwnToOwner(owner,length(lis))
end

function indices(a::BlockPartitionLocalIndices,::Global,::Own)
    BlockPartitionGlobalToOwn(a.n,a.ranges)
end

function indices(a::BlockPartitionLocalIndices,::Ghost,::Global)
    a.ghost.ghost_to_global
end

function indices(a::BlockPartitionLocalIndices,::Ghost,::Owner)
    a.ghost.ghost_to_owner
end

function indices(a::BlockPartitionLocalIndices,::Global,::Ghost)
    a.ghost.global_to_ghost
end

function indices(a::BlockPartitionLocalIndices,::Own,::Local)
    n_own = prod(map(length,a.ranges))
    1:n_own
end

function indices(a::BlockPartitionLocalIndices,::Ghost,::Local)
    n_own = prod(map(length,a.ranges))
    n_ghost = length(a.ghost.ghost_to_global)
    (1:n_ghost).+n_own
end

function indices(a::BlockPartitionLocalIndices,::Local,::Own)
    n_own = prod(map(length,a.ranges))
    n_ghost = length(a.ghost.ghost_to_global)
    n_local = n_own + n_ghost
    LocalToOwn(n_own,n_local)
end

function indices(a::BlockPartitionLocalIndices,::Local,::Ghost)
    n_own = prod(map(length,a.ranges))
    n_ghost = length(a.ghost.ghost_to_global)
    n_local = n_own + n_ghost
    LocalToGhost(n_own,n_local)
end

function indices(a::BlockPartitionLocalIndices,::Global,::Local)
    n_own = Int32(prod(map(length,a.ranges)))
    global_to_own = indices(a,GLOBAL,OWN)
    global_to_ghost = indices(a,GLOBAL,GHOST)
    GlobalToLocal(n_own,global_to_own,global_to_ghost)
end

function indices(a::BlockPartitionLocalIndices,::Local,::Global)
    own_to_global = indices(a,OWN,GLOBAL)
    ghost_to_global = indices(a,GHOST,GLOBAL)
    n_own = length(own_to_global)
    n_ghost = length(ghost_to_global)
    n_local = n_own + n_ghost
    perm = 1:n_local
    LocalToGlobal(own_to_global,ghost_to_global,perm)
end

function indices(a::BlockPartitionLocalIndices,::Local,::Owner)
    own_to_owner = indices(a,OWN,OWNER)
    ghost_to_owner = indices(a,GHOST,OWNER)
    n_own = length(own_to_owner)
    n_ghost = length(ghost_to_owner)
    n_local = n_own + n_ghost
    perm = 1:n_local
    LocalToOwner(own_to_owner,ghost_to_owner,perm)
end

function block(rank::Integer,np::Integer,n::Integer)
    block(rank,(Int(np),),(Int(n),))
end

function block(rank::Integer,np::Dims{N},n::Dims{N}) where N
    p = CartesianIndices(np)[rank]
    ranges = map(local_range,Tuple(p),np,n)
    ghost = GhostIndices(prod(n))
    BlockPartitionLocalIndices(p,np,n,ranges,ghost)
end

function block_with_ghost(rank::Integer,np::Integer,n::Integer,periodic::Bool=false)
    block_with_ghost(rank,(Int(np),),(Int(n),),(periodic,))
end

function block_with_ghost(
    rank::Integer,np::Dims{N},n::Dims{N},
    periodic::NTuple{N,Bool}=ntuple(i->false,N),
    ghost::NTuple{N,Bool}=ntuple(i->true,N)) where N

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
    lis = LinearIndices(n)
    local_cis = CartesianIndices(local_ranges)
    owner_lis = LinearIndices(np)
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
    BlockPartitionLocalIndices(p,np,n,own_ranges,ghostids)
end

myblock = block_with_ghost(2,(2,2),(4,4))

oid_to_gid = indices(myblock,OWN,GLOBAL)
oid_to_owner = indices(myblock,OWN,OWNER)
gid_to_oid = indices(myblock,GLOBAL,OWN)

hid_to_gid = indices(myblock,GHOST,GLOBAL)
hid_to_owner = indices(myblock,GHOST,OWNER)
gid_to_hid = indices(myblock,GLOBAL,GHOST)

oid_to_lid = indices(myblock,OWN,LOCAL)
hid_to_lid = indices(myblock,GHOST,LOCAL)

lid_to_oid = indices(myblock,LOCAL,OWN)
lid_to_hid = indices(myblock,LOCAL,GHOST)
gid_to_lid = indices(myblock,GLOBAL,LOCAL)

lid_to_gid = indices(myblock,LOCAL,GLOBAL)
lid_to_owner = indices(myblock,LOCAL,OWNER)

#         OWNER GLOBAL OWN GHOST LOCAL
# GLOBAL      ?          x     x     x
# OWN         x      x               x
# GHOST       x      x               x
# LOCAL       x      x   x     x

#display(oid_to_gid)
#display(oid_to_owner)
#display(gid_to_oid)
#display(hid_to_gid)
#display(hid_to_owner)
#display(gid_to_hid)
#display(oid_to_lid)
#display(hid_to_lid)

#display(lid_to_oid)
#display(lid_to_hid)
#display(gid_to_lid)
display(lid_to_gid)
display(lid_to_owner)


#prange = Prange(block,ranks,3,10)


end # module
