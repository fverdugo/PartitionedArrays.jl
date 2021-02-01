
mutable struct LidToOHId <: AbstractVector{Int32}
  nlids::Int
  noids::Int32
end

function Base.copy(a::LidToOHId)
  LidToOHId(
    copy(a.nlids),
    copy(a.noids))
end

Base.size(a::LidToOHId) = (a.nlids,)
Base.IndexStyle(::Type{<:LidToOHId}) = IndexLinear()

@inline function Base.getindex(a::LidToOHId,lid::Integer)
  @boundscheck begin
    if !( 1<=lid && lid<=length(a) )
      throw(BoundsError(a,lid))
    end
  end
  if lid<=a.noids
    Int32(lid)
  else
    a.noids-Int32(lid)
  end
end

@inline function Base.push!(a::LidToOHId,ohid::Integer)
  @boundscheck begin
    if (a.noids - ohid) != (length(a) + 1)
      throw(DomainError(ohid))
    end
  end
  a.nlids += 1
  a
end

struct LidToGid <: AbstractVector{Int}
  oid_to_gid::UnitRange{Int}
  hid_to_gid::Vector{Int}
end

function Base.copy(a::LidToGid)
  LidToGid(
    copy(a.oid_to_gid),
    copy(a.hid_to_gid))
end

Base.size(a::LidToGid) = (length(a.oid_to_gid)+length(a.hid_to_gid),)
Base.IndexStyle(::Type{<:LidToGid}) = IndexLinear()

@inline function Base.getindex(a::LidToGid,lid::Integer)
  @boundscheck begin
    if !( 1<=lid && lid<=length(a) )
      throw(BoundsError(a,gid))
    end
  end
  noids = length(a.oid_to_gid)
  if lid <= noids
    oid = lid
    a.oid_to_gid[oid]
  else
    hid = lid - noids
    a.hid_to_gid[hid]
  end
end

@inline function Base.push!(a::LidToGid,gid::Integer)
  push!(a.hid_to_gid,gid)
  a
end

struct LidToPart <: AbstractVector{Int32}
  noids::Int
  part::Int32
  hid_to_part::Vector{Int32}
end

function Base.copy(a::LidToPart)
  LidToPart(
    copy(a.noids),
    copy(a.part),
    copy(a.hid_to_part))
end

Base.size(a::LidToPart) = (a.noids+length(a.hid_to_part),)
Base.IndexStyle(::Type{<:LidToPart}) = IndexLinear()

@inline function Base.getindex(a::LidToPart,lid::Integer)
  @boundscheck begin
    if !( 1<=lid && lid<=length(a) )
      throw(BoundsError(a,gid))
    end
  end
  if lid <= a.noids
    a.part
  else
    hid = lid - a.noids
    a.hid_to_part[hid]
  end
end

@inline function Base.push!(a::LidToPart,part::Integer)
  push!(a.hid_to_part,part)
  a
end

struct GidToLid{T} <: AbstractDict{Int,Int32}
  oid_to_gid::UnitRange{Int}
  oid_to_lid::T
  gid_to_lid::Dict{Int,Int32} # Only for ghost
end

function Base.copy(a::GidToLid)
  GidToLid(
    copy(a.oid_to_gid),
    copy(a.oid_to_lid),
    copy(a.gid_to_lid))
end

Base.length(a::GidToLid) = length(a.oid_to_gid) + length(a.gid_to_lid)
Base.haskey(a::GidToLid,gid::Integer) = (gid in a.oid_to_gid) || haskey(a.gid_to_lid,gid)

function Base.keys(a::GidToLid)
  Base.Iterators.flatten((a.oid_to_gid,keys(a.gid_to_lid)))
end

function Base.values(a::GidToLid)
  Base.Iterators.flatten(
    (Int32(1):Int32(length(a.oid_to_gid)),values(a.gid_to_lid)))
end

@inline function Base.getindex(a::GidToLid,gid::Integer)
  @boundscheck begin
    if ! haskey(a,gid)
      throw(KeyError(gid))
    end
  end
  if gid in a.oid_to_gid
    oid = 1 + gid - first(a.oid_to_gid)
    a.oid_to_lid[oid]
  else
    a.gid_to_lid[gid]
  end
end

@inline function Base.setindex!(a::GidToLid,lid::Integer,gid::Integer)
  if ! haskey(a,gid)
    a.gid_to_lid[gid] = lid
  end
  gid
end

function Base.iterate(a::GidToLid)
  itr = zip(keys(a),values(a))
  next = iterate(itr)
  if next == nothing
    return nothing
  end
  item, state = next
  item[1]=>item[2], (itr,state)
end

function Base.iterate(a::GidToLid,(itr,state))
  next = iterate(itr,state)
  if next == nothing
    return nothing
  end
  item, state = next
  item[1]=>item[2], (itr,state)
end

struct LinearGidToPart <:AbstractVector{Int}
  ngids::Int
  part_to_firstgid::Vector{Int}
end

function Base.copy(a::LinearGidToPart)
  LinearGidToPart(a.ngids,copy(a.part_to_firstgid))
end

Base.size(a::LinearGidToPart) = (a.ngids,)
Base.IndexStyle(::Type{<:LinearGidToPart}) = IndexLinear()

function Base.getindex(a::LinearGidToPart,gid::Integer)
  @boundscheck begin
    if !( 1<=gid && gid<=length(a) )
      throw(BoundsError(a,gid))
    end
  end
  searchsortedlast(a.part_to_firstgid,gid)
end

struct CartesianGidToPart{N} <: AbstractVector{Int}
  ngids::NTuple{N,Int}
  part_to_firstgid::NTuple{N,Vector{Int}}
end

function Base.copy(a::CartesianGidToPart)
  CartesianGidToPart(a.ngids,map(copy,a.part_to_firstgid))
end

Base.size(a::CartesianGidToPart) = (prod(a.ngids),)
Base.IndexStyle(::Type{<:CartesianGidToPart}) = IndexLinear()

function Base.getindex(a::CartesianGidToPart,gid::Integer)
  cgid = Tuple(CartesianIndices(a.ngids)[gid])
  cpart = map(searchsortedlast,a.part_to_firstgid,cgid)
  nparts = map(length,a.part_to_firstgid)
  part = LinearIndices(nparts)[CartesianIndex(cpart)]
  part
end

struct IndexSet <: AbstractIndexSet
  part::Int
  lid_to_gid::Vector{Int}
  lid_to_part::Vector{Int32}
  oid_to_lid::Vector{Int32}
  hid_to_lid::Vector{Int32}
  lid_to_ohid::Vector{Int32}
  gid_to_lid::Dict{Int,Int32}
end

function IndexSet(
  part::Int,
  lid_to_gid::Vector{Int},
  lid_to_part::Vector{Int32},
  oid_to_lid::Vector{Int32},
  hid_to_lid::Vector{Int32},
  lid_to_ohid::Vector{Int32})

  gid_to_lid = Dict{Int,Int32}()
  for (lid,gid) in enumerate(lid_to_gid)
    gid_to_lid[gid] = lid
  end
  IndexSet(
    part,
    lid_to_gid,
    lid_to_part,
    oid_to_lid,
    hid_to_lid,
    lid_to_ohid,
    gid_to_lid)
end

function IndexSet(
  part::Int,
  lid_to_gid::Vector{Int},
  lid_to_part::Vector{Int32},
  oid_to_lid::Vector{Int32},
  hid_to_lid::Vector{Int32})

  lid_to_ohid = zeros(Int32,length(lid_to_gid))
  lid_to_ohid[oid_to_lid] = 1:length(oid_to_lid)
  lid_to_ohid[hid_to_lid] = -(1:length(hid_to_lid))

  IndexSet(
    part,
    lid_to_gid,
    lid_to_part,
    oid_to_lid,
    hid_to_lid,
    lid_to_ohid)
end

function IndexSet(
  part::Int,
  lid_to_gid::Vector{Int},
  lid_to_part::Vector{Int32})

  oid_to_lid = collect(Int32,findall(owner->owner==part,lid_to_part))
  hid_to_lid = collect(Int32,findall(owner->owner!=part,lid_to_part))
  IndexSet(
    part,
    lid_to_gid,
    lid_to_part,
    oid_to_lid,
    hid_to_lid)
end

function Base.copy(a::IndexSet)
  IndexSet(
    copy(a.part),
    copy(a.lid_to_gid),
    copy(a.lid_to_part),
    copy(a.oid_to_lid),
    copy(a.hid_to_lid),
    copy(a.lid_to_ohid),
    copy(a.gid_to_lid))
end

struct ExtendedIndexRange <: AbstractIndexSet
  part::Int
  lid_to_gid::Vector{Int}
  lid_to_part::Vector{Int32}
  oid_to_lid::Vector{Int32}
  hid_to_lid::Vector{Int32}
  lid_to_ohid::Vector{Int32}
  gid_to_lid::GidToLid{Vector{Int32}}
end

function ExtendedIndexRange(
  part::Integer,
  lid_to_gid::Vector{Int},
  lid_to_part::Vector{Int32},
  firstgid::Integer)

  oid_to_lid = collect(Int32,findall(owner->owner==part,lid_to_part))
  hid_to_lid = collect(Int32,findall(owner->owner!=part,lid_to_part))
  noids = length(oid_to_lid)
  lid_to_ohid = zeros(Int32,length(lid_to_gid))
  lid_to_ohid[oid_to_lid] = 1:length(oid_to_lid)
  lid_to_ohid[hid_to_lid] = -(1:length(hid_to_lid))
  offset = firstgid-1
  oid_to_gid = Int(1+offset):Int(noids+offset)
  gid_to_lid_for_ghost = Dict{Int,Int32}()
  for lid in hid_to_lid
    gid_to_lid_for_ghost[lid_to_gid[lid]] = lid
  end
  gid_to_lid = GidToLid(oid_to_gid,oid_to_lid,gid_to_lid_for_ghost)
  ExtendedIndexRange(
    part,
    lid_to_gid,
    lid_to_part,
    oid_to_lid,
    hid_to_lid,
    lid_to_ohid,
    gid_to_lid)
end

function Base.copy(a::ExtendedIndexRange)
  ExtendedIndexRange(
    copy(a.part),
    copy(a.lid_to_gid),
    copy(a.lid_to_part),
    copy(a.oid_to_lid),
    copy(a.hid_to_lid),
    copy(a.lid_to_ohid),
    copy(a.gid_to_lid))
end

struct IndexRange <: AbstractIndexSet
  part::Int
  lid_to_gid::LidToGid
  lid_to_part::LidToPart
  oid_to_lid::UnitRange{Int32}
  hid_to_lid::Vector{Int32}
  lid_to_ohid::LidToOHId
  gid_to_lid::GidToLid{UnitRange{Int32}}
end

function Base.copy(a::IndexRange)
  IndexRange(
    copy(a.part),
    copy(a.lid_to_gid),
    copy(a.lid_to_part),
    copy(a.oid_to_lid),
    copy(a.hid_to_lid),
    copy(a.lid_to_ohid),
    copy(a.gid_to_lid))
end

function IndexRange(
  part::Integer,
  noids::Integer,
  firstgid::Integer)

  offset = firstgid-1
  oid_to_gid = Int(1+offset):Int(noids+offset)
  hid_to_gid = Int[]
  lid_to_gid = LidToGid(oid_to_gid,hid_to_gid)
  hid_to_part = Int32[]
  lid_to_part = LidToPart(noids,part,hid_to_part)
  oid_to_lid = Int32(1):Int32(noids)
  hid_to_lid = Int32[]
  nlids = Int(noids)
  lid_to_ohid = LidToOHId(nlids,Int32(noids))
  gid_to_lid_for_ghost = Dict{Int,Int32}()
  gid_to_lid = GidToLid(oid_to_gid,oid_to_lid,gid_to_lid_for_ghost)
  IndexRange(
    part,
    lid_to_gid,
    lid_to_part,
    oid_to_lid,
    hid_to_lid,
    lid_to_ohid,
    gid_to_lid)
end

function IndexRange(
  part::Integer,
  noids::Integer,
  firstgid::Integer,
  hid_to_gid::Vector{Int},
  hid_to_part::Vector{Int32})

  offset = firstgid-1
  oid_to_gid = Int(1+offset):Int(noids+offset)
  lid_to_gid = LidToGid(oid_to_gid,hid_to_gid)
  lid_to_part = LidToPart(noids,part,hid_to_part)
  oid_to_lid = Int32(1):Int32(noids)
  nhids = length(hid_to_part)
  hid_to_lid = collect(Int32, (1:nhids) .+ noids )
  gid_to_lid_for_ghost = Dict{Int,Int32}()
  for (hid,gid) in enumerate(hid_to_gid)
    lid = hid + noids
    gid_to_lid_for_ghost[gid] = lid
  end
  nlids = Int(noids+nhids)
  lid_to_ohid = LidToOHId(nlids,Int32(noids))
  gid_to_lid = GidToLid(oid_to_gid,oid_to_lid,gid_to_lid_for_ghost)
  IndexRange(
    part,
    lid_to_gid,
    lid_to_part,
    oid_to_lid,
    hid_to_lid,
    lid_to_ohid,
    gid_to_lid)
end

