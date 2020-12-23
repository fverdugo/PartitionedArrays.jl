
abstract type Communicator end

function num_parts(::Communicator)
  @abstractmethod
end

function num_workers(::Communicator)
  @abstractmethod
end

function do_on_parts(task::Function,::Communicator,args...)
  @abstractmethod
end

function do_on_parts(task::Function,args...)
  comm = get_comm(get_distributed_data(first(args)))
  do_on_parts(task,comm,args...)
end

# Like do_on_parts put task does not take part as its first argument
function map_on_parts(task::Function,args...)
  do_on_parts(args...) do part, x...
    task(x...)
  end
end

function i_am_master(::Communicator)
  @abstractmethod
end

# We need to compare communicators to perform some checks
function Base.:(==)(a::Communicator,b::Communicator)
  @abstractmethod
end

# All communicators that are to be executed in the master to workers
# model inherit from these one
abstract type OrchestratedCommunicator <: Communicator end

function i_am_master(::OrchestratedCommunicator)
  true
end

# This is for the communicators to be executed in MPI mode
abstract type CollaborativeCommunicator <: Communicator end

# Data distributed in parts of type T in a communicator
# Formerly, ScatteredVector
abstract type DistributedData{T} end

function get_comm(a::DistributedData)
  @abstractmethod
end

function num_parts(a)
  num_parts(get_comm(a))
end

# Construct a DistributedData object in a communicator
function DistributedData{T}(initializer::Function,::Communicator,args...) where T
  @abstractmethod
end

function DistributedData(initializer::Function,::Communicator,args...)
  @abstractmethod
end

# The comm argument can be omitted if it can be determined from the first
# data argument.
function DistributedData{T}(initializer::Function,args...) where T
  comm = get_comm(get_distributed_data(first(args)))
  DistributedData{T}(initializer,comm,args...)
end

function DistributedData(initializer::Function,args...)
  comm = get_comm(get_distributed_data(first(args)))
  DistributedData(initializer,comm,args...)
end

get_part_type(::Type{<:DistributedData{T}}) where T = T

get_part_type(::DistributedData{T}) where T = T

Base.iterate(a::DistributedData)  = @abstractmethod

Base.iterate(a::DistributedData,state)  = @abstractmethod

function gather!(a::AbstractVector,b::DistributedData)
  @abstractmethod
end

function gather(b::DistributedData{T}) where T
  if i_am_master(get_comm(b))
    a = Vector{T}(undef,num_parts(b))
  else
    a = Vector{T}(undef,0)
  end
  gather!(a,b)
  a
end

function scatter(comm::Communicator,b::AbstractVector)
  @abstractmethod
end

function bcast(comm::Communicator,v)
  if i_am_master(comm)
    part_to_v = fill(v,num_parts(comm))
  else
    T = eltype(v)
    part_to_v = T[]
  end
  scatter(comm,part_to_v)
end

# return an object for which
# its restriction to the parts of a communicator is defined.
# The returned object is not necessarily an instance
# of DistributedData
# Do nothing by default.
function get_distributed_data(object)
  object
end

get_comm(a) = get_comm(get_distributed_data(a))

# Non-blocking in-place exchange
# In this version, sending a number per part is enough
# We have another version below to send a vector of numbers per part (compressed in a Table)
# Starts a non-blocking exchange. It returns a DistributedData of Julia Tasks. Calling wait on these
# tasks will wait until the exchange is done in the corresponding part
# (i.e., at this point it is save to read/write the buffers again).
function async_exchange!(
  data_rcv::DistributedData,
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  @abstractmethod
end

# Blocking in-place exchange
function exchange!(args...;kwargs...)
  t = async_exchange!(args...;kwargs...)
  map_on_parts(wait,t)
  first(args)
end

# Blocking allocating exchange
function exchange(
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  data_rcv = DistributedData(data_snd,parts_rcv) do partid, data_snd, parts_rcv
    similar(data_snd,eltype(data_snd),length(parts_rcv))
  end

  exchange!(data_rcv,data_snd,parts_rcv,parts_snd)
  data_rcv
end

# Non-blocking in-place exchange variable length (compressed in a Table)
function async_exchange!(
  data_rcv::DistributedData{<:Table},
  data_snd::DistributedData{<:Table},
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  @abstractmethod
end

# Blocking allocating exchange variable length (compressed in a Table)
function exchange(
  data_snd::DistributedData{<:Table},
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  # Count how many we snd to each part
  n_snd = DistributedData(data_snd) do part, data_snd
    n_snd = zeros(eltype(data_snd.ptrs),length(data_snd))
    for i in 1:length(n_snd)
      n_snd[i] = data_snd.ptrs[i+1] - data_snd.ptrs[i]
    end
    n_snd
  end

  # Count how many we rcv from each part
  n_rcv = exchange(n_snd,parts_rcv,parts_snd)

  # Allocate rcv tables
  data_rcv = DistributedData(n_rcv,data_snd) do part, n_rcv, data_snd
    ptrs = similar(data_snd.ptrs,eltype(data_snd.ptrs),length(n_rcv)+1)
    for i in 1:length(n_rcv)
      ptrs[i+1] = n_rcv[i]
    end
    length_to_ptrs!(ptrs)
    ndata = ptrs[end]-1
    data = similar(data_snd.data,eltype(data_snd.data),ndata)
    Table(data,ptrs)
  end

  # Do the exchange
  exchange!(data_rcv,data_snd,parts_rcv,parts_snd)
  data_rcv
end

# Discover snd parts from rcv assuming that srd is a subset of neighbors
# Assumes that neighbors is a symmetric communication graph
function discover_parts_snd(parts_rcv::DistributedData, neighbors::DistributedData)
  @assert get_comm(parts_rcv) == get_comm(neighbors)

  # Tell the neighbors whether I want to receive data from them
  data_snd = DistributedData(neighbors,parts_rcv) do part, neighbors, parts_rcv
    dict_snd = Dict(( n=>-1 for n in neighbors))
    for i in parts_rcv
      dict_snd[i] = part
    end
    [ dict_snd[n] for n in neighbors ]
  end
  data_rcv = exchange(data_snd,neighbors,neighbors)

  # build parts_snd
  parts_snd = DistributedData(data_rcv) do part, data_rcv
    k = findall(j->j>0,data_rcv)
    data_rcv[k]
  end

  parts_snd
end

# If neighbors not provided, all procs are considered neighbors (to be improved)
function discover_parts_snd(parts_rcv::DistributedData)
  comm = get_comm(parts_rcv)
  nparts = num_parts(comm)
  neighbors = DistributedData(parts_rcv) do part, parts_rcv
    T = eltype(parts_rcv)
    [T(i) for i in 1:nparts if i!=part]
  end
  discover_parts_snd(parts_rcv,neighbors)
end

function discover_parts_snd(parts_rcv::DistributedData,::Nothing)
  discover_parts_snd(parts_rcv)
end

# TODO create abstract type IndexSet and some specializations instead of having
# so many type parameters in the current IndexSet

# Arbitrary set of global indices stored in a part
# gid_to_part can be omitted with nothing since only for some particular parallel
# data layouts (e.g. uniform partitions) it is efficient to recover this information. 
# oid: owned id
# hig: ghost (aka halo) id
# gid: global id
# lid: local id (ie union of owned + ghost)
struct IndexSet{A,B,C,D,E,F,G}
  part::Int
  ngids::Int
  lid_to_gid::A
  lid_to_part::B
  gid_to_part::C
  oid_to_lid::D
  hid_to_lid::E
  lid_to_ohid::F
  gid_to_lid::G
  function IndexSet(
    part::Integer,
    ngids::Integer,
    lid_to_gid::AbstractVector,
    lid_to_part::AbstractVector,
    gid_to_part::Union{AbstractVector,Nothing},
    oid_to_lid::Union{AbstractVector,AbstractRange},
    hid_to_lid::Union{AbstractVector,AbstractRange},
    lid_to_ohid::AbstractVector,
    gid_to_lid::AbstractDict)
    A = typeof(lid_to_gid)
    B = typeof(lid_to_part)
    C = typeof(gid_to_part)
    D = typeof(oid_to_lid)
    E = typeof(hid_to_lid)
    F = typeof(lid_to_ohid)
    G = typeof(gid_to_lid)
    new{A,B,C,D,E,F,G}(
      part,
      ngids,
      lid_to_gid,
      lid_to_part,
      gid_to_part,
      oid_to_lid,
      hid_to_lid,
      lid_to_ohid,
      gid_to_lid)
  end
end

num_gids(a::IndexSet) = a.ngids
num_lids(a::IndexSet) = length(a.lid_to_part)
num_oids(a::IndexSet) = length(a.oid_to_lid)
num_hids(a::IndexSet) = length(a.hid_to_lid)

function IndexSet(
  part::Integer,
  ngids::Integer,
  lid_to_gid::AbstractVector,
  lid_to_part::AbstractVector,
  gid_to_part::Union{AbstractVector,Nothing},
  oid_to_lid::Union{AbstractVector,AbstractRange},
  hid_to_lid::Union{AbstractVector,AbstractRange},
  lid_to_ohid::AbstractVector)

  gid_to_lid = Dict{Int,Int32}()
  for (lid,gid) in enumerate(lid_to_gid)
    gid_to_lid[gid] = lid
  end
  IndexSet(
    part,
    ngids,
    lid_to_gid,
    lid_to_part,
    gid_to_part,
    oid_to_lid,
    hid_to_lid,
    lid_to_ohid,
    gid_to_lid)
end

function IndexSet(
  part::Integer,
  ngids::Integer,
  lid_to_gid::AbstractVector,
  lid_to_part::AbstractVector,
  gid_to_part::Union{AbstractVector,Nothing},
  oid_to_lid::Union{AbstractVector,AbstractRange},
  hid_to_lid::Union{AbstractVector,AbstractRange})

  lid_to_ohid = zeros(Int32,length(lid_to_gid))
  lid_to_ohid[oid_to_lid] = 1:length(oid_to_lid)
  lid_to_ohid[hid_to_lid] = -(1:length(hid_to_lid))

  IndexSet(
    part,
    ngids,
    lid_to_gid,
    lid_to_part,
    gid_to_part,
    oid_to_lid,
    hid_to_lid,
    lid_to_ohid)
end

function IndexSet(
  part::Integer,
  ngids::Integer,
  lid_to_gid::AbstractVector,
  lid_to_part::AbstractVector,
  gid_to_part::Union{AbstractVector,Nothing}=nothing)

  oid_to_lid = findall(owner->owner==part,lid_to_part)
  hid_to_lid = findall(owner->owner!=part,lid_to_part)
  IndexSet(
    part,
    ngids,
    lid_to_gid,
    lid_to_part,
    gid_to_part,
    oid_to_lid,
    hid_to_lid)
end

function IndexSet(a::IndexSet,xid_to_lid::AbstractVector)
  xid_to_gid = a.lid_to_gid[xid_to_lid]
  xid_to_part = a.lid_to_part[xid_to_lid]
  IndexSet(a.part,a.ngids,xid_to_gid,xid_to_part,a.gid_to_part)
end

function remove_ghost(a::IndexSet)
  IndexSet(a,a.oid_to_lid)
end

# Note that this increases the underlying IndexSet to accomodate new gids
# By default reduce_op is + (make sure that you have initialized lid_to_value)
function setgid!(lid_to_value::AbstractVector,a::IndexSet,value,gid::Integer;reduce_op=+,grow=true)
  if haskey(a.gid_to_lid,gid)
    lid = a.gid_to_lid[gid]
    lid_to_value[lid] = reduce_op(lid_to_value[lid],value)
  else
    @assert grow == true
    part = a.gid_to_part[gid]
    lid = Int32(num_lids(a)+1)
    hid = Int32(num_hids(a)+1)
    push!(a.lid_to_gid,gid)
    push!(a.lid_to_part,part)
    push!(a.hid_to_lid,lid)
    push!(a.lid_to_ohid,-hid)
    a.gid_to_lid[gid] = lid
    v = reduce_op(zero(eltype(lid_to_value)),value)
    push!(lid_to_value,v)
  end
  lid_to_value
end

function setgid!(t::Tuple{AbstractVector,IndexSet},args...;kwargs...)
  lid_to_value, a = t
  setgid!(lid_to_value,a,args...;kwargs...)
end

#function IndexSet(a::IndexSet,b::IndexSet,glue::PosNegPartition)
#  lid_to_gid = lazy_map(PosNegReindex(a.lid_to_gid,b.lid_to_gid),glue)
#  lid_to_part = lazy_map(PosNegReindex(a.lid_to_part,b.lid_to_part),glue)
#  gid_to_lid = GidToLid(a.gid_to_lid,b.gid_to_lid,glue)
#  IndexSet(a.part,a.ngids,lid_to_gid,lid_to_part,gid_to_lid,a.gid_to_part)
#end

#function UniformIndexSet(ngids,np,part)
#  gids = _oid_to_gid(ngids,np,part)
#  oids = Int32(1):Int32(length(gids))
#  oid_to_gid = OidToGid(gids)
#  oid_to_part = Fill(part,length(oids))
#  gid_to_oid = GidToOid(gids,oids)
#  gid_to_part = GidToPart(ngids,np)
#  IndexSet(
#    part,
#    ngids,
#    oid_to_gid,
#    oid_to_part,
#    gid_to_oid,
#    gid_to_part)
#end


#struct OidToGid <: AbstractVector{Int}
#  gids::UnitRange{Int}
#end
#Base.size(a::OidToGid) = (length(a.gids),)
#Base.IndexStyle(::Type{<:OidToGid}) = IndexLinear()
#Base.getindex(a::OidToGid,oid::Integer) = a.gids[oid]
#
#struct GidToOid <: AbstractDict{Int,Int32}
#  gids::UnitRange{Int}
#  oids::UnitRange{Int32}
#end
#Base.length(a::GidToOid) = length(a.gids)
#Base.keys(a::GidToOid) = a.gids
#Base.haskey(a::GidToOid,gid::Int) = gid in a.gids
#Base.values(a::GidToOid) = a.oids
#function Base.getindex(a::GidToOid,gid::Int)
#  @boundscheck begin
#    if ! haskey(a,gid)
#      throw(KeyError(gid))
#    end
#  end
#  oid = Int32(gid - a.gids.start + 1)
#  oid
#end
#function Base.iterate(a::GidToOid)
#  if length(a) == 0
#    return nothing
#  end
#  state = 1
#  a.gids[state]=>a.oids[state], state
#end
#function Base.iterate(a::GidToOid,state)
#  if length(a) <= state
#    return nothing
#  end
#  s = state + 1
#  a.gids[s]=>a.oids[s], s
#end

struct GidToPart <: AbstractVector{Int}
  ngids::Int
  np::Int
end
Base.size(a::GidToPart) = (a.ngids,)
Base.IndexStyle(::Type{<:GidToPart}) = IndexLinear()
function Base.getindex(a::GidToPart,gid::Integer)
  @boundscheck begin
    if !( 1<=gid && gid<=length(a) )
      throw(BoundsError(a,gid))
    end
  end
  _gid_to_part(a.ngids,a.np,gid)
end

function _oid_to_gid(ngids,np,p)
  _olength = ngids ÷ np
  _offset = _olength * (p-1)
  _rem = ngids % np
  if _rem < (np-p+1)
    olength = _olength
    offset = _offset
  else
    olength = _olength + 1
    offset = _offset + p - (np-_rem) - 1
  end
  Int(1+offset):Int(olength+offset)
end

function _gid_to_part(ngids,np,gid)
  @check 1<=gid && gid<=ngids "gid=$gid is not in [1,$ngids]"
  # TODO this can be heavily optimized
  for p in 1:np
    if _is_gid_in_part(ngids,np,p,gid)
      return p
    end
  end
end

function _is_gid_in_part(ngids,np,p,gid)
  gids = _oid_to_gid(ngids,np,p)
  gid >= gids.start && gid <= gids.stop
end

#struct GidToLid{A,B,C} <: AbstractDict{Int,Int32}
#  gid_to_oid::A
#  gid_to_hid::B
#  glue::C
#  function GidToLid(
#    gid_to_oid::AbstractDict,
#    gid_to_hid::AbstractDict,
#    glue::PosNegPartition)
#
#    A = typeof(gid_to_oid)
#    B = typeof(gid_to_hid)
#    C = typeof(glue)
#    new{A,B,C}(
#      gid_to_oid,
#      gid_to_hid,
#      glue)
#  end
#end
#Base.length(a::GidToLid) = length(a.gid_to_oid) + length(a.gid_to_hid)
#Base.haskey(a::GidToLid,gid::Int) = haskey(a.gid_to_oid,gid) || haskey(a.gid_to_hid,gid)
#Base.keys(a::GidToLid) = lazy_map(PosNegReindex(keys(a.gid_to_oid),keys(a.gid_to_hid)),a.glue)
#Base.values(a::GidToLid) = lazy_map(PosNegReindex(values(a.gid_to_oid),values(a.gid_to_hid)),a.glue)
#function Base.getindex(a::GidToLid,gid::Int)
#  @boundscheck begin
#    if ! haskey(a,gid)
#      throw(KeyError(gid))
#    end
#  end
#  if haskey(a.gid_to_oid,gid)
#    oid = a.gid_to_oid[gid]
#    oid_to_lid = a.glue.ipos_to_i
#    lid = oid_to_lid[oid]
#  else
#    hid = a.gid_to_hid[gid]
#    hid_to_lid = a.glue.ineg_to_i
#    lid = hid_to_lid[hid]
#  end
#  lid
#end
#function Base.iterate(a::GidToLid)
#  if length(a) == 0
#    return nothing
#  end
#  state = 1
#  k = keys(a)
#  v = values(a)
#  k[state]=>v[state], (state,k,v)
#end
#function Base.iterate(a::GidToLid,(state,k,v))
#  if length(a) <= state
#    return nothing
#  end
#  s = state + 1
#  k[s]=>v[s], (s,k,v)
#end

struct Exchanger{B,C}
  parts_rcv::B
  parts_snd::B
  lids_rcv::C
  lids_snd::C
  function Exchanger(
    parts_rcv::DistributedData{<:AbstractVector{<:Integer}},
    parts_snd::DistributedData{<:AbstractVector{<:Integer}},
    lids_rcv::DistributedData{<:Table{<:Integer}},
    lids_snd::DistributedData{<:Table{<:Integer}})

    B = typeof(parts_rcv)
    C = typeof(lids_rcv)
    new{B,C}(parts_rcv,parts_snd,lids_rcv,lids_snd)
  end
end

function Exchanger(ids::DistributedData{<:IndexSet},neighbors=nothing)

  parts_rcv = DistributedData(ids) do part, ids
    parts_rcv = Dict((owner=>true for owner in ids.lid_to_part if owner!=part))
    sort(collect(keys(parts_rcv)))
  end

  lids_rcv, gids_rcv = DistributedData(ids,parts_rcv) do part, ids, parts_rcv

    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_rcv) ))

    ptrs = zeros(Int32,length(parts_rcv)+1)
    for owner in ids.lid_to_part
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)

    data_lids = zeros(Int32,ptrs[end]-1)
    data_gids = zeros(Int,ptrs[end]-1)

    for (lid,owner) in enumerate(ids.lid_to_part)
      if owner != part
        p = ptrs[owner_to_i[owner]]
        data_lids[p]=lid
        data_gids[p]=ids.lid_to_gid[lid]
        ptrs[owner_to_i[owner]] += 1
      end
    end
    rewind_ptrs!(ptrs)

    lids_rcv = Table(data_lids,ptrs)
    gids_rcv = Table(data_gids,ptrs)

    lids_rcv, gids_rcv
  end

  parts_snd = discover_parts_snd(parts_rcv,neighbors)

  gids_snd = exchange(gids_rcv,parts_snd,parts_rcv)

  lids_snd = DistributedData(ids, gids_snd) do part, ids, gids_snd
    ptrs = gids_snd.ptrs
    data_lids = zeros(Int32,ptrs[end]-1)
    for (k,gid) in enumerate(gids_snd.data)
      data_lids[k] = ids.gid_to_lid[gid]
    end
    lids_snd = Table(data_lids,ptrs)
    lids_snd
  end

  Exchanger(parts_rcv,parts_snd,lids_rcv,lids_snd)
end

function Base.reverse(a::Exchanger)
  Exchanger(a.parts_snd,a.parts_rcv,a.lids_snd,a.lids_rcv)
end

function allocate_rcv_buffer(::Type{T},a::Exchanger) where T
  data_rcv = DistributedData(a.lids_rcv) do part, lids_rcv
    ptrs = lids_rcv.ptrs
    data = zeros(T,ptrs[end]-1)
    Table(data,ptrs)
  end
  data_rcv
end

function allocate_snd_buffer(::Type{T},a::Exchanger) where T
  data_snd = DistributedData(a.lids_snd) do part, lids_snd
    ptrs = lids_snd.ptrs
    data = zeros(T,ptrs[end]-1)
    Table(data,ptrs)
  end
  data_snd
end

function async_exchange!(
  values::DistributedData{<:AbstractVector{T}},
  exchanger::Exchanger;
  reduce_op=_replace) where T

  # Allocate buffers
  data_rcv = allocate_rcv_buffer(T,exchanger)
  data_snd = allocate_snd_buffer(T,exchanger)

  # Fill snd buffer
  do_on_parts(values,data_snd,exchanger.lids_snd) do part,values,data_snd,lids_snd 
    for p in 1:length(lids_snd.data)
      lid = lids_snd.data[p]
      data_snd.data[p] = values[lid]
    end
  end

  # communicate
  task = async_exchange!(
    data_rcv,
    data_snd,
    exchanger.parts_rcv,
    exchanger.parts_snd)

  # Fill values from rcv buffer
  # asynchronously
  return DistributedData(task,values,data_rcv,exchanger.lids_rcv) do part,task,values,data_rcv,lids_rcv 
    @async begin
      wait(task)
      for p in 1:length(lids_rcv.data)
        lid = lids_rcv.data[p]
        values[lid] = reduce_op(values[lid],data_rcv.data[p])
      end
    end
  end

end

_replace(x,y) = y

struct DistributedRange{A,B} <: AbstractUnitRange{Int}
  ngids::Int
  lids::A
  exchanger::B
  function DistributedRange(
    ngids::Integer,
    lids::DistributedData{<:IndexSet},
    exchanger::Exchanger=Exchanger(lids))
  
    A = typeof(lids)
    B = typeof(exchanger)
    new{A,B}(
      ngids,
      lids,
      exchanger)
  end
end

function DistributedRange(comm::Communicator,ngids::Integer)
  np = num_parts(comm)
  lids = DistributedData(comm) do part
    gids = _oid_to_gid(ngids,np,part)
    lid_to_gid = collect(gids)
    lid_to_part = fill(part,length(gids))
    gid_to_part = GidToPart(ngids,np)
    oid_to_lid = Int32(1):Int32(length(gids))
    hid_to_lid = collect(Int32(1):Int32(0))
    IndexSet(
      part,
      ngids,
      lid_to_gid,
      lid_to_part,
      gid_to_part,
      oid_to_lid,
      hid_to_lid)
  end
  DistributedRange(ngids,lids)
end

Base.first(a::DistributedRange) = 1
Base.last(a::DistributedRange) = a.ngids

get_distributed_data(a::DistributedRange) = a.lids
num_gids(a::DistributedRange) = a.ngids

#function async_exchange!(
#  values::DistributedData{<:AbstractVector},
#  r::DistributedRange)
#  async_exchange!(values,r.exchanger)
#end

function remove_ghost(a::DistributedRange)
  oids = DistributedData(a.lids) do p,lids
    remove_ghost(lids)
  end
  # TODO build an empty exchanger
  DistributedRange(a.ngids,oids)
end


#struct DistributedIndexSet{A,B}
#  ngids::Int
#  ids::A
#  exchanger::B
#  function DistributedIndexSet(
#    ngids::Integer,
#    ids::DistributedData{<:IndexSet},
#    exchanger::Exchanger=Exchanger(ids))
#
#    A = typeof(ids)
#    B = typeof(exchanger)
#    new{A,B}(ngids,ids,exchanger)
#  end
#end
#
#get_distributed_data(a::DistributedIndexSet) = a.ids
#num_gids(a::DistributedIndexSet) = a.ngids
#
#function async_exchange!(
#  values::DistributedData{<:AbstractVector},
#  ids::DistributedIndexSet)
#  async_exchange!(values,ids.exchanger)
#end

## Numeric data build on top of a IndexLayout
## i.e. the data stored locally in a distributed vector
## the three stored vectors have to be views to the same data
#struct ValueLayout{T,A,B,C}
#  lid_to_value::A
#  oid_to_value::B
#  hid_to_value::C
#  function ValueLayout(
#    lid_to_value::AbstractVector{T},
#    oid_to_value::AbstractVector{T},
#    hid_to_value::AbstractVector{T}) where T
#    A = typeof(lid_to_value)
#    B = typeof(oid_to_value)
#    C = typeof(hid_to_value)
#    new{T,A,B,C}(
#      lid_to_value,
#      oid_to_value,
#      hid_to_value)
#  end
#end
#
#function ValueLayout(lid_to_value::AbstractVector{T}, glue::PosNegPartition) where T
#  oid_to_lid = glue.ipos_to_i
#  hid_to_lid = glue.ineg_to_i
#  oid_to_value = view(lid_to_value,oid_to_lid)
#  hid_to_value = view(lid_to_value,hid_to_lid)
#  ValueLayout(lid_to_value,oid_to_value,hid_to_value)
#end
#
## IndexSet + some metadata about owned and ghost ids,
## This is the symbolic data fully describing a distributed vector
## oid: owned id
## hig: ghost (aka halo) id
## gid: global id
## lid: local id (ie union of owned + ghost)
#struct IndexLayout{A,B,C,D}
#  lids::A
#  oids::B
#  hids::C
#  glue::D
#  function IndexLayout(
#    lids::IndexSet,
#    oids::IndexSet,
#    hids::IndexSet,
#    glue::PosNegPartition)
#    A = typeof(lids)
#    B = typeof(oids)
#    C = typeof(hids)
#    D = typeof(glue)
#    new{A,B,C,D}(
#      lids,
#      oids,
#      hids,
#      glue)
#  end
#end
#
#function IndexLayout(lids::IndexSet,glue::PosNegPartition)
#  oid_to_lid = glue.ipos_to_i
#  hid_to_lid = glue.ineg_to_i
#  oids = IndexSet(lids,oid_to_lid)
#  hids = IndexSet(lids,hid_to_lid)
#  IndexLayout(lids,oids,hids,glue)
#end
#
#function IndexLayout(lids::IndexSet)
#  part = lids.part
#  oid_to_lid = findall(owner->owner==part,lids.lid_to_part)
#  glue = PosNegPartition(oid_to_lid,num_lids(lids))
#  IndexLayout(lids,glue)
#end
#
#function IndexLayout(oids::IndexSet,hids::IndexSet,glue::PosNegPartition)
#  lids = IndexSet(oids,hids,glue)
#  IndexLayout(lids,oids,hids,glue)
#end
#
#function Base.getproperty(x::IndexLayout, sym::Symbol)
#  if sym == :part
#    x.oids.part
#  elseif sym == :ngids
#    x.oids.ngids
#  elseif sym == :oid_to_gid
#    x.oids.lid_to_gid
#  elseif sym == :oid_to_part
#    x.oids.lid_to_part
#  elseif sym == :gid_to_oid
#    x.oids.gid_to_lid
#  elseif sym == :gid_to_part
#    x.oids.gid_to_part
#  elseif sym == :lid_to_gid
#    x.lids.lid_to_gid
#  elseif sym == :lid_to_part
#    x.lids.lid_to_part
#  elseif sym == :gid_to_lid
#    x.lids.gid_to_lid
#  elseif sym == :hid_to_gid
#    x.hids.lid_to_gid
#  elseif sym == :hid_to_part
#    x.hids.lid_to_part
#  elseif sym == :gid_to_hid
#    x.hids.gid_to_lid
#  elseif sym == :lid_to_ohid
#    x.glue.i_to_iposneg
#  elseif sym == :oid_to_lid
#    x.glue.ipos_to_i
#  elseif sym == :hid_to_lid
#    x.glue.ineg_to_i
#  else
#    getfield(x, sym)
#  end
#end
#
#function Base.propertynames(x::IndexLayout, private=false)
#  (
#   fieldnames(typeof(x))...,
#  :part,
#  :ngids,
#  :oid_to_gid,
#  :oid_to_part,
#  :gid_to_oid,
#  :gid_to_part,
#  :lid_to_gid,
#  :lid_to_part,
#  :gid_to_lid,
#  :hid_to_gid,
#  :hid_to_part,
#  :gid_to_hid,
#  :lid_to_ohid,
#  :oid_to_lid,
#  :hid_to_lid)
#end
#
#num_gids(a::IndexLayout) = a.ngids
#num_lids(a::IndexLayout) = length(a.lid_to_part)
#
#function Exchanger(ids::DistributedData{<:IndexLayout},neighbors=nothing)
#  hids = DistributedData(ids) do part, ids
#    ids.hids
#  end
#  ghost_exchanger = Exchanger(hids,neighbors)
#  ghost_exchanger
#end
#
#function async_exchange!(
#  values::DistributedData{<:ValueLayout},
#  ghost_exchanger::Exchanger)
#
#  ghost_values = DistributedData(values) do part, values
#    values.hid_to_value
#  end
#  async_exchange!(ghost_values,ghost_exchanger)
#end
#
#struct DistributedIndexLayout{A,B}
#  ngids::Int
#  ids::A
#  exchanger::B
#  function DistributedIndexLayout(
#    ngids::Integer,
#    ids::DistributedData{<:IndexLayout},
#    exchanger::Exchanger=Exchanger(ids))
#
#    A = typeof(ids)
#    B = typeof(exchanger)
#    new{A,B}(ngids,ids,exchanger)
#  end
#end
#
#function DistributedIndexLayout(i::DistributedIndexSet)
#  ids = DistributedData(i) do part, i
#    IndexLayout(i)
#  end
#end
#
#get_distributed_data(a::DistributedIndexLayout) = a.ids
#num_gids(a::DistributedIndexLayout) = a.ngids
#
#function async_exchange!(
#  values::DistributedData{<:ValueLayout},
#  ids::DistributedIndexLayout)
#  async_exchange!(values,ids.exchanger)
#end

struct DistributedVector{T,A,B} <: AbstractVector{T}
  values::A
  ids::B
  function DistributedVector(
    values::DistributedData{<:AbstractVector{T}},
    ids::DistributedRange) where T

    A = typeof(values)
    B = typeof(ids)
    new{T,A,B}(values,ids)
  end
end

function DistributedVector{T}(
  ::UndefInitializer,
  ids::DistributedRange) where T

  values = DistributedData(ids) do part, ids
    nlids = num_lids(ids)
    Vector{T}(undef,nlids)
  end
  DistributedVector(values,ids)
end

struct DistributedVectorSeed{T,A,B}
  ngids::Int
  values::A
  ids::B
  function DistributedVectorSeed(
    ngids::Integer,
    values::DistributedData{<:AbstractVector{T}},
    ids::DistributedData{<:IndexSet}) where T

    A = typeof(values)
    B = typeof(ids)
    new{T,A,B}(ngids,values,ids)
  end
end

function DistributedVectorSeed{T}(comm::Communicator,ngids::Integer) where T
  # TODO build ids without initializing an Exchanger
  ids = DistributedRange(comm,ngids).lids
  values = DistributedData(ids) do part, ids
    zeros(T,num_lids(ids))
  end
  DistributedVectorSeed(ngids,values,ids)
end

function get_distributed_data(a::DistributedVectorSeed)
  DistributedData(a.values,a.ids) do part, values, ids
    (values,ids)
  end
end

function DistributedVector(v::DistributedVectorSeed;assemble::Bool=true)
  ids = DistributedRange(v.ngids,v.ids)
  u = DistributedVector(v.values,ids)
  if assemble
    assemble!(u)
  end
  u
end

function Base.fill!(a::DistributedVector,v)
  do_on_parts(a.values) do part, lid_to_value
    fill!(lid_to_value,v)
  end
  a
end

Base.length(a::DistributedVector) = length(a.ids)

# TODO a better name?
function async_exchange!(a::DistributedVector)
  async_exchange!(a.values,a.ids.exchanger)
end

#TODO async_assemble!
# To implement this we would need to input a task in async_exchange!
function assemble!(a::DistributedVector;reduce_op=+)
  exchanger_rcv = a.ids.exchanger # receives data at ghost ids from remote parts
  exchanger_snd = reverse(exchanger_rcv) # sends data at ghost ids to remote parts
  exchange!(a.values,exchanger_snd;reduce_op=reduce_op)
  exchange!(a.values,exchanger_rcv)
  a
end

struct DistributedSparseMatrix{T,A,B,C} <: AbstractMatrix{T}
  owned_values::A
  ghost_values::A
  row_ids::B
  col_ids::C
  function DistributedSparseMatrix(
    owned_values::DistributedData{<:AbstractSparseMatrix{T}},
    ghost_values::DistributedData{<:AbstractSparseMatrix{T}},
    row_ids::DistributedRange,
    col_ids::DistributedRange) where T

    A = typeof(owned_values)
    B = typeof(row_ids)
    C = typeof(col_ids)
    new{T,A,B,C}(owned_values,ghost_values,row_ids,col_ids)
  end
end

Base.size(a::DistributedSparseMatrix) = (num_gids(a.row_ids),num_gids(a.col_ids))

function LinearAlgebra.mul!(
  c::DistributedVector,
  a::DistributedSparseMatrix,
  b::DistributedVector,
  α::Number,
  β::Number)

  @assert c.ids === a.row_ids
  @assert b.ids === a.col_ids
  t = async_exchange!(b)
  do_on_parts(c.values,a.owned_values,a.ghost_values,a.row_ids,a.col_ids,b.values,t) do part,c,ao,ah,rlids,clids,b,t
    scale_entries!(c,β)
    co = view(c,rlids.oid_to_lid)
    bo = view(b,clids.oid_to_lid)
    mul!(co,ao,bo,α,1)
    wait(t)
    bh = view(b,clids.hid_to_lid)
    mul!(co,ah,bh,α,1)
  end
  c
end
#
#function Base.getindex(
#  a::DistributedSparseMatrix,
#  row_ids::DistributedIndexSet,
#  col_ids::DistributedIndexSet)
#
#  @notimplementedif a.row_ids !== row_ids
#  if a.col_ids === col_ids
#    a
#  else
#    ids_out = col_ids
#    ids_in = a.col_ids
#    values_in = a.values
#    values_out = DistributedData(values_in,ids_in,ids_out) do part, values_in, ids_in, ids_out
#      i_to_lid_in = Int32[]
#      i_to_lid_out = Int32[]
#      for lid_in in 1:num_lids(ids_in)
#         gid = ids_in.lid_to_gid[lid_in]
#         if haskey(ids_out.gid_to_lid,gid)
#           lid_out = ids_out.gid_to_lid[gid]
#           push!(i_to_lid_in,lid_in)
#           push!(i_to_lid_out,lid_out)
#         end
#      end
#      I,J_in,V = findnz(values_in[:,i_to_lid_in])
#      J_out = similar(J_in)
#      J_out .= i_to_lid_out[J_in]
#      sparse(I,J_out,V,size(values_in,1),num_lids(ids_out))
#    end
#    DistributedSparseMatrix(values_out,row_ids,col_ids)
#  end
#end
#
struct AdditiveSchwarz{A,B,C,D}
  problems::A
  solvers::B
  row_ids::C
  col_ids::D
end

function AdditiveSchwarz(a::DistributedSparseMatrix)
  problems = a.owned_values
  solvers = DistributedData(problems) do part, problem
    return \
  end
  AdditiveSchwarz(problems,solvers,a.col_ids,a.row_ids)
end

function LinearAlgebra.mul!(c::DistributedVector,a::AdditiveSchwarz,b::DistributedVector)
  @assert c.ids === a.row_ids
  @assert b.ids === a.col_ids
  do_on_parts(c.values,a.problems,a.solvers,b.values,a.row_ids,a.col_ids) do part,c,p,s,b,row_ids,col_ids
    # TODO not all solvers would accept a view
    # but if oids before hids one can do a reinterpretation of memory instead of a view.
    bo = view(b,col_ids.oid_to_lid)
    c[row_ids.oid_to_lid] = s(p,bo)
  end
  c
end




