
abstract type Backend end

# Should return a DistributedData{Int}
function get_part_ids(b::Backend,nparts::Integer)
  @abstractmethod
end

function get_part_ids(b::Backend,nparts::Tuple)
  get_part_ids(b,prod(nparts))
end

# This can be overwritten to add a finally clause
function distributed_run(driver::Function,b::Backend,nparts)
  part = get_part_ids(b,nparts)
  driver(part)
end

# Data distributed in parts of type T
abstract type DistributedData{T,N} end

Base.size(a::DistributedData) = @abstractmethod

Base.length(a::DistributedData) = prod(size(a))

num_parts(a::DistributedData) = length(a)

get_backend(a::DistributedData) = @abstractmethod

Base.iterate(a::DistributedData)  = @abstractmethod

Base.iterate(a::DistributedData,state)  = @abstractmethod

get_part_ids(a::DistributedData) = get_part_ids(get_backend(a),size(a))

map_parts(task::Function,a::DistributedData...) = @abstractmethod

i_am_master(::DistributedData) = @abstractmethod

Base.eltype(a::DistributedData{T}) where T = T
Base.eltype(::Type{<:DistributedData{T}}) where T = T

Base.ndims(a::DistributedData{T,N}) where {T,N} = N
Base.ndims(::Type{<:DistributedData{T,N}}) where {T,N} = N

#function map_parts(task::Function,a...)
#  map_parts(task,map(DistributedData,a)...)
#end
#
#DistributedData(a::DistributedData) = a

const MASTER = 1

# import the master part to the main scope
# in MPI this will broadcast the master part to all procs
get_master_part(a::DistributedData) = get_part(a,MASTER)

# This one is safe to use only when all parts contain the same value, e.g. the result of a gather_all call.
get_part(a::DistributedData) = @abstractmethod

get_part(a::DistributedData,part::Integer) = @abstractmethod

gather!(rcv::DistributedData,snd::DistributedData) = @abstractmethod

gather_all!(rcv::DistributedData,snd::DistributedData) = @abstractmethod

function gather(snd::DistributedData)
  np = num_parts(snd)
  parts = get_part_ids(snd)
  rcv = map_parts(parts,snd) do part, snd
    T = typeof(snd)
    if part == MASTER
      Vector{T}(undef,np)
    else
      Vector{T}(undef,0)
    end
  end
  gather!(rcv,snd)
  rcv
end

function gather_all(snd::DistributedData)
  np = num_parts(snd)
  rcv = map_parts(snd) do snd
    T = typeof(snd)
    Vector{T}(undef,np)
  end
  gather_all!(rcv,snd)
  rcv
end

function scatter(snd::DistributedData)
  @abstractmethod
end

function bcast(snd::DistributedData)
  np = num_parts(snd)
  parts = get_part_ids(snd)
  snd2 = map_parts(parts,snd) do part, snd
    T = typeof(snd)
    if part == MASTER
      v = Vector{T}(undef,np)
      fill!(v,snd)
    else
      v = Vector{T}(undef,0)
    end
    v
  end
  scatter(snd2)
end

function reduce_master(op,snd::DistributedData;init)
  a = gather(snd)
  map_parts(i->reduce(op,i;init=init),a)
end

function reduce_all(args...;kwargs...)
  b = reduce_master(args...;kwargs...)
  bcast(b)
end

function Base.reduce(op,a::DistributedData;init)
  b = reduce_master(op,a;init=init)
  get_master_part(b)
end

function Base.sum(a::DistributedData)
  reduce(+,a,init=zero(eltype(a)))
end

# Non-blocking in-place exchange
# In this version, sending a number per part is enough
# We have another version below to send a vector of numbers per part (compressed in a Table)
# Starts a non-blocking exchange. It returns a DistributedData of Julia Tasks. Calling schedule and wait on these
# tasks will wait until the exchange is done in the corresponding part
# (i.e., at this point it is save to read/write the buffers again).
function async_exchange!(
  data_rcv::DistributedData,
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData,
  t_in::DistributedData)

  @abstractmethod
end

function async_exchange!(
  data_rcv::DistributedData,
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  t_in = _empty_tasks(parts_rcv)
  async_exchange!(data_rcv,data_snd,parts_rcv,parts_snd,t_in)
end

function _empty_tasks(a)
  map_parts(a) do a
    @task nothing
  end
end

# Non-blocking allocating exchange
# the returned data_rcv cannot be consumed in a part until the corresponding task in t is done.
function async_exchange(
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData,
  t_in::DistributedData=_empty_tasks(parts_rcv))

  data_rcv = map_parts(data_snd,parts_rcv) do data_snd, parts_rcv
    similar(data_snd,eltype(data_snd),length(parts_rcv))
  end

  t_out = async_exchange!(data_rcv,data_snd,parts_rcv,parts_snd,t_in)

  data_rcv, t_out
end

# Non-blocking in-place exchange variable length (compressed in a Table)
function async_exchange!(
  data_rcv::DistributedData{<:Table},
  data_snd::DistributedData{<:Table},
  parts_rcv::DistributedData,
  parts_snd::DistributedData,
  t_in::DistributedData)

  @abstractmethod
end

# Non-blocking allocating exchange variable length (compressed in a Table)
function async_exchange(
  data_snd::DistributedData{<:Table},
  parts_rcv::DistributedData,
  parts_snd::DistributedData,
  t_in::DistributedData)

  # Allocate empty data
  data_rcv = map_parts(empty_table,data_snd)
  n_snd = map_parts(parts_snd) do parts_snd
    Int[]
  end

  # wait data_snd to be in a correct state and
  # Count how many we snd to each part
  t1 = map_parts(n_snd,data_snd,t_in) do n_snd,data_snd,t_in
    @task begin
      wait(schedule(t_in))
      resize!(n_snd,length(data_snd))
      for i in 1:length(n_snd)
        n_snd[i] = data_snd.ptrs[i+1] - data_snd.ptrs[i]
      end
    end
  end

  # Count how many we rcv from each part
  n_rcv, t2 = async_exchange(n_snd,parts_rcv,parts_snd,t1)

  # Wait n_rcv to be in a correct state and
  # resize data_rcv to the correct size
  t3 = map_parts(n_rcv,t2,data_rcv) do n_rcv,t2,data_rcv
    @task begin
      wait(schedule(t2))
      resize!(data_rcv.ptrs,length(n_rcv)+1)
      for i in 1:length(n_rcv)
        data_rcv.ptrs[i+1] = n_rcv[i]
      end
      length_to_ptrs!(data_rcv.ptrs)
      ndata = data_rcv.ptrs[end]-1
      resize!(data_rcv.data,ndata)
    end
  end

  # Do the actual exchange
  t4 = async_exchange!(data_rcv,data_snd,parts_rcv,parts_snd,t3)

  data_rcv, t4
end

# Blocking in-place exchange
function exchange!(args...;kwargs...)
  t = async_exchange!(args...;kwargs...)
  map_parts(schedule,t)
  map_parts(wait,t)
  first(args)
end

# Blocking allocating exchange
function exchange(args...;kwargs...)
  data_rcv, t = async_exchange(args...;kwargs...)
  map_parts(schedule,t)
  map_parts(wait,t)
  data_rcv
end

# Discover snd parts from rcv assuming that srd is a subset of neighbors
# Assumes that neighbors is a symmetric communication graph
function discover_parts_snd(parts_rcv::DistributedData, neighbors::DistributedData)
  @assert num_parts(parts_rcv) == num_parts(neighbors)

  parts = get_part_ids(parts_rcv)

  # Tell the neighbors whether I want to receive data from them
  data_snd = map_parts(parts,neighbors,parts_rcv) do part, neighbors, parts_rcv
    dict_snd = Dict(( n=>-1 for n in neighbors))
    for i in parts_rcv
      dict_snd[i] = part
    end
    [ dict_snd[n] for n in neighbors ]
  end
  data_rcv = exchange(data_snd,neighbors,neighbors)

  # build parts_snd
  parts_snd = map_parts(data_rcv) do data_rcv
    k = findall(j->j>0,data_rcv)
    data_rcv[k]
  end

  parts_snd
end

# If neighbors not provided, all procs are considered neighbors (to be improved)
function discover_parts_snd(parts_rcv::DistributedData)
  parts = get_part_ids(parts_rcv)
  nparts = num_parts(parts)
  neighbors = map_parts(parts,parts_rcv) do part, parts_rcv
    T = eltype(parts_rcv)
    [T(i) for i in 1:nparts if i!=part]
  end
  discover_parts_snd(parts_rcv,neighbors)
end

function discover_parts_snd(parts_rcv::DistributedData,::Nothing)
  discover_parts_snd(parts_rcv)
end

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

function add_gid!(a::IndexSet,gid::Integer)
  if !haskey(a.gid_to_lid,gid)
    part = a.gid_to_part[gid]
    lid = Int32(num_lids(a)+1)
    hid = Int32(num_hids(a)+1)
    push!(a.lid_to_gid,gid)
    push!(a.lid_to_part,part)
    push!(a.hid_to_lid,lid)
    push!(a.lid_to_ohid,-hid)
    a.gid_to_lid[gid] = lid
  end
  a
end

function to_lid!(ids::AbstractArray{<:Integer},a::IndexSet)
  for i in eachindex(ids)
    gid = ids[i]
    lid = a.gid_to_lid[gid]
    ids[i] = lid
  end
  ids
end

function to_gid!(ids::AbstractArray{<:Integer},a::IndexSet)
  for i in eachindex(ids)
    lid = ids[i]
    gid = a.lid_to_gid[lid]
    ids[i] = gid
  end
  ids
end

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

  parts = get_part_ids(ids)

  parts_rcv = map_parts(parts,ids) do part, ids
    parts_rcv = Dict((owner=>true for owner in ids.lid_to_part if owner!=part))
    sort(collect(keys(parts_rcv)))
  end

  lids_rcv, gids_rcv = map_parts(parts,ids,parts_rcv) do part,ids,parts_rcv

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

  lids_snd = map_parts(ids,gids_snd) do ids,gids_snd
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
  data_rcv = map_parts(a.lids_rcv) do lids_rcv
    ptrs = lids_rcv.ptrs
    data = zeros(T,ptrs[end]-1)
    Table(data,ptrs)
  end
  data_rcv
end

function allocate_snd_buffer(::Type{T},a::Exchanger) where T
  data_snd = map_parts(a.lids_snd) do lids_snd
    ptrs = lids_snd.ptrs
    data = zeros(T,ptrs[end]-1)
    Table(data,ptrs)
  end
  data_snd
end

function async_exchange!(
  values::DistributedData{<:AbstractVector{T}},
  exchanger::Exchanger,
  t0::DistributedData=_empty_tasks(exchanger.parts_rcv);
  reduce_op=_replace) where T

  # Allocate buffers
  data_rcv = allocate_rcv_buffer(T,exchanger)
  data_snd = allocate_snd_buffer(T,exchanger)

  # Fill snd buffer
  t1 = map_parts(t0,values,data_snd,exchanger.lids_snd) do t0,values,data_snd,lids_snd 
    @task begin
      wait(schedule(t0))
      for p in 1:length(lids_snd.data)
        lid = lids_snd.data[p]
        data_snd.data[p] = values[lid]
      end
    end
  end

  # communicate
  t2 = async_exchange!(
    data_rcv,
    data_snd,
    exchanger.parts_rcv,
    exchanger.parts_snd,
    t1)

  # Fill values from rcv buffer
  # asynchronously
  t3 = map_parts(t2,values,data_rcv,exchanger.lids_rcv) do t2,values,data_rcv,lids_rcv 
    @task begin
      wait(schedule(t2))
      for p in 1:length(lids_rcv.data)
        lid = lids_rcv.data[p]
        values[lid] = reduce_op(values[lid],data_rcv.data[p])
      end
    end
  end

  t3
end

_replace(x,y) = y

# TODO mutable is needed to correctly implement add_gid!
mutable struct DistributedRange{A,B} <: AbstractUnitRange{Int}
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

Base.first(a::DistributedRange) = 1
Base.last(a::DistributedRange) = a.ngids

num_gids(a::DistributedRange) = a.ngids
num_parts(a::DistributedRange) = num_parts(a.lids)

function DistributedRange(parts::DistributedData{<:Integer},ngids::Integer)
  np = num_parts(parts)
  lids = map_parts(parts) do part
    gids = _oid_to_gid(ngids,np,part)
    lid_to_gid = collect(gids)
    lid_to_part = fill(part,length(gids))
    part_to_gid = _part_to_gid(ngids,np)
    gid_to_part = GidToPart(ngids,part_to_gid)
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

function DistributedRange(
  parts::DistributedData{<:Integer},
  ngids::NTuple{N,<:Integer}) where N

  np = size(parts)
  lids = map_parts(parts) do part
    gids = _oid_to_gid(ngids,np,part)
    lid_to_gid = gids
    lid_to_part = fill(part,length(gids))
    part_to_gid = _part_to_gid(ngids,np)
    gid_to_part = GidToPart(ngids,part_to_gid)
    oid_to_lid = Int32(1):Int32(length(gids))
    hid_to_lid = collect(Int32(1):Int32(0))
    IndexSet(
      part,
      prod(ngids),
      lid_to_gid,
      lid_to_part,
      gid_to_part,
      oid_to_lid,
      hid_to_lid)
  end
  DistributedRange(prod(ngids),lids)
end

function _oid_to_gid(ngids::Integer,np::Integer,p::Integer)
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

function _oid_to_gid(ngids::Tuple,np::Tuple,p::Integer)
  cp = Tuple(CartesianIndices(np)[p])
  _oid_to_gid(ngids,np,cp)
end

function _oid_to_gid(ngids::Tuple,np::Tuple,p::Tuple)
  D = length(np)
  @assert length(ngids) == D
  d_to_odid_to_gdid = map(_oid_to_gid,ngids,np,p)
  _id_tensor_product(d_to_odid_to_gdid,ngids)
end

function _id_tensor_product(d_to_dlid_to_gdid::Tuple,d_to_ngdids::Tuple)
  d_to_nldids = map(length,d_to_dlid_to_gdid)
  lcis = CartesianIndices(d_to_nldids)
  llis = LinearIndices(d_to_nldids)
  glis = LinearIndices(d_to_ngdids)
  D = length(d_to_ngdids)
  gci = zeros(Int,D)
  lid_to_gid = zeros(Int,length(lcis))
  for lci in lcis
    for d in 1:D
      ldid = lci[d]
      gdid = d_to_dlid_to_gdid[d][ldid]
      gci[d] = gdid
    end
    lid = llis[lci]
    lid_to_gid[lid] = glis[CartesianIndex(Tuple(gci))]
  end
  lid_to_gid
end

function _part_to_gid(ngids::Integer,np::Integer)
  [first(_oid_to_gid(ngids,np,p)) for p in 1:np]
end

function _part_to_gid(ngids::Tuple,np::Tuple)
  map(_part_to_gid,ngids,np)
end

struct GidToPart{A,B} <: AbstractVector{Int}
  ngids::A
  part_to_gid::B
end

Base.size(a::GidToPart) = (prod(a.ngids),)
Base.IndexStyle(::Type{<:GidToPart}) = IndexLinear()

function Base.getindex(a::GidToPart,gid::Integer)
  @boundscheck begin
    if !( 1<=gid && gid<=length(a) )
      throw(BoundsError(a,gid))
    end
  end
  _find_part_from_gid(a.ngids,a.part_to_gid,gid)
end

function _find_part_from_gid(ngids::Int,part_to_gid::Vector{Int},gid::Integer)
  searchsortedlast(part_to_gid,gid)
end

function _find_part_from_gid(
  ngids::NTuple{N,Int},part_to_gid::NTuple{N,Vector{Int}},gid::Integer) where N
  cgid = Tuple(CartesianIndices(ngids)[gid])
  cpart = map(searchsortedlast,part_to_gid,cgid)
  nparts = map(length,part_to_gid)
  part = LinearIndices(nparts)[CartesianIndex(cpart)]
  part
end

function add_gid!(a::DistributedRange,gids::DistributedData{<:AbstractArray{<:Integer}})
  map_parts(a.lids,gids) do lids,gids
    for gid in gids
      add_gid!(lids,gid)
    end
  end
  a.exchanger = Exchanger(a.lids)
  a
end

function add_gid(a::DistributedRange,gids::DistributedData{<:AbstractArray{<:Integer}})
  lids = map_parts(deepcopy,a.lids)
  b = DistributedRange(a.ngids,lids)
  add_gid!(b,gids)
  b
end

function to_lid!(ids::DistributedData{<:AbstractArray{<:Integer}},a::DistributedRange)
  map_parts(to_lid!,ids,a.lids)
end

function to_gid!(ids::DistributedData{<:AbstractArray{<:Integer}},a::DistributedRange)
  map_parts(to_gid!,ids,a.lids)
end

struct DistributedVector{T,A,B} <: AbstractVector{T}
  values::A
  rows::B
  function DistributedVector(
    values::DistributedData{<:AbstractVector{T}},
    rows::DistributedRange) where T

    A = typeof(values)
    B = typeof(rows)
    new{T,A,B}(values,rows)
  end
end

Base.size(a::DistributedVector) = (length(a.rows),)
Base.axes(a::DistributedVector) = (a.rows,)
Base.IndexStyle(::Type{<:DistributedVector}) = IndexLinear()
function Base.getindex(a::DistributedVector,gid::Integer)
  # In practice this function should not be used
  @notimplemented
end

function Base.similar(a::DistributedVector)
  similar(a,eltype(a),axes(a))
end

function Base.similar(a::DistributedVector,::Type{T}) where T
  similar(a,T,axes(a))
end

function Base.similar(a::DistributedVector,::Type{T},axes::Tuple{Int}) where T
  @notimplemented
end

function Base.similar(a::DistributedVector,::Type{T},axes::Tuple{<:DistributedRange}) where T
  rows = axes[1]
  values = map_parts(a.values,rows.lids) do values, lids
    similar(values,T,num_lids(lids))
  end
  DistributedVector(values,rows)
end

function Base.similar(
  ::Type{<:DistributedVector{T,<:DistributedData{A}}},axes::Tuple{Int}) where {T,A}
  @notimplemented
end

function Base.similar(
  ::Type{<:DistributedVector{T,<:DistributedData{A}}},axes::Tuple{<:DistributedRange}) where {T,A}
  rows = axes[1]
  values = map_parts(rows.lids) do lids
    similar(A,num_lids(lids))
  end
  DistributedVector(values,rows)
end

function Base.copy!(a::DistributedVector,b::DistributedVector)
  map_parts(copy!,a.values,b.values)
  a
end

function Base.copyto!(a::DistributedVector,b::DistributedVector)
  map_parts(copyto!,a.values,b.values)
  a
end

function Base.copy(b::DistributedVector)
  a = similar(b)
  copy!(a,b)
  a
end

struct DistributedBroadcasted{A,B}
  values::A
  rows::B
end

@inline function Base.materialize(bc::DistributedBroadcasted)
  values = map_parts(Base.materialize,bc.values)
  DistributedVector(values,bc.rows)
end

@inline function Base.materialize!(a::DistributedVector,b::DistributedBroadcasted)
  @assert a.rows === b.rows
  map_parts(a.values,b.values) do dest, x
    Base.materialize!(dest,x)
  end
  a
end

function Base.broadcasted(
  f,
  args::Union{DistributedVector,DistributedBroadcasted}...)

  values = map(i->i.values,args)
  values = map_parts((largs...)->Base.broadcasted(f,largs...),values...)
  a1 = first(args)
  @notimplementedif any(ai->ai.rows!==a1.rows,args)
  DistributedBroadcasted(values,a1.rows)
end

function Base.broadcasted(
  f,
  a::Number,
  b::Union{DistributedVector,DistributedBroadcasted})

  values = map_parts(b->Base.broadcasted(f,a,b),b.values)
  DistributedBroadcasted(values,b.rows)
end

function Base.broadcasted(
  f,
  a::Union{DistributedVector,DistributedBroadcasted},
  b::Number)

  values = map_parts(a->Base.broadcasted(f,a,b),a.values)
  DistributedBroadcasted(values,a.rows)
end

function LinearAlgebra.norm(a::DistributedVector,p::Real=2)
  contibs = map_parts(a.values,a.rows.lids) do lid_to_value, lids
    oid_to_value = view(lid_to_value,lids.oid_to_lid)
    norm(oid_to_value,p)^p
  end
  reduce(+,contibs;init=zero(eltype(contibs)))^(1/p)
end

function DistributedVector{T}(
  ::UndefInitializer,
  rows::DistributedRange) where T

  values = map_parts(rows.lids) do lids
    nlids = num_lids(lids)
    Vector{T}(undef,nlids)
  end
  DistributedVector(values,rows)
end

function DistributedVector(v::Number, rows::DistributedRange)
  a = DistributedVector{typeof(v)}(undef,rows)
  fill!(a,v)
  a
end

function DistributedVector(
  init,
  I::DistributedData{<:AbstractArray{<:Integer}},
  V::DistributedData{<:AbstractArray},
  rows::DistributedRange;
  ids::Symbol)

  @assert ids in (:global,:local)
  if ids == :global
    to_lid!(I,rows)
  end

  values = map_parts(rows.lids,I,V) do lids,I,V
    values = init(num_lids(lids))
    fill!(values,zero(eltype(values)))
    for i in 1:length(I)
      lid = I[i]
      values[lid] += V[i]
    end
    values
  end

  DistributedVector(values,rows)
end

function DistributedVector(
  I::DistributedData{<:AbstractArray{<:Integer}},
  V::DistributedData{<:AbstractArray{T}},
  rows;
  ids::Symbol) where T
  DistributedVector(n->zeros(T,n),I,V,rows;ids=ids)
end

function DistributedVector(
  init,
  I::DistributedData{<:AbstractArray{<:Integer}},
  V::DistributedData{<:AbstractArray},
  n::Integer;
  ids::Symbol)

  @assert ids == :global
  parts = get_part_ids(I)
  rows = DistributedRange(parts,n)
  add_gid!(rows,I)
  DistributedVector(init,I,V,rows;ids=ids)
end

function Base.:*(a::Number,b::DistributedVector)
  values = map_parts(b.values) do values
    a*values
  end
  DistributedVector(values,b.rows)
end

function Base.:*(b::DistributedVector,a::Number)
  a*b
end

for op in (:+,:-)
  @eval begin

    function Base.$op(a::DistributedVector)
      values = map_parts(a.values) do a
        $op(a)
      end
      DistributedVector(values,a.rows)
    end

    function Base.$op(a::DistributedVector,b::DistributedVector)
      @assert a.rows === b.rows
      values = map_parts(a.values,b.values) do a,b
        $op(a,b)
      end
      DistributedVector(values,a.rows)
    end

  end
end

function Base.fill!(a::DistributedVector,v)
  map_parts(a.values) do lid_to_value
    fill!(lid_to_value,v)
  end
  a
end

function Base.reduce(op,a::DistributedVector;init)
  b = map_parts(a.values,a.rows.lids) do values,lids
    owned_values = view(values,lids.oid_to_lid)
    reduce(op,owned_values,init=init)
  end
  reduce(op,b,init=init)
end

function Base.sum(a::DistributedVector)
  reduce(+,a,init=zero(eltype(a)))
end

function LinearAlgebra.dot(a::DistributedVector,b::DistributedVector)
  c = map_parts(a.values,b.values,a.rows.lids,b.rows.lids) do a,b,alids,blids
    a_owned = view(a,alids.oid_to_lid)
    b_owned = view(b,blids.oid_to_lid)
    dot(a_owned,b_owned)
  end
  sum(c)
end

function local_view(a::DistributedVector)
  a.values
end

function global_view(a::DistributedVector)
  map_parts(a.values,a.rows.lids) do values, lids
    GlobalView(values,(lids.gid_to_lid,),(lids.ngids,))
  end
end

struct GlobalView{T,N,A,B} <: AbstractArray{T,N}
  values::A
  d_to_gid_to_lid::B
  global_size::NTuple{N,Int}
  function GlobalView(
    values::AbstractArray{T,N},
    d_to_gid_to_lid::NTuple{N},
    global_size::NTuple{N}) where {T,N}

    A = typeof(values)
    B = typeof(d_to_gid_to_lid)
    new{T,N,A,B}(values,d_to_gid_to_lid,global_size)
  end
end

Base.size(a::GlobalView) = a.global_size
Base.IndexStyle(::Type{<:GlobalView}) = IndexCartesian()
function Base.getindex(a::GlobalView{T,N},gid::Vararg{Integer,N}) where {T,N}
  lid = map((g,gid_to_lid)->gid_to_lid[g],gid,a.d_to_gid_to_lid)
  a.values[lid...]
end
function Base.setindex!(a::GlobalView{T,N},v,gid::Vararg{Integer,N}) where {T,N}
  lid = map((g,gid_to_lid)->gid_to_lid[g],gid,a.d_to_gid_to_lid)
  a.values[lid...] = v
end

function async_exchange!(
  a::DistributedVector,
  t0::DistributedData=_empty_tasks(a.rows.exchanger.parts_rcv))
  async_exchange!(a.values,a.rows.exchanger,t0)
end

# Non-blocking assembly
function async_assemble!(
  a::DistributedVector,
  t0::DistributedData=_empty_tasks(a.rows.exchanger.parts_rcv);
  reduce_op=+)

  exchanger_rcv = a.rows.exchanger # receives data at ghost ids from remote parts
  exchanger_snd = reverse(exchanger_rcv) # sends data at ghost ids to remote parts
  t1 = async_exchange!(a.values,exchanger_snd,t0;reduce_op=reduce_op)
  map_parts(t1,a.values,a.rows.lids) do t1,values,lids
    @task begin
      wait(schedule(t1))
      values[lids.hid_to_lid] .= zero(eltype(values))
    end
  end
end

# Blocking assembly
function assemble!(args...;kwargs...)
  t = async_assemble!(args...;kwargs...)
  map_parts(schedule,t)
  map_parts(wait,t)
  first(args)
end

struct DistributedSparseMatrix{T,A,B,C,D} <: AbstractMatrix{T}
  values::A
  rows::B
  cols::C
  exchanger::D
  function DistributedSparseMatrix(
    values::DistributedData{<:AbstractSparseMatrix{T}},
    rows::DistributedRange,
    cols::DistributedRange,
    exchanger=_matrix_exchanger(values,rows.exchanger,rows.lids,cols.lids)) where T

    A = typeof(values)
    B = typeof(rows)
    C = typeof(cols)
    D = typeof(exchanger)
    new{T,A,B,C,D}(values,rows,cols,exchanger)
  end
end

Base.size(a::DistributedSparseMatrix) = (num_gids(a.rows),num_gids(a.cols))
Base.axes(a::DistributedSparseMatrix) = (a.rows,a.cols)
Base.IndexStyle(::Type{<:DistributedSparseMatrix}) = IndexCartesian()
function Base.getindex(a::DistributedSparseMatrix,gi::Integer,gj::Integer)
  #This should not be used in practice
  @notimplemented
end

function DistributedSparseMatrix(
  init,
  I::DistributedData{<:AbstractArray{<:Integer}},
  J::DistributedData{<:AbstractArray{<:Integer}},
  V::DistributedData{<:AbstractArray},
  rows::DistributedRange,
  cols::DistributedRange;
  ids::Symbol)

  @assert ids in (:global,:local)
  if ids == :global
    to_lid!(I,rows)
    to_lid!(J,cols)
  end

  values = map_parts(I,J,V,rows.lids,cols.lids) do I,J,V,rlids,clids
    init(I,J,V,num_lids(rlids),num_lids(clids))
  end

  DistributedSparseMatrix(values,rows,cols)
end

function DistributedSparseMatrix(
  init,
  I::DistributedData{<:AbstractArray{<:Integer}},
  J::DistributedData{<:AbstractArray{<:Integer}},
  V::DistributedData{<:AbstractArray},
  nrows::Integer,
  ncols::Integer;
  ids::Symbol)

  @assert ids == :global
  parts = get_part_ids(I)
  rows = DistributedRange(parts,nrows)
  if nrows == ncols
    cols = rows
  else
    cols = DistributedRange(parts,ncols)
  end
  add_gid!(rows,I)
  add_gid!(cols,J)
  DistributedSparseMatrix(init,I,J,V,rows,cols;ids=ids)
end

function DistributedSparseMatrix(
  I::DistributedData{<:AbstractArray{<:Integer}},
  J::DistributedData{<:AbstractArray{<:Integer}},
  V::DistributedData{<:AbstractArray},
  rows,
  cols;
  ids::Symbol)

  DistributedSparseMatrix(sparse,I,J,V,rows,cols;ids=ids)
end

function LinearAlgebra.mul!(
  c::DistributedVector,
  a::DistributedSparseMatrix,
  b::DistributedVector,
  α::Number,
  β::Number)

  @assert c.rows === a.rows
  @assert b.rows === a.cols
  t = async_exchange!(b)
  map_parts(t,c.values,a.values,b.values) do t,c,a,b
    # TODO start multiplying the diagonal block
    # before waiting for the ghost values of b
    wait(schedule(t))
    mul!(c,a,b,α,β)
  end
  c
end

function local_view(a::DistributedSparseMatrix)
  a.values
end

function global_view(a::DistributedSparseMatrix)
  map_parts(a.values,a.rows.lids,a.cols.lids) do values,rlids,clids
    GlobalView(values,(rlids.gid_to_lid,clids.gid_to_lid),(rlids.ngids,clids.ngids))
  end
end

function _matrix_exchanger(values,row_exchanger,row_lids,col_lids)

  part = get_part_ids(row_lids)
  parts_rcv = row_exchanger.parts_rcv
  parts_snd = row_exchanger.parts_snd
  findnz_values = map_parts(findnz,values)

  function setup_rcv(part,parts_rcv,row_lids,col_lids,findnz_values)
    k_li, k_lj, = findnz_values
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_rcv) ))
    ptrs = zeros(Int32,length(parts_rcv)+1)
    for k in 1:length(k_li)
      li = k_li[k]
      owner = row_lids.lid_to_part[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)
    k_rcv_data = zeros(Int,ptrs[end]-1)
    gi_rcv_data = zeros(Int,ptrs[end]-1)
    gj_rcv_data = zeros(Int,ptrs[end]-1)
    for k in 1:length(k_li)
      li = k_li[k]
      lj = k_lj[k]
      owner = row_lids.lid_to_part[li]
      if owner != part
        p = ptrs[owner_to_i[owner]]
        k_rcv_data[p] = k
        gi_rcv_data[p] = row_lids.lid_to_gid[li]
        gj_rcv_data[p] = col_lids.lid_to_gid[lj]
        ptrs[owner_to_i[owner]] += 1
      end
    end
    rewind_ptrs!(ptrs)
    k_rcv = Table(k_rcv_data,ptrs)
    gi_rcv = Table(gi_rcv_data,ptrs)
    gj_rcv = Table(gj_rcv_data,ptrs)
    k_rcv, gi_rcv, gj_rcv
  end
  
  k_rcv, gi_rcv, gj_rcv = map_parts(setup_rcv,part,parts_rcv,row_lids,col_lids,findnz_values)

  gi_snd = exchange(gi_rcv,parts_snd,parts_rcv)
  gj_snd = exchange(gj_rcv,parts_snd,parts_rcv)

  function setup_snd(part,row_lids,col_lids,gi_snd,gj_snd,findnz_values)
    ptrs = gi_snd.ptrs
    k_snd_data = zeros(Int,ptrs[end]-1)
    k_li, k_lj, = findnz_values
    k_v = collect(1:length(k_li))
    # TODO this can be optimized:
    li_lj_to_k = sparse(k_li,k_lj,k_v,num_lids(row_lids),num_lids(col_lids))
    for p in 1:length(gi_snd.data)
      gi = gi_snd.data[p]
      gj = gj_snd.data[p]
      li = row_lids.gid_to_lid[gi]
      lj = col_lids.gid_to_lid[gj]
      k = li_lj_to_k[li,lj]
      @assert k > 0 "The sparsity patern of the ghost layer is inconsistent"
      k_snd_data[p] = k
    end
    k_snd = Table(k_snd_data,ptrs)
    k_snd
  end

  k_snd = map_parts(setup_snd,part,row_lids,col_lids,gi_snd,gj_snd,findnz_values)

  Exchanger(parts_rcv,parts_snd,k_rcv,k_snd)
end

# Non-blocking exchange
function async_exchange!(
  a::DistributedSparseMatrix,
  t0::DistributedData=_empty_tasks(a.exchanger.parts_rcv))
  nzval = map_parts(nonzeros,a.values)
  async_exchange!(nzval,a.exchanger,t0)
end

# Non-blocking assembly
function async_assemble!(
  a::DistributedSparseMatrix,
  t0::DistributedData=_empty_tasks(a.exchanger.parts_rcv);
  reduce_op=+)

  exchanger_rcv = a.exchanger # receives data at ghost ids from remote parts
  exchanger_snd = reverse(exchanger_rcv) # sends data at ghost ids to remote parts
  nzval = map_parts(nonzeros,a.values)
  t1 = async_exchange!(nzval,exchanger_snd,t0;reduce_op=reduce_op)
  map_parts(t1,nzval,exchanger_snd.lids_snd) do t1,nzval,lids_snd
    @task begin
      wait(schedule(t1))
      nzval[lids_snd.data] .= zero(eltype(nzval))
    end
  end
end

function Base.:*(a::Number,b::DistributedSparseMatrix)
  values = map_parts(b.values) do values
    a*values
  end
  DistributedSparseMatrix(values,b.rows,b.cols,b.exchanger)
end

function Base.:*(b::DistributedSparseMatrix,a::Number)
  a*b
end

function Base.:*(a::DistributedSparseMatrix{Ta},b::DistributedVector{Tb}) where {Ta,Tb}
  T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
  c = DistributedVector{T}(undef,a.rows)
  mul!(c,a,b)
  c
end

for op in (:+,:-)
  @eval begin

    function Base.$op(a::DistributedSparseMatrix)
      values = map_parts(a.values) do a
        $op(a)
      end
      DistributedSparseMatrix(values,a.rows,a.cols,a.exchanger)
    end

  end
end

# This structs implements the same methods as the Identity preconditioner
# in the IterativeSolvers package.
# TODO swap row and cols
struct Jacobi{T,A,B,C}
  diaginv::A
  rows::B
  cols::C
  function Jacobi(
    diaginv::DistributedData{<:AbstractVector{T}},
    rows::DistributedRange,
    cols::DistributedRange) where T

    A = typeof(diaginv)
    B = typeof(rows)
    C = typeof(cols)
    new{T,A,B,C}(diaginv,rows,cols)
  end
end

function Jacobi(a::DistributedSparseMatrix)
  diaginv = map_parts(a.values,a.rows.lids,a.cols.lids) do values, rlids, clids
    @assert num_oids(rlids) == num_oids(clids)
    @notimplementedif rlids.oid_to_lid != clids.oid_to_lid
    ldiag = collect(diag(values))
    odiag = ldiag[rlids.oid_to_lid]
    diaginv = inv.(odiag)
  end
  Jacobi(diaginv,a.rows,a.cols)
end

function LinearAlgebra.ldiv!(
  c::DistributedVector,
  a::Jacobi,
  b::DistributedVector)

  @assert c.rows === a.cols
  @assert b.rows === a.rows
  map_parts(c.values,a.diaginv,b.values,a.rows.lids,a.cols.lids) do c,diaginv,b,rlids,clids
    @assert num_oids(rlids) == num_oids(clids)
    for oid in 1:num_oids(rlids)
      li = rlids.oid_to_lid[oid]
      lj = clids.oid_to_lid[oid]
      c[lj] = diaginv[oid]*b[li]
    end
  end
  c
end

function LinearAlgebra.ldiv!(
  a::Jacobi,
  b::DistributedVector)
  ldiv!(b,a,b)
end

function Base.:\(a::Jacobi{Ta},b::DistributedVector{Tb}) where {Ta,Tb}
  T = typeof(zero(Ta)/one(Tb)+zero(Ta)/one(Tb))
  c = DistributedVector{T}(undef,a.cols)
  ldiv!(c,a,b)
  c
end

# Misc functions that could be removed if IterativeSolvers would be implemented in terms
# of axes(A,d) instead of size(A,d)
function IterativeSolvers.zerox(A::DistributedSparseMatrix,b::DistributedVector)
  T = IterativeSolvers.Adivtype(A, b)
  x = similar(b, T, axes(A, 2))
  fill!(x, zero(T))
  return x
end

