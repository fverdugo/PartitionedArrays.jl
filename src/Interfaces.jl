
abstract type Backend end

# Should return a PData{Int}
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
abstract type PData{T,N} end

Base.size(a::PData) = @abstractmethod

Base.length(a::PData) = prod(size(a))

num_parts(a::PData) = length(a)

get_backend(a::PData) = @abstractmethod

Base.iterate(a::PData)  = @abstractmethod

Base.iterate(a::PData,state)  = @abstractmethod

get_part_ids(a::PData) = get_part_ids(get_backend(a),size(a))

map_parts(task,a::PData...) = @abstractmethod

i_am_main(::PData) = @abstractmethod

Base.eltype(a::PData{T}) where T = T
Base.eltype(::Type{<:PData{T}}) where T = T

Base.ndims(a::PData{T,N}) where {T,N} = N
Base.ndims(::Type{<:PData{T,N}}) where {T,N} = N

#function map_parts(task,a...)
#  map_parts(task,map(PData,a)...)
#end
#
#PData(a::PData) = a

const MAIN = 1

# import the main part to the main scope
# in MPI this will broadcast the main part to all procs
get_main_part(a::PData) = get_part(a,MAIN)

# This one is safe to use only when all parts contain the same value, e.g. the result of a gather_all call.
get_part(a::PData) = @abstractmethod

get_part(a::PData,part::Integer) = @abstractmethod

# rcv can contain vectors or tables
gather!(rcv::PData,snd::PData) = @abstractmethod

gather_all!(rcv::PData,snd::PData) = @abstractmethod

function allocate_gather(snd::PData)
  np = num_parts(snd)
  parts = get_part_ids(snd)
  rcv = map_parts(parts,snd) do part, snd
    T = typeof(snd)
    if part == MAIN
      Vector{T}(undef,np)
    else
      Vector{T}(undef,0)
    end
  end
  rcv
end

function allocate_gather(snd::PData{<:AbstractVector})
  l = map_parts(length,snd)
  l_main = gather(l)
  parts = get_part_ids(snd)
  rcv = map_parts(parts,l_main,snd) do part, l, snd
    if part == MAIN
      ptrs = counts_to_ptrs(l)
      ndata = ptrs[end]-1
      data = Vector{eltype(snd)}(undef,ndata)
      Table(data,ptrs)
    else
      ptrs = Vector{Int32}(undef,1)
      data = Vector{eltype(snd)}(undef,0)
      Table(data,ptrs)
    end
  end
  rcv
end

function gather(snd::PData)
  rcv = allocate_gather(snd)
  gather!(rcv,snd)
  rcv
end

function allocate_gather_all(snd::PData)
  np = num_parts(snd)
  rcv = map_parts(snd) do snd
    T = typeof(snd)
    Vector{T}(undef,np)
  end
  rcv
end

function allocate_gather_all(snd::PData{<:AbstractVector})
  l = map_parts(length,snd)
  l_all = gather_all(l)
  parts = get_part_ids(snd)
  rcv = map_parts(parts,l_all,snd) do part, l, snd
    ptrs = counts_to_ptrs(l)
    ndata = ptrs[end]-1
    data = Vector{eltype(snd)}(undef,ndata)
    Table(data,ptrs)
  end
  rcv
end

function gather_all(snd::PData)
  rcv = allocate_gather_all(snd)
  gather_all!(rcv,snd)
  rcv
end

# The back-end need to support these cases:
# i.e. PData{AbstractVector{<:Number}} and PData{AbstractVector{<:AbstractVector{<:Number}}}
function scatter(snd::PData)
  @abstractmethod
end

# AKA broadcast
function emit(snd::PData)
  np = num_parts(snd)
  parts = get_part_ids(snd)
  snd2 = map_parts(parts,snd) do part, snd
    T = typeof(snd)
    if part == MAIN
      v = Vector{T}(undef,np)
      fill!(v,snd)
    else
      v = Vector{T}(undef,0)
    end
    v
  end
  scatter(snd2)
end

function reduce_main(op,snd::PData;init)
  a = gather(snd)
  map_parts(i->reduce(op,i;init=init),a)
end

function reduce_all(args...;kwargs...)
  b = reduce_main(args...;kwargs...)
  emit(b)
end

function Base.reduce(op,a::PData;init)
  b = reduce_main(op,a;init=init)
  get_main_part(b)
end

function Base.sum(a::PData)
  reduce(+,a,init=zero(eltype(a)))
end

# inclusive prefix reduction
function iscan(op,a::PData;init)
  b = iscan_main(op,a,init=init)
  scatter(b)
end

function iscan_all(op,a::PData;init)
  b = iscan_main(op,a,init=init)
  emit(b)
end

function iscan_main(op,a;init)
  b = gather(a)
  parts = get_part_ids(a)
  map_parts(parts,b) do part, b
    if part == MAIN
      b[1] = op(init,b[1])
      @inbounds for i in 1:(length(b)-1)
        b[i+1] = op(b[i],b[i+1])
      end
    end
  end
  b
end

# exclusive prefix reduction
function xscan(op,a::PData;init)
  b = xscan_main(op,a,init=init)
  scatter(b)
end

function xscan_all(op,a::PData;init)
  b = xscan_main(op,a,init=init)
  emit(b)
end

function xscan_main(op,a::PData;init)
  b = gather(a)
  parts = get_part_ids(a)
  map_parts(parts,b) do part, b
    if part == MAIN
      @inbounds for i in (length(b)-1):-1:1
        b[i+1] = b[i]
      end
      b[1] = init
      @inbounds for i in 1:(length(b)-1)
        b[i+1] = op(b[i],b[i+1])
      end
    end
  end
  b
end

# Non-blocking in-place exchange
# In this version, sending a number per part is enough
# We have another version below to send a vector of numbers per part (compressed in a Table)
# Starts a non-blocking exchange. It returns a PData of Julia Tasks. Calling schedule and wait on these
# tasks will wait until the exchange is done in the corresponding part
# (i.e., at this point it is save to read/write the buffers again).
function async_exchange!(
  data_rcv::PData,
  data_snd::PData,
  parts_rcv::PData,
  parts_snd::PData,
  t_in::PData)

  @abstractmethod
end

function async_exchange!(
  data_rcv::PData,
  data_snd::PData,
  parts_rcv::PData,
  parts_snd::PData)

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
  data_snd::PData,
  parts_rcv::PData,
  parts_snd::PData,
  t_in::PData=_empty_tasks(parts_rcv))

  data_rcv = map_parts(data_snd,parts_rcv) do data_snd, parts_rcv
    similar(data_snd,eltype(data_snd),length(parts_rcv))
  end

  t_out = async_exchange!(data_rcv,data_snd,parts_rcv,parts_snd,t_in)

  data_rcv, t_out
end

# Non-blocking in-place exchange variable length (compressed in a Table)
function async_exchange!(
  data_rcv::PData{<:Table},
  data_snd::PData{<:Table},
  parts_rcv::PData,
  parts_snd::PData,
  t_in::PData)

  @abstractmethod
end

# Non-blocking allocating exchange variable length (compressed in a Table)
function async_exchange(
  data_snd::PData{<:Table},
  parts_rcv::PData,
  parts_snd::PData,
  t_in::PData)

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
function discover_parts_snd(parts_rcv::PData, neighbors::PData)
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

# If neighbors not provided, we need to gather in main
function discover_parts_snd(parts_rcv::PData)
  parts = get_part_ids(parts_rcv)
  parts_rcv_main = gather(parts_rcv)
  parts_snd_main = map_parts(parts,parts_rcv_main) do part,parts_rcv
    if part == MAIN
      parts_snd = _parts_rcv_to_parts_snd(parts_rcv)
    else
      ptrs = similar(parts_rcv.ptrs,eltype(parts_rcv.ptrs),1)
      data = similar(parts_rcv.data,eltype(parts_rcv.data),0)
      parts_snd = Table(data,ptrs)
    end
    parts_snd
  end
  parts_snd = scatter(parts_snd_main)
  parts_snd
end

function _parts_rcv_to_parts_snd(parts_rcv::Table)
  I = Int32[]
  J = Int32[]
  np = length(parts_rcv)
  for p in 1:np
    kini = parts_rcv.ptrs[p]
    kend = parts_rcv.ptrs[p+1]-1
    for k in kini:kend
      push!(I,p)
      push!(J,parts_rcv.data[k])
    end
  end
  graph = sparse(I,J,I,np,np)
  ptrs = similar(parts_rcv.ptrs)
  fill!(ptrs,zero(eltype(ptrs)))
  for (i,j,_) in nziterator(graph)
    ptrs[j+1] += 1
  end
  length_to_ptrs!(ptrs)
  ndata = ptrs[end]-1
  data = similar(parts_rcv.data,eltype(parts_rcv.data),ndata)
  for (i,j,_) in nziterator(graph)
    data[ptrs[j]] = i
    ptrs[j] += 1
  end
  rewind_ptrs!(ptrs)
  parts_snd = Table(data,ptrs)
end

function discover_parts_snd(parts_rcv::PData,::Nothing)
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
get_part(a::IndexSet) = a.part
get_lid_to_gid(a::IndexSet) = a.lid_to_gid
get_lid_to_part(a::IndexSet) = a.lid_to_part
get_gid_to_part(a::IndexSet) = a.gid_to_part
get_oid_to_lid(a::IndexSet) = a.oid_to_lid
get_hid_to_lid(a::IndexSet) = a.hid_to_lid
get_lid_to_ohid(a::IndexSet) = a.lid_to_ohid
get_gid_to_lid(a::IndexSet) = a.gid_to_lid

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

function oids_are_equal(a::IndexSet,b::IndexSet)
  view(a.lid_to_gid,a.oid_to_lid) == view(b.lid_to_gid,b.oid_to_lid)
end

function hids_are_equal(a::IndexSet,b::IndexSet)
  view(a.lid_to_gid,a.hid_to_lid) == view(b.lid_to_gid,b.hid_to_lid)
end

function lids_are_equal(a::IndexSet,b::IndexSet)
  a.lid_to_gid == b.lid_to_gid
end

struct Exchanger{B,C}
  parts_rcv::B
  parts_snd::B
  lids_rcv::C
  lids_snd::C
  function Exchanger(
    parts_rcv::PData{<:AbstractVector{<:Integer}},
    parts_snd::PData{<:AbstractVector{<:Integer}},
    lids_rcv::PData{<:Table{<:Integer}},
    lids_snd::PData{<:Table{<:Integer}})

    B = typeof(parts_rcv)
    C = typeof(lids_rcv)
    new{B,C}(parts_rcv,parts_snd,lids_rcv,lids_snd)
  end
end

function Exchanger(ids::PData{<:IndexSet},neighbors=nothing)

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

function empty_exchanger(a::PData)
  parts_rcv = map_parts(i->Int[],a)
  parts_snd = map_parts(i->Int[],a)
  lids_rcv = map_parts(i->Table(Vector{Int32}[]),a)
  lids_snd = map_parts(i->Table(Vector{Int32}[]),a)
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
  values::PData{<:AbstractVector{T}},
  exchanger::Exchanger,
  t0::PData=_empty_tasks(exchanger.parts_rcv);
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

function async_exchange!(
  values::PData{<:Table},
  exchanger::Exchanger,
  t0::PData=_empty_tasks(exchanger.parts_rcv);
  reduce_op=_replace)

  data, ptrs = map_parts(t->(t.data,t.ptrs),values)
  t_exchanger = _table_exchanger(exchanger,ptrs)
  async_exchange!(data,t_exchanger,t0;reduce_op=reduce_op)
end

function _table_exchanger(exchanger,values)
  lids_rcv = _table_lids_snd(exchanger.lids_rcv,values)
  lids_snd = _table_lids_snd(exchanger.lids_snd,values)
  parts_rcv = exchanger.parts_rcv
  parts_snd = exchanger.parts_snd
  Exchanger(parts_rcv,parts_snd,lids_rcv,lids_snd)
end

function _table_lids_snd(lids_snd,tptrs)
  k_snd = map_parts(tptrs,lids_snd) do tptrs,lids_snd
    ptrs = similar(lids_snd.ptrs)
    fill!(ptrs,zero(eltype(ptrs)))
    np = length(ptrs)-1
    for p in 1:np
      iini = lids_snd.ptrs[p]
      iend = lids_snd.ptrs[p+1]-1
      for i in iini:iend
        d = lids_snd.data[i]
        ptrs[p+1] += tptrs[d+1]-tptrs[d]
      end
    end
    length_to_ptrs!(ptrs)
    ndata = ptrs[end]-1
    data = similar(lids_snd.data,eltype(lids_snd.data),ndata)
    for p in 1:np
      iini = lids_snd.ptrs[p]
      iend = lids_snd.ptrs[p+1]-1
      for i in iini:iend
        d = lids_snd.data[i]
        jini = tptrs[d]
        jend = tptrs[d+1]-1
        for j in jini:jend
          data[ptrs[p]] = j
          ptrs[p] += 1
        end
      end
    end
    rewind_ptrs!(ptrs)
    Table(data,ptrs)
  end
  k_snd
end

# TODO mutable is needed to correctly implement add_gid!
mutable struct PRange{A,B} <: AbstractUnitRange{Int}
  ngids::Int
  lids::A
  ghost::Bool
  exchanger::B
  function PRange(
    ngids::Integer,
    lids::PData{<:IndexSet},
    ghost::Bool,
    exchanger::Exchanger)
  
    A = typeof(lids)
    B = typeof(exchanger)
    new{A,B}(
      ngids,
      lids,
      ghost,
      exchanger)
  end
end

# TODO in MPI this causes to copy the world comm
# and makes some assertions to fail.
function Base.copy(a::PRange)
  ngids = copy(a.ngids)
  lids = deepcopy(a.lids)
  ghost = copy(a.ghost)
  exchanger = deepcopy(a.exchanger)
  PRange(ngids,lids,ghost,exchanger)
end

function PRange(
  ngids::Integer,
  lids::PData{<:IndexSet},
  ghost::Bool=true)

  exchanger =  ghost ? Exchanger(lids) : empty_exchanger(lids)
  PRange(ngids,lids,ghost,exchanger)
end

Base.first(a::PRange) = 1
Base.last(a::PRange) = a.ngids

num_gids(a::PRange) = a.ngids
num_parts(a::PRange) = num_parts(a.lids)

function PRange(parts::PData{<:Integer},ngids::Integer)
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
  ghost = false
  PRange(ngids,lids,ghost)
end

function PRange(
  parts::PData{<:Integer,1},
  ngids::NTuple{N,<:Integer}) where N
  PRange(parts,prod(ngids))
end

function PRange(parts::PData{<:Integer},noids::PData{<:Integer})
  ngids = reduce(+,noids,init=0)
  firstgids = xscan_all(+,noids,init=1)
  lids = map_parts(parts,noids,firstgids) do part,noids,firstgids
    lid_to_gid = collect(Int,(1:noids) .+ (firstgids[part]-1))
    lid_to_part = fill(part,length(lid_to_gid))
    gid_to_part = GidToPart(ngids,firstgids)
    oid_to_lid = Int32(1):Int32(length(lid_to_gid))
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
  ghost = false
  PRange(ngids,lids,ghost)
end


# TODO this is type instable
function PRange(
  parts::PData{<:Integer,N},
  ngids::NTuple{N,<:Integer};
  ghost::Bool=false) where N

  np = size(parts)
  lids = map_parts(parts) do part
    if ghost
      cp = Tuple(CartesianIndices(np)[part])
      d_to_ldid_to_gdid = map(_lid_to_gid,ngids,np,cp)
      lid_to_gid = _id_tensor_product(Int,d_to_ldid_to_gdid,ngids)
      d_to_nldids = map(length,d_to_ldid_to_gdid)
      lid_to_part = _lid_to_part(d_to_nldids,np,cp)
      oid_to_lid = collect(Int32,findall(lid_to_part .== part))
      hid_to_lid = collect(Int32,findall(lid_to_part .!= part))
    else
      gids = _oid_to_gid(ngids,np,part)
      lid_to_gid = gids
      lid_to_part = fill(part,length(gids))
      oid_to_lid = Int32(1):Int32(length(gids))
      hid_to_lid = collect(Int32(1):Int32(0))
    end
    part_to_gid = _part_to_gid(ngids,np)
    gid_to_part = GidToPart(ngids,part_to_gid)
    IndexSet(
      part,
      prod(ngids),
      lid_to_gid,
      lid_to_part,
      gid_to_part,
      oid_to_lid,
      hid_to_lid)
  end
  PRange(prod(ngids),lids,ghost)
end

# TODO this is type instable
function PCartesianIndices(
  parts::PData{<:Integer,N},
  ngids::NTuple{N,<:Integer};
  ghost::Bool= false) where N

  np = size(parts)
  lids = map_parts(parts) do part
    cis_parts = CartesianIndices(np)
    p = Tuple(cis_parts[part])
    if ghost
      d_to_odid_to_gdid = map(_lid_to_gid,ngids,np,p)
    else
      d_to_odid_to_gdid = map(_oid_to_gid,ngids,np,p)
    end
    CartesianIndices(d_to_odid_to_gdid)
  end
  lids
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

function _lid_to_gid(ngids::Integer,np::Integer,p::Integer)
  oid_to_gid = _oid_to_gid(ngids,np,p)
  gini = first(oid_to_gid)
  gend = last(oid_to_gid)
  if np == 1
    lid_to_gid = oid_to_gid
  elseif p == 1
    lid_to_gid = gini:(gend+1)
  elseif p != np
    lid_to_gid = (gini-1):(gend+1)
  else
    lid_to_gid = (gini-1):gend
  end
  lid_to_gid
end

function _lid_to_part(nlids::Integer,np::Integer,p::Integer)
  lid_to_part = Vector{Int32}(undef,nlids)
  fill!(lid_to_part,p)
  if np == 1
    lid_to_part
  elseif p == 1
    lid_to_part[end] = p+1
  elseif p != np
    lid_to_part[1] = p-1
    lid_to_part[end] = p+1
  else
    lid_to_part[1] = p-1
  end
  lid_to_part
end

function _oid_to_gid(ngids::Tuple,np::Tuple,p::Integer)
  cp = Tuple(CartesianIndices(np)[p])
  _oid_to_gid(ngids,np,cp)
end

function _oid_to_gid(ngids::Tuple,np::Tuple,p::Tuple)
  D = length(np)
  @assert length(ngids) == D
  d_to_odid_to_gdid = map(_oid_to_gid,ngids,np,p)
  _id_tensor_product(Int,d_to_odid_to_gdid,ngids)
end

function _lid_to_gid(ngids::Tuple,np::Tuple,p::Integer)
  cp = Tuple(CartesianIndices(np)[p])
  _lid_to_gid(ngids,np,cp)
end

function _lid_to_gid(ngids::Tuple,np::Tuple,p::Tuple)
  D = length(np)
  @assert length(ngids) == D
  d_to_ldid_to_gdid = map(_lid_to_gid,ngids,np,p)
  _id_tensor_product(Int,d_to_ldid_to_gdid,ngids)
end

function _lid_to_part(nlids::Tuple,np::Tuple,p::Integer)
  cp = Tuple(CartesianIndices(np)[p])
  _lid_to_part(nlids,np,cp)
end

function _lid_to_part(nlids::Tuple,np::Tuple,p::Tuple)
  D = length(np)
  @assert length(nlids) == D
  d_to_ldid_to_dpart = map(_lid_to_part,nlids,np,p)
  _id_tensor_product(Int,d_to_ldid_to_dpart,np)
end

function _id_tensor_product(::Type{T},d_to_dlid_to_gdid::Tuple,d_to_ngdids::Tuple) where T
  d_to_nldids = map(length,d_to_dlid_to_gdid)
  lcis = CartesianIndices(d_to_nldids)
  llis = LinearIndices(d_to_nldids)
  glis = LinearIndices(d_to_ngdids)
  D = length(d_to_ngdids)
  gci = zeros(T,D)
  lid_to_gid = zeros(T,length(lcis))
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

#function _part_tensor_product(d_to_dlid_to_dpart::Tuple,d_to_ndparts::Tuple)
#  d_to_nldids = map(length,d_to_dlid_to_dpart)
#  lcis = CartesianIndices(d_to_nldids)
#  llis = LinearIndices(d_to_nldids)
#  plis = LinearIndices(d_to_ndparts)
#  lid_to_part = zeros(Int32,length(lcis))
#  D = length(d_to_ndparts)
#  pci = zeros(Int32,D)
#  for lci in lcis
#    for d in 1:D
#      ldid = lci[d]
#      dpart = d_to_dlid_to_dpart[d][ldid]
#      pci[d] = dpart
#    end
#    lid = llis[lci]
#    lid_to_part[lid] = plis[CartesianIndex(Tuple(pci))]
#  end
#  lid_to_part
#end

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

function add_gid!(a::PRange,gids::PData{<:AbstractArray{<:Integer}})
  map_parts(a.lids,gids) do lids,gids
    for gid in gids
      add_gid!(lids,gid)
    end
  end
  a.exchanger = Exchanger(a.lids)
  a.ghost = true
  a
end

function add_gid(a::PRange,gids::PData{<:AbstractArray{<:Integer}})
  lids = map_parts(deepcopy,a.lids)
  b = PRange(a.ngids,lids)
  add_gid!(b,gids)
  b
end

function to_lid!(ids::PData{<:AbstractArray{<:Integer}},a::PRange)
  map_parts(to_lid!,ids,a.lids)
end

function to_gid!(ids::PData{<:AbstractArray{<:Integer}},a::PRange)
  map_parts(to_gid!,ids,a.lids)
end

function oids_are_equal(a::PRange,b::PRange)
  if a.lids === b.lids
    true
  else
    c = map_parts(oids_are_equal,a.lids,b.lids)
    reduce(&,c,init=true)
  end
end

function hids_are_equal(a::PRange,b::PRange)
  if a.lids === b.lids
    true
  else
    c = map_parts(hids_are_equal,a.lids,b.lids)
    reduce(&,c,init=true)
  end
end

function lids_are_equal(a::PRange,b::PRange)
  if a.lids === b.lids
    true
  else
    c = map_parts(lids_are_equal,a.lids,b.lids)
    reduce(&,c,init=true)
  end
end

struct PVector{T,A,B} <: AbstractVector{T}
  values::A
  rows::B
  function PVector(
    values::PData{<:AbstractVector{T}},
    rows::PRange) where T

    A = typeof(values)
    B = typeof(rows)
    new{T,A,B}(values,rows)
  end
end

function Base.getproperty(x::PVector, sym::Symbol)
  if sym == :owned_values
    map_parts(x.values,x.rows.lids) do v,r
      view(v,r.oid_to_lid)
    end
  elseif sym == :ghost_values
    map_parts(x.values,x.rows.lids) do v,r
      view(v,r.hid_to_lid)
    end
  else
    getfield(x, sym)
  end
end

function Base.propertynames(x::PVector, private=false)
  (fieldnames(typeof(x))...,:owned_values,:ghost_values)
end

Base.size(a::PVector) = (length(a.rows),)
Base.axes(a::PVector) = (a.rows,)
Base.IndexStyle(::Type{<:PVector}) = IndexLinear()
function Base.getindex(a::PVector,gid::Integer)
  # In practice this function should not be used
  @notimplemented
end

function Base.similar(a::PVector)
  similar(a,eltype(a),axes(a))
end

function Base.similar(a::PVector,::Type{T}) where T
  similar(a,T,axes(a))
end

function Base.similar(a::PVector,::Type{T},axes::Tuple{Int}) where T
  @notimplemented
end

function Base.similar(a::PVector,::Type{T},axes::Tuple{<:PRange}) where T
  rows = axes[1]
  values = map_parts(a.values,rows.lids) do values, lids
    similar(values,T,num_lids(lids))
  end
  PVector(values,rows)
end

function Base.similar(
  ::Type{<:PVector{T,<:PData{A}}},axes::Tuple{Int}) where {T,A}
  @notimplemented
end

function Base.similar(
  ::Type{<:PVector{T,<:PData{A}}},axes::Tuple{<:PRange}) where {T,A}
  rows = axes[1]
  values = map_parts(rows.lids) do lids
    similar(A,num_lids(lids))
  end
  PVector(values,rows)
end

function Base.copy!(a::PVector,b::PVector)
  map_parts(copy!,a.values,b.values)
  a
end

function Base.copyto!(a::PVector,b::PVector)
  map_parts(copyto!,a.values,b.values)
  a
end

function Base.copy(b::PVector)
  a = similar(b)
  copy!(a,b)
  a
end

struct DistributedBroadcasted{A,B,C}
  owned_values::A
  ghost_values::B
  rows::C
end

@inline function Base.materialize(b::DistributedBroadcasted)
  owned_values = map_parts(Base.materialize,b.owned_values)
  T = eltype(eltype(owned_values))
  a = PVector{T}(undef,b.rows)
  map_parts(a.owned_values,owned_values) do dest, src
    dest .= src
  end
  if b.ghost_values !== nothing
    ghost_values = map_parts(Base.materialize,b.ghost_values)
    map_parts(a.ghost_values,ghost_values) do dest, src
      dest .= src
    end
  end
  a
end

@inline function Base.materialize!(a::PVector,b::DistributedBroadcasted)
  map_parts(a.owned_values,b.owned_values) do dest, x
    Base.materialize!(dest,x)
  end
  if b.ghost_values !== nothing
    map_parts(a.ghost_values,b.ghost_values) do dest, x
      Base.materialize!(dest,x)
    end
  end
  a
end

function Base.broadcasted(
  f,
  args::Union{PVector,DistributedBroadcasted}...)

  a1 = first(args)
  @check all(ai->oids_are_equal(ai.rows,a1.rows),args)
  owned_values_in = map(arg->arg.owned_values,args)
  owned_values = map_parts((largs...)->Base.broadcasted(f,largs...),owned_values_in...)
  if all(ai->ai.rows===a1.rows,args) && !any(ai->ai.ghost_values===nothing,args)
    ghost_values_in = map(arg->arg.ghost_values,args)
    ghost_values = map_parts((largs...)->Base.broadcasted(f,largs...),ghost_values_in...)
  else
    ghost_values = nothing
  end
  DistributedBroadcasted(owned_values,ghost_values,a1.rows)
end

function Base.broadcasted(
  f,
  a::Number,
  b::Union{PVector,DistributedBroadcasted})

  owned_values = map_parts(b->Base.broadcasted(f,a,b),b.owned_values)
  if b.ghost_values !== nothing
    ghost_values = map_parts(b->Base.broadcasted(f,a,b),b.ghost_values)
  else
    ghost_values = nothing
  end
  DistributedBroadcasted(owned_values,ghost_values,b.rows)
end

function Base.broadcasted(
  f,
  a::Union{PVector,DistributedBroadcasted},
  b::Number)

  owned_values = map_parts(a->Base.broadcasted(f,a,b),a.owned_values)
  if a.ghost_values !== nothing
    ghost_values = map_parts(a->Base.broadcasted(f,a,b),a.ghost_values)
  else
    ghost_values = nothing
  end
  DistributedBroadcasted(owned_values,ghost_values,a.rows)
end

function LinearAlgebra.norm(a::PVector,p::Real=2)
  contibs = map_parts(a.owned_values) do oid_to_value
    norm(oid_to_value,p)^p
  end
  reduce(+,contibs;init=zero(eltype(contibs)))^(1/p)
end

function PVector{T}(
  ::UndefInitializer,
  rows::PRange) where T

  values = map_parts(rows.lids) do lids
    nlids = num_lids(lids)
    Vector{T}(undef,nlids)
  end
  PVector(values,rows)
end

function PVector(v::Number, rows::PRange)
  a = PVector{typeof(v)}(undef,rows)
  fill!(a,v)
  a
end

# If one chooses ids=:global the ids are translated in-place in I.
function PVector(
  init,
  I::PData{<:AbstractArray{<:Integer}},
  V::PData{<:AbstractArray},
  rows::PRange;
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

  PVector(values,rows)
end

function PVector(
  I::PData{<:AbstractArray{<:Integer}},
  V::PData{<:AbstractArray{T}},
  rows;
  ids::Symbol) where T
  PVector(n->zeros(T,n),I,V,rows;ids=ids)
end

function PVector(
  init,
  I::PData{<:AbstractArray{<:Integer}},
  V::PData{<:AbstractArray},
  n::Integer;
  ids::Symbol)

  @assert ids == :global
  parts = get_part_ids(I)
  rows = PRange(parts,n)
  add_gid!(rows,I)
  PVector(init,I,V,rows;ids=ids)
end

function Base.:*(a::Number,b::PVector)
  values = map_parts(b.values) do values
    a*values
  end
  PVector(values,b.rows)
end

function Base.:*(b::PVector,a::Number)
  a*b
end

for op in (:+,:-)
  @eval begin

    function Base.$op(a::PVector)
      values = map_parts(a.values) do a
        $op(a)
      end
      PVector(values,a.rows)
    end

    function Base.$op(a::PVector,b::PVector)
      $op.(a,b)
    end

  end
end

function Base.fill!(a::PVector,v)
  map_parts(a.values) do lid_to_value
    fill!(lid_to_value,v)
  end
  a
end

function Base.reduce(op,a::PVector;init)
  b = map_parts(a.values,a.rows.lids) do values,lids
    owned_values = view(values,lids.oid_to_lid)
    reduce(op,owned_values,init=init)
  end
  reduce(op,b,init=init)
end

function Base.sum(a::PVector)
  reduce(+,a,init=zero(eltype(a)))
end

function LinearAlgebra.dot(a::PVector,b::PVector)
  c = map_parts(a.values,b.values,a.rows.lids,b.rows.lids) do a,b,alids,blids
    a_owned = view(a,alids.oid_to_lid)
    b_owned = view(b,blids.oid_to_lid)
    dot(a_owned,b_owned)
  end
  sum(c)
end

function local_view(a::PVector)
  a.values
end

function global_view(a::PVector)
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
  a::PVector,
  t0::PData=_empty_tasks(a.rows.exchanger.parts_rcv))
  async_exchange!(a.values,a.rows.exchanger,t0)
end

# Non-blocking assembly
# TODO reduce op as first argument and init kwargument
function async_assemble!(
  a::PVector,
  t0::PData=_empty_tasks(a.rows.exchanger.parts_rcv);
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

struct PSparseMatrix{T,A,B,C,D} <: AbstractMatrix{T}
  values::A
  rows::B
  cols::C
  exchanger::D
  function PSparseMatrix(
    values::PData{<:AbstractSparseMatrix{T}},
    rows::PRange,
    cols::PRange,
    exchanger::Exchanger) where T

    A = typeof(values)
    B = typeof(rows)
    C = typeof(cols)
    D = typeof(exchanger)
    new{T,A,B,C,D}(values,rows,cols,exchanger)
  end
end

function PSparseMatrix(
  values::PData{<:AbstractSparseMatrix{T}},
  rows::PRange,
  cols::PRange) where T

  if rows.ghost
    exchanger = matrix_exchanger(values,rows.exchanger,rows.lids,cols.lids)
  else
    exchanger = empty_exchanger(rows.lids)
  end
  PSparseMatrix(values,rows,cols,exchanger)
end

function Base.getproperty(x::PSparseMatrix, sym::Symbol)
  if sym == :owned_owned_values
    map_parts(x.values,x.rows.lids,x.cols.lids) do v,r,c
      indices = (r.oid_to_lid,c.oid_to_lid)
      inv_indices = (r.lid_to_ohid,c.lid_to_ohid)
      flag = (1,1)
      SubSparseMatrix(v,indices,inv_indices,flag)
    end
  elseif sym == :owned_ghost_values
    map_parts(x.values,x.rows.lids,x.cols.lids) do v,r,c
      indices = (r.oid_to_lid,c.hid_to_lid)
      inv_indices = (r.lid_to_ohid,c.lid_to_ohid)
      flag = (1,-1)
      SubSparseMatrix(v,indices,inv_indices,flag)
    end
  elseif sym == :ghost_owned_values
    map_parts(x.values,x.rows.lids,x.cols.lids) do v,r,c
      indices = (r.hid_to_lid,c.oid_to_lid)
      inv_indices = (r.lid_to_ohid,c.lid_to_ohid)
      flag = (-1,1)
      SubSparseMatrix(v,indices,inv_indices,flag)
    end
  elseif sym == :ghost_ghost_values
    map_parts(x.values,x.rows.lids,x.cols.lids) do v,r,c
      indices = (r.hid_to_lid,c.hid_to_lid)
      inv_indices = (r.lid_to_ohid,c.lid_to_ohid)
      flag = (-1,-1)
      SubSparseMatrix(v,indices,inv_indices,flag)
    end
  else
    getfield(x, sym)
  end
end

function Base.propertynames(x::PSparseMatrix, private=false)
  (
    fieldnames(typeof(x))...,
    :owned_owned_values,
    :owned_ghost_values,
    :ghost_owned_values,
    :ghost_ghost_values)
end

Base.size(a::PSparseMatrix) = (num_gids(a.rows),num_gids(a.cols))
Base.axes(a::PSparseMatrix) = (a.rows,a.cols)
Base.IndexStyle(::Type{<:PSparseMatrix}) = IndexCartesian()
function Base.getindex(a::PSparseMatrix,gi::Integer,gj::Integer)
  #This should not be used in practice
  @notimplemented
end

# If one chooses ids=:global the ids are translated in-place in I and J
function PSparseMatrix(
  init,
  I::PData{<:AbstractArray{<:Integer}},
  J::PData{<:AbstractArray{<:Integer}},
  V::PData{<:AbstractArray},
  rows::PRange,
  cols::PRange;
  ids::Symbol)

  @assert ids in (:global,:local)
  if ids == :global
    to_lid!(I,rows)
    to_lid!(J,cols)
  end

  values = map_parts(I,J,V,rows.lids,cols.lids) do I,J,V,rlids,clids
    init(I,J,V,num_lids(rlids),num_lids(clids))
  end

  PSparseMatrix(values,rows,cols)
end

function PSparseMatrix(
  init,
  I::PData{<:AbstractArray{<:Integer}},
  J::PData{<:AbstractArray{<:Integer}},
  V::PData{<:AbstractArray},
  nrows::Integer,
  ncols::Integer;
  ids::Symbol)

  @assert ids == :global
  parts = get_part_ids(I)
  rows = PRange(parts,nrows)
  cols = PRange(parts,ncols)
  add_gid!(rows,I)
  add_gid!(cols,J)
  PSparseMatrix(init,I,J,V,rows,cols;ids=ids)
end

function PSparseMatrix(
  I::PData{<:AbstractArray{<:Integer}},
  J::PData{<:AbstractArray{<:Integer}},
  V::PData{<:AbstractArray},
  rows,
  cols;
  ids::Symbol)

  PSparseMatrix(sparse,I,J,V,rows,cols;ids=ids)
end

function LinearAlgebra.mul!(
  c::PVector,
  a::PSparseMatrix,
  b::PVector,
  α::Number,
  β::Number)

  @check oids_are_equal(c.rows,a.rows)
  @check oids_are_equal(a.cols,b.rows)
  @check hids_are_equal(a.cols,b.rows)

  # Start the exchange
  t = async_exchange!(b)

  # Meanwhile, process the owned blocks
  map_parts(c.owned_values,a.owned_owned_values,b.owned_values) do co,aoo,bo
    if β != 1
        β != 0 ? rmul!(co, β) : fill!(co,zero(eltype(co)))
    end
    mul!(co,aoo,bo,α,1)
  end

  # Wait for the exchange to finish and process the ghost block
  map_parts(t,c.owned_values,a.owned_ghost_values,b.ghost_values) do t,co,aoh,bh
    wait(schedule(t))
    mul!(co,aoh,bh,α,1)
  end

  c
end

struct SubSparseMatrix{T,A,B,C} <: AbstractMatrix{T}
  parent::A
  indices::B
  inv_indices::C
  flag::Tuple{Int,Int}
  function SubSparseMatrix(
    parent::AbstractSparseMatrix{T},
    indices::Tuple,
    inv_indices::Tuple,
    flag::Tuple{Int,Int}) where T

    A = typeof(parent)
    B = typeof(indices)
    C = typeof(inv_indices)
    new{T,A,B,C}(parent,indices,inv_indices,flag)
  end
end

Base.size(a::SubSparseMatrix) = map(length,a.indices)
Base.IndexStyle(::Type{<:SubSparseMatrix}) = IndexCartesian()
function Base.getindex(a::SubSparseMatrix,i::Integer,j::Integer)
  I = a.indices[1][i]
  J = a.indices[2][j]
  a.parent[I,J]
end

function LinearAlgebra.mul!(
  C::AbstractVector,
  A::SubSparseMatrix,
  B::AbstractVector,
  α::Number,
  β::Number)

  @notimplemented
end

function LinearAlgebra.mul!(
  C::AbstractVector,
  A::SubSparseMatrix{T,<:SparseArrays.AbstractSparseMatrixCSC} where T,
  B::AbstractVector,
  α::Number,
  β::Number)

  size(A, 2) == size(B, 1) || throw(DimensionMismatch())
  size(A, 1) == size(C, 1) || throw(DimensionMismatch())
  size(B, 2) == size(C, 2) || throw(DimensionMismatch())
  if β != 1
      β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
  end
  rows, cols = A.indices
  invrows, invcols = A.inv_indices
  rflag, cflag = A.flag
  Ap = A.parent
  nzv = nonzeros(Ap)
  rv = rowvals(Ap)
  colptrs = SparseArrays.getcolptr(Ap)
  for (j,J) in enumerate(cols)
      αxj = B[j] * α
      for p = colptrs[J]:(colptrs[J+1]-1)
        I = rv[p]
        i = invrows[I]*rflag
        if i>0
          C[i] += nzv[p]*αxj
        end
      end
  end
  C
end

function local_view(a::PSparseMatrix)
  a.values
end

function global_view(a::PSparseMatrix)
  map_parts(a.values,a.rows.lids,a.cols.lids) do values,rlids,clids
    GlobalView(values,(rlids.gid_to_lid,clids.gid_to_lid),(rlids.ngids,clids.ngids))
  end
end

function matrix_exchanger(values,row_exchanger,row_lids,col_lids)

  part = get_part_ids(row_lids)
  parts_rcv = row_exchanger.parts_rcv
  parts_snd = row_exchanger.parts_snd

  function setup_rcv(part,parts_rcv,row_lids,col_lids,values)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_rcv) ))
    ptrs = zeros(Int32,length(parts_rcv)+1)
    for (li,lj,v) in nziterator(values)
      owner = row_lids.lid_to_part[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)
    k_rcv_data = zeros(Int,ptrs[end]-1)
    gi_rcv_data = zeros(Int,ptrs[end]-1)
    gj_rcv_data = zeros(Int,ptrs[end]-1)
    for (k,(li,lj,v)) in enumerate(nziterator(values))
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
  
  k_rcv, gi_rcv, gj_rcv = map_parts(setup_rcv,part,parts_rcv,row_lids,col_lids,values)

  gi_snd = exchange(gi_rcv,parts_snd,parts_rcv)
  gj_snd = exchange(gj_rcv,parts_snd,parts_rcv)

  function setup_snd(part,row_lids,col_lids,gi_snd,gj_snd,values)
    ptrs = gi_snd.ptrs
    k_snd_data = zeros(Int,ptrs[end]-1)
    for p in 1:length(gi_snd.data)
      gi = gi_snd.data[p]
      gj = gj_snd.data[p]
      li = row_lids.gid_to_lid[gi]
      lj = col_lids.gid_to_lid[gj]
      k = nzindex(values,li,lj)
      @check k > 0 "The sparsity patern of the ghost layer is inconsistent"
      k_snd_data[p] = k
    end
    k_snd = Table(k_snd_data,ptrs)
    k_snd
  end

  k_snd = map_parts(setup_snd,part,row_lids,col_lids,gi_snd,gj_snd,values)

  Exchanger(parts_rcv,parts_snd,k_rcv,k_snd)
end

function nzindex(A::AbstractSparseMatrix, i0::Integer, i1::Integer)
  @notimplemented
end

function nzindex(A::SparseArrays.AbstractSparseMatrixCSC, i0::Integer, i1::Integer)
    if !(1 <= i0 <= size(A, 1) && 1 <= i1 <= size(A, 2)); throw(BoundsError()); end
    ptrs = SparseArrays.getcolptr(A)
    r1 = Int(ptrs[i1])
    r2 = Int(ptrs[i1+1]-1)
    (r1 > r2) && return -1
    r1 = searchsortedfirst(rowvals(A), i0, r1, r2, Base.Order.Forward)
    ((r1 > r2) || (rowvals(A)[r1] != i0)) ? -1 : r1
end

nziterator(a::AbstractSparseMatrix) = @notimplemented

nziterator(a::SparseArrays.AbstractSparseMatrixCSC) = NZIteratorCSC(a)

struct NZIteratorCSC{A}
  matrix::A
end

Base.length(a::NZIteratorCSC) = nnz(a.matrix)
Base.eltype(::Type{<:NZIteratorCSC{A}}) where A = Tuple{Int,Int,eltype(A)}
Base.eltype(::T) where T <: NZIteratorCSC = eltype(T)
Base.IteratorSize(::Type{<:NZIteratorCSC}) = Base.HasLength()
Base.IteratorEltype(::Type{<:NZIteratorCSC}) = Base.HasEltype()

@inline function Base.iterate(a::NZIteratorCSC)
  if nnz(a.matrix) == 0
    return nothing
  end
  col = 0
  ptrs = SparseArrays.getcolptr(a.matrix)
  knext = nothing
  while knext === nothing
    col += 1
    ks = ptrs[col]:(ptrs[col+1]-1)
    knext = iterate(ks)
  end
  k, kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = nonzeros(a.matrix)[k]
  (i,j,v), (col,kstate)
end

@inline function Base.iterate(a::NZIteratorCSC,state)
  ptrs = SparseArrays.getcolptr(a.matrix)
  col, kstate = state
  ks = ptrs[col]:(ptrs[col+1]-1)
  knext = iterate(ks,kstate)
  if knext === nothing
    while knext === nothing
      if col == size(a.matrix,2)
        return nothing
      end
      col += 1
      ks = ptrs[col]:(ptrs[col+1]-1)
      knext = iterate(ks)
    end
  end
  k, kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = nonzeros(a.matrix)[k]
  (i,j,v), (col,kstate)
end

# Non-blocking exchange
function async_exchange!(
  a::PSparseMatrix,
  t0::PData=_empty_tasks(a.exchanger.parts_rcv))
  nzval = map_parts(nonzeros,a.values)
  async_exchange!(nzval,a.exchanger,t0)
end

# Non-blocking assembly
function async_assemble!(
  a::PSparseMatrix,
  t0::PData=_empty_tasks(a.exchanger.parts_rcv);
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

function async_assemble!(
  I::PData{<:AbstractVector{<:Integer}},
  J::PData{<:AbstractVector{<:Integer}},
  V::PData{<:AbstractVector},
  rows::PRange,
  t0::PData=_empty_tasks(rows.exchanger.parts_rcv);
  reduce_op=+)

  map_parts(wait∘schedule,t0)

  part = get_part_ids(rows.lids)
  parts_rcv = rows.exchanger.parts_rcv
  parts_snd = rows.exchanger.parts_snd

  to_lid!(I,rows)
  coo_values = map_parts(tuple,I,J,V)

  function setup_rcv(part,parts_rcv,row_lids,coo_values)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_rcv) ))
    ptrs = zeros(Int32,length(parts_rcv)+1)
    k_li, k_gj, k_v = coo_values
    for k in 1:length(k_li)
      li = k_li[k]
      owner = row_lids.lid_to_part[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)
    gi_rcv_data = zeros(Int,ptrs[end]-1)
    gj_rcv_data = zeros(Int,ptrs[end]-1)
    v_rcv_data = zeros(eltype(k_v),ptrs[end]-1)
    for k in 1:length(k_li)
      li = k_li[k]
      owner = row_lids.lid_to_part[li]
      if owner != part
        gi = row_lids.lid_to_gid[li]
        gj = k_gj[k]
        v = k_v[k]
        p = ptrs[owner_to_i[owner]]
        gi_rcv_data[p] = gi
        gj_rcv_data[p] = gj
        v_rcv_data[p] = v
        k_v[k] = zero(v)
        ptrs[owner_to_i[owner]] += 1
      end
    end
    rewind_ptrs!(ptrs)
    gi_rcv = Table(gi_rcv_data,ptrs)
    gj_rcv = Table(gj_rcv_data,ptrs)
    v_rcv = Table(v_rcv_data,ptrs)
    gi_rcv, gj_rcv, v_rcv 
  end
  
  gi_rcv, gj_rcv, v_rcv = map_parts(setup_rcv,part,parts_rcv,rows.lids,coo_values)

  gi_snd, t1 = async_exchange(gi_rcv,parts_snd,parts_rcv)
  gj_snd, t2 = async_exchange(gj_rcv,parts_snd,parts_rcv)
  v_snd, t3 = async_exchange(v_rcv,parts_snd,parts_rcv)

  function setup_snd(t1,t2,t3,part,row_lids,gi_snd,gj_snd,v_snd,coo_values)
    @task begin
      wait(schedule(t1))
      wait(schedule(t2))
      wait(schedule(t3))
      k_li, k_gj, k_v = coo_values
      to_gid!(k_li,row_lids)
      ptrs = gi_snd.ptrs
      for p in 1:length(gi_snd.data)
        gi = gi_snd.data[p]
        gj = gj_snd.data[p]
        v = v_snd.data[p]
        push!(k_li,gi)
        push!(k_gj,gj)
        push!(k_v,v)
      end
    end
  end

  t4 = map_parts(setup_snd,t1,t2,t3,part,rows.lids,gi_snd,gj_snd,v_snd,coo_values)

  t4
end

function Base.:*(a::Number,b::PSparseMatrix)
  values = map_parts(b.values) do values
    a*values
  end
  PSparseMatrix(values,b.rows,b.cols,b.exchanger)
end

function Base.:*(b::PSparseMatrix,a::Number)
  a*b
end

function Base.:*(a::PSparseMatrix{Ta},b::PVector{Tb}) where {Ta,Tb}
  T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
  c = PVector{T}(undef,a.rows)
  mul!(c,a,b)
  c
end

for op in (:+,:-)
  @eval begin

    function Base.$op(a::PSparseMatrix)
      values = map_parts(a.values) do a
        $op(a)
      end
      PSparseMatrix(values,a.rows,a.cols,a.exchanger)
    end

  end
end

# This structs implements the same methods as the Identity preconditioner
# in the IterativeSolvers package.
struct Jacobi{T,A,B,C}
  diaginv::A
  rows::B
  cols::C
  function Jacobi(
    diaginv::PData{<:AbstractVector{T}},
    rows::PRange,
    cols::PRange) where T

    A = typeof(diaginv)
    B = typeof(rows)
    C = typeof(cols)
    new{T,A,B,C}(diaginv,rows,cols)
  end
end

function Jacobi(a::PSparseMatrix)
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
  c::PVector,
  a::Jacobi,
  b::PVector)

  @check oids_are_equal(c.rows,a.cols)
  @check oids_are_equal(b.rows,a.rows)
  map_parts(c.owned_values,a.diaginv,b.owned_values) do c,diaginv,b
    for oid in 1:length(b)
      c[oid] = diaginv[oid]*b[oid]
    end
  end
  c
end

function LinearAlgebra.ldiv!(
  a::Jacobi,
  b::PVector)
  ldiv!(b,a,b)
end

function Base.:\(a::Jacobi{Ta},b::PVector{Tb}) where {Ta,Tb}
  T = typeof(zero(Ta)/one(Tb)+zero(Ta)/one(Tb))
  c = PVector{T}(undef,a.cols)
  ldiv!(c,a,b)
  c
end

# Misc functions that could be removed if IterativeSolvers was implemented in terms
# of axes(A,d) instead of size(A,d)
function IterativeSolvers.zerox(A::PSparseMatrix,b::PVector)
  T = IterativeSolvers.Adivtype(A, b)
  x = similar(b, T, axes(A, 2))
  fill!(x, zero(T))
  return x
end

