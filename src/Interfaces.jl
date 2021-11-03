
"""
    abstract type AbstractBackend end

Abstract type representing a message passing model used to run
a distributed computation.

At this moment, these specializations are available
- [`SequentialBackend`](@ref)
- [`MPIBackend`](@ref)
"""
abstract type AbstractBackend end

# Should return a AbstractPData{Int}
"""
    get_part_ids(b::AbstractBackend,nparts::Integer) -> AbstractPData{Int}
    get_part_ids(b::AbstractBackend,nparts::Tuple) -> AbstractPData{Int}

Return a partitioned data consisting of `nparts`,
where each part stores its part id. The concrete type of
the returned object depends on the back-end `b`. If `nparts` is a tuple
a Cartesian partition is considered with `nparts[i]` parts in direction `i`.
"""
function get_part_ids(b::AbstractBackend,nparts::Integer)
  @abstractmethod
end

function get_part_ids(b::AbstractBackend,nparts::Tuple)
  get_part_ids(b,prod(nparts))
end

# This can be overwritten to add a finally clause
function prun(driver::Function,b::AbstractBackend,nparts)
  part = get_part_ids(b,nparts)
  driver(part)
end

# Data distributed in parts of type T
"""
    abstract type AbstractPData{T,N} end

Abstract type representing a data partitioned into parts of type `T` where `N`
is the tensor order of the partition. I.e., `N==1` for linear partitions and
`N>1` for Cartesian partitions.

At this moment, these specializations are available
- [`SequentialData`](@ref)
- [`MPIData`](@ref)
"""
abstract type AbstractPData{T,N} end

"""
    size(a::AbstractPData) -> Tuple

Number of parts per direction in the partitioned data `a`.
"""
Base.size(a::AbstractPData) = @abstractmethod

"""
    length(a::AbstractPData) -> Int

Same as `num_parts(a)`.
"""
Base.length(a::AbstractPData) = prod(size(a))

"""
    num_parts(a::AbstractPData) -> Int

Total number of parts in `a`.
"""
num_parts(a::AbstractPData) = length(a)

"""
    get_backend(a::AbstractPData) -> AbstractBackend

Get the back-end associated with `a`.
"""
get_backend(a::AbstractPData) = @abstractmethod

Base.iterate(a::AbstractPData)  = @abstractmethod

Base.iterate(a::AbstractPData,state)  = @abstractmethod

get_part_ids(a::AbstractPData) = get_part_ids(get_backend(a),size(a))

map_parts(task,a::AbstractPData...) = @abstractmethod

i_am_main(::AbstractPData) = @abstractmethod

Base.eltype(a::AbstractPData{T}) where T = T
Base.eltype(::Type{<:AbstractPData{T}}) where T = T

Base.ndims(a::AbstractPData{T,N}) where {T,N} = N
Base.ndims(::Type{<:AbstractPData{T,N}}) where {T,N} = N

Base.copy(a::AbstractPData) = map_parts(copy,a)

#function map_parts(task,a...)
#  map_parts(task,map(AbstractPData,a)...)
#end
#
#AbstractPData(a::AbstractPData) = a

const MAIN = 1

function map_main(f,args::AbstractPData...)
  parts = get_part_ids(first(args))
  map_parts(parts,args...) do part,args...
    if part == MAIN
      f(args...)
    else
      nothing
    end
  end
end

# import the main part to the main scope
# in MPI this will broadcast the main part to all procs
get_main_part(a::AbstractPData) = get_part(a,MAIN)

# This one is safe to use only when all parts contain the same value, e.g. the result of a gather_all call.
get_part(a::AbstractPData) = @abstractmethod

get_part(a::AbstractPData,part::Integer) = @abstractmethod

# rcv can contain vectors or tables
gather!(rcv::AbstractPData,snd::AbstractPData) = @abstractmethod

gather_all!(rcv::AbstractPData,snd::AbstractPData) = @abstractmethod

function allocate_gather(snd::AbstractPData)
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

function allocate_gather(snd::AbstractPData{<:AbstractVector})
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

function gather(snd::AbstractPData)
  rcv = allocate_gather(snd)
  gather!(rcv,snd)
  rcv
end

function allocate_gather_all(snd::AbstractPData)
  np = num_parts(snd)
  rcv = map_parts(snd) do snd
    T = typeof(snd)
    Vector{T}(undef,np)
  end
  rcv
end

function allocate_gather_all(snd::AbstractPData{<:AbstractVector})
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

function gather_all(snd::AbstractPData)
  rcv = allocate_gather_all(snd)
  gather_all!(rcv,snd)
  rcv
end

# The back-end need to support these cases:
# i.e. AbstractPData{AbstractVector{<:Number}} and AbstractPData{AbstractVector{<:AbstractVector{<:Number}}}
function scatter(snd::AbstractPData)
  @abstractmethod
end

# AKA broadcast
function emit(snd::AbstractPData)
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

function reduce_main(op,snd::AbstractPData;init)
  a = gather(snd)
  map_parts(i->reduce(op,i;init=init),a)
end

function reduce_all(args...;kwargs...)
  b = reduce_main(args...;kwargs...)
  emit(b)
end

function Base.reduce(op,a::AbstractPData;init)
  b = reduce_main(op,a;init=init)
  get_main_part(b)
end

function Base.sum(a::AbstractPData)
  reduce(+,a,init=zero(eltype(a)))
end

# inclusive prefix reduction
function iscan(op,a::AbstractPData;init)
  b = iscan_main(op,a,init=init)
  scatter(b)
end

function iscan(op,::typeof(reduce),a::AbstractPData;init)
  b,n = iscan_main(op,reduce,a,init=init)
  scatter(b), get_main_part(n)
end

function iscan_all(op,a::AbstractPData;init)
  b = iscan_main(op,a,init=init)
  emit(b)
end

function iscan_all(op,::typeof(reduce),a::AbstractPData;init)
  b,n = iscan_main(op,reduce,a,init=init)
  emit(b), get_main_part(n)
end

function iscan_main(op,a;init)
  b = gather(a)
  map_parts(b) do b
    _iscan_local!(op,b,init)
  end
  b
end

function iscan_main(op,::typeof(reduce),a;init)
  b = gather(a)
  n = map_parts(b) do b
    reduce(op,b,init=init)
  end
  map_parts(b) do b
    _iscan_local!(op,b,init)
  end
  b,n
end

function _iscan_local!(op,b,init)
  if length(b)!=0
    b[1] = op(init,b[1])
  end
  @inbounds for i in 1:(length(b)-1)
    b[i+1] = op(b[i],b[i+1])
  end
  b
end

# exclusive prefix reduction
function xscan(op,a::AbstractPData;init)
  b = xscan_main(op,a,init=init)
  scatter(b)
end

function xscan(op,::typeof(reduce),a::AbstractPData;init)
  b,n = xscan_main(op,reduce,a,init=init)
  scatter(b), get_main_part(n)
end

function xscan_all(op,a::AbstractPData;init)
  b = xscan_main(op,a,init=init)
  emit(b)
end

function xscan_all(op,::typeof(reduce),a::AbstractPData;init)
  b,n = xscan_main(op,reduce,a,init=init)
  emit(b), get_main_part(n)
end

function xscan_main(op,a::AbstractPData;init)
  b = gather(a)
  map_parts(b) do b
    _xscan_local!(op,b,init)
  end
  b
end

function xscan_main(op,::typeof(reduce),a::AbstractPData;init)
  b = gather(a)
  n = map_parts(b) do b
    reduce(op,b,init=init)
  end
  map_parts(b) do b
    _xscan_local!(op,b,init)
  end
  b,n
end

function _xscan_local!(op,b,init)
  @inbounds for i in (length(b)-1):-1:1
    b[i+1] = b[i]
  end
  if length(b) != 0
    b[1] = init
  end
  @inbounds for i in 1:(length(b)-1)
    b[i+1] = op(b[i],b[i+1])
  end
end

# TODO improve the mechanism for waiting
# Non-blocking in-place exchange
# In this version, sending a number per part is enough
# We have another version below to send a vector of numbers per part (compressed in a Table)
# Starts a non-blocking exchange. It returns a AbstractPData of Julia Tasks. Calling schedule and wait on these
# tasks will wait until the exchange is done in the corresponding part
# (i.e., at this point it is save to read/write the buffers again).
function async_exchange!(
  data_rcv::AbstractPData,
  data_snd::AbstractPData,
  parts_rcv::AbstractPData,
  parts_snd::AbstractPData,
  t_in::AbstractPData)

  @abstractmethod
end

function async_exchange!(
  data_rcv::AbstractPData,
  data_snd::AbstractPData,
  parts_rcv::AbstractPData,
  parts_snd::AbstractPData)

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
  data_snd::AbstractPData,
  parts_rcv::AbstractPData,
  parts_snd::AbstractPData,
  t_in::AbstractPData=_empty_tasks(parts_rcv))

  data_rcv = map_parts(data_snd,parts_rcv) do data_snd, parts_rcv
    similar(data_snd,eltype(data_snd),length(parts_rcv))
  end

  t_out = async_exchange!(data_rcv,data_snd,parts_rcv,parts_snd,t_in)

  data_rcv, t_out
end

# Non-blocking in-place exchange variable length (compressed in a Table)
function async_exchange!(
  data_rcv::AbstractPData{<:Table},
  data_snd::AbstractPData{<:Table},
  parts_rcv::AbstractPData,
  parts_snd::AbstractPData,
  t_in::AbstractPData)

  @abstractmethod
end

# Non-blocking allocating exchange variable length (compressed in a Table)
function async_exchange(
  data_snd::AbstractPData{<:Table},
  parts_rcv::AbstractPData,
  parts_snd::AbstractPData,
  t_in::AbstractPData)

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

# Discover snd parts from rcv assuming that snd is a subset of neighbors
# Assumes that neighbors_snd is a symmetric communication graph
# if neighbors_rcv not provided
function discover_parts_snd(
  parts_rcv::AbstractPData,
  neighbors_snd::AbstractPData,
  neighbors_rcv::AbstractPData=neighbors_snd)
  @assert num_parts(parts_rcv) == num_parts(neighbors_rcv)

  parts = get_part_ids(parts_rcv)

  # Tell the neighbors whether I want to receive data from them
  data_rcv = map_parts(parts,neighbors_rcv,parts_rcv) do part, neighbors_rcv, parts_rcv
    dict_snd = Dict(( n=>Int32(-1) for n in neighbors_rcv))
    for i in parts_rcv
      dict_snd[i] = part
    end
    [ dict_snd[n] for n in neighbors_rcv ]
  end
  data_snd = exchange(data_rcv,neighbors_snd,neighbors_rcv)

  # build parts_snd
  parts_snd = map_parts(data_snd) do data_snd
    k = findall(j->j>0,data_snd)
    data_snd[k]
  end

  parts_snd
end

function _warn_message_on_main_task_discover_parts_snd(data::AbstractPData)
   map_main(data) do data
        warn_msg="""
                 [PartitionedArrays.jl] Warning: Using a non-scalable implementation
                 to discover reciprocal parts in sparse communication kernel among nearest
                 neighbours. This might cause trouble when running the code at medium/large scales.
                 You can avoid this using the Exchanger constructor with a superset of
                 the actual receivers/senders
                 """
        @warn warn_msg
	      Base.show_backtrace(stderr,backtrace())
        println(stderr,"")
   end
end

# If neighbors not provided, we need to gather in main
function discover_parts_snd(parts_rcv::AbstractPData)
  _warn_message_on_main_task_discover_parts_snd(parts_rcv)
  parts_rcv_main = gather(parts_rcv)
  parts_snd_main = map_parts(_parts_rcv_to_parts_snd,parts_rcv_main)
  parts_snd = scatter(parts_snd_main)
  parts_snd
end

# This also works in part != MAIN since it is able to deal
# with an empty table (the result is also an empty table in this case)
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

function discover_parts_snd(parts_rcv::AbstractPData,::Nothing)
  discover_parts_snd(parts_rcv)
end

function discover_parts_snd(parts_rcv::AbstractPData,::Nothing,::Nothing)
  discover_parts_snd(parts_rcv)
end

function discover_parts_snd(parts_rcv::AbstractPData,neighbors::AbstractPData,::Nothing)
  discover_parts_snd(parts_rcv,neighbors)
end

abstract type AbstractIndexSet end

num_lids(a::AbstractIndexSet) = length(a.lid_to_part)
num_oids(a::AbstractIndexSet) = length(a.oid_to_lid)
num_hids(a::AbstractIndexSet) = length(a.hid_to_lid)
get_part(a::AbstractIndexSet) = a.part
get_lid_to_gid(a::AbstractIndexSet) = a.lid_to_gid
get_lid_to_part(a::AbstractIndexSet) = a.lid_to_part
get_oid_to_lid(a::AbstractIndexSet) = a.oid_to_lid
get_hid_to_lid(a::AbstractIndexSet) = a.hid_to_lid
get_lid_to_ohid(a::AbstractIndexSet) = a.lid_to_ohid
get_gid_to_lid(a::AbstractIndexSet) = a.gid_to_lid

function add_gid!(a::AbstractIndexSet,gid::Integer,part::Integer)
  if (part != a.part) && (!haskey(a.gid_to_lid,gid))
    _add_gid_ghost!(a,gid,part)
  end
  a
end

function add_gid!(gid_to_part::AbstractArray,a::AbstractIndexSet,gid::Integer)
  if !haskey(a.gid_to_lid,gid)
    part = gid_to_part[gid]
    _add_gid_ghost!(a,gid,part)
  end
  a
end

#TODO use resize + setindex instead of push! when possible
@inline function _add_gid_ghost!(a,gid,part)
  lid = Int32(num_lids(a)+1)
  hid = Int32(num_hids(a)+1)
  push!(a.lid_to_gid,gid)
  push!(a.lid_to_part,part)
  push!(a.hid_to_lid,lid)
  push!(a.lid_to_ohid,-hid)
  a.gid_to_lid[gid] = lid
end

function add_gids!(
  a::AbstractIndexSet,
  i_to_gid::AbstractVector{<:Integer},
  i_to_part::AbstractVector{<:Integer})

  for i in 1:length(i_to_gid)
    gid = i_to_gid[i]
    part = i_to_part[i]
    add_gid!(a,gid,part)
  end
  a
end

function add_gids!(
  gid_to_part::AbstractArray,
  a::AbstractIndexSet,
  gids::AbstractVector{<:Integer})

  for gid in gids
    add_gid!(gid_to_part,a,gid)
  end
  a
end

function to_lids!(ids::AbstractArray{<:Integer},a::AbstractIndexSet)
  for i in eachindex(ids)
    gid = ids[i]
    lid = a.gid_to_lid[gid]
    ids[i] = lid
  end
  ids
end

function to_gids!(ids::AbstractArray{<:Integer},a::AbstractIndexSet)
  for i in eachindex(ids)
    lid = ids[i]
    gid = a.lid_to_gid[lid]
    ids[i] = gid
  end
  ids
end

function oids_are_equal(a::AbstractIndexSet,b::AbstractIndexSet)
  view(a.lid_to_gid,a.oid_to_lid) == view(b.lid_to_gid,b.oid_to_lid)
end

function hids_are_equal(a::AbstractIndexSet,b::AbstractIndexSet)
  view(a.lid_to_gid,a.hid_to_lid) == view(b.lid_to_gid,b.hid_to_lid)
end

function lids_are_equal(a::AbstractIndexSet,b::AbstractIndexSet)
  a.lid_to_gid == b.lid_to_gid
end

function find_lid_map(a::AbstractIndexSet,b::AbstractIndexSet)
  alid_to_blid = fill(Int32(-1),num_lids(a))
  for blid in 1:num_lids(b)
    gid = b.lid_to_gid[blid]
    alid = a.gid_to_lid[gid]
    alid_to_blid[alid] = blid
  end
  alid_to_blid
end

# The given ids are assumed to be a sub-set of the lids
function touched_hids(a::AbstractIndexSet,gids::AbstractVector{<:Integer})
  i = 0
  hid_touched = fill(false,num_hids(a))
  for gid in gids
    lid = a.gid_to_lid[gid]
    ohid = a.lid_to_ohid[lid]
    hid = - ohid
    if ohid < 0 && !hid_touched[hid]
      hid_touched[hid] = true
      i += 1
    end
  end
  i_to_hid = Vector{Int32}(undef,i)
  i = 0
  hid_touched .= false
  for gid in gids
    lid = a.gid_to_lid[gid]
    ohid = a.lid_to_ohid[lid]
    hid = - ohid
    if ohid < 0 && !hid_touched[hid]
      hid_touched[hid] = true
      i += 1
      i_to_hid[i] = hid
    end
  end
  i_to_hid
end

struct Exchanger{B,C}
  parts_rcv::B
  parts_snd::B
  lids_rcv::C
  lids_snd::C
  function Exchanger(
    parts_rcv::AbstractPData{<:AbstractVector{<:Integer}},
    parts_snd::AbstractPData{<:AbstractVector{<:Integer}},
    lids_rcv::AbstractPData{<:Table{<:Integer}},
    lids_snd::AbstractPData{<:Table{<:Integer}})

    B = typeof(parts_rcv)
    C = typeof(lids_rcv)
    new{B,C}(parts_rcv,parts_snd,lids_rcv,lids_snd)
  end
end

function Base.copy(a::Exchanger)
  Exchanger(
    copy(a.parts_rcv),
    copy(a.parts_snd),
    copy(a.lids_rcv),
    copy(a.lids_snd))
end

function Exchanger(
  ids::AbstractPData{<:AbstractIndexSet},
  neighbors_snd=nothing,
  neighbors_rcv=nothing;
  reuse_parts_rcv=false)

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

  if reuse_parts_rcv
    parts_snd = parts_rcv
  else
    parts_snd = discover_parts_snd(parts_rcv,neighbors_snd,neighbors_rcv)
  end

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

function empty_exchanger(a::AbstractPData)
  parts_rcv = map_parts(i->Int32[],a)
  parts_snd = map_parts(i->Int32[],a)
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
  values::AbstractPData{<:AbstractVector{T}},
  exchanger::Exchanger,
  t0::AbstractPData=_empty_tasks(exchanger.parts_rcv)) where T

  async_exchange!(_replace,values,exchanger,t0)
end

function async_exchange!(
  values_rcv::AbstractPData{<:AbstractVector{T}},
  values_snd::AbstractPData{<:AbstractVector{T}},
  exchanger::Exchanger,
  t0::AbstractPData=_empty_tasks(exchanger.parts_rcv)) where T

  async_exchange!(_replace,values_rcv,values_snd,exchanger,t0)
end

_replace(x,y) = y

function async_exchange!(
  combine_op,
  values::AbstractPData{<:AbstractVector{Trcv}},
  exchanger::Exchanger,
  t0::AbstractPData=_empty_tasks(exchanger.parts_rcv)) where {Trcv,Tsnd}

  async_exchange!(combine_op,values,values,exchanger,t0)
end

function async_exchange!(
  combine_op,
  values_rcv::AbstractPData{<:AbstractVector{Trcv}},
  values_snd::AbstractPData{<:AbstractVector{Tsnd}},
  exchanger::Exchanger,
  t0::AbstractPData=_empty_tasks(exchanger.parts_rcv)) where {Trcv,Tsnd}

  # Allocate buffers
  data_rcv = allocate_rcv_buffer(Trcv,exchanger)
  data_snd = allocate_snd_buffer(Tsnd,exchanger)

  # Fill snd buffer
  t1 = map_parts(t0,values_snd,data_snd,exchanger.lids_snd) do t0,values_snd,data_snd,lids_snd
    @task begin
      wait(schedule(t0))
      for p in 1:length(lids_snd.data)
        lid = lids_snd.data[p]
        data_snd.data[p] = values_snd[lid]
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

  # Fill values_rcv from rcv buffer
  # asynchronously
  t3 = map_parts(t2,values_rcv,data_rcv,exchanger.lids_rcv) do t2,values_rcv,data_rcv,lids_rcv
    @task begin
      wait(schedule(t2))
      for p in 1:length(lids_rcv.data)
        lid = lids_rcv.data[p]
        values_rcv[lid] = combine_op(values_rcv[lid],data_rcv.data[p])
      end
    end
  end

  t3
end

function async_exchange!(
  values::AbstractPData{<:Table},
  exchanger::Exchanger,
  t0::AbstractPData=_empty_tasks(exchanger.parts_rcv))

  async_exchange!(_replace,values,exchanger,t0)
end

function async_exchange!(
  combine_op,
  values::AbstractPData{<:Table},
  exchanger::Exchanger,
  t0::AbstractPData=_empty_tasks(exchanger.parts_rcv))

  data, ptrs = map_parts(t->(t.data,t.ptrs),values)
  t_exchanger = _table_exchanger(exchanger,ptrs)
  async_exchange!(combine_op,data,t_exchanger,t0)
end

function async_exchange!(
  combine_op,
  values_rcv::AbstractPData{<:Table},
  values_snd::AbstractPData{<:Table},
  exchanger::Exchanger,
  t0::AbstractPData=_empty_tasks(exchanger.parts_rcv))

  @notimplemented
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

# mutable is needed to correctly implement add_gids!
mutable struct PRange{A,B,C} <: AbstractUnitRange{Int}
  ngids::Int
  partition::A
  exchanger::B
  gid_to_part::C
  ghost::Bool
  function PRange(
    ngids::Integer,
    partition::AbstractPData{<:AbstractIndexSet},
    exchanger::Exchanger,
    gid_to_part::Union{AbstractPData{<:AbstractArray{<:Integer}},Nothing}=nothing,
    ghost::Bool=true)

    A = typeof(partition)
    B = typeof(exchanger)
    C = typeof(gid_to_part)
    new{A,B,C}(
      ngids,
      partition,
      exchanger,
      gid_to_part,
      ghost)
  end
end

function Base.copy(a::PRange)
  PRange(
    copy(a.ngids),
    copy(a.partition),
    copy(a.exchanger),
    a.gid_to_part===nothing ? nothing : copy(a.gid_to_part),
    copy(a.ghost))
end

function PRange(
  ngids::Integer,
  partition::AbstractPData{<:AbstractIndexSet},
  gid_to_part::Union{AbstractPData{<:AbstractArray{<:Integer}},Nothing}=nothing,
  ghost::Bool=true)

  exchanger =  ghost ? Exchanger(partition) : empty_exchanger(partition)
  PRange(ngids,partition,exchanger,gid_to_part,ghost)
end

Base.first(a::PRange) = 1
Base.last(a::PRange) = a.ngids

num_gids(a::PRange) = a.ngids
num_parts(a::PRange) = num_parts(a.partition)

function PRange(parts::AbstractPData{<:Integer},ngids::Integer)
  np = num_parts(parts)
  partition, gid_to_part = map_parts(parts) do part
    oid_to_gid = _oid_to_gid(ngids,np,part)
    noids = length(oid_to_gid)
    part_to_firstgid = _part_to_firstgid(ngids,np)
    firstgid = first(oid_to_gid)
    partition = IndexRange(
      part,
      noids,
      firstgid)
    gid_to_part = LinearGidToPart(ngids,part_to_firstgid)
    partition, gid_to_part
  end
  ghost = false
  PRange(ngids,partition,gid_to_part,ghost)
end

function PRange(
  parts::AbstractPData{<:Integer,1},
  ngids::NTuple{N,<:Integer}) where N
  PRange(parts,prod(ngids))
end

function PRange(parts::AbstractPData{<:Integer},noids::AbstractPData{<:Integer})
  ngids = reduce(+,noids,init=0)
  PRange(parts,ngids,noids)
end

function PRange(
  parts::AbstractPData{<:Integer},
  ngids::Integer,
  noids::AbstractPData{<:Integer})
  part_to_firstgid = xscan_all(+,noids,init=1)
  PRange(parts,ngids,noids,part_to_firstgid)
end

function PRange(
  parts::AbstractPData{<:Integer},
  ngids::Integer,
  noids::AbstractPData{<:Integer},
  part_to_firstgid::AbstractPData{<:AbstractVector{<:Integer}})

  partition, gid_to_part = map_parts(parts,noids,part_to_firstgid) do part,noids,part_to_firstgid
    firstgid = part_to_firstgid[part]
    partition = IndexRange(
      part,
      noids,
      firstgid)
    gid_to_part = LinearGidToPart(ngids,part_to_firstgid)
    partition, gid_to_part
  end
  ghost = false
  PRange(ngids,partition,gid_to_part,ghost)
end

function PRange(
  parts::AbstractPData{<:Integer},
  ngids::Integer,
  noids::AbstractPData{<:Integer},
  firstgid::AbstractPData{<:Integer})

  partition = map_parts(parts,noids,firstgid) do part,noids,firstgid
    IndexRange(
      part,
      noids,
      firstgid)
  end
  ghost = false
  gid_to_part = nothing
  PRange(ngids,partition,gid_to_part,ghost)
end

function PRange(
  parts::AbstractPData{<:Integer},
  ngids::Integer,
  noids::AbstractPData{<:Integer},
  firstgid::AbstractPData{<:Integer},
  hid_to_gid::AbstractPData{Vector{Int}},
  hid_to_part::AbstractPData{Vector{Int32}},
  neighbors_snd=nothing,
  neighbors_rcv=nothing,
  ;
  kwargs...)

  partition = map_parts(
    parts,noids,firstgid,hid_to_gid,hid_to_part) do part,noids,firstgid,hid_to_gid,hid_to_part
    IndexRange(
      part,
      noids,
      firstgid,
      hid_to_gid,
      hid_to_part)
  end
  exchanger = Exchanger(partition,neighbors_snd,neighbors_rcv;kwargs...)
  ghost = true
  gid_to_part = nothing
  PRange(ngids,partition,exchanger,gid_to_part,ghost)
end

function PRange(
  parts::AbstractPData{<:Integer,N},
  ngids::NTuple{N,<:Integer}) where N

  np = size(parts)
  partition, gid_to_part = map_parts(parts) do part
    gids = _oid_to_gid(ngids,np,part)
    lid_to_gid = gids
    lid_to_part = fill(Int32(part),length(gids))
    oid_to_lid = collect(Int32(1):Int32(length(gids)))
    hid_to_lid = collect(Int32(1):Int32(0))
    part_to_firstgid = _part_to_firstgid(ngids,np)
    gid_to_part = CartesianGidToPart(ngids,part_to_firstgid)
    partition = IndexSet(
      part,
      lid_to_gid,
      lid_to_part,
      oid_to_lid,
      hid_to_lid)
    partition, gid_to_part
  end
  ghost = false
  PRange(prod(ngids),partition,gid_to_part,ghost)
end

function touched_hids(
  a::PRange,
  gids::AbstractPData{<:AbstractVector{<:Integer}})

  map_parts(touched_hids,a.partition,gids)
end

function PCartesianIndices(
  parts::AbstractPData{<:Integer,N},
  ngids::NTuple{N,<:Integer}) where N

  np = size(parts)
  lids = map_parts(parts) do part
    cis_parts = CartesianIndices(np)
    p = Tuple(cis_parts[part])
    d_to_odid_to_gdid = map(_oid_to_gid,ngids,np,p)
    CartesianIndices(d_to_odid_to_gdid)
  end
  lids
end

struct WithGhost end
with_ghost = WithGhost()

struct NoGhost end
no_ghost = NoGhost()

function PRange(
  parts::AbstractPData{<:Integer,N},
  ngids::NTuple{N,<:Integer},
  ::WithGhost) where N

  np = size(parts)
  partition, gid_to_part = map_parts(parts) do part
    cp = Tuple(CartesianIndices(np)[part])
    d_to_ldid_to_gdid = map(_lid_to_gid,ngids,np,cp)
    lid_to_gid = _id_tensor_product(Int,d_to_ldid_to_gdid,ngids)
    d_to_nldids = map(length,d_to_ldid_to_gdid)
    lid_to_part = _lid_to_part(d_to_nldids,np,cp)
    oid_to_lid = collect(Int32,findall(lid_to_part .== part))
    hid_to_lid = collect(Int32,findall(lid_to_part .!= part))
    part_to_firstgid = _part_to_firstgid(ngids,np)
    gid_to_part = CartesianGidToPart(ngids,part_to_firstgid)
    partition = IndexSet(
      part,
      lid_to_gid,
      lid_to_part,
      oid_to_lid,
      hid_to_lid)
    partition, gid_to_part
  end
  ghost = true
  exchanger = Exchanger(partition;reuse_parts_rcv=true)
  PRange(prod(ngids),partition,exchanger,gid_to_part,ghost)
end

function PRange(
  parts::AbstractPData{<:Integer,N},
  ngids::NTuple{N,<:Integer},
  ::NoGhost) where N

  PRange(parts,ngids)
end

function PCartesianIndices(
  parts::AbstractPData{<:Integer,N},
  ngids::NTuple{N,<:Integer},
  ::WithGhost) where N

  np = size(parts)
  lids = map_parts(parts) do part
    cis_parts = CartesianIndices(np)
    p = Tuple(cis_parts[part])
    d_to_odid_to_gdid = map(_lid_to_gid,ngids,np,p)
    CartesianIndices(d_to_odid_to_gdid)
  end
  lids
end

function PCartesianIndices(
  parts::AbstractPData{<:Integer,N},
  ngids::NTuple{N,<:Integer},
  ::NoGhost) where N

  PCartesianIndices(parts,ngids)
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
  _id_tensor_product(Int32,d_to_ldid_to_dpart,np)
end

function _id_tensor_product(::Type{T},d_to_dlid_to_gdid::Tuple,d_to_ngdids::Tuple) where T
  d_to_nldids = map(length,d_to_dlid_to_gdid)
  lcis = CartesianIndices(d_to_nldids)
  llis = LinearIndices(d_to_nldids)
  glis = LinearIndices(d_to_ngdids)
  D = length(d_to_ngdids)
  d = ntuple(identity,Val{D}())
  lid_to_gid = zeros(T,length(lcis))
  for lci in lcis
    gci = map(d) do d
      ldid = lci[d]
      gdid = d_to_dlid_to_gdid[d][ldid]
      gdid
    end
    lid = llis[lci]
    lid_to_gid[lid] = glis[CartesianIndex(gci)]
  end
  lid_to_gid
end

function _part_to_firstgid(ngids::Integer,np::Integer)
  [first(_oid_to_gid(ngids,np,p)) for p in 1:np]
end

function _part_to_firstgid(ngids::Tuple,np::Tuple)
  map(_part_to_firstgid,ngids,np)
end

function add_gids!(
  a::PRange,
  i_to_gid::AbstractPData{<:AbstractArray{<:Integer}},
  i_to_part::AbstractPData{<:AbstractArray{<:Integer}},
  neighbors_snd=nothing,
  neighbors_rcv=nothing;
  kwargs...)

  map_parts(add_gids!,a.partition,i_to_gid,i_to_part)
  a.exchanger = Exchanger(a.partition,neighbors_snd,neighbors_rcv;kwargs...)
  a.ghost = true
  a
end

function add_gids!(
  a::PRange,
  gids::AbstractPData{<:AbstractArray{<:Integer}},
  neighbors_snd=nothing,
  neighbors_rcv=nothing;
  kwargs...)

  if a.gid_to_part === nothing
    msg = """ The given PRange object has not enough information to perform this operation.
    Make sure that you have built the PRange object with a suitable `gid_to_part`, or
    explicitly provide the owner part of the given gids.
    """
    throw(DomainError(a,msg))
  end
  map_parts(add_gids!,a.gid_to_part,a.partition,gids)
  a.exchanger = Exchanger(a.partition,neighbors_snd,neighbors_rcv;kwargs...)
  a.ghost = true
  a
end

function add_gids(a::PRange,args...;kwargs...)
  b = copy(a)
  add_gids!(b,args...;kwargs...)
  b
end

function to_lids!(ids::AbstractPData{<:AbstractArray{<:Integer}},a::PRange)
  map_parts(to_lids!,ids,a.partition)
end

function to_gids!(ids::AbstractPData{<:AbstractArray{<:Integer}},a::PRange)
  map_parts(to_gids!,ids,a.partition)
end

function oids_are_equal(a::PRange,b::PRange)
  if a.partition === b.partition
    true
  else
    c = map_parts(oids_are_equal,a.partition,b.partition)
    reduce(&,c,init=true)
  end
end

function hids_are_equal(a::PRange,b::PRange)
  if a.partition === b.partition
    true
  else
    c = map_parts(hids_are_equal,a.partition,b.partition)
    reduce(&,c,init=true)
  end
end

function lids_are_equal(a::PRange,b::PRange)
  if a.partition === b.partition
    true
  else
    c = map_parts(lids_are_equal,a.partition,b.partition)
    reduce(&,c,init=true)
  end
end

struct PVector{T,A,B} <: AbstractVector{T}
  values::A
  rows::B
  function PVector(
    values::AbstractPData{<:AbstractVector{T}},
    rows::PRange) where T

    A = typeof(values)
    B = typeof(rows)
    new{T,A,B}(values,rows)
  end
end

function Base.getproperty(x::PVector, sym::Symbol)
  if sym == :owned_values
    map_parts(x.values,x.rows.partition) do v,r
      view(v,r.oid_to_lid)
    end
  elseif sym == :ghost_values
    map_parts(x.values,x.rows.partition) do v,r
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
  values = map_parts(a.values,rows.partition) do values, partition
    similar(values,T,num_lids(partition))
  end
  PVector(values,rows)
end

function Base.similar(
  ::Type{<:PVector{T,<:AbstractPData{A}}},axes::Tuple{Int}) where {T,A}
  @notimplemented
end

function Base.similar(
  ::Type{<:PVector{T,<:AbstractPData{A}}},axes::Tuple{<:PRange}) where {T,A}
  rows = axes[1]
  values = map_parts(rows.partition) do partition
    similar(A,num_lids(partition))
  end
  PVector(values,rows)
end

function Base.copy!(a::PVector,b::PVector)
  @check oids_are_equal(a.rows,b.rows)
  if a.rows.partition === b.rows.partition
    map_parts(copy!,a.values,b.values)
  else
    map_parts(copy!,a.owned_values,b.owned_values)
  end
  a
end

function Base.copyto!(a::PVector,b::PVector)
  @check oids_are_equal(a.rows,b.rows)
  if a.rows.partition === b.rows.partition
    map_parts(copyto!,a.values,b.values)
  else
    map_parts(copyto!,a.owned_values,b.owned_values)
  end
  a
end

function Base.copy(b::PVector)
  a = similar(b)
  copy!(a,b)
  a
end

function LinearAlgebra.rmul!(a::PVector,v::Number)
  map_parts(a.values) do l
    rmul!(l,v)
  end
  a
end

function Base.:(==)(a::PVector,b::PVector)
  length(a) == length(b) &&
  num_parts(a.values) == num_parts(b.values) &&
  reduce(&,map_parts(==,a.owned_values,b.owned_values),init=true)
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
  if b.ghost_values !== nothing && a.rows === b.rows
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

# Distances.jl related (needed eg for non-linear solvers)

for M in Distances.metrics
  @eval begin
    function (dist::$M)(a::PVector,b::PVector)
      _eval_dist(dist,a,b,Distances.parameters(dist))
    end
  end
end

function _eval_dist(d,a,b,::Nothing)
  partials = map_parts(a.owned_values,b.owned_values) do a,b
    _eval_dist_local(d,a,b,nothing)
  end
  s = reduce(
    (i,j)->Distances.eval_reduce(d,i,j),
    partials,
    init=Distances.eval_start(d, a, b))
  Distances.eval_end(d,s)
end

Base.@propagate_inbounds function _eval_dist_local(d,a,b,::Nothing)
  @boundscheck if length(a) != length(b)
    throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
  end
  if length(a) == 0
    return zero(Distances.result_type(d, a, b))
  end
  @inbounds begin
    s = Distances.eval_start(d, a, b)
    if (IndexStyle(a, b) === IndexLinear() && eachindex(a) == eachindex(b)) || axes(a) == axes(b)
      @simd for I in eachindex(a, b)
        ai = a[I]
        bi = b[I]
        s = Distances.eval_reduce(d, s, Distances.eval_op(d, ai, bi))
      end
    else
      for (ai, bi) in zip(a, b)
        s = Distances.eval_reduce(d, s, Distances.eval_op(d, ai, bi))
      end
    end
    return s
  end
end

function _eval_dist(d,a,b,p)
  @notimplemented
end

function _eval_dist_local(d,a,b,p)
  @notimplemented
end

function Base.any(f::Function,x::PVector)
  partials = map_parts(x.owned_values) do o
    any(f,o)
  end
  reduce(|,partials,init=false)
end

function Base.all(f::Function,x::PVector)
  partials = map_parts(x.owned_values) do o
    all(f,o)
  end
  reduce(&,partials,init=true)
end

function Base.maximum(x::PVector)
  partials = map_parts(maximum,x.owned_values)
  reduce(max,partials,init=typemin(eltype(x)))
end

function Base.maximum(f::Function,x::PVector)
  partials = map_parts(x.owned_values) do o
    maximum(f,o)
  end
  reduce(max,partials,init=typemin(eltype(x)))
end

function Base.minimum(x::PVector)
  partials = map_parts(minimum,x.owned_values)
  reduce(min,partials,init=typemax(eltype(x)))
end

function Base.minimum(f::Function,x::PVector)
  partials = map_parts(x.owned_values) do o
    minimum(f,o)
  end
  reduce(min,partials,init=typemax(eltype(x)))
end

function Base.findall(f::Function,x::PVector)
  @notimplemented
end

function PVector{T}(
  ::UndefInitializer,
  rows::PRange) where T

  values = map_parts(rows.partition) do partition
    nlids = num_lids(partition)
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
  I::AbstractPData{<:AbstractArray{<:Integer}},
  V::AbstractPData{<:AbstractArray},
  rows::PRange;
  ids::Symbol)

  @assert ids in (:global,:local)
  if ids == :global
    to_lids!(I,rows)
  end

  values = map_parts(rows.partition,I,V) do partition,I,V
    values = init(num_lids(partition))
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
  I::AbstractPData{<:AbstractArray{<:Integer}},
  V::AbstractPData{<:AbstractArray{T}},
  rows;
  ids::Symbol) where T
  PVector(n->zeros(T,n),I,V,rows;ids=ids)
end

function PVector(
  init,
  I::AbstractPData{<:AbstractArray{<:Integer}},
  V::AbstractPData{<:AbstractArray},
  n::Integer;
  ids::Symbol)

  @assert ids == :global
  parts = get_part_ids(I)
  rows = PRange(parts,n)
  add_gids!(rows,I)
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
  b = map_parts(a.values,a.rows.partition) do values,partition
    owned_values = view(values,partition.oid_to_lid)
    reduce(op,owned_values,init=init)
  end
  reduce(op,b,init=init)
end

function Base.sum(a::PVector)
  reduce(+,a,init=zero(eltype(a)))
end

function LinearAlgebra.dot(a::PVector,b::PVector)
  c = map_parts(a.values,b.values,a.rows.partition,b.rows.partition) do a,b,alids,blids
    a_owned = view(a,alids.oid_to_lid)
    b_owned = view(b,blids.oid_to_lid)
    dot(a_owned,b_owned)
  end
  sum(c)
end

function local_view(a::PVector,rows::PRange)
  if a.rows === rows
    a.values
  else
    map_parts(a.values,rows.partition,a.rows.partition) do values,rows,arows
      LocalView(values,(find_lid_map(rows,arows),))
    end
  end
end

struct LocalView{T,N,A,B} <:AbstractArray{T,N}
  plids_to_value::A
  d_to_lid_to_plid::B
  local_size::NTuple{N,Int}
  function LocalView(
    plids_to_value::AbstractArray{T,N},d_to_lid_to_plid::NTuple{N}) where {T,N}
    A = typeof(plids_to_value)
    B = typeof(d_to_lid_to_plid)
    local_size = map(length,d_to_lid_to_plid)
    new{T,N,A,B}(plids_to_value,d_to_lid_to_plid,local_size)
  end
end

Base.size(a::LocalView) = a.local_size
Base.IndexStyle(::Type{<:LocalView}) = IndexCartesian()
function Base.getindex(a::LocalView{T,N},lids::Vararg{Integer,N}) where {T,N}
  plids = map(_lid_to_plid,lids,a.d_to_lid_to_plid)
  if all(i->i>0,plids)
    a.plids_to_value[plids...]
  else
    zero(T)
  end
end
function Base.setindex!(a::LocalView{T,N},v,lids::Vararg{Integer,N}) where {T,N}
  plids = map(_lid_to_plid,lids,a.d_to_lid_to_plid)
  @check all(i->i>0,plids) "You are trying to set a value that is not stored in the local portion"
  a.plids_to_value[plids...] = v
end
function _lid_to_plid(lid,lid_to_plid)
  plid = lid_to_plid[lid]
  plid
end

function global_view(a::PVector,rows::PRange)
  @notimplementedif a.rows !== rows
  n = length(rows)
  map_parts(a.values,a.rows.partition) do values, partition
    GlobalView(values,(partition.gid_to_lid,),(n,))
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
  t0::AbstractPData=_empty_tasks(a.rows.exchanger.parts_rcv))
  async_exchange!(a.values,a.rows.exchanger,t0)
end

# Non-blocking assembly
function async_assemble!(
  a::PVector,
  t0::AbstractPData=_empty_tasks(a.rows.exchanger.parts_rcv))
  async_assemble!(+,a,t0)
end

function async_assemble!(
  combine_op,
  a::PVector,
  t0::AbstractPData=_empty_tasks(a.rows.exchanger.parts_rcv))

  exchanger_rcv = a.rows.exchanger # receives data at ghost ids from remote parts
  exchanger_snd = reverse(exchanger_rcv) # sends data at ghost ids to remote parts
  t1 = async_exchange!(combine_op,a.values,exchanger_snd,t0)
  map_parts(t1,a.values,a.rows.partition) do t1,values,partition
    @task begin
      wait(schedule(t1))
      values[partition.hid_to_lid] .= zero(eltype(values))
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
    values::AbstractPData{<:AbstractSparseMatrix{T}},
    rows::PRange,
    cols::PRange,
    exchanger::Exchanger=matrix_exchanger(values,rows,cols)) where T

    A = typeof(values)
    B = typeof(rows)
    C = typeof(cols)
    D = typeof(exchanger)
    new{T,A,B,C,D}(values,rows,cols,exchanger)
  end
end

function LinearAlgebra.fillstored!(a::PSparseMatrix,v)
  map_parts(a.values) do values
    LinearAlgebra.fillstored!(values,v)
  end
  a
end

function Base.copy(a::PSparseMatrix)
  PSparseMatrix(
    copy(a.values),
    copy(a.rows),
    copy(a.cols),
    copy(a.exchanger))
end

function Base.getproperty(x::PSparseMatrix, sym::Symbol)
  if sym == :owned_owned_values
    map_parts(x.values,x.rows.partition,x.cols.partition) do v,r,c
      indices = (r.oid_to_lid,c.oid_to_lid)
      inv_indices = (r.lid_to_ohid,c.lid_to_ohid)
      flag = (1,1)
      SubSparseMatrix(v,indices,inv_indices,flag)
    end
  elseif sym == :owned_ghost_values
    map_parts(x.values,x.rows.partition,x.cols.partition) do v,r,c
      indices = (r.oid_to_lid,c.hid_to_lid)
      inv_indices = (r.lid_to_ohid,c.lid_to_ohid)
      flag = (1,-1)
      SubSparseMatrix(v,indices,inv_indices,flag)
    end
  elseif sym == :ghost_owned_values
    map_parts(x.values,x.rows.partition,x.cols.partition) do v,r,c
      indices = (r.hid_to_lid,c.oid_to_lid)
      inv_indices = (r.lid_to_ohid,c.lid_to_ohid)
      flag = (-1,1)
      SubSparseMatrix(v,indices,inv_indices,flag)
    end
  elseif sym == :ghost_ghost_values
    map_parts(x.values,x.rows.partition,x.cols.partition) do v,r,c
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
  I::AbstractPData{<:AbstractArray{<:Integer}},
  J::AbstractPData{<:AbstractArray{<:Integer}},
  V::AbstractPData{<:AbstractArray},
  rows::PRange,
  cols::PRange,
  args...;
  ids::Symbol)

  @assert ids in (:global,:local)
  if ids == :global
    to_lids!(I,rows)
    to_lids!(J,cols)
  end

  values = map_parts(I,J,V,rows.partition,cols.partition) do I,J,V,rlids,clids
    init(I,J,V,num_lids(rlids),num_lids(clids))
  end

  PSparseMatrix(values,rows,cols,args...)
end

function PSparseMatrix(
  init,
  I::AbstractPData{<:AbstractArray{<:Integer}},
  J::AbstractPData{<:AbstractArray{<:Integer}},
  V::AbstractPData{<:AbstractArray},
  nrows::Integer,
  ncols::Integer,
  args...;
  ids::Symbol)

  @assert ids == :global
  parts = get_part_ids(I)
  rows = PRange(parts,nrows)
  cols = PRange(parts,ncols)
  add_gids!(rows,I)
  add_gids!(cols,J)
  PSparseMatrix(init,I,J,V,rows,cols,args...;ids=ids)
end

# Using sparse as default
function PSparseMatrix(
  I::AbstractPData{<:AbstractArray{<:Integer}},
  J::AbstractPData{<:AbstractArray{<:Integer}},
  V::AbstractPData{<:AbstractArray},
  args...; kwargs...)

  PSparseMatrix(sparse,I,J,V,args...;kwargs...)
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

function local_view(a::PSparseMatrix,rows::PRange,cols::PRange)
  if a.rows === rows && a.cols === cols
    a.values
  else
    map_parts(
      a.values,rows.partition,cols.partition,a.rows.partition,a.cols.partition) do values,rows,cols,arows,acols
      rmap = find_lid_map(rows,arows)
      cmap = (cols === rows && acols === arows) ? rmap : find_lid_map(cols,acols)
      LocalView(values,(rmap,cmap))
    end
  end
end

function global_view(a::PSparseMatrix,rows::PRange,cols::PRange)
  @notimplementedif a.rows !== rows
  @notimplementedif a.cols !== cols
  nrows = length(rows)
  ncols = length(cols)
  map_parts(a.values,a.rows.partition,a.cols.partition) do values,rlids,clids
    GlobalView(values,(rlids.gid_to_lid,clids.gid_to_lid),(nrows,ncols))
  end
end

function matrix_exchanger(
  values::AbstractPData{<:AbstractSparseMatrix},
  rows::PRange,
  cols::PRange)

  if rows.ghost
    matrix_exchanger(values,rows.exchanger,rows.partition,cols.partition)
  else
    empty_exchanger(rows.partition)
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
      @check k > 0 "The sparsity pattern of the ghost layer is inconsistent"
      k_snd_data[p] = k
    end
    k_snd = Table(k_snd_data,ptrs)
    k_snd
  end

  k_snd = map_parts(setup_snd,part,row_lids,col_lids,gi_snd,gj_snd,values)

  Exchanger(parts_rcv,parts_snd,k_rcv,k_snd)
end

# Non-blocking exchange
function async_exchange!(
  a::PSparseMatrix,
  t0::AbstractPData=_empty_tasks(a.exchanger.parts_rcv))
  nzval = map_parts(nonzeros,a.values)
  async_exchange!(nzval,a.exchanger,t0)
end

# Non-blocking assembly
function async_assemble!(
  a::PSparseMatrix,
  t0::AbstractPData=_empty_tasks(a.exchanger.parts_rcv))
  async_assemble!(+,a,t0)
end

function async_assemble!(
  combine_op,
  a::PSparseMatrix,
  t0::AbstractPData=_empty_tasks(a.exchanger.parts_rcv))

  exchanger_rcv = a.exchanger # receives data at ghost ids from remote parts
  exchanger_snd = reverse(exchanger_rcv) # sends data at ghost ids to remote parts
  nzval = map_parts(nonzeros,a.values)
  t1 = async_exchange!(combine_op,nzval,exchanger_snd,t0)
  map_parts(t1,nzval,exchanger_snd.lids_snd) do t1,nzval,lids_snd
    @task begin
      wait(schedule(t1))
      nzval[lids_snd.data] .= zero(eltype(nzval))
    end
  end
end

function async_assemble!(
  I::AbstractPData{<:AbstractVector{<:Integer}},
  J::AbstractPData{<:AbstractVector{<:Integer}},
  V::AbstractPData{<:AbstractVector},
  rows::PRange,
  t0::AbstractPData=_empty_tasks(rows.exchanger.parts_rcv))

  map_parts(wait∘schedule,t0)

  part = get_part_ids(rows.partition)
  parts_rcv = rows.exchanger.parts_rcv
  parts_snd = rows.exchanger.parts_snd

  to_lids!(I,rows)
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

  gi_rcv, gj_rcv, v_rcv = map_parts(setup_rcv,part,parts_rcv,rows.partition,coo_values)

  gi_snd, t1 = async_exchange(gi_rcv,parts_snd,parts_rcv)
  gj_snd, t2 = async_exchange(gj_rcv,parts_snd,parts_rcv)
  v_snd, t3 = async_exchange(v_rcv,parts_snd,parts_rcv)

  function setup_snd(t1,t2,t3,part,row_lids,gi_snd,gj_snd,v_snd,coo_values)
    @task begin
      wait(schedule(t1))
      wait(schedule(t2))
      wait(schedule(t3))
      k_li, k_gj, k_v = coo_values
      to_gids!(k_li,row_lids)
      ptrs = gi_snd.ptrs
      current_n = length(k_li)
      new_n = current_n + length(gi_snd.data)
      resize!(k_li,new_n)
      resize!(k_gj,new_n)
      resize!(k_v,new_n)
      for p in 1:length(gi_snd.data)
        gi = gi_snd.data[p]
        gj = gj_snd.data[p]
        v = v_snd.data[p]
        k_li[current_n+p] = gi
        k_gj[current_n+p] = gj
        k_v[current_n+p] = v
      end
    end
  end

  t4 = map_parts(setup_snd,t1,t2,t3,part,rows.partition,gi_snd,gj_snd,v_snd,coo_values)

  t4
end

function async_exchange!(
  I::AbstractPData{<:AbstractVector{<:Integer}},
  J::AbstractPData{<:AbstractVector{<:Integer}},
  V::AbstractPData{<:AbstractVector},
  rows::PRange,
  t0::AbstractPData=_empty_tasks(rows.exchanger.parts_rcv))

  map_parts(wait∘schedule,t0)

  part = get_part_ids(rows.partition)
  parts_rcv = rows.exchanger.parts_rcv
  parts_snd = rows.exchanger.parts_snd
  lids_snd = rows.exchanger.lids_snd

  to_lids!(I,rows)
  coo_values = map_parts(tuple,I,J,V)

  function setup_snd(part,parts_snd,lids_snd,row_lids,coo_values)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    unset = zero(Int32)
    lid_to_i = fill(unset,num_lids(row_lids))
    for i in 1:length(lids_snd)
      for p in lids_snd.ptrs[i]:(lids_snd.ptrs[i+1]-1)
        lid = lids_snd.data[p]
        lid_to_i[lid] = i
      end
    end
    ptrs = zeros(Int32,length(parts_snd)+1)
    k_li, k_gj, k_v = coo_values
    Tv = eltype(k_v)
    for k in 1:length(k_li)
      li = k_li[k]
      i = lid_to_i[li]
      if i != unset
        ptrs[i+1] +=1
      elseif row_lids.lid_to_part[li] != part
        k_v[k] = zero(Tv)
      end
    end
    length_to_ptrs!(ptrs)
    gi_snd_data = zeros(Int,ptrs[end]-1)
    gj_snd_data = zeros(Int,ptrs[end]-1)
    v_snd_data = zeros(Tv,ptrs[end]-1)
    for k in 1:length(k_li)
      li = k_li[k]
      i = lid_to_i[li]
      if i != unset
        gi = row_lids.lid_to_gid[li]
        gj = k_gj[k]
        v = k_v[k]
        p = ptrs[i]
        gi_snd_data[p] = gi
        gj_snd_data[p] = gj
        v_snd_data[p] = v
        ptrs[i] +=1
      end
    end
    rewind_ptrs!(ptrs)
    gi_snd = Table(gi_snd_data,ptrs)
    gj_snd = Table(gj_snd_data,ptrs)
    v_snd = Table(v_snd_data,ptrs)
    gi_snd, gj_snd, v_snd
  end

  gi_snd, gj_snd, v_snd = map_parts(
    setup_snd,part,parts_snd,lids_snd,rows.partition,coo_values)

  gi_rcv, t1 = async_exchange(gi_snd,parts_rcv,parts_snd)
  gj_rcv, t2 = async_exchange(gj_snd,parts_rcv,parts_snd)
  v_rcv, t3 = async_exchange(v_snd,parts_rcv,parts_snd)

  function setup_rcv(t1,t2,t3,part,row_lids,gi_rcv,gj_rcv,v_rcv,coo_values)
    @task begin
      wait(schedule(t1))
      wait(schedule(t2))
      wait(schedule(t3))
      k_li, k_gj, k_v = coo_values
      to_gids!(k_li,row_lids)
      ptrs = gi_rcv.ptrs
      current_n = length(k_li)
      new_n = current_n + length(gi_rcv.data)
      resize!(k_li,new_n)
      resize!(k_gj,new_n)
      resize!(k_v,new_n)
      for p in 1:length(gi_rcv.data)
        gi = gi_rcv.data[p]
        gj = gj_rcv.data[p]
        v = v_rcv.data[p]
        k_li[current_n+p] = gi
        k_gj[current_n+p] = gj
        k_v[current_n+p] = v
      end
    end
  end

  t4 = map_parts(setup_rcv,t1,t2,t3,part,rows.partition,gi_rcv,gj_rcv,v_rcv,coo_values)

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

# Not efficient, just for convenience and debugging purposes
function Base.:\(a::PSparseMatrix{Ta},b::PVector{Tb}) where {Ta,Tb}
  T = typeof(one(Ta)\one(Tb)+one(Ta)\one(Tb))
  c = PVector{T}(undef,a.cols)
  a_in_main = gather(a)
  b_in_main = gather(b,a_in_main.rows)
  c_in_main = gather(c,a_in_main.cols)
  map_main(c_in_main.values,a_in_main.values,b_in_main.values) do c, a, b
    c .= a\b
    nothing
  end
  scatter!(c,c_in_main)
  c
end

# Not efficient, just for convenience and debugging purposes
struct PLU{A,B,C}
  lu_in_main::A
  rows::B
  cols::C
end
function LinearAlgebra.lu(a::PSparseMatrix)
  a_in_main = gather(a)
  lu_in_main = map_main(lu,a_in_main.values)
  PLU(lu_in_main,a_in_main.rows,a_in_main.cols)
end
function LinearAlgebra.lu!(b::PLU,a::PSparseMatrix)
  a_in_main = gather(a,b.rows,b.cols)
  map_main(lu!,b.lu_in_main,a_in_main.values)
  b
end
function LinearAlgebra.ldiv!(c::PVector,a::PLU,b::PVector)
  b_in_main = gather(b,a.rows)
  c_in_main = gather(c,a.cols)
  map_main(ldiv!,c_in_main.values,a.lu_in_main,b_in_main.values)
  scatter!(c,c_in_main)
  c
end

function gather(
  a::PSparseMatrix{Ta},
  rows_in_main::PRange=_to_main(a.rows),
  cols_in_main::PRange=_to_main(a.cols)) where {Ta}

  I,J,V = map_parts(a.values,a.rows.partition,a.cols.partition) do a,rows,cols
    n = 0
    for (i,j,v) in nziterator(a)
      if rows.lid_to_part[i] == rows.part
        n += 1
      end
    end
    I = zeros(Int,n)
    J = zeros(Int,n)
    V = zeros(Ta,n)
    n = 0
    for (i,j,v) in nziterator(a)
      if rows.lid_to_part[i] == rows.part
        n += 1
        I[n] = rows.lid_to_gid[i]
        J[n] = cols.lid_to_gid[j]
        V[n] = v
      end
    end
    I,J,V
  end
  assemble!(I,J,V,rows_in_main)
  I,J,V = map_parts(a.rows.partition,I,J,V) do rows,I,J,V
    if rows.part == MAIN
      I,J,V
    else
      similar(I,eltype(I),0),similar(J,eltype(J),0),similar(V,eltype(V),0)
    end
  end
  T = eltype(a.values)
  exchanger = empty_exchanger(rows_in_main.partition)
  a_in_main = PSparseMatrix(
    (args...)->compresscoo(T,args...),
    I,J,V,rows_in_main,cols_in_main,exchanger;ids=:global)
  a_in_main
end

function gather(b::PVector,rows_in_main::PRange=_to_main(b.rows))
  T = eltype(b)
  b_in_main = PVector(zero(T),rows_in_main)
  map_parts(b.values,b_in_main.values,b.rows.partition) do b,b_in_main,rows
    part = rows.part
    if part == MAIN
      b_in_main[view(rows.lid_to_gid,rows.oid_to_lid)] .= view(b,rows.oid_to_lid)
    else
      b_in_main .= view(b,rows.oid_to_lid)
    end
  end
  assemble!(b_in_main)
  b_in_main
end

function scatter!(c::PVector,c_in_main::PVector)
  exchange!(c_in_main)
  map_parts(c.values,c_in_main.values,c.rows.partition) do c, c_in_main, rows
    part = rows.part
    if part == MAIN
      c[rows.oid_to_lid] .= view(c_in_main,view(rows.lid_to_gid,rows.oid_to_lid))
    else
      c[rows.oid_to_lid] .= c_in_main
    end
  end
  c
end

function _to_main(rows::PRange)
  parts = get_part_ids(rows.partition)
  ngids = length(rows)
  partition = map_parts(parts,rows.partition) do part,rows
    if part == MAIN
      lid_to_gid = collect(1:ngids)
      lid_to_part = fill(Int32(MAIN),ngids)
    else
      lid_to_gid = collect(view(rows.lid_to_gid,rows.oid_to_lid))
      lid_to_part = fill(Int32(MAIN),num_oids(rows))
    end
    IndexSet(part,lid_to_gid,lid_to_part)
  end
  mrows = PRange(ngids,partition)
end

# Misc functions that could be removed if IterativeSolvers was implemented in terms
# of axes(A,d) instead of size(A,d)
function IterativeSolvers.zerox(A::PSparseMatrix,b::PVector)
  T = IterativeSolvers.Adivtype(A, b)
  x = similar(b, T, axes(A, 2))
  fill!(x, zero(T))
  return x
end
