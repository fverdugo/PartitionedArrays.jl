
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

# In this version, sending a number per part is enough
# We have another version below to send a vector of numbers per part (compressed in a Table)
function exchange!(
  data_rcv::DistributedData,
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  @abstractmethod
end

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

function exchange!(
  data_rcv::DistributedData{<:Table},
  data_snd::DistributedData{<:Table},
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  @abstractmethod
end

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

# A, B, C should be the type of some indexable collection, e.g. ranges or vectors or dicts
struct IndexSet{A,B,C}
  ngids::Int
  lid_to_gid::A
  lid_to_owner::B
  gid_to_lid::C
end

function IndexSet(ngids,lid_to_gid,lid_to_owner)
  gid_to_lid = Dict{Int,Int32}()
  for (lid,gid) in enumerate(lid_to_gid)
    gid_to_lid[gid] = lid
  end
  IndexSet(ngids,lid_to_gid,lid_to_owner,gid_to_lid)
end

num_gids(a::IndexSet) = a.ngids
num_lids(a::IndexSet) = length(a.lid_to_owner)

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
    parts_rcv = Dict((owner=>true for owner in ids.lid_to_owner if owner!=part))
    sort(collect(keys(parts_rcv)))
  end

  lids_rcv, gids_rcv = DistributedData(ids,parts_rcv) do part, ids, parts_rcv

    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_rcv) ))

    ptrs = zeros(Int32,length(parts_rcv)+1)
    for owner in ids.lid_to_owner
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)

    data_lids = zeros(Int32,ptrs[end]-1)
    data_gids = zeros(Int,ptrs[end]-1)

    for (lid,owner) in enumerate(ids.lid_to_owner)
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

struct DistributedIndexSet{A,B}
  ngids::Int
  lids::A
  exchanger::B
  function DistributedIndexSet(
    ngids::Integer,
    lids::DistributedData{<:IndexSet},
    exchanger::Exchanger=Exchanger(lids))

    A = typeof(lids)
    B = typeof(exchanger)
    new{A,B}(ngids,lids,exchanger)
  end
end

get_distributed_data(a::DistributedIndexSet) = a.lids
num_gids(a::DistributedIndexSet) = a.ngids

function non_overlaping(ids::DistributedIndexSet)
  lids = DistributedData(ids) do part, ids
    lid_to_gid = similar(ids.lid_to_gid,eltype(ids.lid_to_gid),0)
    for (lid,owner) in enumerate(ids.lid_to_owner)
      if owner == part
        gid = ids.lid_to_gid[lid]
        push!(lid_to_gid,gid)
      end
    end
    lid_to_owner = similar(ids.lid_to_owner,eltype(ids.lid_to_owner),length(lid_to_gid))
    fill!(lid_to_owner,part)
    IndexSet(ids.ngids,lid_to_gid,lid_to_owner)
  end
  neighbors = DistributedData(ids.exchanger.parts_rcv) do part,parts_rcv
    similar(parts_rcv,eltype(parts_rcv),0)
  end
  exchanger = Exchanger(lids,neighbors)
  DistributedIndexSet(ids.ngids,lids,exchanger)
end

struct DistributedVector{T,A,B} <: AbstractVector{T}
  values::A
  ids::B
  function DistributedVector(
    values::DistributedData{<:AbstractVector{T}},
    ids::DistributedIndexSet) where T

    A = typeof(values)
    B = typeof(ids)
    new{T,A,B}(values,ids)
  end
end

function DistributedVector{T}(::UndefInitializer,ids::DistributedIndexSet) where T
  values = DistributedData(ids) do part, ids
    nlids = length(ids.lid_to_owner)
    Vector{T}(undef,nlids)
  end
  DistributedVector(values,ids)
end

function Base.fill!(a::DistributedVector,v)
  do_on_parts(a.values) do part, values
    fill!(values,v)
  end
  a
end

#get_distributed_data(a::DistributedVector) = a.values

Base.length(a::DistributedVector) = num_gids(a.ids)

function exchange!(a::DistributedVector{T}) where T

  # Allocate buffers
  data_rcv = allocate_rcv_buffer(T,a.ids.exchanger)
  data_snd = allocate_snd_buffer(T,a.ids.exchanger)

  # Fill snd buffer
  do_on_parts(a.values,data_snd,a.ids.exchanger.lids_snd) do part,values,data_snd,lids_snd 
    for p in 1:length(lids_snd.data)
      lid = lids_snd.data[p]
      data_snd.data[p] = values[lid]
    end
  end

  # communicate
  exchange!(
    data_rcv,
    data_snd,
    a.ids.exchanger.parts_rcv,
    a.ids.exchanger.parts_snd)

  # Fill non-owned values from rcv buffer
  do_on_parts( a.values,data_rcv,a.ids.exchanger.lids_rcv) do part,values,data_rcv,lids_rcv 
    for p in 1:length(lids_rcv.data)
      lid = lids_rcv.data[p]
      values[lid] = data_rcv.data[p]  
    end
  end

  a
end

struct DistributedSparseMatrix{T,A,B} <: AbstractMatrix{T}
  values::A
  row_ids::B
  col_ids::B
  function DistributedSparseMatrix(
    values::DistributedData{<:AbstractSparseMatrix{T}},
    row_ids::DistributedIndexSet,
    col_ids::DistributedIndexSet) where T

    A = typeof(values)
    B = typeof(col_ids)
    new{T,A,B}(values,row_ids,col_ids)
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
  exchange!(b)
  do_on_parts(c.values,a.values,b.values) do part,c,a,b
    mul!(c,a,b,α,β)
  end
  c
end

