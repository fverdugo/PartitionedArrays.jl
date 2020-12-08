
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


struct DistributedIndexSet{T<:DistributedData{<:IndexSet}}
  lids::T
  ngids::Int
end

get_distributed_data(a::DistributedIndexSet) = a.lids
num_gids(a::DistributedIndexSet) = a.ngids

function DistributedIndexSet(initializer::Function,comm::Communicator,ngids::Integer,args...)
  lids = DistributedData(initializer,comm,args...)
  DistributedIndexSet(lids,ngids)
end

# The comm argument can be omitted if it can be determined from the first
# data argument.
function DistributedIndexSet(initializer::Function,ngids::Integer,args...) where T
  comm = get_comm(get_distributed_data(first(args)))
  DistributedIndexSet(initializer,comm,args...)
end

#struct DistributedVector{T,A<:DistributedData{<:AbstractVector{T}},B<:DistributedIndexSet} <: AbstractVector{T}
#  values::A
#  ids::B
#end
#
#get_distributed_data(a::DistributedVector) = a.values
#
#Base.length(a::DistributedVector) = num_gids(a.ids)




