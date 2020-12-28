
abstract type Backend end

# Should return a DistributedData{Part}
function Partition(b::Backend,nparts::Integer)
  @abstractmethod
end

function Partition(b::Backend,nparts::Tuple)
  Partition(b,prod(nparts))
end

# This can be overwritten to add a finally clause
function distributed_run(driver::Function,b::Backend,nparts)
  part = Partition(b,nparts)
  driver(part)
end

# Data distributed in parts of type T
abstract type DistributedData{T} end

num_parts(a::DistributedData) = @abstractmethod

function map_parts(task::Function,a::DistributedData...)
  @abstractmethod
end

function i_am_master(::DistributedData)
  @abstractmethod
end

struct Part
  id::Int
  num_parts::Int
end

num_parts(a::Part) = a.num_parts
Base.:(==)(a::Part,b::Integer) = a.id == b
Base.:(==)(a::Integer,b::Part) = b == a
Base.:(==)(a::Part,b::Part) = b.id == a.id

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
  parts_snd::DistributedData)

  @abstractmethod
end

# Blocking in-place exchange
function exchange!(args...;kwargs...)
  t = async_exchange!(args...;kwargs...)
  map_parts(schedule,t)
  map_parts(wait,t)
  first(args)
end

# Non-blocking allocating exchange
# the returned data_rcv cannot be consumed in a part until the corresponding task in t is done.
function async_exchange(
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  data_rcv = map_parts(data_snd,parts_rcv) do data_snd, parts_rcv
    similar(data_snd,eltype(data_snd),length(parts_rcv))
  end

  t = async_exchange!(data_rcv,data_snd,parts_rcv,parts_snd)

  data_rcv, t
end

# Blocking allocating exchange
function exchange(args...;kwargs...)
  data_rcv, t = async_exchange(args...;kwargs...)
  map_parts(schedule,t)
  map_parts(wait,t)
  data_rcv
end
