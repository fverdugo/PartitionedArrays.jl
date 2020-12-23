struct SequentialCommunicator <: OrchestratedCommunicator
  nparts::Int
end

function SequentialCommunicator(user_driver_function,nparts)
  comm=SequentialCommunicator(nparts)
  user_driver_function(comm)
end

function SequentialCommunicator(nparts::Tuple)
  SequentialCommunicator(prod(nparts))
end

# All objects to be used with this communicator need to implement this
# function
function get_part(comm::SequentialCommunicator,object,part::Integer)
  @abstractmethod
end

function get_part(comm::SequentialCommunicator,object::Number,part::Integer)
  object
end

function num_parts(a::SequentialCommunicator)
  a.nparts
end

function num_workers(a::SequentialCommunicator)
  1
end

function Base.:(==)(a::SequentialCommunicator,b::SequentialCommunicator)
  a.nparts == b.nparts
end

function do_on_parts(task::Function,comm::SequentialCommunicator,args...)
  for part in 1:num_parts(comm)
    largs = map(a->get_part(comm,get_distributed_data(a),part),args)
    task(part,largs...)
  end
end

struct SequentialDistributedData{T} <: DistributedData{T}
  comm::SequentialCommunicator
  parts::Vector{T}
end

function Base.show(io::IO,k::MIME"text/plain",data::SequentialDistributedData)
  for part in 1:num_parts(data)
    if part != 1
      println(io,"")
    end
    println(io,"On part $part of $(num_parts(data)):")
    show(io,k,data.parts[part])
  end
end

function Base.iterate(a::SequentialDistributedData)
  next = DistributedData(a) do part, a
    iterate(a)
  end
  if eltype(next.parts) == Nothing || any(i->i==Nothing,next.parts)
    return nothing
  end
  item = DistributedData(next) do part, next
    next[1]
  end
  state = DistributedData(next) do part, next
    next[2]
  end
  item, state
end

function Base.iterate(a::SequentialDistributedData,state)
  next = DistributedData(a,state) do part, a, state
    iterate(a,state)
  end
  if eltype(next.parts) == Nothing || any(i->i==Nothing,next.parts)
    return nothing
  end
  item = DistributedData(next) do part, next
    next[1]
  end
  state = DistributedData(next) do part, next
    next[2]
  end
  item, state
end

get_part(comm::SequentialCommunicator,a::SequentialDistributedData,part::Integer) = a.parts[part]

get_comm(a::SequentialDistributedData) = a.comm

function DistributedData{T}(initializer::Function,comm::SequentialCommunicator,args...) where T
  nparts = num_parts(comm)
  parts = [initializer(i,map(a->get_part(comm,get_distributed_data(a),i),args)...) for i in 1:nparts]
  SequentialDistributedData{T}(comm,parts)
end

function DistributedData(initializer::Function,comm::SequentialCommunicator,args...)
  nparts = num_parts(comm)
  parts = [initializer(i,map(a->get_part(comm,get_distributed_data(a),i),args)...) for i in 1:nparts]
  SequentialDistributedData(comm,parts)
end

function gather!(a::AbstractVector,b::SequentialDistributedData)
  @assert length(a) == num_parts(b)
  copyto!(a,b.parts)
end

function scatter(comm::SequentialCommunicator,b::AbstractVector)
  @assert length(b) == num_parts(comm)
  SequentialDistributedData(comm,b)
end

function async_exchange!(
  data_rcv::SequentialDistributedData,
  data_snd::SequentialDistributedData,
  parts_rcv::SequentialDistributedData,
  parts_snd::SequentialDistributedData)

  @check get_comm(parts_rcv) == get_comm(data_rcv)
  @check get_comm(parts_rcv) == get_comm(data_snd)

  @boundscheck _check_rcv_and_snd_match(parts_rcv,parts_snd)
  for part_rcv in 1:num_parts(parts_rcv)
    for (i, part_snd) in enumerate(parts_rcv.parts[part_rcv])
      j = first(findall(k->k==part_rcv,parts_snd.parts[part_snd]))
      data_rcv.parts[part_rcv][i] = data_snd.parts[part_snd][j]
    end
  end
  DistributedData(get_comm(parts_rcv)) do part
    @async nothing
  end
end

function _check_rcv_and_snd_match(parts_rcv::SequentialDistributedData,parts_snd::SequentialDistributedData)
  @check get_comm(parts_rcv) == get_comm(parts_snd)
  for part in 1:num_parts(parts_rcv)
    for i in parts_rcv.parts[part]
      @check length(findall(k->k==part,parts_snd.parts[i])) == 1
    end
    for i in parts_snd.parts[part]
      @check length(findall(k->k==part,parts_rcv.parts[i])) == 1
    end
  end
  true
end

function async_exchange!(
  data_rcv::SequentialDistributedData{<:Table},
  data_snd::SequentialDistributedData{<:Table},
  parts_rcv::SequentialDistributedData,
  parts_snd::SequentialDistributedData)

  @check get_comm(parts_rcv) == get_comm(data_rcv)
  @check get_comm(parts_rcv) == get_comm(data_snd)

  @boundscheck _check_rcv_and_snd_match(parts_rcv,parts_snd)
  for part_rcv in 1:num_parts(parts_rcv)
    for (i, part_snd) in enumerate(parts_rcv.parts[part_rcv])
      j = first(findall(k->k==part_rcv,parts_snd.parts[part_snd]))
      ptrs_rcv = data_rcv.parts[part_rcv].ptrs
      ptrs_snd = data_snd.parts[part_snd].ptrs
      @check ptrs_rcv[i+1]-ptrs_rcv[i] == ptrs_snd[j+1]-ptrs_snd[j]
      for p in 1:(ptrs_rcv[i+1]-ptrs_rcv[i])
        p_rcv = p+ptrs_rcv[i]-1
        p_snd = p+ptrs_snd[j]-1
        data_rcv.parts[part_rcv].data[p_rcv] = data_snd.parts[part_snd].data[p_snd]
      end
    end
  end
  DistributedData(get_comm(parts_rcv)) do part
    @async nothing
  end
end

