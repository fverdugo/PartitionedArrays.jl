
struct SequentialBackend <: Backend end

const sequential = SequentialBackend()

function get_parts(b::SequentialBackend,nparts::Integer)
  parts = [ part for part in 1:nparts ]
  SequentialDistributedData(parts)
end

struct SequentialDistributedData{T} <: DistributedData{T}
  parts::Vector{T}
end

num_parts(a::SequentialDistributedData) = length(a.parts)

get_backend(a::SequentialDistributedData) = sequential

function Base.iterate(a::SequentialDistributedData)
  next = map_parts(iterate,a)
  if eltype(next.parts) == Nothing || any(i->i==Nothing,next.parts)
    return nothing
  end
  item = map_parts(first,next)
  state = map_parts(_second,next)
  item, state
end

_second(a) = a[2]

function Base.iterate(a::SequentialDistributedData,state)
  next = map_parts(iterate,a,state)
  if eltype(next.parts) == Nothing || any(i->i==Nothing,next.parts)
    return nothing
  end
  item = map_parts(first,next)
  state = map_parts(_second,next)
  item, state
end

function map_parts(task::Function,args::SequentialDistributedData...)
  @assert length(args) > 0
  @assert all(a->length(a.parts)==length(first(args).parts),args)
  parts_in = map(a->a.parts,args)
  parts_out = map(task,parts_in...)
  SequentialDistributedData(parts_out)
end

i_am_master(a::SequentialDistributedData) = true

function Base.show(io::IO,k::MIME"text/plain",data::SequentialDistributedData)
  for part in 1:num_parts(data)
    if part != 1
      println(io,"")
    end
    println(io,"On part $part of $(num_parts(data)):")
    show(io,k,data.parts[part])
  end
end

function async_exchange!(
  data_rcv::SequentialDistributedData,
  data_snd::SequentialDistributedData,
  parts_rcv::SequentialDistributedData,
  parts_snd::SequentialDistributedData,
  t_in::DistributedData)

  @check num_parts(parts_rcv) == num_parts(data_rcv)
  @check num_parts(parts_rcv) == num_parts(data_snd)
  @check num_parts(parts_rcv) == num_parts(t_in)

  map_parts(schedule,t_in)
  map_parts(wait,t_in)

  @boundscheck _check_rcv_and_snd_match(parts_rcv,parts_snd)
  for part_rcv in 1:num_parts(parts_rcv)
    for (i, part_snd) in enumerate(parts_rcv.parts[part_rcv])
      j = first(findall(k->k==part_rcv,parts_snd.parts[part_snd]))
      data_rcv.parts[part_rcv][i] = data_snd.parts[part_snd][j]
    end
  end

  t_out = map_parts(data_rcv) do data_rcv
    @task nothing
  end
  t_out
end

function _check_rcv_and_snd_match(parts_rcv::SequentialDistributedData,parts_snd::SequentialDistributedData)
  @check num_parts(parts_rcv) == num_parts(parts_snd)
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
  parts_snd::SequentialDistributedData,
  t_in::SequentialDistributedData)

  @check num_parts(parts_rcv) == num_parts(data_rcv)
  @check num_parts(parts_rcv) == num_parts(data_snd)
  @check num_parts(parts_rcv) == num_parts(t_in)

  map_parts(schedule,t_in)
  map_parts(wait,t_in)

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

  t_out = map_parts(data_rcv) do data_rcv
    @task nothing
  end
  t_out
end
