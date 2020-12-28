
struct SequentialBackend <: Backend end

const sequential = SequentialBackend()

function Partition(b::SequentialBackend,nparts::Integer)
  parts = [ Part(part,nparts) for part in 1:nparts ]
  SequentialDistributedData(parts)
end

struct SequentialDistributedData{T} <: DistributedData{T}
  parts::Vector{T}
end

num_parts(a::SequentialDistributedData) = length(a.parts)

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
  parts_snd::SequentialDistributedData)

  @check num_parts(parts_rcv) == num_parts(data_rcv)
  @check num_parts(parts_rcv) == num_parts(data_snd)

  @boundscheck _check_rcv_and_snd_match(parts_rcv,parts_snd)
  for part_rcv in 1:num_parts(parts_rcv)
    for (i, part_snd) in enumerate(parts_rcv.parts[part_rcv])
      j = first(findall(k->k==part_rcv,parts_snd.parts[part_snd]))
      data_rcv.parts[part_rcv][i] = data_snd.parts[part_snd][j]
    end
  end
  map_parts(data_rcv) do data_rcv
    @async nothing
  end
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
