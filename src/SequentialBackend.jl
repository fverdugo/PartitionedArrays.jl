
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
