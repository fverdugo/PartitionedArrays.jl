
# TODO remove this in the future
struct SequentialBackendDeprecated <: AbstractBackend end
const sequential = SequentialBackendDeprecated()
function get_part_ids(b::SequentialBackendDeprecated,nparts::Integer)
  @warn "The usage of the constant PartitionedArrays.sequential is deprecated. Use SequentialBackend() instead."
  parts = [ part for part in 1:nparts ]
  SequentialData(parts)
end
function get_part_ids(b::SequentialBackendDeprecated,nparts::Tuple)
  @warn "The usage of the constant PartitionedArrays.sequential is deprecated. Use SequentialBackend() instead."
  parts = collect(LinearIndices(nparts))
  SequentialData(parts)
end
function prun_debug(driver::Function,b::SequentialBackendDeprecated,nparts)
  @warn "The usage of the constant PartitionedArrays.sequential is deprecated. Use SequentialBackend() instead."
  @warn "Function `prun_debug` is deprecated, use `with_backend` instead."
  with_backend(driver,b,nparts)
end

struct SequentialBackend <: AbstractBackend end

function get_part_ids(b::SequentialBackend,nparts::Integer)
  parts = [ part for part in 1:nparts ]
  SequentialData(parts)
end

function get_part_ids(b::SequentialBackend,nparts::Tuple)
  parts = collect(LinearIndices(nparts))
  SequentialData(parts)
end

function prun_debug(driver::Function,b::SequentialBackend,nparts)
  @warn "Function `prun_debug` is deprecated, use `with_backend` instead."
  with_backend(driver,b,nparts)
end

struct SequentialData{T,N} <: AbstractPData{T,N}
  parts::Array{T,N}
end

Base.size(a::SequentialData) = size(a.parts)

i_am_main(a::SequentialData) = true

get_backend(a::SequentialData) = SequentialBackend()

function Base.iterate(a::SequentialData)
  next = map_parts(iterate,a)
  if eltype(next.parts) == Nothing || any(i->i==Nothing,next.parts)
    return nothing
  end
  item = map_parts(first,next)
  state = map_parts(_second,next)
  item, state
end

_second(a) = a[2]

function Base.iterate(a::SequentialData,state)
  next = map_parts(iterate,a,state)
  if eltype(next.parts) == Nothing || any(i->i==Nothing,next.parts)
    return nothing
  end
  item = map_parts(first,next)
  state = map_parts(_second,next)
  item, state
end

function map_parts(task,args::SequentialData...)
  @assert length(args) > 0
  @assert all(a->length(a.parts)==length(first(args).parts),args)
  parts_in = map(a->a.parts,args)
  parts_out = map(task,parts_in...)
  SequentialData(parts_out)
end

function Base.show(io::IO,k::MIME"text/plain",data::SequentialData)
  for part in 1:num_parts(data)
    if part != 1
      println(io,"")
    end
    println(io,"On part $part of $(num_parts(data)):")
    show(io,k,data.parts[part])
  end
end

get_part(a::SequentialData,part::Integer) = a.parts[part]
get_part(a::SequentialData) = get_main_part(a)

function gather!(rcv::SequentialData,snd::SequentialData)
  @assert num_parts(rcv) == num_parts(snd)
  @assert length(rcv.parts[MAIN]) == num_parts(snd)
  for part in 1:num_parts(snd)
    rcv.parts[MAIN][part] = snd.parts[part]
  end
  rcv
end

function gather!(rcv::SequentialData{<:Table},snd::SequentialData)
  @assert num_parts(rcv) == num_parts(snd)
  @assert length(rcv.parts[MAIN]) == num_parts(snd)
  for part in 1:num_parts(snd)
    offset = rcv.parts[MAIN].ptrs[part]-1
    for i in 1:length(snd.parts[part])
      rcv.parts[MAIN].data[i+offset] = snd.parts[part][i]
    end
  end
  rcv
end

function gather_all!(rcv::SequentialData,snd::SequentialData)
  @assert num_parts(rcv) == num_parts(snd)
  for part_rcv in 1:num_parts(rcv)
    @assert length(rcv.parts[part_rcv]) == num_parts(snd)
    for part_snd in 1:num_parts(snd)
      rcv.parts[part_rcv][part_snd] = snd.parts[part_snd]
    end
  end
  rcv
end

function gather_all!(rcv::SequentialData{<:Table},snd::SequentialData)
  @assert num_parts(rcv) == num_parts(snd)
  for part_rcv in 1:num_parts(rcv)
    @assert length(rcv.parts[part_rcv]) == num_parts(snd)
    for part_snd in 1:num_parts(snd)
      offset = rcv.parts[part_rcv].ptrs[part_snd]-1
      for i in 1:length(snd.parts[part_snd])
        rcv.parts[part_rcv].data[i+offset] = snd.parts[part_snd][i]
      end
    end
  end
  rcv
end

function scatter(snd::SequentialData)
  @assert length(snd.parts[MAIN]) == num_parts(snd)
  parts = similar(snd.parts,eltype(snd.parts[MAIN]),size(snd.parts))
  copyto!(parts,snd.parts[MAIN])
  SequentialData(parts)
end

function async_exchange!(
  data_rcv::SequentialData,
  data_snd::SequentialData,
  parts_rcv::SequentialData,
  parts_snd::SequentialData,
  t_in::AbstractPData)

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

function _check_rcv_and_snd_match(parts_rcv::SequentialData,parts_snd::SequentialData)
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
  data_rcv::SequentialData{<:Table},
  data_snd::SequentialData{<:Table},
  parts_rcv::SequentialData,
  parts_snd::SequentialData,
  t_in::SequentialData)

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
