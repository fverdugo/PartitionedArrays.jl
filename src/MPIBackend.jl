
get_part_id(comm::MPI.Comm) = MPI.Comm_rank(comm)+1
num_parts(comm::MPI.Comm) = MPI.Comm_size(comm)

struct MPIBackend <: Backend end

const mpi = MPIBackend()

function get_part_ids(b::MPIBackend,nparts::Integer)
  comm = MPI.COMM_WORLD
  @notimplementedif num_parts(comm) != nparts
  MPIChunkyData(get_part_id(comm),comm,(nparts,))
end

function get_part_ids(b::MPIBackend,nparts::Tuple)
  comm = MPI.COMM_WORLD
  @notimplementedif num_parts(comm) != prod(nparts)
  MPIChunkyData(get_part_id(comm),comm,nparts)
end

function distributed_run(driver::Function,b::MPIBackend,nparts)
  MPI.Init()
  #try 
    part = get_part_ids(b,nparts)
    driver(part)
  #finally
  #  MPI.Finalize()
  #end
end

struct MPIChunkyData{T,N} <: ChunkyData{T,N}
  part::T
  comm::MPI.Comm
  size::NTuple{N,Int}
  function MPIChunkyData(
    part::T,
    comm::MPI.Comm,
    size::NTuple{N,<:Integer}) where {T,N}
    @assert num_parts(comm) == prod(size)
    new{T,N}(part,comm,size)
  end
end

Base.size(a::MPIChunkyData) = a.size
get_part_id(a::MPIChunkyData) = get_part_id(a.comm)
get_backend(a::MPIChunkyData) = mpi
i_am_master(a::MPIChunkyData) = get_part_id(a.comm) == MASTER

function Base.iterate(a::MPIChunkyData)
  next = iterate(a.part)
  if next == nothing
    return next
  end
  item, state = next
  MPIChunkyData(item,a.comm,a.size), state
end

function Base.iterate(a::MPIChunkyData,state)
  next = iterate(a.part,state)
  if next == nothing
    return next
  end
  item, state = next
  MPIChunkyData(item,a.comm,a.size), state
end

function map_parts(task::Function,args::MPIChunkyData...)
  @assert length(args) > 0
  @assert all(a->a.comm===first(args).comm,args)
  parts_in = map(a->a.part,args)
  part_out = task(parts_in...)
  MPIChunkyData(part_out,first(args).comm,first(args).size)
end

function Base.show(io::IO,k::MIME"text/plain",data::MPIChunkyData)
  MPI.Barrier(data.comm)
  str = """
  On part $(get_part_id(data)) of $(num_parts(data)):
  $(data.part)
  """
  print(io,str)
  MPI.Barrier(data.comm)
end

get_part(a::MPIChunkyData) = a.part

function get_part(a::MPIChunkyData,part::Integer)
  part = MPI.Bcast!(Ref(copy(a.part)),part-1,a.comm)
  part[]
end

function gather!(rcv::MPIChunkyData,snd::MPIChunkyData)
  @assert rcv.comm === snd.comm
  if get_part_id(snd) == MASTER
    @assert length(rcv.part) == num_parts(snd)
    rcv.part[MASTER] = snd.part
    MPI.Gather!(MPI.IN_PLACE,MPI.UBuffer(rcv.part,1),MASTER-1,snd.comm)
  else
    MPI.Gather!(Ref(snd.part),nothing,MASTER-1,snd.comm)
  end
  rcv
end

function gather_all!(rcv::MPIChunkyData,snd::MPIChunkyData)
  @assert rcv.comm === snd.comm
  @assert length(rcv.part) == num_parts(snd)
  MPI.Allgather!(Ref(snd.part),MPI.UBuffer(rcv.part,1),snd.comm)
  rcv
end

function scatter(snd::MPIChunkyData)
  if get_part_id(snd) == MASTER
    part = snd.part[MASTER]
    MPI.Scatter!(MPI.UBuffer(snd.part,1),MPI.IN_PLACE,MASTER-1,snd.comm)
  else
    rcv = Vector{eltype(snd.part)}(undef,1)
    MPI.Scatter!(nothing,rcv,MASTER-1,snd.comm)        
    part = rcv[1]
  end
  MPIChunkyData(part,snd.comm,snd.size)
end

function bcast(snd::MPIChunkyData)
  MPIChunkyData(get_master_part(snd),snd.comm,snd.size)
end

function async_exchange!(
  data_rcv::MPIChunkyData,
  data_snd::MPIChunkyData,
  parts_rcv::MPIChunkyData,
  parts_snd::MPIChunkyData,
  t_in::MPIChunkyData)

  @check parts_rcv.comm === data_rcv.comm
  @check parts_rcv.comm === data_snd.comm
  @check parts_rcv.comm === t_in.comm
  comm = parts_rcv.comm
  s = parts_rcv.size

  t0 = t_in.part

  t1 = @async begin

    req_all = MPI.Request[]
    wait(schedule(t0))

    for (i,part_rcv) in enumerate(parts_rcv.part)
      rank_rcv = part_rcv-1
      buff_rcv = view(data_rcv.part,i:i)
      tag_rcv = part_rcv
      reqr = MPI.Irecv!(buff_rcv,rank_rcv,tag_rcv,comm)
      push!(req_all,reqr)
    end

    for (i,part_snd) in enumerate(parts_snd.part)
      rank_snd = part_snd-1
      buff_snd = view(data_snd.part,i:i)
      tag_snd = get_part_id(comm)
      reqs = MPI.Isend(buff_snd,rank_snd,tag_snd,comm)
      push!(req_all,reqs)
    end

    return req_all
  end

  t2 = @task begin
    req_all = fetch(t1)
    MPI.Waitall!(req_all)
  end

  t_out = MPIChunkyData(t2,comm,s)
  t_out
end

function async_exchange!(
  data_rcv::MPIChunkyData{<:Table},
  data_snd::MPIChunkyData{<:Table},
  parts_rcv::MPIChunkyData,
  parts_snd::MPIChunkyData,
  t_in::MPIChunkyData)

  @check parts_rcv.comm === data_rcv.comm
  @check parts_rcv.comm === data_snd.comm
  @check parts_rcv.comm === t_in.comm
  comm = parts_rcv.comm
  s = parts_rcv.size

  t0 = t_in.part

  t1 = @async begin

    req_all = MPI.Request[]
    wait(schedule(t0))

    for (i,part_rcv) in enumerate(parts_rcv.part)
      rank_rcv = part_rcv-1
      ptrs_rcv = data_rcv.part.ptrs
      buff_rcv = view(data_rcv.part.data,ptrs_rcv[i]:(ptrs_rcv[i+1]-1))
      tag_rcv = part_rcv
      reqr = MPI.Irecv!(buff_rcv,rank_rcv,tag_rcv,comm)
      push!(req_all,reqr)
    end

    for (i,part_snd) in enumerate(parts_snd.part)
      rank_snd = part_snd-1
      ptrs_snd = data_snd.part.ptrs
      buff_snd = view(data_snd.part.data,ptrs_snd[i]:(ptrs_snd[i+1]-1))
      tag_snd = get_part_id(comm)
      reqs = MPI.Isend(buff_snd,rank_snd,tag_snd,comm)
      push!(req_all,reqs)
    end

    return req_all
  end

  t2 = @task begin
    req_all = fetch(t1)
    MPI.Waitall!(req_all)
  end

  t_out = MPIChunkyData(t2,comm,s)
  t_out
end

