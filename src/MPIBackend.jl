

get_part(comm::MPI.Comm) = MPI.Comm_rank(comm)+1
num_parts(comm::MPI.Comm) = MPI.Comm_size(comm)

struct MPIBackend <: Backend end

const mpi = MPIBackend()

function get_parts(b::MPIBackend,nparts::Integer)
  comm = MPI.COMM_WORLD
  @notimplementedif num_parts(comm) != nparts
  MPIDistributedData(get_part(comm),comm)
end

function distributed_run(driver::Function,b::MPIBackend,nparts)
  MPI.Init()
  #try 
    part = get_parts(b,nparts)
    driver(part)
  #finally
  #  MPI.Finalize()
  #end
end

struct MPIDistributedData{T} <: DistributedData{T}
  part::T
  comm::MPI.Comm
end

num_parts(a::MPIDistributedData) = num_parts(a.comm)
get_part(a::MPIDistributedData) = get_part(a.comm)
i_am_master(a::MPIDistributedData) = get_part(a) == 1
get_backend(a::MPIDistributedData) = mpi

function Base.iterate(a::MPIDistributedData)
  next = iterate(a.part)
  if next == nothing
    return next
  end
  item, state = next
  MPIDistributedData(item,a.comm), state
end

function Base.iterate(a::MPIDistributedData,state)
  next = iterate(a.part,state)
  if next == nothing
    return next
  end
  item, state = next
  MPIDistributedData(item,a.comm), state
end

function map_parts(task::Function,args::MPIDistributedData...)
  @assert length(args) > 0
  @assert all(a->a.comm===first(args).comm,args)
  parts_in = map(a->a.part,args)
  part_out = task(parts_in...)
  MPIDistributedData(part_out,first(args).comm)
end

function Base.show(io::IO,k::MIME"text/plain",data::MPIDistributedData)
  MPI.Barrier(data.comm)
  str = """
  On part $(get_part(data)) of $(num_parts(data)):
  $(data.part)
  """
  print(io,str)
  MPI.Barrier(data.comm)
end

function async_exchange!(
  data_rcv::MPIDistributedData,
  data_snd::MPIDistributedData,
  parts_rcv::MPIDistributedData,
  parts_snd::MPIDistributedData,
  t_in::MPIDistributedData)

  @check parts_rcv.comm === data_rcv.comm
  @check parts_rcv.comm === data_snd.comm
  @check parts_rcv.comm === t_in.comm
  comm = parts_rcv.comm

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
      tag_snd = get_part(comm)
      reqs = MPI.Isend(buff_snd,rank_snd,tag_snd,comm)
      push!(req_all,reqs)
    end

    return req_all
  end

  t2 = @task begin
    req_all = fetch(t1)
    MPI.Waitall!(req_all)
  end

  t_out = MPIDistributedData(t2,comm)
  t_out
end

function async_exchange!(
  data_rcv::MPIDistributedData{<:Table},
  data_snd::MPIDistributedData{<:Table},
  parts_rcv::MPIDistributedData,
  parts_snd::MPIDistributedData,
  t_in::MPIDistributedData)

  @check parts_rcv.comm === data_rcv.comm
  @check parts_rcv.comm === data_snd.comm
  @check parts_rcv.comm === t_in.comm
  comm = parts_rcv.comm

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
      tag_snd = get_part(comm)
      reqs = MPI.Isend(buff_snd,rank_snd,tag_snd,comm)
      push!(req_all,reqs)
    end

    return req_all
  end

  t2 = @task begin
    req_all = fetch(t1)
    MPI.Waitall!(req_all)
  end

  t_out = MPIDistributedData(t2,comm)
  t_out
end

