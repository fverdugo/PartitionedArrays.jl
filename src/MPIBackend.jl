
# TODO remove this in the future.
struct MPIBackendDeprecated <: AbstractBackend end
const mpi = MPIBackendDeprecated()
function get_part_ids(b::MPIBackendDeprecated,nparts::Integer)
  @warn "The usage of the constant PartitionedArrays.mpi is deprecated. Use MPIBackend() instead."
  get_part_ids(MPIBackend(),nparts)
end
function get_part_ids(b::MPIBackendDeprecated,nparts::Tuple)
  @warn "The usage of the constant PartitionedArrays.mpi is deprecated. Use MPIBackend() instead."
  get_part_ids(MPIBackend(),nparts)
end
function prun_debug(driver::Function,b::MPIBackendDeprecated,nparts)
  @warn "The usage of the constant PartitionedArrays.mpi is deprecated. Use MPIBackend() instead."
  prun_debug(driver,MPIBackend(),nparts)
end

function num_parts(comm::MPI.Comm)
  if comm != MPI.COMM_NULL
    nparts = MPI.Comm_size(comm)
  else
    nparts = -1
  end
  nparts
end

function get_part_id(comm::MPI.Comm)
  if comm != MPI.COMM_NULL
    id = MPI.Comm_rank(comm)+1
  else
    id = -1
  end
  id
end

function i_am_in(comm::MPI.Comm)
  get_part_id(comm) >=0
end

function get_part_ids(comm::MPI.Comm)
  rank = get_part_id(comm)
  nparts = num_parts(comm)
  MPIData(rank,comm,(nparts,))
end

struct MPIBackend <: AbstractBackend end

function get_part_ids(b::MPIBackend,nparts::Integer)
  root_comm = MPI.Comm_dup(MPI.COMM_WORLD)
  size = MPI.Comm_size(root_comm)
  rank = MPI.Comm_rank(root_comm)
  need = prod(nparts)
  _get_part_ids_body(root_comm,rank,size,need,nparts)
end

function get_part_ids(b::MPIBackend,nparts::Tuple)
  root_comm = MPI.Comm_dup(MPI.COMM_WORLD)
  size = MPI.Comm_size(root_comm)
  rank = MPI.Comm_rank(root_comm)
  need = prod(nparts)
  _get_part_ids_body(root_comm,rank,size,need,nparts)
end

function _get_part_ids_body(root_comm,rank,size,need,nparts)
  # For compatibility between MPI post and pre v0.20
  @static if isdefined(MPI,:MPI_UNDEFINED)
    mpi_undefined = MPI.MPI_UNDEFINED[]
  else
    mpi_undefined = MPI.API.MPI_UNDEFINED[]
  end

  if size < need
    throw("Not enough MPI ranks, please run mpiexec with -n $need (at least)")
  elseif size > need
    if rank < need
      comm = MPI.Comm_split(root_comm, 0, 0)
      MPIData(get_part_id(comm),comm,Tuple(nparts))
    else
      comm = MPI.Comm_split(root_comm, mpi_undefined, mpi_undefined)
      MPIData(get_part_id(comm),comm,(-1,))
    end
  else
    comm = root_comm
    MPIData(get_part_id(comm),comm,Tuple(nparts))
  end
end


function with_backend(driver,b::MPIBackend,nparts,args...;kwargs...)
  if !MPI.Initialized()
    MPI.Init()
  end
  if MPI.Comm_size(MPI.COMM_WORLD) == 1
    part = get_part_ids(b,nparts)
    driver(part,args...;kwargs...)
  else
    try
      part = get_part_ids(b,nparts)
      if i_am_in(part)
        driver(part,args...;kwargs...)
      end
    catch e
      @error "" exception=(e, catch_backtrace())
      if MPI.Initialized() && !MPI.Finalized()
        MPI.Abort(MPI.COMM_WORLD,1)
      end
    end
  end
  # We are NOT invoking MPI.Finalize() here because we rely on
  # MPI.jl, which registers MPI.Finalize() in atexit()
end

#TODO not needed any more. Deprecate.
# Useful to debug an MPI program when executed interactively
# on the REPL, i.e., with a single MPI task
function prun_debug(driver::Function,b::MPIBackend,nparts)
  @warn "Function `prun_debug` is deprecated, use `with_backend` instead."
  if !MPI.Initialized()
    MPI.Init()
  end
  if (prod(nparts) != 1)
    MPI.Abort(MPI.COMM_WORLD,1)
  end
  part = get_part_ids(b,nparts)
  driver(part)
  # We are NOT invoking MPI.Finalize() here because we rely on
  # MPI.jl, which registers MPI.Finalize() in atexit()
end

struct MPIData{T,N} <: AbstractPData{T,N}
  part::T
  comm::MPI.Comm
  size::NTuple{N,Int}
  function MPIData(
    part::T,
    comm::MPI.Comm,
    size::NTuple{N,<:Integer}) where {T,N}
    @assert num_parts(comm) == prod(size)
    new{T,N}(part,comm,size)
  end
end

Base.size(a::MPIData) = a.size
get_part_id(a::MPIData) = get_part_id(a.comm)
get_backend(a::MPIData) = MPIBackend()
i_am_main(a::MPIData) = get_part_id(a.comm) == MAIN

function get_part_ids(a::MPIData)
  MPIData(get_part_id(a.comm),a.comm,a.size)
end
function num_parts(parts::MPIData)
  num_parts(parts.comm)
end
function i_am_in(parts::MPIData)
  i_am_in(parts.comm)
end

function map_parts(task,args::MPIData...)
  @assert length(args) > 0
  @assert all(a->a.comm===first(args).comm,args)
  parts_in = map(a->a.part,args)
  part_out = task(parts_in...)
  MPIData(part_out,first(args).comm,first(args).size)
end

#TODO Tables are not displayed correctly
function Base.show(io::IO,k::MIME"text/plain",data::MPIData)
  MPI.Barrier(data.comm)
  str = """
  On part $(get_part_id(data)) of $(num_parts(data)):
  $(data.part)
  """
  print(io,str)
  MPI.Barrier(data.comm)
end

get_part(a::MPIData) = a.part

function get_part(a::MPIData,part::Integer)
  rcv = MPI.Bcast!(Ref(copy(a.part)),part-1,a.comm)
  rcv[]
end

function get_part(a::MPIData{<:AbstractVector},part::Integer)
  l = map_parts(length,a)
  l_at_part = get_part(l,part)
  if get_part_id(a) == part
    rcv = a.part
  else
    rcv = similar(a.part,eltype(a.part),l_at_part)
  end
  MPI.Bcast!(rcv,part-1,a.comm)
  rcv
end

function gather!(rcv::MPIData,snd::MPIData)
  @assert rcv.comm === snd.comm
  if get_part_id(snd) == MAIN
    @assert length(rcv.part) == num_parts(snd)
    rcv.part[MAIN] = snd.part
    MPI.Gather!(MPI.IN_PLACE,MPI.UBuffer(rcv.part,1),MAIN-1,snd.comm)
  else
    MPI.Gather!(Ref(snd.part),nothing,MAIN-1,snd.comm)
  end
  rcv
end

function gather!(rcv::MPIData{<:Table},snd::MPIData)
  @assert rcv.comm === snd.comm
  if get_part_id(snd) == MAIN
    @assert length(rcv.part) == num_parts(snd)
    kini = rcv.part.ptrs[MAIN]
    kend = rcv.part.ptrs[MAIN+1]-1
    rcv.part.data[kini:kend] = snd.part
    counts = ptrs_to_counts(rcv.part.ptrs)
    MPI.Gatherv!(MPI.IN_PLACE,MPI.VBuffer(rcv.part.data,counts),MAIN-1,snd.comm)
  else
    MPI.Gatherv!(snd.part,nothing,MAIN-1,snd.comm)
  end
  rcv
end

function gather_all!(rcv::MPIData,snd::MPIData)
  @assert rcv.comm === snd.comm
  @assert length(rcv.part) == num_parts(snd)
  MPI.Allgather!(Ref(snd.part),MPI.UBuffer(rcv.part,1),snd.comm)
  rcv
end

function gather_all!(rcv::MPIData{<:Table},snd::MPIData)
  @assert rcv.comm === snd.comm
  @assert length(rcv.part) == num_parts(snd)
  counts = ptrs_to_counts(rcv.part.ptrs)
  MPI.Allgatherv!(snd.part,MPI.VBuffer(rcv.part.data,counts),snd.comm)
  rcv
end

function scatter(snd::MPIData)
  if get_part_id(snd) == MAIN
    part = snd.part[MAIN]
    MPI.Scatter!(MPI.UBuffer(snd.part,1),MPI.IN_PLACE,MAIN-1,snd.comm)
  else
    rcv = Vector{eltype(snd.part)}(undef,1)
    MPI.Scatter!(nothing,rcv,MAIN-1,snd.comm)
    part = rcv[1]
  end
  MPIData(part,snd.comm,snd.size)
end

function scatter(snd::MPIData{<:Table})
  counts_main = map_parts(ptrs_to_counts∘get_ptrs,snd)
  counts_scat = scatter(counts_main)
  if get_part_id(snd) == MAIN
    buf = MPI.VBuffer(snd.part.data,counts_main.part)
    MPI.Scatterv!(buf,MPI.IN_PLACE,MAIN-1,snd.comm)
    rcv = snd.part[MAIN]
  else
    T=eltype(snd.part)
    # This void Vbuffer is required to circumvent a deadlock
    # that we found with OpenMPI 4.1.X on Gadi. In particular, the
    # deadlock arises whenever buf is set to nothing
    buf = MPI.VBuffer(eltype(T)[],eltype(T)[])
    rcv = T(undef,counts_scat.part)
    MPI.Scatterv!(buf,rcv,MAIN-1,snd.comm)
  end
  MPIData(rcv,snd.comm,snd.size)
end

function scatter(snd::MPIData{<:AbstractVector{<:AbstractVector}})
  snd_table = map_parts(Table,snd)
  scatter(snd_table)
end

function emit(snd::MPIData)
  MPIData(get_main_part(snd),snd.comm,snd.size)
end

function async_exchange!(
  data_rcv::MPIData,
  data_snd::MPIData,
  parts_rcv::MPIData,
  parts_snd::MPIData,
  t_in::MPIData)

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
    @static if isdefined(MPI,:Waitall)
        MPI.Waitall(req_all,MPI.Status)
    else
        MPI.Waitall!(req_all)
    end
  end

  t_out = MPIData(t2,comm,s)
  t_out
end

function async_exchange!(
  data_rcv::MPIData{<:Table},
  data_snd::MPIData{<:Table},
  parts_rcv::MPIData,
  parts_snd::MPIData,
  t_in::MPIData)

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
    @static if isdefined(MPI,:Waitall)
      MPI.Waitall(req_all,MPI.Status)
    else
      MPI.Waitall!(req_all)
    end
  end

  t_out = MPIData(t2,comm,s)
  t_out
end
