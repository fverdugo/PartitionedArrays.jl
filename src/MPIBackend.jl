
# Wrap mpi calls

get_part_id(comm::MPI.Comm) = MPI.Comm_rank(comm)+1
num_parts(comm::MPI.Comm) = MPI.Comm_size(comm)
get_part(comm::MPI.Comm) = Part(get_part_id(comm),num_parts(comm))

struct MPIBackend <: Backend end

const mpi = MPIBackend()

function Partition(b::MPIBackend,nparts::Integer)
  comm = MPI.COMM_WORLD
  @notimplementedif num_parts(comm) != nparts
  MPIDistributedData(get_part(comm),comm)
end

function distributed_run(driver::Function,b::MPIBackend,nparts)
  MPI.Init()
  try 
    part = Partition(b,nparts)
    driver(part)
  finally
    MPI.Finalize()
  end
end

struct MPIDistributedData{T} <: DistributedData{T}
  part::T
  comm::MPI.Comm
end

num_parts(a::MPIDistributedData) = num_parts(a.comm)
get_part_id(a::MPIDistributedData) = get_part_id(a.comm)
get_part(a::MPIDistributedData) = get_part(a.comm)
i_am_master(a::MPIDistributedData) = get_part_id(a) == 1

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
  On part $(get_part_id(data)) of $(num_parts(data)):
  $(data.part)
  """
  print(io,str)
  MPI.Barrier(data.comm)
end
