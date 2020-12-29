module DistributedDataDraft

export Backend
export distributed_run
export DistributedData
export num_parts
export map_parts
export get_parts
export async_exchange!
export exchange!
export exchange
export async_exchange
export Table

export SequentialBackend
export sequential

export MPIBackend
export mpi

include("Helpers.jl")

include("Interfaces.jl")

include("SequentialBackend.jl")

import MPI
include("MPIBackend.jl")

end
