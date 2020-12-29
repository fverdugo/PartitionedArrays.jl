module DistributedDataDraft

export Backend
export distributed_run
export DistributedData
export num_parts
export map_parts
export get_parts
export get_backend
export async_exchange!
export exchange!
export exchange
export async_exchange
export Table
export discover_parts_snd
export IndexSet
export num_gids
export num_lids
export num_oids
export num_hids
export Exchanger
export allocate_rcv_buffer
export allocate_snd_buffer
export DistributedRange
export DistributedVector
export async_assemble!
export assemble!

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
