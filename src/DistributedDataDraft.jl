module DistributedDataDraft

using Gridap.Arrays: Table, length_to_ptrs!, rewind_ptrs!, lazy_map, PosNegReindex, PosNegPartition, Reindex
using SparseArrays: AbstractSparseMatrix, findnz, sparse
using FillArrays
using LinearAlgebra

export Communicator
export num_parts
export num_workers
export do_on_parts
export map_on_parts
export i_am_master
export OrchestratedCommunicator
export Communicator
export DistributedData
export get_comm
export get_part_type
export gather!
export gather
export scatter
export bcast
export get_distributed_data
export spawn_exchange!
export exchange!
export exchange
export discover_parts_snd
export num_oids
export UniformIndexSet
export IndexSet
export num_lids
export Exchanger
export allocate_rcv_buffer
export allocate_snd_buffer
export DistributedIndexSet
export num_gids
export non_overlaping
export DistributedVector
export DistributedSparseMatrix
export AdditiveSchwarz

export SequentialCommunicator

include("Helpers.jl")

include("Interfaces.jl")

include("Sequential.jl")

end
