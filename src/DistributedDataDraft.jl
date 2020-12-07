module DistributedDataDraft

export Communicator
export num_parts
export num_workers
export do_on_parts
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
export exchange!
export exchange
export discover_parts_snd
export IndexSet
export DistributedIndexSet
export num_gids

export SequentialCommunicator

include("Helpers.jl")

include("Interfaces.jl")

include("Sequential.jl")

end
