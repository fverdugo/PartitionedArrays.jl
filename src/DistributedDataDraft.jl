module DistributedDataDraft

export Backend
export distributed_run
export DistributedData
export num_parts
export map_parts
export Part
export Partition

export SequentialBackend
export sequential


include("Helpers.jl")

include("Interfaces.jl")

include("SequentialBackend.jl")

end
