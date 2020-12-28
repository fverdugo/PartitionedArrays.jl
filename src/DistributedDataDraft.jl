module DistributedDataDraft

export Backend
export distributed_run
export DistributedData
export num_parts
export map_parts
export Part

export SequentialBackend
export sequential


include("Helpers.jl")

include("Interfaces.jl")

include("Sequential.jl")

end
