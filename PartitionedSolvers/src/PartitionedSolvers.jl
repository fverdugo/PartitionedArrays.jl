module PartitionedSolvers

using PartitionedArrays
using PartitionedArrays: val_parameter
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using Printf
import NLsolve
using SparseMatricesCSR

include("new/interfaces.jl")
include("new/wrappers.jl")
include("new/smoothers.jl")
include("new/amg.jl")

end # module
