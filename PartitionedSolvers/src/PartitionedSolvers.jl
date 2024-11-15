module PartitionedSolvers

using PartitionedArrays
using PartitionedArrays: val_parameter
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using Printf
import NLsolve
using SparseMatricesCSR

include("interfaces.jl")
include("wrappers.jl")
include("smoothers.jl")
include("amg.jl")
include("nonlinear_solvers.jl")
include("ode_solvers.jl")

end # module
