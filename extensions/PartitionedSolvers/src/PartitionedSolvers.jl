module PartitionedSolvers

using PartitionedArrays
using SparseArrays
using LinearAlgebra

export setup
export use!
export setup!
export finalize!
export AbstractLinearSolver
export linear_solver
export attach_nullspace
export default_nullspace
include("interfaces.jl")

export lu_solver
export diagonal_solver
export richardson
export jacobi
export additive_schwarz
include("smoothers.jl")

export amg
export smoothed_aggregation
export v_cycle
export w_cycle
include("amg.jl")

end # module
