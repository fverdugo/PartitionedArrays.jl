module PartitionedSolvers

using PartitionedArrays
using SparseArrays
using LinearAlgebra

export setup
export solve!
export setup!
export finalize!
export apply!
export AbstractLinearSolver
export linear_solver
include("interfaces.jl")

export do_nothing_linear_solver
export lu_solver
export ilu_solver
export diagonal_solver
export richardson_solver
export jacobi_solver
export additive_schwartz_solver
include("smoothers.jl")

end # module
