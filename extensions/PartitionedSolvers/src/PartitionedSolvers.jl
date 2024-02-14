module PartitionedSolvers

using PartitionedArrays
using SparseArrays
using LinearAlgebra

export setup
export use!
export setup!
export finalize!
export AbstractLinearProblem
export AbstractLinearSolver
export linear_solver
export linear_problem
export replace_matrix
export replace_rhs
include("interfaces.jl")

export do_nothing_linear_solver
export lu_solver
export diagonal_solver
export richardson
export jacobi
export additive_schwarz
include("smoothers.jl")

export amg
include("amg.jl")

end # module
