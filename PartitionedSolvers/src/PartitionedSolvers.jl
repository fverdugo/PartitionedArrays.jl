module PartitionedSolvers

using PartitionedArrays
using PartitionedArrays: val_parameter
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using Printf

export setup
export solve
export solve!
export update!
export finalize!
export AbstractLinearSolver
export linear_solver
export default_nullspace
export nullspace
export uses_nullspace
export uses_initial_guess
export iterations!
include("interfaces.jl")

export lu_solver
export jacobi_correction
export richardson
export jacobi
export gauss_seidel
export additive_schwarz_correction
export additive_schwarz
export conjugate_gradients
include("smoothers.jl")

export amg
export smoothed_aggregation
export v_cycle
export w_cycle
export amg_level_params
export amg_level_params_linear_elasticity
export amg_fine_params
export amg_coarse_params
export amg_statistics
include("amg.jl")

export LinearProblem
export matrix
export rhs
export jacobian
export tangent
export nonlinear_solver
export nonlinear_operator
include("nonlinear_solvers.jl")

end # module
