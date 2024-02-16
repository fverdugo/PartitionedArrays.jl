module AMGTests

using PartitionedArrays
using PartitionedArrays: laplace_matrix
using PartitionedSolvers
using LinearAlgebra
using Test

# First with a sequential matrix
nodes_per_dir = (100,100)
A = laplace_matrix(nodes_per_dir)
x = ones(axes(A,2))
b = A*x

problem = linear_problem(A,b)
y = similar(x)
y .= 0

solver = amg()
workspace = setup(solver)(problem,y)
use!(solver)(problem,y,workspace)
problem = replace_matrix(problem,2*A)
setup!(solver)(problem,y,workspace)
use!(solver)(problem,y,workspace)
finalize!(solver)(workspace)

level_params = (;
    pre_smoother = jacobi(;maxiters=10,omega=2/3),
    pos_smoother = jacobi(;maxiters=10,omega=2/3),
    coarsening = smoothed_aggregation(;epsilon=0,omega=1),
    cycle = w_cycle()
   )

coarse_params = (;
    coarse_solver = lu_solver(),
    coarse_size = 15,
   )

nfine = 10
fine_params = fill(level_params,nfine)

solver = amg(;fine_params,coarse_params)
workspace = setup(solver)(problem,y)
use!(solver)(problem,y,workspace)

parts_per_dir = (2,2)
np = prod(parts_per_dir)
parts = DebugArray(LinearIndices((np,)))

nodes_per_dir = (4,4)
A = laplace_matrix(nodes_per_dir,parts_per_dir,parts)
x = pones(partition(axes(A,2)))
b = A*x

problem = linear_problem(A,b)
y = similar(x)
y .= 0

solver = amg()
workspace = setup(solver)(problem,y)
use!(solver)(problem,y,workspace)
problem = replace_matrix(problem,2*A)
setup!(solver)(problem,y,workspace)
use!(solver)(problem,y,workspace)
finalize!(solver)(workspace)

end
