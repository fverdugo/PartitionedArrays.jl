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
y = similar(x)
y .= 0

solver = amg()
S = setup(solver)(y,A,b)
use!(solver)(y,S,b)
setup!(solver)(S,2*A)
use!(solver)(y,S,b)
finalize!(solver)(S)

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

# Now with a nullspace

O = attach_nullspace(A,default_nullspace(A))
S = setup(solver)(y,O,b)
use!(solver)(y,S,b)
setup!(solver)(S,2*A)
use!(solver)(y,S,b)
finalize!(solver)(S)

# Now in parallel

parts_per_dir = (2,2)
np = prod(parts_per_dir)
parts = DebugArray(LinearIndices((np,)))

nodes_per_dir = (4,4)
A = laplace_matrix(nodes_per_dir,parts_per_dir,parts)
x = pones(partition(axes(A,2)))
b = A*x

y = similar(x)
y .= 0

solver = amg()
S = setup(solver)(y,A,b)
use!(solver)(y,S,b)
setup!(solver)(S,2*A)
use!(solver)(y,S,b)
finalize!(solver)(S)

# Now with a nullspace

O = attach_nullspace(A,default_nullspace(A))
S = setup(solver)(y,O,b)
use!(solver)(y,S,b)
setup!(solver)(S,2*A)
use!(solver)(y,S,b)
finalize!(solver)(S)

end
