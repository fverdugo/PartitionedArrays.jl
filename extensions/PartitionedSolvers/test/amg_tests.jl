module AMGTests

using PartitionedArrays
using PartitionedArrays: laplace_matrix
using PartitionedSolvers
using LinearAlgebra
using Test
using IterativeSolvers: cg!

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

# Non-default options

level_params = amg_level_params(;
    pre_smoother = jacobi(;maxiters=10,omega=2/3),
    cycle = w_cycle()
   )

fine_params = amg_fine_params(;
    level_params,
    n_fine_levels=5)

coarse_params = (;
    coarse_solver = lu_solver(),
    coarse_size = 15,
   )

solver = amg(;fine_params,coarse_params)

# Now with a nullspace

O = attach_nullspace(A,default_nullspace(A))
S = setup(solver)(y,O,b)
use!(solver)(y,S,b)
setup!(solver)(S,2*A)
use!(solver)(y,S,b)
finalize!(solver)(S)

# Now as a preconditioner
Pl = preconditioner(amg(),y,A,b)
y .= 0
cg!(y,A,b;Pl,verbose=true)

# Now in parallel

parts_per_dir = (2,2)
np = prod(parts_per_dir)
parts = DebugArray(LinearIndices((np,)))

nodes_per_dir = (10,10)
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

Pl = preconditioner(amg(),y,A,b)
y .= 0
cg!(y,A,b;Pl,verbose=true)

end
