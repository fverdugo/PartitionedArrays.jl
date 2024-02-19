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
using Random
Random.seed!(1)
x = rand(size(A,2))
b = A*x
y = similar(x)
y .= 0

solver = amg()
S = setup(solver)(y,A,b)
solve!(solver)(y,S,b)
setup!(solver)(S,2*A)
solve!(solver)(y,S,b)
finalize!(solver)(S)

amg_statistics(S) |> display

# Non-default options

level_params = amg_level_params(;
    pre_smoother = jacobi(;iters=10,omega=2/3),
    cycle = v_cycle,
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
solve!(solver)(y,S,b)
setup!(solver)(S,2*A)
solve!(solver)(y,S,b)
finalize!(solver)(S)

# Now as a preconditioner

level_params = amg_level_params(;
   pre_smoother = gauss_seidel(;iters=1),
   )

fine_params = amg_fine_params(;level_params)

Pl = preconditioner(amg(;fine_params),y,A,b)
y .= 0
cg!(y,A,b;Pl,verbose=true)



# Now in parallel

parts_per_dir = (4,4)
np = prod(parts_per_dir)
parts = DebugArray(LinearIndices((np,)))

nodes_per_dir = (100,100)
A = laplace_matrix(nodes_per_dir,parts_per_dir,parts)
x = pones(partition(axes(A,2)))
b = A*x

y = similar(x)
y .= 0

solver = amg()
S = setup(solver)(y,A,b)
amg_statistics(S) |> display
solve!(solver)(y,S,b)
setup!(solver)(S,2*A)
solve!(solver)(y,S,b)
finalize!(solver)(S)

# Now with a nullspace

O = attach_nullspace(A,default_nullspace(A))
S = setup(solver)(y,O,b)
solve!(solver)(y,S,b)
setup!(solver)(S,2*A)
solve!(solver)(y,S,b)
finalize!(solver)(S)

level_params = amg_level_params(;
    pre_smoother = jacobi(;iters=1,omega=2/3),
   )

fine_params = amg_fine_params(;
    level_params,
    n_fine_levels=5)

solver = amg(;fine_params)

Pl = preconditioner(solver,y,A,b)
y .= 0
cg!(y,A,b;Pl,verbose=true)

level_params = amg_level_params(;
    pre_smoother = richardson(additive_schwarz(gauss_seidel(;iters=1)),iters=1),
   )

fine_params = amg_fine_params(;
    level_params,
    n_fine_levels=5)

solver = amg(;fine_params)

Pl = preconditioner(solver,y,A,b)
y .= 0
cg!(y,A,b;Pl,verbose=true)

end
