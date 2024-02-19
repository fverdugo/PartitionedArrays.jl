module SmoothersTests

using PartitionedArrays
using PartitionedArrays: laplace_matrix
using PartitionedSolvers
using LinearAlgebra
using Test

np = 4
parts = DebugArray(LinearIndices((np,)))

parts_per_dir = (2,2)
nodes_per_dir = (8,8)
A = laplace_matrix(nodes_per_dir,parts_per_dir,parts)
x = pones(partition(axes(A,2)))
b = A*x

solver = lu_solver()
y = similar(x)
S = setup(solver)(y,A,b)
solve!(solver)(y,S,b)
tol = 1.e-8
@test norm(y-x)/norm(x) < tol
setup!(solver)(S,2*A)
solve!(solver)(y,S,b)
@test norm(y-x/2)/norm(x/2) < tol
finalize!(solver)(S)

solver = richardson(lu_solver(),iters=1)
y = similar(x)
y .= 0
S = setup(solver)(y,A,b)
solve!(solver)(y,S,b)
tol = 1.e-8
@test norm(y-x)/norm(x) < tol
setup!(solver)(S,2*A)
solve!(solver)(y,S,b)
@test norm(y-x/2)/norm(x/2) < tol
finalize!(solver)(S)

solver = jacobi(;iters=1000)
y = similar(x)
y .= 0
S = setup(solver)(y,A,b)
solve!(solver)(y,S,b)
tol = 1.e-8
@test norm(y-x)/norm(x) < tol
setup!(solver)(S,2*A)
solve!(solver)(y,S,b)
@test norm(y-x/2)/norm(x/2) < tol
finalize!(solver)(S)

solver = additive_schwarz(lu_solver())
y = similar(x)
y .= 0
S = setup(solver)(y,A,b)
solve!(solver)(y,S,b)
setup!(solver)(S,2*A)
solve!(solver)(y,S,b)
finalize!(solver)(S)

end #module
