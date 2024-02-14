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
problem = linear_problem(A,b)
y = similar(x)
S = setup(solver)(problem,y)
use!(solver)(problem,y,S)
tol = 1.e-8
@test norm(y-x)/norm(x) < tol
problem = replace_matrix(problem,2*A)
setup!(solver)(problem,y,S)
use!(solver)(problem,y,S)
@test norm(y-x/2)/norm(x/2) < tol
finalize!(solver)(S)

solver = richardson(lu_solver(),maxiters=1)
problem = linear_problem(A,b)
y = similar(x)
y .= 0
S = setup(solver)(problem,y)
use!(solver)(problem,y,S)
tol = 1.e-8
@test norm(y-x)/norm(x) < tol
problem = replace_matrix(problem,2*A)
setup!(solver)(problem,y,S)
use!(solver)(problem,y,S)
@test norm(y-x/2)/norm(x/2) < tol
finalize!(solver)(S)

solver = jacobi(;maxiters=1000)
problem = linear_problem(A,b)
y = similar(x)
y .= 0
S = setup(solver)(problem,y)
use!(solver)(problem,y,S)
tol = 1.e-8
@test norm(y-x)/norm(x) < tol
problem = replace_matrix(problem,2*A)
setup!(solver)(problem,y,S)
use!(solver)(problem,y,S)
@test norm(y-x/2)/norm(x/2) < tol
finalize!(solver)(S)

solver = additive_schwarz(lu_solver())
problem = linear_problem(A,b)
y = similar(x)
y .= 0
S = setup(solver)(problem,y)
use!(solver)(problem,y,S)
problem = replace_matrix(problem,2*A)
setup!(solver)(problem,y,S)
use!(solver)(problem,y,S)
finalize!(solver)(S)

end #module
