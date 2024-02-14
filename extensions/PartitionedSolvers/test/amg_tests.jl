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

end
