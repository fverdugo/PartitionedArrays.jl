module SmoothersTests

import PartitionedSolvers as PS
using PartitionedArrays
using LinearAlgebra
using Test
using SparseMatricesCSR

np = 4
parts = DebugArray(LinearIndices((np,)))
parts_per_dir = (2,2)
nodes_per_dir = (8,8)
args = laplacian_fem(nodes_per_dir,parts_per_dir,parts)
A = psparse(args...) |> fetch
x = pones(partition(axes(A,2)))
b = A*x

tol = 1.e-8
y = similar(x)
y .= 0
p = PS.linear_problem(y,A,b)
s = PS.solve(p)
y = PS.solution(s)
@test norm(y-x)/norm(x) < tol

tol = 1.e-8
y = similar(x)
y .= 0
p = PS.linear_problem(y,A,b)
s = PS.jacobi(p;iterations=1000)
s = PS.solve(s)
y = PS.solution(s)
@test norm(y-x)/norm(x) < tol

s = PS.update(s,matrix=2*A)
for x in PS.history(PS.solution,s)
end
@test norm(y-x/2)/norm(x/2) < tol

y .= 0
p = PS.update(p,solution=y)
s = PS.additive_schwarz(p)
s = PS.solve(s)

s = PS.additive_schwarz(p;local_solver=PS.jacobi)
s = PS.solve(s)

y .= 0
p = PS.update(p,solution=y)
s = PS.additive_schwarz(p;local_solver=PS.gauss_seidel)
s = PS.solve(s)

args = laplacian_fdm(nodes_per_dir,parts_per_dir,parts)
T = SparseMatrixCSR{1,Float64,Int32}
A = psparse(T,args...;assembled=true) |> fetch
x = pones(partition(axes(A,2)))
b = A*x
y = similar(x)
y .= 0
p = PS.linear_problem(y,A,b)
s = PS.gauss_seidel(p)
s = PS.solve(s;zero_guess=true)
n1 = norm(PS.solution(s))
y .= 0
s = PS.update(s,solution=y)
s = PS.solve(s)
n2 = norm(PS.solution(s))
n = n1
@test n ≈ n1 ≈ n2

args = laplacian_fdm(nodes_per_dir,parts_per_dir,parts)
T = SparseMatrixCSR{1,Float64,Int32}
A = psparse(T,args...;assembled=true,split_format=false) |> fetch
x = pones(partition(axes(A,2)))
b = A*x
y = similar(x)
y .= 0
p = PS.linear_problem(y,A,b)
s = PS.gauss_seidel(p)
s = PS.solve(s;zero_guess=true)
n1 = norm(PS.solution(s))
y .= 0
s = PS.update(s,solution=y)
s = PS.solve(s)
n2 = norm(PS.solution(s))
@test n ≈ n1 ≈ n2

args = laplacian_fdm(nodes_per_dir,parts_per_dir,parts)
A = psparse(args...;assembled=true,split_format=false) |> fetch
x = pones(partition(axes(A,2)))
b = A*x
y = similar(x)
y .= 0
p = PS.linear_problem(y,A,b)
s = PS.gauss_seidel(p)
s = PS.solve(s;zero_guess=true)
n1 = norm(PS.solution(s))
y .= 0
s = PS.update(s,solution=y)
s = PS.solve(s)
n2 = norm(PS.solution(s))
@test n ≈ n1 ≈ n2

args = laplacian_fdm(nodes_per_dir)
A = sparse_matrix(args...)
x = ones(axes(A,2))
b = A*x
y = similar(x)
y .= 0
p = PS.linear_problem(y,A,b)
s = PS.gauss_seidel(p)
s = PS.solve(s;zero_guess=true)
n1 = norm(PS.solution(s))
n = n1
y .= 0
s = PS.update(s,solution=y)
s = PS.solve(s)
n2 = norm(PS.solution(s))
@test n ≈ n1 ≈ n2

args = laplacian_fdm(nodes_per_dir)
A = sparse_matrix(T,args...)
x = ones(axes(A,2))
b = A*x
y = similar(x)
y .= 0
p = PS.linear_problem(y,A,b)
s = PS.gauss_seidel(p)
s = PS.solve(s;zero_guess=true)
n1 = norm(PS.solution(s))
y .= 0
s = PS.update(s,solution=y)
s = PS.solve(s)
n2 = norm(PS.solution(s))
@test n ≈ n1 ≈ n2


end # module
