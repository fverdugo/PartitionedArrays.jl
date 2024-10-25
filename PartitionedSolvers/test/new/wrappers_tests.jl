module WrappersTests

import PartitionedSolvers as PS
import PartitionedArrays as PA
using Test
using LinearAlgebra

nodes=(10,10)
args = PA.laplacian_fem(nodes)
A = PA.sparse_matrix(args...)
x_exact = ones(axes(A,2))
b = A*x_exact

x = similar(x_exact)
p = PS.linear_problem(x,A,b)

s = PS.LinearAlgebra_lu(p)
s = PS.solve(s)
x = PS.solution(s)
@test x ≈ x_exact

s = PS.update(s,matrix=2*A)
s = PS.solve(s)
x = PS.solution(s)
@test x*2 ≈ x_exact

ldiv!(x,s,b)
PS.smooth!(x,s,b)


r = similar(b)
w = nothing
p = PS.nonlinear_problem(x,r,A,w) do p2
    x2 = PS.solution(p2)
    if PS.residual(p2) !== nothing
        r2 = PS.residual(p2)
        mul!(r2,A,x2)
        r2 .-= b
        p2 = PS.update(p2,residual = r2)
    end
    if PS.jacobian(p2) !== nothing
        p2 = PS.update(p2,jacobian = A)
    end
    p2
end

s = PS.NLSolvers_nlsolve(p;show_trace=true)
s = PS.solve(s)


dx = copy(x)
lp = PS.linear_problem(dx,A,r)
ls = PS.LinearAlgebra_lu(lp)
function linsolve(dx,A,b)
    ls = PS.update(ls,matrix=A)
    ldiv!(dx,ls,b)
    dx
end

x .= 0
s = PS.NLSolvers_nlsolve(p;show_trace=true,linsolve)
s = PS.solve(s)







end # module
