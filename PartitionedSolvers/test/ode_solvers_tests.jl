module ODESolversTests

import PartitionedSolvers as PS
using Test
import PartitionedArrays as PA

function mock_ode_1(u)
    r = zeros(1)
    j = PA.sparse_matrix([1],[1],[0.0],1,1)
    v = 0*u
    x = (0,u,v)
    ts = (0,10)
    coeffs = (1.,1.)
    workspace = nothing
    PS.ode_problem(x,r,j,ts,coeffs,workspace) do ode
        (t,u2,v2) = PS.solution(ode)
        du,dv = PS.coefficients(ode)
        r = PS.residual(ode)
        j = PS.jacobian(ode)
        if r !== nothing
            r .= v2 .- 32
            ode = PS.update(ode,residual = r)
        end
        if j !== nothing
            j .= dv
            ode = PS.update(ode,jacobian = j)
        end
        ode
    end |> PS.update
end

function mock_ode_2(u)
    r = zeros(1)
    j = PA.sparse_matrix([1],[1],[0.0],1,1)
    v = 0*u
    x = (0,u,v)
    ts = (1,10)
    coeffs = (1.,1.)
    workspace = nothing
    PS.ode_problem(x,r,j,ts,coeffs,workspace) do ode
        (t,u2,v2) = PS.solution(ode)
        du,dv = PS.coefficients(ode)
        r = PS.residual(ode)
        j = PS.jacobian(ode)
        if r !== nothing
            r .=  v2 .+ (u2 ./ t) .- 3*t
            #r .=  v2 .+ u2
            ode = PS.update(ode,residual = r)
        end
        if j !== nothing
            j .=  (du ./ t) .+ dv
            #j .=  du .+ dv
            ode = PS.update(ode,jacobian = j)
        end
        ode
    end |> PS.update
end

u = [0.0]
p = mock_ode_1(u)
solver = p -> PS.newton_raphson(p;verbose=false)
s = PS.backward_euler(p;dt=1,solver)
for s in PS.history(s)
    t,u,v = PS.solution(s)
    @show PS.solution(s)
end
t,u,v = PS.solution(s)
@test u == [320.]
@test v == [32.]

u = [1.0]
p = mock_ode_2(u)
solver = p -> PS.newton_raphson(p;verbose=false)
s = PS.backward_euler(p;dt=0.001,solver)
s = PS.solve(s)
t,u,v = PS.solution(s)
tol = 1.0e-4
@test abs(u[1] - 100)/100 < tol
@test abs(v[1] - 20)/20 < tol
@show PS.solution(s)


end # module
