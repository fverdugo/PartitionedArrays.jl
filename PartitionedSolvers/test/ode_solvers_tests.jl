module ODESolversTests

import PartitionedSolvers as PS
using Test
import PartitionedArrays as PA

function mock_ode(u)
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
            r .= 2 .* u2.^2 .+ v2 .- 4*t .+ 1
            ode = PS.update(ode,residual = r)
        end
        if j !== nothing
            j .= 4 .* u2 .* du .+ dv
            ode = PS.update(ode,jacobian = j)
        end
        ode
    end |> PS.update
end

u = [2.0]
p = mock_ode(u)
s = PS.backward_euler(p)
for s in PS.history(s)
    t,u,v = PS.solution(s)
    @show PS.solution(s)
    @test v[1] != 0
end



end # module
