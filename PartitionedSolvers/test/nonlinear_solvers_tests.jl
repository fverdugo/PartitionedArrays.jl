module NonlinearSolversTests

import PartitionedSolvers as PS
import PartitionedArrays as PA
using Test
using LinearAlgebra

function mock_nonlinear_problem(x0)
    r0 = zeros(1)
    j0 = PA.sparse_matrix([1],[1],[0.0],1,1)
    workspace = nothing
    PS.nonlinear_problem(x0,r0,j0,workspace) do p
        x = PS.solution(p)
        r = PS.residual(p)
        j = PS.jacobian(p)
        if r !== nothing
            r .= 2 .* x.^2 .- 4
            p = PS.update(p,residual = r)
        end
        if j !== nothing
            j .= 4 .* x
            p = PS.update(p,jacobian = j)
        end
        p
    end |> PS.update
end

x = [1.0]
p = mock_nonlinear_problem(x)
s = PS.newton_raphson(p;verbose=true)
s = PS.solve(s)

x = [1.0]
p = mock_nonlinear_problem(x)
s = PS.solve(p)

x = [1.0]
p = mock_nonlinear_problem(x)
for s in PS.history(p)
    @show s.workspace.residual_loss
end

x = [1.0]
p = mock_nonlinear_problem(x)
for residual_loss in PS.history(s->s.workspace.residual_loss,p)
    @show residual_loss
end

end # module
