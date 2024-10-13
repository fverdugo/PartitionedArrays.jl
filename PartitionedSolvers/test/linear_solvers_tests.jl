module LinearSolversTests

using PartitionedArrays
import PartitionedSolvers as PS
using TimerOutputs

args = laplacian_fem((10,10,10))
A = sparse_matrix(args...)
x = ones(axes(A,2))
b = A*x

timer_output = TimerOutput()

P = PS.LinearAlgebra_lu(A,verbose=true,verbosity=(;indentation=">>>> "))

P = PS.identity_solver(;timer_output)

P = PS.richardson(x,A,b;preconditioner=P,timer_output)

P = PS.jacobi(x,A,b;timer_output)

x,P = PS.solve!(x,P,b)
x,P = PS.solve!(x,P,b)
P = PS.update!(P,A)
x, P =PS.solve!(x,P,b)

P |> PS.status |> display
P |> PS.timer_output |> display




end # module
