module LinearSolversTests

using PartitionedArrays
import PartitionedSolvers as PS

args = laplacian_fem((10,10,10))
A = sparse_matrix(args...)
x = ones(axes(A,2))
b = A*x

P = PS.LinearAlgebra_lu(A,verbose=true,verbosity=(;indentation=">>>> "))

#P = PS.identity_solver()

#P = PS.richardson(x,A,b)

x,P = PS.solve!(x,P,b)
x,P = PS.solve!(x,P,b)
P = PS.update!(P,A)
x, P =PS.solve!(x,P,b)

P |> PS.status |> display
#P |> PS.timer_output |> display




end # module
