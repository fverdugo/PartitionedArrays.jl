module LinearSolversTests

using PartitionedArrays
import PartitionedSolvers as PS
using TimerOutputs
using LinearAlgebra

args = laplacian_fem((10,10,10))
A = sparse_matrix(args...)
x = ones(axes(A,2))
b = A*x

timer_output = TimerOutput()

P = PS.LinearAlgebra_lu(A,verbose=true,verbosity=(;indentation=">>>> "))

P = PS.identity_solver(;timer_output)

P = PS.richardson(x,A,b;preconditioner=P,timer_output)

P = PS.jacobi(x,A,b;timer_output,verbose=true)

x,P = PS.solve!(x,P,b)
x,P = PS.solve!(x,P,b)
P = PS.update!(P,A)
x, P =PS.solve!(x,P,b)

P |> PS.status |> display
P |> PS.timer_output |> display


timer_output = TimerOutput()

r = copy(b)
j = A
function rj!(r,j,x)
    mul!(r,A,x)
    r .-= b
    r,j
end
p = PS.nonlinear_problem(rj!,r,j;timer_output)


indentation = "    "
linear_solver = PS.jacobi(x,A,r;timer_output,verbose=true,iterations=2,verbosity=PS.verbosity(;indentation))
S = PS.newton_raphson(x,p;verbose=true,iterations=10,timer_output,linear_solver)
x .= 0
x,S,p = PS.solve!(x,S,p)

p |> PS.status |> display
#linear_solver |> PS.status |> display
# TODO make it mutable?
PS.linear_solver(S) |> PS.status |> display
S |> PS.timer_output |> display

# TODO add timer to nonlinear_problem


end # module
