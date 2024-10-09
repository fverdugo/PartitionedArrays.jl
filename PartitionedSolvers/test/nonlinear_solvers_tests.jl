module NonlinearSolversTests

using PartitionedSolvers

function mock_nonlinear_operator()
    function f(x)
        r = zeros(2)
        r_s = nothing
        f!(r,r_s,x)
    end
    function f!(r,r_s,x)
        r[1] = 2x[1]+1
        r[2] = x[2]-2
        r,r_s
    end
    function df(x)
        j = zeros(2,2)
        j_s = nothing
        df!(j,j_s,x)
    end
    function df!(j,j_s,x)
        j[1,1] = 2
        j[1,2] = 0
        j[2,1] = 0
        j[2,2] = 1
        j,j_s
    end
    residual = InplaceFunction(f,f!)
    jacobian = InplaceFunction(df,df!)
    FunctionWithDerivative(residual,jacobian)
    #function_with_tangent(residual,tangent)
end

f = mock_nonlinear_operator()
x = zeros(2)
r,r! = setup(f,x)
r = r!(r,x)

j,j! = setup(jacobian(f),x)
j = j!(j,x)

solver = mock_newton_raphson()
P = setup(solver,f,x)
#x = solve!(x,P)


end # module
