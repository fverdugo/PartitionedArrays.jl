module NonlinearSolversTests

using PartitionedSolvers

function mock_newton_raphson(;linear_solver=lu_solver(),iters=10)
    function nr_setup(f,x,options)
        df = tangent(f)
        t,t! = setup(df,x)
        A = matrix(t)
        b = rhs(t)
        dx = similar(b,axes(A,2))
        P = setup(linear_solver,dx,A,b)
        (;dx,t,t!,P)
    end
    function nr_update!(state,f)
        (;dx,t,t!,P) = state
        df = tangent(f)
        t! = update!(t!,df)
        (;dx,t,t!,P)
    end
    function nr_step!(x,state,options,step=0)
        if iters == step
            nothing
        end
        (;dx,t,t!,P) = state
        t = t!(t,x)
        A = matrix(t)
        b = rhs(t)
        P = update!(P,A)
        dx = ldiv!(dx,P,b)
        x += dx
        x, step+1
    end
    function nr_finalize!(state)
        (;dx,t,t!,P) = state
        finalize!(P)
    end
    nonlinear_solver(;
        setup = nr_setup,
        update! = nr_update!,
        step! = nr_step!,
        finalize! = nr_finalize!,
       )
end

function mock_nonlinear_operator()
    function r_setup(x)
        r = zeros(2)
        state = nothing
        r_call!(r,state,x)
        r, state
    end
    function r_call!(r,state,x)
        r[1] = 2x[1]-x[2]+1
        r[2] = x[2]
        r
    end
    function j_setup(x)
        j = zeros(2,2)
        state = nothing
        j_call!(j,state,x)
        j, state
    end
    function j_call!(j,state,x)
        j[1,1] = 2
        j[1,2] = -1
        j[2,1] = 0
        j[2,2] = 1
        j
    end
    jop = nonlinear_operator(;setup=j_setup,call! =j_call!)
    rop = nonlinear_operator(;setup=r_setup,call! = r_call!,jacobian=jop)
    rop
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
