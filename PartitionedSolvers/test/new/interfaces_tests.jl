module InterfacesTests

import PartitionedSolvers as PS
using Test

function mock_linear_solver(p)
    Ainv = 1/PS.matrix(p)
    function update(Ainv,A)
        Ainv = 1/A
    end
    function step(x,Ainv,b,phase=:start)
        if phase === :stop
            return nothing
        end
        x = Ainv*b
        phase = :stop
        x,Ainv,phase
    end
    uses_initial_guess = false
    PS.linear_solver(update,step,p,Ainv;uses_initial_guess)
end

#function main()
#    x = 0.0
#    A = 2.0
#    b = 12.0
#    @time lp = PS.linear_problem(x,A,b)
#    @time ls = mock_linear_solver(lp)
#    @time ls = PS.solve(ls)
#    @time x = PS.solution(ls)
#    @time ls = PS.update(ls,matrix=2*A)
#    @time ls = PS.solve(ls)
#end
#main()

x = 0.0
A = 2.0
b = 12.0
lp = PS.linear_problem(x,A,b)
ls = mock_linear_solver(lp)
ls = PS.solve(ls)
x = PS.solution(ls)
@test x == A\b

ls,phase = PS.step(ls)
next = PS.step(ls,phase)
@test next === nothing

ls = PS.update(ls,matrix=2*A)
ls = PS.solve(ls)
x = PS.solution(ls)
@test x == (2*A)\b

ls = PS.update(ls,rhs=4*b)
ls = PS.solve(ls)
x = PS.solution(ls)
@test x == (2*A)\(4*b)

for x in PS.history(ls)
    @show x
end

function mock_nonlinear_problem(x)
    r = 0*x
    j = 0*x
    PS.nonlinear_problem(r,j,x) do r,j,x
        if r !== nothing
            r = 2*x^2 - 4
        end
        if j !== nothing
            j = 4*x
        end
        r,j
    end
end

function mock_nonlinear_solver_update(ws,p)
    (;ls,iteration,iterations) = ws
    ls = PS.update(ls;matrix=PS.jacobian(p),rhs=PS.residual(p))
    (;ls,iteration,iterations)
end

function mock_nonlinear_solver_step(ws,p,phase=:start)
    if phase === :stop
        return nothing
    end
    (;ls,iteration,iterations) = ws
    if phase === :start
        iteration = 0
        phase = :advance
    end
    ls = PS.solve(ls)
    x = PS.solution(p)
    x -= PS.solution(ls)
    p = PS.update(p,solution=x)
    iteration += 1
    if iteration == iterations
        phase = :stop
    end
    ws = (;ls,iteration,iterations)
    ws = mock_nonlinear_solver_update(ws,p)
    ws,p,phase
end

function mock_nonlinear_solver(p;solver=mock_linear_solver,iterations=10)
    iteration = 0
    dx = PS.solution(p)
    lp = PS.linear_problem(dx,PS.jacobian(p),PS.residual(p))
    ls = solver(lp)
    workspace = (;ls,iteration,iterations)
    PS.nonlinear_solver(
        mock_nonlinear_solver_update,
        mock_nonlinear_solver_step,
        p,
        workspace)
end

#function main()
#    x = 1.0
#    @time p = mock_nonlinear_problem(x)
#    @time s = mock_nonlinear_solver(p)
#    @time s = PS.solve(s)
#end
#
#main()
#main()

x = 1.0
p = mock_nonlinear_problem(x)
@show PS.residual(p)
@show PS.jacobian(p)

s = mock_nonlinear_solver(p)
s = PS.solve(s)

x = 1.0
s = PS.update(s,solution=x)
for x in PS.history(s)
    @show x
end

function mock_ode(u)
    r = 0*u
    j = 0*u
    v = 0*u
    x = (0,u,v)
    ts = (0,10)
    PS.ode_problem(ts,r,j,x) do r,j,x,dx
        (t,u,v) = x
        du,dv = dx
        if r !== nothing
            r = 2*u^2 + v - 4*t + 1
        end
        if j !== nothing
            j = 4*u*du + dv
        end
        r,j
    end
end

function mock_ode_solver(ode;
        dt = (PS.interval(ode)[2]-PS.interval(ode)[1])/10,
        solver = mock_nonlinear_solver)
    t,u,v = PS.solution(ode)
    tend = last(PS.interval(ode))
    x = copy(u)
    p = PS.nonlinear_problem(PS.residual(ode),PS.jacobian(ode),x) do r,j,x
        v = (x - u) / dt
        rj! = PS.inplace(ode)
        r,j = rj!(r,j,(t,x,v),(1.,1/dt))
        r,j
    end
    S = solver(p)
    function update(;problem=ode,kwargs...)
        ode = PS.update(problem;kwargs...)
        t,u,v = PS.solution(ode)
        tend = last(PS.interval(ode))
    end
    function step(phase=:start)
        if phase === :stop
            return nothing
        end
        if phase === :start
            t = first(PS.interval(ode))
            phase = :advance
        end
        S = PS.solve(S)
        x = PS.solution(S)
        t += dt
        v = (x - u) / dt
        u = x
        ode = PS.update(ode,solution=(t,u,v))
        if t >= tend
            phase = :stop
        end
        phase
    end
    problem() = ode
    status() = nothing
    PS.ode_solver(update,step,problem,status)
end

u = 2.0
p = mock_ode(u)
s = mock_ode_solver(p)
for x in PS.history(s)
    @show x
end
s = PS.update(s,solution=(0.0,u,0.0))
@show PS.solution(s)
@time s = PS.solve(s)
@show PS.solution(s)


end # module


