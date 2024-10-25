module InterfacesTests

import PartitionedSolvers as PS

using Test

function mock_linear_solver(p)
    @assert ! PS.uses_mutable_types(p)
    Ainv = 1/PS.matrix(p)
    function update(Ainv,A)
        Ainv = 1/A
    end
    function step(x,Ainv,b,phase=:start)
        #if phase === :stop
        #    return nothing
        #end
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
lp = PS.linear_problem(x,A,b;uses_mutable_types=false)
ls = mock_linear_solver(lp)
ls = PS.solve(ls)
x = PS.solution(ls)
@test x == A\b

ls,phase = PS.step(ls)
ls,phase = PS.step(ls,phase)
@test phase === :stop

ls = PS.update(ls,matrix=2*A)
ls = PS.solve(ls)
x = PS.solution(ls)
@test x == (2*A)\b

ls = PS.update(ls,rhs=4*b)
ls = PS.solve(ls)
x = PS.solution(ls)
@test x == (2*A)\(4*b)

for ls in PS.history(ls)
    @show ls
end

for x in PS.history(PS.solution,ls)
    @show x
end

function mock_nonlinear_problem(x0)
    r0 = 0*x0
    j0 = 0*x0
    workspace = nothing
    PS.nonlinear_problem(x0,r0,j0,workspace;uses_mutable_types=false) do p
        x = PS.solution(p)
        if PS.residual(p) !== nothing
            p = PS.update(p,residual = 2*x^2 - 4)
        end
        if PS.jacobian(p) !== nothing
            p = PS.update(p,jacobian = 4*x)
        end
        p
    end |> PS.update
end

function mock_nonlinear_solver_update(ws,p)
    (;ls,iteration,iterations) = ws
    ls = PS.update(ls;matrix=PS.jacobian(p),rhs=PS.residual(p))
    (;ls,iteration,iterations)
end

function mock_nonlinear_solver_step(ws,p,phase=:start)
    #if phase === :stop
    #    return nothing
    #end
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
    @assert ! PS.uses_mutable_types(p)
    iteration = 0
    dx = PS.solution(p)
    lp = PS.linear_problem(dx,PS.jacobian(p),PS.residual(p);uses_mutable_types=false)
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

x = 1.0
p = mock_nonlinear_problem(x)
@show PS.residual(p)
@show PS.jacobian(p)

s = mock_nonlinear_solver(p)
s = PS.solve(s)

x = 1.0
s = PS.update(s,solution=x)
for x in PS.history(PS.solution,s)
    @show x
end

function mock_ode(u)
    r = 0*u
    j = 0*u
    v = 0*u
    x = (0,u,v)
    ts = (0,10)
    dx = (u,u)
    workspace = nothing
    PS.ode_problem(x,r,j,ts,dx,workspace;uses_mutable_types=false) do ode
        (t,u2,v2) = PS.solution(ode)
        du,dv = PS.coefficients(ode)
        if PS.residual(ode) !== nothing
            ode = PS.update(ode,residual = 2*u2^2 + v2 - 4*t + 1)
        end
        if PS.jacobian(ode) !== nothing
            ode = PS.update(ode,jacobian = 4*u2*du + dv)
        end
        ode
    end |> PS.update
end

function mock_ode_solver_problem(x0,dt,ode0)
    t,u, = PS.solution(ode0)
    workspace = nothing
    PS.nonlinear_problem(PS.residual(ode0),PS.jacobian(ode0),x0,workspace;uses_mutable_types=false) do p
        x = PS.solution(p)
        v = (x - u) / dt
        r = PS.residual(p)
        j = PS.jacobian(p)
        ode = PS.update(ode0,residual=r,jacobian=p,solution=(t,x,v))
        r = PS.residual(ode)
        j = PS.jacobian(ode)
        p = PS.update(p,residual=r,jacobian=j)
    end
end

function mock_ode_solver_update(workspace,ode0)
    (;s,dt) = workspace
    ode = PS.update(ode0,coefficients=(1.0,1/dt))
    x = PS.solution(s)
    p = mock_ode_solver_problem(x,dt,ode)
    s = PS.update(s,problem=p)
    (;s,dt)
end

function mock_ode_solver_step(workspace,ode,phase=:start)
    #if phase === :stop
    #    return nothing
    #end
    (;s,dt) = workspace
    t,u,v = PS.solution(ode)
    if phase === :start
        t = first(PS.interval(ode))
        phase = :advance
    end
    s = PS.solve(s)
    x = PS.solution(s)
    t += dt
    v = (x - u) / dt
    u = x
    ode = PS.update(ode,solution=(t,u,v))
    tend = last(PS.interval(ode))
    if t >= tend
        phase = :stop
    end
    workspace = (;s,dt)
    workspace = mock_ode_solver_update(workspace,ode)
    workspace,ode,phase
end

function mock_ode_solver(ode0;
        dt = (PS.interval(ode0)[2]-PS.interval(ode0)[1])/10,
        solver = mock_nonlinear_solver)

    @assert ! PS.uses_mutable_types(ode0)
    ode = PS.update(ode0,coefficients=(1.0,1/dt))
    _,u,_ = PS.solution(ode)
    x = u
    p = mock_ode_solver_problem(x,dt,ode)
    s = solver(p)
    workspace = (;s,dt)
    PS.ode_solver(mock_ode_solver_update,mock_ode_solver_step,ode,workspace)
end

#function main()
#    u = 2.0
#    p = mock_ode(u)
#    s = mock_ode_solver(p)
#    for x in PS.history(s)
#        @show x
#    end
#    s = PS.update(s,solution=(0.0,u,0.0))
#    @time s = PS.solve(s)
#end
#
#main()

u = 2.0
p = mock_ode(u)
s = mock_ode_solver(p)

for x in PS.history(PS.solution,s)
    @show x
end
#s = PS.update(s,solution=(0.0,u,0.0))
#PS.solve(s)

end # module


