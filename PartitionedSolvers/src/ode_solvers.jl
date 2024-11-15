
function nonlinear_stage_problem(f,ode0,t,x0,coeffs)
    workspace = nothing
    nonlinear_problem(x0,residual(ode0),jacobian(ode0),workspace) do p
        x = solution(p)
        r = residual(p)
        j = jacobian(p)
        ode = update(ode0,coefficients=coeffs,residual=r,jacobian=j,solution=(t,f(x)...))
        r = residual(ode)
        j = jacobian(ode)
        p = update(p,residual=r,jacobian=j)
    end
end

function backward_euler_update(workspace,ode)
    (;s,x,dt,ex) = workspace
    t, = solution(ode)
    x = solution(s)
    coeffs = coefficients(ode)
    p = nonlinear_stage_problem(ex,ode,t,x,coeffs)
    p = update(p,solution=x)
    s = update(s,problem=p)
    workspace = (;s,x,dt,ex)
end

function backward_euler_step(workspace,ode,phase=:start)
    (;s,x,dt,ex) = workspace
    t,u,v = solution(ode)
    if phase === :start
        t = first(interval(ode))
        phase = :advance
    end
    s = solve(s)
    x = solution(s)
    t += dt
    u,v = ex(x)
    u .= x
    ode = update(ode,solution=(t,u,v))
    tend = last(interval(ode))
    if t >= tend
        phase = :stop
    end
    workspace = (;s,x,dt,ex)
    workspace = backward_euler_update(workspace,ode)
    workspace,ode,phase
end

function backward_euler(ode;
        dt = (interval(ode)[end]-interval(ode)[1])/10,
        solver = default_solver,
    )
    @assert uses_mutable_types(ode)
    coeffs = (1.0,1/dt)
    t,u,v = solution(ode)
    t = interval(ode)[1]
    x = copy(u)
    function ex(x)
        v .= (x .- u) ./ dt
        (x,v)
    end
    p = nonlinear_stage_problem(ex,ode,t,x,coeffs)
    s = solver(p)
    workspace = (;s,x,dt,ex)
    ode_solver(backward_euler_update,backward_euler_step,ode,workspace)
end

