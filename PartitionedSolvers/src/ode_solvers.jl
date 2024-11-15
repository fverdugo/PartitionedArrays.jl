
function nonlinear_stage_problem_statement(p)
    (;scheme,coeffs,ode,x1,x0,dt) = workspace(p)
    x = solution(p)
    r = residual(p)
    j = jacobian(p)
    x1 = scheme(x1,x0,x,dt)
    ode = update(ode,residual=r,jacobian=j,solution=x1,coefficients=coeffs(dt))
    r = residual(ode)
    j = jacobian(ode)
    p = update(p,residual=r,jacobian=j)
end

function nonlinear_stage_problem(scheme,coeffs,x1,x0,x,ode,dt)
    workspace = (;scheme,coeffs,x1,x0,ode,dt)
    nonlinear_problem(
        nonlinear_stage_problem_statement,
        x,residual(ode),jacobian(ode),workspace)
end

function update_nonlinear_stage_problem(x1,x0,p,dt)
    (;scheme,coeffs,ode,dt) = workspace(p)
    update(p,workspace=(;scheme,coeffs,x1,x0,ode,dt))
end

function single_stage_solver(scheme,coeffs,ode;
        dt = (interval(ode)[end]-interval(ode)[1])/10,
        solver = default_solver,
    )
    t,u,v = solution(ode)
    t = interval(ode)[1]
    ode = update(ode,solution=(t,u,v),coefficients=coeffs(dt))
    x = copy(u)
    x0 = (t,u,v)
    x1 = map(copy,x0) # TODO Too much copies?
    p = nonlinear_stage_problem(scheme,coeffs,x1,x0,x,ode,dt)
    s = solver(p)
    workspace = (;s,dt,scheme,p,x1)
    ode_solver(single_stage_solver_update,single_stage_solver_step,ode,workspace)
end

function single_stage_solver_update(workspace,ode)
    (;s,dt,scheme,p,x1) = workspace
    t,u,v = solution(ode)
    x = solution(s)
    x0 = (t,u,v)
    p = update_nonlinear_stage_problem(x1,x0,p,dt)
    s = update(s,problem=p)
    workspace = (;s,dt,scheme,p,x1)
end

function single_stage_solver_step(workspace,ode,phase=:start)
    (;s,dt,scheme,p,x1) = workspace
    t,u,v = solution(ode)
    if phase === :start
        t = first(interval(ode))
        phase = :advance
    end
    s = solve(s)
    x = solution(s)
    x0 = solution(ode)
    x0 = scheme(x0,x0,x,dt)
    t, = x0
    ode = update(ode,solution=x0)
    tend = last(interval(ode))
    if t > tend || t≈tend
        phase = :stop
    end
    workspace = (;s,dt,scheme,p,x1)
    workspace = single_stage_solver_update(workspace,ode)
    workspace,ode,phase
end

function backward_euler_scheme!((t1,u1,v1),(t0,u0,v0),x,dt)
    t1 = t0 + dt
    v1 .= (x .- u0) ./ dt
    u1 .= x
    (t1,u1,v1)
end

function backward_euler_coefficients(dt)
    (1.0,1.0/dt)
end

function backward_euler(ode;kwargs...)
    coeffs = backward_euler_coefficients
    scheme = backward_euler_scheme!
    single_stage_solver(scheme,coeffs,ode;kwargs...)
end

