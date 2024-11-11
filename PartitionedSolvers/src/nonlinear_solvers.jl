# TODO This is a very vanilla NR solver for the moment
function newton_raphson(p;
        solver=default_solver,
        iterations=1000,
        residual_tol = convert(real(eltype(solution(p))), 1e-8),
        solution_tol = zero(real(eltype(solution(p)))),
        verbose = false,
        output_prefix = "",
    )
    @assert uses_mutable_types(p)
    iteration = 0
    x = solution(p)
    # Just to make sure that the problem is in the right state
    p = update(p,solution=x)
    dx = similar(x,axes(jacobian(p),2))
    lp = linear_problem(dx,jacobian(p),residual(p))
    ls = solver(lp)
    solution_loss = maximum(abs,dx)
    residual_loss = maximum(abs,residual(p))
    workspace = (;ls,iteration,iterations,solution_loss,residual_loss,solution_tol,residual_tol,verbose,output_prefix)
    nonlinear_solver(
        newton_raphson_update,
        newton_raphson_step,
        p,
        workspace)
end

function newton_raphson_update(ws,p)
    (;ls,iteration,iterations,solution_loss,residual_loss,solution_tol,residual_tol,verbose,output_prefix) = ws
    ls = update(ls;matrix=jacobian(p),rhs=residual(p))
    (;ls,iteration,iterations,solution_loss,residual_loss,solution_tol,residual_tol,verbose,output_prefix)
end

function newton_raphson_step(ws,p,phase=:start)
    (;ls,iteration,iterations,solution_loss,residual_loss,solution_tol,residual_tol,verbose,output_prefix) = ws
    if phase === :start
        iteration = 0
        ws = (;ls,iteration,iterations,solution_loss,residual_loss,solution_tol,residual_tol,verbose,output_prefix)
        phase = :advance
        print_progress_header(ws)
        print_progress(ws)
    end
    ls = solve(ls)
    dx = solution(ls)
    x = solution(p)
    x .-= dx
    p = update(p,solution=x)
    iteration += 1
    solution_loss = maximum(abs,dx)
    residual_loss = maximum(abs,residual(p))
    if solution_loss <= solution_tol || residual_loss <= residual_tol
        phase = :stop
    end
    if iteration == iterations
        phase = :stop
    end
    ws = (;ls,iteration,iterations,solution_loss,residual_loss,solution_tol,residual_tol,verbose,output_prefix)
    ws = newton_raphson_update(ws,p)
    print_progress(ws)
    ws,p,phase
end

function print_progress_header(a)
    s = a.output_prefix
    v = a.verbose
    c = "current"
    t = "target"
    v && @printf "%s%20s %20s %20s\n" s "iterations" "residual" "solution"
    v && @printf "%s%10s %10s %10s %10s %10s %10s\n" s c t c t c t
end

function print_progress(a)
    s = a.output_prefix
    v = a.verbose
    v && @printf "%s%10i %10i %10.2e %10.2e %10.2e %10.2e\n" s a.iteration a.iterations a.residual_loss a.residual_tol a.solution_loss a.solution_tol
end


