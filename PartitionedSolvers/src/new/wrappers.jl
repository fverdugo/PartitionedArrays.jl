
function LinearAlgebra_lu(p)
    F = lu(matrix(p))
    function update(;problem=p,kwargs...)
        p = update(problem;kwargs...)
        if haskey(kwargs,:matrix) || problem !== p
            lu!(F,matrix(p))
        end
        p
    end
    function step(phase=:start)
        if phase === :stop
            return nothing
        end
        x = solution(p)
        b = rhs(p)
        ldiv!(x,F,b)
        phase = :stop
    end
    problem() = p
    status() = nothing
    uses_initial_guess = false
    linear_solver(update,step,problem,status;uses_initial_guess)
end

function NLSolvers_nlsolve(p;kwargs...)
    result = nothing
    fj! = inplace(p)
    function f!(r,x)
        fj!(r,nothing,x)
        r
    end
    function j!(j,x)
        fj!(nothing,j,x)
        j
    end
    df = OnceDifferentiable(f!,j!,fj!,solution(p),residual(p),jacobian(p))
    function update(newp)
        p = newp
        result = nothing
        fj! = inplace(p)
    end
    function step(phase=:start)
        if phase === :stop
            return nothing
        end
        x = solution(p)
        result = nlsolve(df,x;kwargs...)
        copyto!(x,result.x)
        phase = :stop
    end
    status() = result
    problem() = p
    nonlinear_solver(update,step,problem,status)
end

#function newton_raphson(p;solver=LinearAlgebra_lu,iterations=100)
#    iteration = 0
#    dx = similar(solution(p),axes(jacobian(p),2))
#    lp = linear_problem(dx,jacobian(p),residual(p))
#    S = solver(lp)
#    function update(;problem=p)
#        p = problem
#        lp = linear_problem(dx,jacobian(p),residual(p))
#        S = update(S;problem=lp)
#        p
#    end
#    function step(phase=:start)
#        if phase === :stop
#            return nothing
#        end
#        if phase === :start
#            iteration = 0
#            phase = :advance
#        end
#        dx = solution(lp)
#        ldiv!(dx,S,rhs(lp))
#        x = solution(p)
#        x .-= dx
#        p = update(p,solution=x)
#        S = update(S;matrix=jacobian(p),rhs=residual(p))
#        iteration += 1
#        if iteration == iterations
#            phase = :stop
#        end
#        phase
#    end
#    problem() = p
#    status() = nothing
#    nonlinear_solver(update,step,problem,status)
#end
#
#function backward_euler(ode;
#        dt = (interval(ode)[2]-interval(ode)[1])/100,
#        solver = newton_raphson,
#        verbosity = PartitionedSolvers.verbosity())
#    t,u,v = solution(ode)
#    tend = last(interval(ode))
#    x = copy(u)
#    p = nonlinear_problem(residual(ode),jacobian(ode),x) do r,j,x
#        v .= (x .- u) ./ dt
#        rj! = inplace(ode)
#        rj(r,j,(t,x,v),(1.,1/dt))
#        r,j
#    end
#    S = solver(p)
#    function update(newode)
#        ode = newode
#        t,u,v = solution(ode)
#        tend = last(interval(ode))
#    end
#    function step(phase=:start)
#        if phase === :stop
#            return nothing
#        end
#        if phase === :start
#            t = first(interval(ode))
#            phase = :advance
#            print_time_step(verbosity,t,tend)
#        end
#        S = solve(S)
#        x = solution(S)
#        t += dt
#        v .= (x .- u) ./ dt
#        u .= x
#        ode = update(ode,solution=(t,u,v))
#        print_time_step(verbosity,t,tend)
#        if t > tend
#            phase = :stop
#        end
#        phase
#    end
#    problem() = ode
#    status() = nothing
#    ode_solver(update,step,problem,status)
#end


