
function do_nothing_linear_solver()
    setup(x,problem) = nothing
    solve!(x,problem,state) = copy!(x,b)
    setup!(x,problem,state) = nothing
    algebraic_solver(;setup,setup!,solve!)
end

function lu_solver()
    setup(x,problem) = lu(linear_operator(problem))
    setup!(x,problem,state) = lu!(state,linear_operator(problem))
    solve!(x,problem,state) = ldiv!(x,state,rhs(problem))
    algebraic_solver(;setup,solve!,setup!)
end

function diagonal_solver()
    setup(x,problem) = diag(linear_operator(problem))
    setup!(x,problem,D) = diag!(D,linear_operator(problem))
    solve!(x,problem,D) = x .= D .\ rhs(problem)
    algebraic_solver(;setup,setup!,solve!)
end

function richardson(solver;niters)
    function setup(x,problem)
        A = linear_operator(problem)
        b = rhs(problem)
        r = similar(b)
        dx = similar(x,axes(A,2))
        P = preconditioner(solver,dx,problem)
        state = (r,dx,P)
    end
    function setup!(x,problem,state)
        (r,dx,P) = state
        preconditioner!(P,dx,problem)
        state
    end
    function solve!(x,problem,state)
        A = linear_operator(problem)
        b = rhs(problem)
        (r,dx,P) = state
        for iter in 1:niters
            dx .= x
            mul!(r,A,dx)
            r .= r .- b
            ldiv!(dx,P,r)
            x .-= dx
        end
        (;niters)
    end
    function finalize!(state)
        (r,dx,P) = state
        PartitionedSolvers.finalize!(P)
    end
    algebraic_solver(;setup,setup!,solve!,finalize!)
end

function jacobi(;kwargs...)
    solver = diagonal_solver()
    richardson(solver;kwargs...)
end

function additive_schwarz(local_solver)
    function setup(x,problem)
        A = linear_operator(problem)
        b = rhs(problem)
        local_problems = map(linear_problem,own_own_values(A),own_values(b))
        f = (x,p) -> PartitionedSolvers.setup(local_solver,x,p)
        local_setups = map(f,own_values(x),local_problems)
        local_setups
    end
    function setup!(x,problem,local_setups)
        A = linear_operator(problem)
        b = rhs(problem)
        local_problems = map(linear_problem,own_own_values(A),own_values(b))
        f = (x,p,s) -> PartitionedSolvers.setup!(local_solver,x,p,s)
        map(f,own_values(x),local_problems,local_setups)
        local_setups
    end
    function solve!(x,problem,local_setups)
        A = linear_operator(problem)
        b = rhs(problem)
        local_problems = map(linear_problem,own_own_values(A),own_values(b))
        f = (x,p,s) -> PartitionedSolvers.solve!(local_solver,x,p,s)
        map(f,own_values(x),local_problems,local_setups)
    end
    function finalize!(local_setups)
        f = (S) -> PartitionedSolvers.finalize!(local_solver,S)
        map(f,local_setups)
        nothing
    end
    linear_solver(;setup,setup!,solve!,finalize!)
end

