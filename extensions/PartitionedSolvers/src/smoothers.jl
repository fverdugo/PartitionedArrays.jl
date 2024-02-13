
function lu_solver()
    setup(problem,x) = lu(matrix(problem))
    setup!(problem,x,state) = lu!(state,matrix(problem))
    use!(problem,x,state) = ldiv!(x,state,rhs(problem))
    linear_solver(;setup,use!,setup!)
end

function diagonal_solver()
    setup(problem,x) = diag(matrix(problem))
    setup!(problem,x,state) = diag!(state,matrix(problem))
    use!(problem,x,state) = x .= state .\ rhs(problem)
    linear_solver(;setup,setup!,use!)
end

function richardson(solver;niters)
    function setup(problem,x)
        A = matrix(problem)
        b = rhs(problem)
        r = similar(b)
        dx = similar(x,axes(A,2))
        P = preconditioner(solver,problem,dx)
        state = (r,dx,P)
    end
    function setup!(problem,x,state)
        (r,dx,P) = state
        preconditioner!(P,problem,dx)
        state
    end
    function use!(problem,x,state)
        (r,dx,P) = state
        A = matrix(problem)
        b = rhs(problem)
        for iter in 1:niters
            dx .= x
            mul!(r,A,dx)
            r .-= b
            ldiv!(dx,P,r)
            x .-= dx
        end
        (;niters)
    end
    function finalize!(state)
        (r,dx,P) = state
        PartitionedSolvers.finalize!(P)
    end
    linear_solver(;setup,setup!,use!,finalize!)
end

function jacobi(;kwargs...)
    solver = diagonal_solver()
    richardson(solver;kwargs...)
end

function build_local_problems(problem)
    A = matrix(problem)
    b = rhs(problem)
    ns = nullspace(problem)
    if ns === nothing
        map(linear_problem,own_own_values(A),own_values(b))
    else
        map(linear_problem,own_own_values(A),own_values(b),ns)
    end
end

function additive_schwarz(local_solver)
    function setup(problem,x)
        local_problems = build_local_problems(problem)
        local_setups = map(PartitionedSolvers.setup(local_solver),local_problems,own_values(x))
        local_setups
    end
    function setup!(problem,x,local_setups)
        local_problems = build_local_problems(problem)
        map(PartitionedSolvers.setup!(local_solver),local_problems,own_values(x),local_setups)
        local_setups
    end
    function use!(problem,x,local_setups)
        local_problems = build_local_problems(problem)
        map(PartitionedSolvers.use!(local_solver),local_problems,own_values(x),local_setups)
    end
    function finalize!(local_setups)
        map(PartitionedSolvers.finalize!(local_solver),local_setups)
        nothing
    end
    linear_solver(;setup,setup!,use!,finalize!)
end

