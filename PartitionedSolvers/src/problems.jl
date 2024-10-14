
abstract type AbstractType end

function Base.show(io::IO,data::AbstractType)
    print(io,"PartitionedSolvers.$(nameof(typeof(data)))(…)")
end

abstract type AbstractLinearProblem <: AbstractType end

function linear_problem(A,b;attributes...)
    LinearProblem(A,b,attributes)
end

struct LinearProblem{A,B,C} <: AbstractLinearProblem
    matrix::A
    rhs::B
    attributes::C
end

matrix(p::LinearProblem) = p.matrix
rhs(p::LinearProblem) = p.rhs
attributes(p::LinearProblem) = p.attributes

struct NonlinearProblemStatus
    residual_updates::Int
    jacobian_updates::Int
end

function NonlinearProblemStatus()
    NonlinearProblemStatus(0,0)
end

function update(p::NonlinearProblemStatus)
    rups = p.residual_updates + 1
    jups = p.jacobian_updates + 1
    NonlinearProblemStatus(rups,jups)
end

function update_residual(p::NonlinearProblemStatus)
    rups = p.residual_updates + 1
    jups = p.jacobian_updates
    NonlinearProblemStatus(rups,jups)
end

function update_jacobian(p::NonlinearProblemStatus)
    rups = p.residual_updates
    jups = p.jacobian_updates + 1
    NonlinearProblemStatus(rups,jups)
end

function Base.show(io::IO,k::MIME"text/plain",data::NonlinearProblemStatus)
    println("Residual updates since creation: $(data.residual_updates)")
    println("Jacobian updates since creation: $(data.jacobian_updates)")
end

abstract type AbstractNonlinearProblem <: AbstractType end

function nonlinear_problem(rj!,r,j;
        timer_output=TimerOutput(),
        attributes...
    )
    tangent = linear_problem(j,r;attributes...)
    status = NonlinearProblemStatus()
    workspace = (;rj!,tangent,status,timer_output)
    NonlinearProblem(workspace)
end

struct NonlinearProblem{A} <: AbstractNonlinearProblem
    workspace::A
end

residual(a::NonlinearProblem) = rhs(tangent(a))
jacobian(a::NonlinearProblem) = matrix(tangent(a))
tangent(a::NonlinearProblem) = a.workspace.tangent
status(a::NonlinearProblem) = a.workspace.status

function update!(p::NonlinearProblem,x;kwargs...)
    (;rj!,tangent,status,timer_output) = p.workspace
    @timeit timer_output "nonlinear_problem update!" begin
        r = rhs(tangent)
        j = matrix(tangent)
        r,j = rj!(r,j,x)
        tangent = LinearProblem(j,r,attributes(tangent))
        status = update(status)
        workspace = (;rj!,tangent,status,timer_output)
        NonlinearProblem(workspace)
    end
end

function update_residual!(p::NonlinearProblem,x;kwargs...)
    (;rj!,tangent,status,timer_output) = p.workspace
    @timeit timer_output "nonlinear_problem update_residual!" begin
        r = rhs(tangent)
        j = matrix(tangent)
        r,_ = rj!(r,nothing,x)
        tangent = LinearProblem(j,r,attributes(tangent))
        status = update_residual(status)
        workspace = (;rj!,tangent,status,timer_output)
        NonlinearProblem(workspace)
    end
end

function update_jacobian!(p::NonlinearProblem,x;kwargs...)
    (;rj!,tangent,status,timer_output) = p.workspace
    @timeit timer_output "nonlinear_problem update_jacobian!" begin
        r = rhs(tangent)
        j = matrix(tangent)
        _,j = rj!(nothing,j,x)
        tangent = LinearProblem(j,r,attributes(tangent))
        status = update_jacobian(status)
        workspace = (;rj!,tangent,status,timer_output)
        NonlinearProblem(workspace)
    end
end

## TODO implement residual, jacobian, tangent
#
#function ode_problem(r,j=nothing,t=nothing;attributes...)
#    ODEProblem(r,j,t,attributes)
#end
#
#abstract type AbstractODEProblem <: AbstractType end
#
#struct ODEProblem{A,B,C,D} <: AbstractODEProblem
#    residual::A
#    jacobian::B
#    tangent::C
#    attributes::D
#end
#
#function optimization_problem(r,j=nothing,t=nothing;
#    attributes...)
#    OptimizationProblem(r,j,t,attributes)
#end
#
#abstract type AbstractOptimizationProblem <: AbstractType end
#
#struct OptimizationProblem{A,B,C,D} <: AbstractOptimizationProblem
#    objective::A
#    gradient::B
#    hessian::C
#    attributes::D
#end
#
#struct InplaceFunction{A,B} <: Function
#    parent::A
#    parent!::B
#end
#
#function (f::InplaceFunction)(x...)
#    f.parent(x...)
#end
#
#function inplace(f::InplaceFunction)
#    f.parent!
#end
#
#function inplace(f)
#    function inplace!(x,args...)
#        f(args...)
#    end
#end


