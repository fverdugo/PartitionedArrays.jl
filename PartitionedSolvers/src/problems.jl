
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

function nonlinear_problem(rj!,r,j;attributes...)
    t = linear_problem(j,r;attributes...)
    status = NonlinearProblemStatus()
    NonlinearProblem(rj!,t,status)
end

struct NonlinearProblem{A,B,C} <: AbstractNonlinearProblem
    update!::A
    tangent::B
    status::C
end

residual(a::NonlinearProblem) = rhs(tangent(a))
jacobian(a::NonlinearProblem) = matrix(tangent(a))
tangent(a::NonlinearProblem) = a.tangent
status(a::NonlinearProblem) = a.status

function update!(p::NonlinearProblem,x;kwargs...)
    t = tangent(p)
    r = residual(p)
    j = jacobian(p)
    r,j = p.update!(r,j,x)
    t = LinearProblem(j,r,attributes(t))
    status = update(p.status)
    NonlinearProblem(p.update!,t,status)
end

function update_residual!(p::NonlinearProblem,x;kwargs...)
    t = tangent(p)
    r = residual(p)
    r,_ = p.update!(r,nothing,x;kwargs...)
    t = LinearProblem(matrix(t),r,attributes(t))
    status = update_residual(p.status)
    NonlinearProblem(p.update!,t,status)
end

function update_jacobian!(p::NonlinearProblem,x;kwargs...)
    t = tangent(p)
    j = jacobian(p)
    _,j = p.update!(nothing,j,x;kwargs...)
    t = LinearProblem(j,rhs(t),attributes(t))
    status = update_jacobian(p.status)
    NonlinearProblem(p.update!,t,status)
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


