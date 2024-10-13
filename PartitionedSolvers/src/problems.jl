
abstract type AbstractType end

function Base.show(io::IO,data::AbstractType)
    print(io,"PartitionedSolvers.$(nameof(typeof(data)))(…)")
end

function nonlinear_problem(r,j=nothing,t=nothing;attributes...)
    NonlinearProblem(r,j,t,attributes)
end

abstract type AbstractNonlinearProblem <: AbstractType end

struct NonlinearProblem{A,B,C,D} <: AbstractNonlinearProblem
    residual::A
    jacobian::B
    tangent::C
    attributes::D
end

abstract type AbstractLinearProblem <: AbstractNonlinearProblem end

function linear_problem(A,b;attributes...)
    LinearProblem(A,b,attributes)
end

struct LinearProblem{A,B,C} <: AbstractLinearProblem
    matrix::A
    rhs::B
    attributes::C
end

# TODO implement residual, jacobian, tangent

function ode_problem(r,j=nothing,t=nothing;attributes...)
    ODEProblem(r,j,t,attributes)
end

abstract type AbstractODEProblem <: AbstractType end

struct ODEProblem{A,B,C,D} <: AbstractODEProblem
    residual::A
    jacobian::B
    tangent::C
    attributes::D
end

function optimization_problem(r,j=nothing,t=nothing;
    attributes...)
    OptimizationProblem(r,j,t,attributes)
end

abstract type AbstractOptimizationProblem <: AbstractType end

struct OptimizationProblem{A,B,C,D} <: AbstractOptimizationProblem
    objective::A
    gradient::B
    hessian::C
    attributes::D
end

struct InplaceFunction{A,B} <: Function
    parent::A
    parent!::B
end

function (f::InplaceFunction)(x...)
    f.parent(x...)
end

function inplace(f::InplaceFunction)
    f.parent!
end

function inplace(f)
    function inplace!(x,args...)
        f(args...)
    end
end


