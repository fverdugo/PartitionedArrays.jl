
abstract type Backend end

function distributed_run(driver::Function,b::Backend,nparts::Integer)
  @abstractmethod
end

function distributed_run(driver::Function,b::Backend,nparts::Tuple)
  distributed_run(driver,b,prod(nparts))
end

# Data distributed in parts of type T
abstract type DistributedData{T} end

num_parts(a::DistributedData) = @abstractmethod

function map_parts(task::Function,a::DistributedData...)
  @abstractmethod
end

struct Part
  id::Int
  num_parts::Int
end

num_parts(a::Part) = a.num_parts
