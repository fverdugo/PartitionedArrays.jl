
abstract type Backend end

# Should return a DistributedData{Part}
function Partition(b::Backend,nparts::Integer)
  @abstractmethod
end

function Partition(b::Backend,nparts::Tuple)
  Partition(b,prod(nparts))
end

# This can be overwritten to add a finally clause
function distributed_run(driver::Function,b::Backend,nparts)
  part = Partition(b,nparts)
  driver(part)
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

