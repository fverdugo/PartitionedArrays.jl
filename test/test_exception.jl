
using PartitionedArrays

function throw_assert(parts)
  nparts=length(parts)
  map_parts(parts) do part
      @assert rand(1:nparts) != part
  end
end

function test_exception(parts)
  throw_assert(parts)
end
