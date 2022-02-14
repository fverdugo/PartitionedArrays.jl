
using PartitionedArrays

function throw_assert(parts)
  nparts=length(parts)
  p_main = map_parts(parts) do part
    if part == MAIN
      part_fail = rand(1:nparts)
    else
      0
    end
  end
  p = emit(p_main)
  map_parts(parts,p) do part,part_fail
      @assert  part_fail != part
  end
end

function test_exception(parts)
  throw_assert(parts)
end
