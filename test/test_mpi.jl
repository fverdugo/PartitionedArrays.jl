
using DistributedDataDraft
using Test

nparts = 4
distributed_run(mpi,nparts) do parts

  display(parts)

  values = map_parts(parts) do part
    10*part.id
  end
  
  map_parts(parts,values) do part, value
    @test 10*part.id == value
  end

end
