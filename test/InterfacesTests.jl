module InterfacesTests

using DistributedDataDraft
using Test

nparts = 4
distributed_run(sequential,nparts) do parts1

  parts2 = Partition(sequential,nparts)
  map_parts(parts1,parts2) do part1, part2
    @test part1 == part2
  end

end



end # module
