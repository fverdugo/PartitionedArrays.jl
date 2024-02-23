module HelpersTests

import PartitionedArraysBenchmarks as pb
using PartitionedArrays

project = Base.active_project()

params = (;
          nodes = 2,
          ntasks_per_node = 1,
          np = 2,
          method="PartitionedArrays",
          cells_per_dir = (10,10,10),
          parts_per_dir = (2,1,1),
          nruns = 10
         )

pb.experiment(pb.benchmark_spmv,"experiment",DebugArray,params)

pb.runjob(:bash,:benchmark_spmv,params;project)

end  # module
