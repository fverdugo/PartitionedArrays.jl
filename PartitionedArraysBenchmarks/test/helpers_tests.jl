module HelpersTests

import PartitionedArraysBenchmarks as pb
using PartitionedArrays

results_dir = mkpath("results")

params = (;
          nodes = 2,
          ntasks_per_node = 1,
          np = 2,
          method="PartitionedArrays",
          cells_per_dir = (10,10,10),
          parts_per_dir = (2,1,1),
          nruns = 10,
          benchmark = :benchmark_spmv
         )
#pb.string_dict(params) |> display
#pb.symbol_dict(params) |> display
#pb.symbol_dict(params) |> pb.string_dict |> display
#pb.string_dict(params) |> pb.jobparams |> display
#pb.symbol_dict(params) |> pb.jobparams |> display
#pb.symbol_dict(params) |> pb.string_dict |> pb.jobparams |> display

pb.experiment(pb.benchmark_spmv,"experiment",DebugArray,params;results_dir)
pb.runjob(:bash,pb.string_dict(params);results_dir)
pb.runjob(:bash,params;results_dir)

end  # module
