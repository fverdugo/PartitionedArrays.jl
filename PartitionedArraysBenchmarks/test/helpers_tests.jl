module HelpersTests

import PartitionedArraysBenchmarks as pb
using PartitionedArrays

path = mkpath("results")
filename = params -> joinpath(path,pb.jobname(params))

params = (;
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

pb.experiment(pb.benchmark_spmv,filename(params),DebugArray,params)

template = raw"""
julia --project={{{__PROJECT__}}} -O3 --check-bounds=no -e '
code = quote
   import PartitionedArraysBenchmarks as pb
   import PartitionedArrays as pa
   params = {{{__PARAMS__}}}
   jobname = "{{{__JOBNAME__}}}"
   pa.with_mpi() do distribute
       pb.experiment(pb.{{{benchmark}}},jobname,distribute,params)
   end
end
using MPI
cmd = mpiexec()
run(`$cmd -np {{{np}}} julia --project={{{__PROJECT__}}} -O3 --check-bounds=no -e $code`)
'
"""

pb.runjob(:bash,template,pb.string_dict(params);filename)
pb.runjob(:bash,template,params;filename)

end  # module
