module BenchmarkTests

import PartitionedArraysBenchmarks as pb
import PartitionedArrays as pa

cells_per_dir=(10,10,10)
parts_per_dir=(1,1,1)
nruns = 10

params = (;nruns,cells_per_dir,parts_per_dir,method="psprse")
jobparams = (;)
out = pa.with_mpi(distribute->pb.benchmark_psparse(distribute,params,jobparams))

params = (;nruns,cells_per_dir,parts_per_dir,method="petsc_setvalues")
jobparams = (;)
out = pa.with_mpi(distribute->pb.benchmark_psparse(distribute,params,jobparams))

params = (;nruns,cells_per_dir,parts_per_dir,method="petsc_coo")
jobparams = (;)
out = pa.with_mpi(distribute->pb.benchmark_psparse(distribute,params,jobparams))

params = (;nruns,cells_per_dir,parts_per_dir,method="PartitionedArrays")
jobparams = (;)
out = pa.with_mpi(distribute->pb.benchmark_spmv(distribute,params,jobparams))

params = (;nruns,cells_per_dir,parts_per_dir,method="Pestc")
jobparams = (;)
out = pa.with_mpi(distribute->pb.benchmark_spmv(distribute,params,jobparams))

end # module
