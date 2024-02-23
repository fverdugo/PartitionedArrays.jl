module BenchmarkTests

import PartitionedArraysBenchmarks as pb
import PartitionedArrays as pa
import MPI

cells_per_dir=(10,10,10)
parts_per_dir=(1,1,1)
nruns = 10

params = (;nruns,cells_per_dir,parts_per_dir,method="psprse")
out = pa.with_mpi(distribute->pb.benchmark_psparse(distribute,params))

params = (;nruns,cells_per_dir,parts_per_dir,method="petsc_setvalues")
out = pa.with_mpi(distribute->pb.benchmark_psparse(distribute,params))

params = (;nruns,cells_per_dir,parts_per_dir,method="petsc_coo")
out = pa.with_mpi(distribute->pb.benchmark_psparse(distribute,params))

params = (;nruns,cells_per_dir,parts_per_dir,method="PartitionedArrays")
out = pa.with_mpi(distribute->pb.benchmark_spmv(distribute,params))

params = (;nruns,cells_per_dir,parts_per_dir,method="Pestc")
out = pa.with_mpi(distribute->pb.benchmark_spmv(distribute,params))

code = quote
    import PartitionedArraysBenchmarks as pb
    import PartitionedArrays as pa
    pa.with_mpi() do distribute
        cells_per_dir=(10,10,10)
        parts_per_dir = (2,2,1)
        nruns = 10
        params = (;nruns,cells_per_dir,parts_per_dir,method="PartitionedArrays")
        out = pb.benchmark_spmv(distribute,params)
        params = (;nruns,cells_per_dir,parts_per_dir,method="Pestc")
        out = pb.benchmark_spmv(distribute,params)
        params = (;nruns,cells_per_dir,parts_per_dir,method="psprse")
        out = pb.benchmark_psparse(distribute,params)
        params = (;nruns,cells_per_dir,parts_per_dir,method="petsc_coo")
        out = pb.benchmark_psparse(distribute,params)
        params = (;nruns,cells_per_dir,parts_per_dir,method="petsc_setvalues")
        out = pb.benchmark_psparse(distribute,params)
    end
end
project = Base.active_project()
cmd = MPI.mpiexec()
run(`$cmd -np 4 julia --project=$project -e $code`)


end # module
