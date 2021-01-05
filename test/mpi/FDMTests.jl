module FDMTests

include("mpiexec.jl")
run_mpi_driver(procs=8,file="driver_fdm.jl")

end # module
