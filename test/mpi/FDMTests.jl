module FDMTests

include("mpiexec.jl")
run_mpi_driver(procs=4,file="driver_fdm.jl")

end # module
