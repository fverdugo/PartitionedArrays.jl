module MPIBackendTests

include("mpiexec.jl")
run_mpi_driver(procs=4,file="driver_mpi_backend.jl")

end # module
