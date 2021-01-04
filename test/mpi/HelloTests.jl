module HelloTests

include("mpiexec.jl")
run_mpi_driver(procs=3,file="driver_hello.jl")

end # module
