module ExceptionTests

include("mpiexec.jl")
run_mpi_driver(procs=8,file="driver_exception.jl")

end # module
