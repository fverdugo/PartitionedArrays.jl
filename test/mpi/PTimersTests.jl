module PTimersTests

include("mpiexec.jl")
run_mpi_driver(procs=4,file="driver_p_timers.jl")

end # module
