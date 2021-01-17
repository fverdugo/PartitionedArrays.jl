module FEMSATests

include("mpiexec.jl")
run_mpi_driver(procs=4,file="driver_fem_sa.jl")

end # module
