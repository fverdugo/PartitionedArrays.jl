using MPI
include("run_mpi_driver.jl")
file = joinpath(@__DIR__,"drivers","fdm_example.jl")
run_mpi_driver(file;procs=4)

