using MPI
include("run_mpi_driver.jl")
file = joinpath(@__DIR__,"drivers","p_range_tests.jl")
run_mpi_driver(file;procs=4)

