using MPI
using Test
include("run_mpi_driver.jl")
file = joinpath(@__DIR__,"drivers","exception_tests.jl")
failed = Ref(false)
try
    run_mpi_driver(file;procs=4)
catch e
    failed[] = true
end
@test failed[]
