module BlockArraysTests

using MPI

project = joinpath(@__DIR__,"..","..")
code = quote
    include(joinpath($project,"test","block_arrays_tests.jl"))
    with_mpi(block_arrays_tests)
end

run(`$(mpiexec()) -np 4 julia --project=$project -e $code `)

end # module

