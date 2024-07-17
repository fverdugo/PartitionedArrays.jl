module GalleryTests

using MPI

project = joinpath(@__DIR__,"..","..")
code = quote
    include(joinpath($project,"test","gallery_tests.jl"))
    with_mpi(gallery_tests)
end

run(`$(mpiexec()) -np 4 julia --project=$project -e $code `)

end # module
