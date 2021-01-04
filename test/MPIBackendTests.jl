module MPIBackendTests

using MPI
using Test

testdir = @__DIR__
repodir = joinpath(testdir,"..")

files_and_procs = [
  "test_mpi_hello.jl"=>3,
  "test_mpi.jl"=>4,
  "test_interfaces_mpi.jl"=>4]

for (file,procs) in files_and_procs
  mpiexec() do cmd
    run(`$cmd -n $procs $(Base.julia_cmd()) --project=$repodir $(joinpath(testdir, file))`)
    @test true
  end
end

end # module
