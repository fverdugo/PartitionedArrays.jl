module MPITests

using MPI
using Test

testdir = joinpath(@__DIR__,"..")
repodir = joinpath(testdir,"..")

files_and_procs = [
  "HelloTests.jl"=>3,
  "MPIBackendTests.jl"=>4,
  "MPIBackendTests.jl"=>4]

for (file,procs) in files_and_procs
  mpiexec() do cmd
    run(`$cmd -n $procs $(Base.julia_cmd()) --project=$repodir $(joinpath(@__DIR__,file))`)
    @test true
  end
end

end
