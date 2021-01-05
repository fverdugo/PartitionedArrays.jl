using MPI
using Test
function run_mpi_driver(;procs,file)
  mpidir = @__DIR__
  testdir = joinpath(mpidir,"..")
  repodir = joinpath(testdir,"..")
  mpiexec() do cmd
    run(`$cmd -n $procs $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`)
    @test true
  end
end
