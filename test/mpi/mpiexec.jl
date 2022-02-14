using MPI
using Test
function run_mpi_driver(;procs,file)
  mpidir = @__DIR__
  testdir = joinpath(mpidir,"..")
  repodir = joinpath(testdir,"..")
  mpiexec() do cmd
    if MPI.MPI_LIBRARY == MPI.OpenMPI
      @test success(`$cmd -n $procs --oversubscribe $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`)
    else
      @test success(`$cmd -n $procs $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`)
    end
  end
end
