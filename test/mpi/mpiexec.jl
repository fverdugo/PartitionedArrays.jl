using MPI
using Test
function run_mpi_driver(;procs,file)
  mpidir = @__DIR__
  testdir = joinpath(mpidir,"..")
  repodir = joinpath(testdir,"..")
  mpiexec() do cmd
    if MPI.MPI_LIBRARY == MPI.OpenMPI
      run(`$cmd -n $procs --oversubscribe $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`)
    else
      run(`$cmd -n $procs $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`)
    end
    @test true
  end
end
