module ExceptionTests
using Test

include("mpiexec.jl")
failed = Ref(false)
try
  run_mpi_driver(procs=8,file="driver_exception.jl")
catch e
  failed[] = true
end
@test failed[]

end # module
