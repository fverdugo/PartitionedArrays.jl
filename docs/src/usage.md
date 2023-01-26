# Usage

PartitionedArrays considers a data-oriented programming model that allows one to write distributed algorithms
in a generic way, independent from the message passing back-end used to run them.
The basic abstraction in this model is that distributed data can be expressed using array containers.
The particular container type will depend on the back-end used to run the code in parallel (e.g., MPI).

## Basic usage

We want each rank in a distributed system to print its rank id and the total number of ranks. Here, the distributed
data are the rank ids. If we have an array with all rank ids, printing the messages is trivial with `map` function.

```julia
np = 4
ranks = LinearIndices((np,))
map(ranks) do rank
   println("I am proc $rank of $np.")
end
```
Previous code is not parallel (yet). However, it can be easily parallelized if one considers a suitable distributed
array type and if this type overloads `map` function with a parallel implementation.

```julia
# File examples/hello_mpi.jl
using PartitionedArrays
using MPI; MPI.Init()
np = 4
ranks = mpi_distribute(LinearIndices((np,)))
map(ranks) do rank
   println("I am proc $rank of $np.")
end
```

Now this code is parallel. Function `mpi_distribute` takes an array and distributes it over the different
ranks of a given MPI communicator, a duplicate of `MPI.COMM_WORLD` by default. The result is of a new
array type called `MPIData`, which overloads function `map` with a parallel implementation.
Function `mpi_distribute` assigns exactly
one item in the input array to each rank in the communicator. Thus, the resulting array `ranks` will be distributed
in such a way that each MPI rank will get an integer corresponding to its (1-based) rank id. If we place
  this code in a file called `"hello_mpi.jl"`, we can run it as any Julia applications using the Julia API for MPI in
  `MPI.jl`. For instance,

```julia
using MPI
mpiexec(cmd->run(`$cmd -np4 hello_mpi.jl`))
```

The construction of the array `ranks` containing the rank ids is just the first step of a computation
using PartitionedArrays. See the examples for more interesting cases.


## Running MPI code safely

MPI applications should call `MPI.Abort` if they need to stop prematurely (e.g., by an error).
The Julia error handling system is not aware of that. For this reasons, codes like the following one
will crash and stop without calling `MPI.Abort`.

```julia
using MPI; MPI.Init()
using PartitionedArrays
np = 3
ranks = mpi_distribute(LinearIndices((np,)))
map(ranks) do rank
    if rank == 2
        error("I have crashed")
    end
end
```
Even worse, the code will crash only in the 2nd MPI process. The other ones will finish normally.
This can lead to zombie MPI processes running (and provably consuming quota in your cluster account).
To fix this, PartitionedArrays provides function `with_mpi`. We rewrite the previous example using it.

```julia
using PartitionedArrays
with_mpi() do distribute
    np = 3
    ranks = distribute(LinearIndices((np,)))
    map(ranks) do rank
        if rank == 2
            error("I have crashed")
        end
    end
end
```
Essentially, `with_mpi(f)` calls `f(mpi_distribute)` in a `try`-`catch` block. If some error is cached, 
`MPI.Abort` will be called, safely finishing all the MPI processes, also the ones that did not experienced
the error. Function `with_mpi` also initializes MPI if needed. So, you do not need to explicitly call
`MPI.Init()` in the driver.

## Debugging

One of the main advantages of PartitionedArrays is that it allows you to write and debug your parallel
code without using MPI. This makes possible to use the standard Julia development workflow (e.g., Revise)
which will make your life much easier when implementing distributed applications. This ability comes from the
fact that one can use standard serial Julia arrays to test your code. However, the array type `MPIData` resulting
after distributing data over MPI processes, is not as flexible as the standard arrays in Julia. There are operations
that are not allowed for `MPIData` for performance reasons in general. One of them is scalar indexing.
For this reason, PartitionedArrays provides an special array type called `SequentialData` for debugging purposes.
The type `SequentialData` tries to mimic the limitations of `MPIData` but it is just a wrapper to a standard
Julia array, and therefore can be used in a standard Julia session.

```julia
using PartitionedArrays
np = 4
ranks = SequentialData(LinearIndices((np,)))
ranks[3] # Error!
```
The last line of code will throw an error telling that scalar indexing is not allowed. This is to mimic the error
you would get in production when using MPI.

```julia
using PartitionedArrays
with_mpi() do distribute
    np = 4
    ranks = distribute(LinearIndices((np,)))
    ranks[3] # Error!
end
```
We also provide function `with_sequential_data` which allows to easily switch from one back-end to the other.
For instance, if we define the following main function

```julia
using PartitionedArrays
function main(distribute)
    np = 4
    ranks = distribute(LinearIndices((np,)))
    map(ranks) do rank
       println("I am proc $rank of $np")
    end
end
```
then `with_sequential_data(main)` and `with_mpi(main)` will run the code using the
debug back-end and MPI respectively.

