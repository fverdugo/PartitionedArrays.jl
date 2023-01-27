# Usage

PartitionedArrays considers a data-oriented programming model that allows one to write distributed algorithms
in a generic way, independent from the message passing back-end used to run them.
The basic abstraction of this model consists in expressing distributed data using array containers.
The particular container type will depend on the back-end used to run the code in parallel. MPI is one of the possible
back-ends, used to run large cases on computational clusters.

## Basic usage

We want each rank in a distributed system to print its rank id and the total number of ranks. The distributed
data are the rank ids. If we have an array with all rank ids, printing the messages is trivial with `map` function.

```julia
np = 4
ranks = LinearIndices((np,))
map(ranks) do rank
   println("I am proc $rank of $np.")
end
```
Previous code is not parallel (yet). However, it can be easily parallelized if one considers a suitable distributed
array type that overloads `map` with a parallel implementation.

```julia
# hello_mpi.jl
using PartitionedArrays
np = 4
ranks = distribute_with_mpi(LinearIndices((np,)))
map(ranks) do rank
   println("I am proc $rank of $np.")
end
```

Now this code is parallel. Function `distribute_with_mpi` takes an array and distributes it over the different
ranks of a given MPI communicator (a duplicate of `MPI.COMM_WORLD` by default). The type of the result is an
array type called `MPIData`, which overloads function `map` with a parallel implementation.
Function `distribute_with_mpi` assigns exactly
one item in the input array to each rank in the communicator. Thus, the resulting array `ranks` will be distributed
in such a way that each MPI rank will get an integer corresponding to its (1-based) rank id. If we place
  this code in a file called `"hello_mpi.jl"`, we can run it as any Julia applications using the MPI API in
  `MPI.jl`. For instance,

```julia
using MPI
mpiexec(cmd->run(`$cmd -np4 hello_mpi.jl`))
```

The construction of the array `ranks` containing the rank ids is just the first step of a computation
using PartitionedArrays. See the [Examples](@ref) for more interesting cases.


## Running MPI code safely

MPI applications should call `MPI.Abort` if they stop prematurely (e.g., by an error).
The Julia error handling system is not aware of that. For this reasons, codes like the following one
will crash and stop without calling `MPI.Abort`.

```julia
using PartitionedArrays
np = 3
ranks = distribute_with_mpi(LinearIndices((np,)))
map(ranks) do rank
    if rank == 2
        error("I have crashed")
    end
end
```
Even worse, the code will crash only in the 2nd MPI process. The other processes will finish normally.
This can lead to zombie MPI processes running in the background (and provably consuming quota in your cluster account until
the queuing system kills them).
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
Essentially, `with_mpi(f)` calls `f(distribute_with_mpi)` in a `try`-`catch` block. If some error is cached, 
`MPI.Abort` will be called, safely finishing all the MPI processes, also the ones that did not experienced
the error.

## Debugging

One of the main advantages of PartitionedArrays is that it allows one to write and debug your parallel
code without using MPI. This makes possible to use the standard Julia development workflow (e.g., Revise)
 when implementing distributed applications, which is certainly useful. This ability comes from the
fact that one can use standard serial Julia arrays to test your application based on PartitionedArrays.
However, the array type `MPIData` resulting
after distributing data over MPI processes, is not as flexible as the standard arrays in Julia. There are operations
that are not allowed for `MPIData`, mainly for performance reasons. One of them is indexing the array at arbitrary indices.
In consequence, code that runs with the common Julia arrays might fall when switching to MPI.
In order to anticipate these type of errors,
PartitionedArrays provides an special array type called `SequentialData` for debugging purposes.
The type `SequentialData` tries to mimic the limitations of `MPIData` but it is just a wrapper to a standard
Julia array and therefore can be used in a standard Julia session.

```julia
using PartitionedArrays
np = 4
ranks = SequentialData(LinearIndices((np,)))
ranks[3] # Error!
```
The last line of previous code will throw an error telling that scalar indexing is not allowed. This is to mimic the error
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
debug back-end and MPI respectively. If you want to run in using native Julia arrays, you can simply call `main(identity)`.
Make sure that your code works using `SequentialData` before moving to MPI.

