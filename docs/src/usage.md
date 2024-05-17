# Usage

PartitionedArrays considers a data-oriented programming model that allows one to write distributed algorithms
in a generic way, independent from the message passing back-end used to run them.
The basic abstraction of this model consists in expressing distributed data using array containers.
The particular container type will depend on the back-end used to run the code in parallel. MPI is one of the possible
back-ends, used to run large cases on computational clusters. However, one can also use serial arrays to
prototype and debug complex codes in an effective way.

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
array type called `MPIArray`, which overloads function `map` with a parallel implementation.
Function `distribute_with_mpi` assigns exactly
one item in the input array to each rank in the communicator. Thus, the resulting array `ranks` will be distributed
in such a way that each MPI rank will get an integer corresponding to its (1-based) rank id. If we place
  this code in a file called `"hello_mpi.jl"`, we can run it as any Julia applications using the MPI API in
  `MPI.jl`. For instance,

```julia
using MPI
mpiexec(cmd->run(`$cmd -np 4 julia --project=. hello_mpi.jl`))
```

The construction of the array `ranks` containing the rank ids is just the first step of a computation
using PartitionedArrays. See the [Examples](@ref) for more interesting cases.

## Debugging

One of the main advantages of PartitionedArrays is that it allows one to write and debug your parallel
code without using MPI. This makes possible to use the standard Julia development workflow (e.g., Revise)
 when implementing distributed applications, which is certainly useful. This ability comes from the
fact that one can use standard serial Julia arrays to test your application based on PartitionedArrays.
However, the array type `MPIArray` resulting
after distributing data over MPI processes, is not as flexible as the standard arrays in Julia. There are operations
that are not allowed for `MPIArray`, mainly for performance reasons. One of them is indexing the array at arbitrary indices.
In consequence, code that runs with the common Julia arrays might fall when switching to MPI.
In order to anticipate these type of errors,
PartitionedArrays provides an special array type called `DebugArray` for debugging purposes.
The type `DebugArray` tries to mimic the limitations of `MPIArray` but it is just a wrapper to a standard
Julia array and therefore can be used in a standard Julia session.

```julia
using PartitionedArrays
np = 4
ranks = DebugArray(LinearIndices((np,)))
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
We also provide function `with_debug` which allows to easily switch from one back-end to the other.
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
then `with_debug(main)` and `with_mpi(main)` will run the code using the
debug back-end and MPI respectively. If you want to run in using native Julia arrays, you can simply call `main(identity)`.
Make sure that your code works using `DebugArray` before moving to MPI.


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

## Benchmarking distributed codes

When using MPI, the computational time to run some code can be different for each one of
the processes. Usually, one measures the time for each process and computes some statistics
of the resulting values. This is done by doing time measurements with the tool of your choice and then `gather`ing the results
on the root for further analysis. Note that this is possible thanks to the changes in version 0.4.1
that allow one to use  `gather` on arbitrary objects.

In the following example, we force different computation times at each of the processes
by sleeping a value proportional to the rank id. We gather all the timings in the main process and compute some statistics:

```julia
using PartitionedArrays
using Statistics
with_mpi() do distribute
    np = 3
    ranks = distribute(LinearIndices((np,)))
    t = @elapsed map(ranks) do rank
        sleep(rank)
    end
    ts = gather(map(rank->t,ranks))
    map_main(ts) do ts
        @show ts
        @show maximum(ts)
        @show minimum(ts)
        @show Statistics.mean(ts)
    end
end
```

```
ts = [1.001268313, 2.0023204, 3.001216396]
maximum(ts) = 3.001216396
minimum(ts) = 1.001268313
Statistics.mean(ts) = 2.001601703
```

This mechanism also works for the other back-ends. For sequential ones, it provides the time
spend by all parts combined. Note how we define `t` (outside the call to `map`) and the object passed to `gather`.

```julia
using PartitionedArrays
using Statistics
with_debug() do distribute
    np = 3
    ranks = distribute(LinearIndices((np,)))
    t = @elapsed map(ranks) do rank
        sleep(rank)
    end
    ts = gather(map(rank->t,ranks))
    map_main(ts) do ts
        @show ts
        @show maximum(ts)
        @show minimum(ts)
        @show Statistics.mean(ts)
    end
end;
```

```
ts = [6.009726399, 6.009726399, 6.009726399]
maximum(ts) = 6.009726399
minimum(ts) = 6.009726399
Statistics.mean(ts) = 6.009726398999999
```

We can also consider more sophisticated ways of measuring the times, e.g., with [TimerOutputs](https://github.com/KristofferC/TimerOutputs.jl).

```julia
using PartitionedArrays
using Statistics
using TimerOutputs
with_mpi() do distribute
    np = 3
    ranks = distribute(LinearIndices((np,)))
    to = TimerOutput()
    @timeit to "phase 1" map(ranks) do rank
        sleep(rank)
    end
    @timeit to "phase 2" map(ranks) do rank
        sleep(2*rank)
    end
    tos = gather(map(rank->to,ranks))
    map_main(tos) do tos
        # check the timings on the first rank
        display(tos[1])
        # compute statistics for phase 1
        ts = map(tos) do to
            TimerOutputs.time(to["phase 1"])
        end
        @show ts
        @show maximum(ts)
        @show minimum(ts)
        @show Statistics.mean(ts)
    end
end
```

```
 ────────────────────────────────────────────────────────────────────
                            Time                    Allocations      
                   ───────────────────────   ────────────────────────
 Tot / % measured:      10.3s /  29.3%           44.9MiB /   0.0%    

 Section   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────
 phase 2        1    2.00s   66.6%   2.00s      120B   50.0%     120B
 phase 1        1    1.00s   33.4%   1.00s      120B   50.0%     120B
 ────────────────────────────────────────────────────────────────────
ts = [1002323746, 2001614329, 3004363808]
maximum(ts) = 3004363808
minimum(ts) = 1002323746
Statistics.mean(ts) = 2.0027672943333333e9
```

In addition, the library provides a special timer type called [`PTimer`](@ref).

!!! note
    `PTimer` has been deprecated. Do time measurements with the tool of your choice and then `gather` the results
    on the root for further analysis (see above).


In the following example we force different computation times at each of the processes
by sleeping a value proportional to the rank id.
When displayed, the instance of [`PTimer`](@ref) shows some statistics of the
times over the different processes.

```julia
using PartitionedArrays
with_mpi() do distribute
    np = 3
    ranks = distribute(LinearIndices((np,)))
    t = PTimer(ranks)
    tic!(t)
    map(ranks) do rank
        sleep(rank)
    end
    toc!(t,"Sleep")
    display(t)
end
```

```
───────────────────────────────────────────
Section         max         min         avg
───────────────────────────────────────────
Sleep     3.021e+00   1.021e+00   2.021e+00
───────────────────────────────────────────
```

This mechanism also works for the other back-ends. For sequential ones, it provides the time
spend by all parts combined.

```julia
using PartitionedArrays
with_debug() do distribute
    np = 3
    ranks = distribute(LinearIndices((np,)))
    t = PTimer(ranks)
    tic!(t)
    map(ranks) do rank
        sleep(rank)
    end
    toc!(t,"Sleep")
    display(t)
end
```

```
───────────────────────────────────────────
Section         max         min         avg
───────────────────────────────────────────
Sleep     6.010e+00   6.010e+00   6.010e+00
───────────────────────────────────────────
```

