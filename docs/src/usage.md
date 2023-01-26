# Usage

PartitionedArrays.jl considers a data-oriented programming model that allows one to write distributed algorithms
in a generic way, independent from the message passing back-end used to run them.
The basic abstraction in this model is that distributed data can be expressed using array containers.
The particular container type will depend on the back-end used to run the code in parallel (e.g., MPI).

## Basic example

We want each rank in a distributed system to print its rank id and the total number of ranks. Here, the distributed
data are the rank ids. If we have an array with all rank ids, printing the messages is trivial with `map` function.

```julia
np = 4
ranks = LinearIndices((np,))
map(ranks) do rank
   println("I am proc $rank of $np")
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
   println("I am proc $rank of $np")
end
```

Now this code is parallel. Function `mpi_distribute` takes an array and distributes it over the different
ranks of a given MPI communicator, a duplicate of `MPI.COMM_WORLD` by default. The result is of a new
array type called `MPIData`, which overloads function `map` with a parallel implemention.
Function `mpi_distribute` assigns exactly
one item in the input array to each rank in the communicator. Thus, the resulting array `rank` will be distributed
in such a way that each MPI rank will get an integer corresponding to its (1-based) rank id. If we place
  this code in a file called `"hello_mpi.jl"`, we can run it as any Julia applications using the Julia API for MPI in
  `MPI.jl`. For instance,

```julia
using MPI
mpiexec(cmd->run(`$cmd -np4 hello_mpi.jl`))
```

## Communication primitives

A number of communication primitives are available in this package. For the moment, the package includes
the basic primitives needed to implement distributed-memory finite differences and finite element codes, but other
primitives can be added in the future. Communications are represented in PartitionedArrays.jl as operations on arrays.
The data in the input and output arrays in the communication directives are usually the same, but arranged in a different
way. When these arrays are distributed, this re-arrangement of the data results in the desired communications.
In the following code, the first rank generates an array of random integers and scatters it over all ranks. Each rank
counts the number of even items in its part. Finally, the partial sums are reduced in the first rank.

```julia
using PartitionedArrays
np = 4
ranks = LinearIndices((np,))
a_snd = map(ranks) do rank
    if rank == 1
          n = 10
          a = rand(1:30,n)
          load = div(n,np)
          [ a[(1:load).+(i-1)*load] for i in 1:np ]
    else
          [Int[]]
    end
end
a_rcv = scatter(a_snd)
b_snd = map(ai->count(isodd,ai),a_rcv)
b_rcv = reduction(+,b_snd,init=0)
```


## Parallel vectors and sparse matrices

## Running MPI code safely

## Debugging


