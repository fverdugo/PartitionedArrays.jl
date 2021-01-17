# ChunkyAlgebra.jl

🚧 ⛏️ WIP

## What

[![Build Status](https://github.com/fverdugo/ChunkyAlgebra.jl/workflows/CI/badge.svg)](https://github.com/fverdugo/ChunkyAlgebra.jl/actions)
[![Coverage](https://codecov.io/gh/fverdugo/ChunkyAlgebra.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/fverdugo/ChunkyAlgebra.jl)

This package provides a data-oriented parallel implementation of the basic linear algebra objects needed in FD and FE simulations. The long-term goal of this package is to provide (when combined with other Julia packages as `IterativeSolvers.jl`) a Julia alternative to well-known distributed algebra back ends such as `PETSc` or `Trilinos`.

At this moment, a simple FD system can be assembled and solved in parallel with this package together with a Conjugate Gradient method from `IterativeSolvers.jl` . See the file [test_fdm.jl]( https://github.com/fverdugo/ChunkyAlgebra.jl/blob/master/test/test_fdm.jl).

These basic types are currently implemented:
- `ChunkyData`: The low level type representing some data distributed over several chunks or parts. This is the core component of the data-oriented parallel implementation.
- `ChunkyRange`: A specialization of `AbstractUnitRange` that has information about how the ids in the range are distributed in different chunks. This type is used to describe the parallel data layout of rows and cols in `ChunkyVector` and `ChunkySparseMatrix` objects.
- `ChunkyVector`: A vector distributed in (overlapping or non-overlapping) chunks.
- `ChunkySparseMatrix`: A sparse matrix distributed in (overlapping or non-overlapping)) chunks of rows.

On these types, several communication operations are defined:

- `gather!`, `gather`, `gather_all!`, `gather_all`
- `reduce`, `reduce_all`, `reduce_master`
- `scatter`, `bcast`
- `exchange!` `exchange`, `async_exchange!` `async_exchange`
- `assemble!`, `async_assemble!`

## Why

One can use PETSc bindings like [PETSc.jl](https://github.com/JuliaParallel/PETSc.jl) for parallel computations in Julia, but this approach has some limitations:

- PETSc is hard-codded for vectors/matrices of some particular element types (e.g. Float64 and ComplexF64).

- PETSc forces one to use MPI as the parallel execution model. Drivers are executed as `mpirun -np 4 julia --project=. input.jl`, which means no interactive Julia sessions, no `Revise`, no `Debugger`. This is a major limitation to develop parallel algorithms.

This package aims to overcome these limitations. It implements (and allows to implement) parallel algorithms in a generic way independently of the underlying hardware / message passing software that is eventually used. At this moment, this library provides two back ends for running the generic parallel algorithms:
- `SequentialBackend`: The parallel data is split in chunks, which are stored in a conventional (sequential) Julia session (typically in an `Array`). The tasks in the parallel algorithms are executed one after the other. Note that the sequential back end does not mean to distribute the data in a single part. The data can be split in an arbitrary number of parts. 
- `MPIBackend`: Chunks of parallel data and parallel tasks are mapped to different MPI processes. The drivers are to be executed in MPI mode, e.g., `mpirun -n 4 julia --project=. input.jl`.


The `SequentialBackend` is specially handy for developing new code. Since it runs in a standard Julia session, one can use tools like `Revise` and `Debugger` that will certainly do your live easier at the developing stage. Once the code works with the `SequentialBackend`, it can be automatically deployed in a super computer via the `MPIBackend`.  Other back ends like a `ThreadedBacked`, `DistributedBackend`, or `MPIXBackend` can be added in the future.

