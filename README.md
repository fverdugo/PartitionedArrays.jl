# DistributedDataDraft

🚧 ⛏️ WIP (DistributedDataDraft is just a provisional dummy package name)

## What

[![Build Status](https://github.com/fverdugo/DistributedDataDraft.jl/workflows/CI/badge.svg)](https://github.com/fverdugo/DistributedDataDraft.jl/actions)
[![Coverage](https://codecov.io/gh/fverdugo/DistributedDataDraft.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/fverdugo/DistributedDataDraft.jl)

This package provides a data-oriented parallel implementation of the basic linear algebra objects needed in FD and FE simulations. The long-term goal of this package is to provide (when combined with other Julia packages as `IterativeSolvers.jl`) a Julia alternative to well-known distributed algebra backends such as `PETSc` or `Trilinos`.

At this moment, a simple FD system can be assembled and solver in parallel with this package with a Conjugate Gradients method from `IterativeSolvers.jl` . See the file [test_fdm.jl]( https://github.com/fverdugo/DistributedDataDraft.jl/blob/master/test/test_fdm.jl).

Three basic types are currently implemented:
- `DistributedData`: The low level type representing some data distributed over several parts. This is the core component of the data-oritented parallel implementation.
- `DistributedVector`: A vector distributed in (overlapping or non-overlapping) chunks.
- `DistributedSparseMatrix`: A sparse matrix distributed in several (overlapping or non-overlapping)) chunks.

On these types, several communiction operations are defined:

- `gather!`, `gather`, `gather_all!`, `gather_all`
- `reduce`, `reduce_all`, `reduce_master`
- `scatter`, `bcast`
- `exchange!` `exchange`, `async_exchange!` `async_exchange`
- `assemble!`, `async_assemble!`

## Why

The basic desing noverly of this lirbary is that it implements (and allows to implement) parallel algorithms in a generic way independently of the underelying harware / message passing software that is eventually used. At this moment, this library provides two backends for running the generic parallel algorithms (others like a `ThreadedBacked` or `MPIXBackend` can be added in the future):
- `SequantialBackend`: The parallel data is splitted in chuncks, which are stored in a conventional (sequential) Julia session (typically in an `Array`). The tasks in the parallel algorithms are executed one after the other. Note that the sequential backend does not mean to distribute the data in a single part. The data can be splitted in an arbitrary number of parts. 
- `MPIBackend`: Chunks of parallel data and parallel tasks are mapped to different MPI processes. The drivers are to be executed in MPI mode, e.g., `mpirun julia --project=. input.jl`.

The `SequantialBackend` is specially handy for developing new code. Since it runs in a standard Julia session, one can use tools like `Revise` and `Debugger` that will certainly do your live easier at the developing stage. Once the code works with the `SequantialBackend` can be automatically deployed in a super computer via the `MPIBackend`.


