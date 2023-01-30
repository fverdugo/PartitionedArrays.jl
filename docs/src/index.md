# PartitionedArrays.jl

Welcome to the documentation for PartitionedArrays.jl!

## What

This package provides distributed (a.k.a. partitioned) vectors and sparse matrices like the ones needed in
distributed finite differences, finite volumes, or finite element computations. Packages such [`GridapDistributed`](https://github.com/gridap/GridapDistributed.jl) have shown weak and strong scaling up to tens of thousands of CPU cores in
the distributed assembly of sparse linear systems when using PartitionedArrays as their distributed linear algebra back-end. See this publication for further details:

> Santiago Badia, Alberto F. Martín, and Francesc Verdugo (2022). "GridapDistributed: a massively parallel finite element toolbox in Julia". Journal of Open Source Software, 7(74), 4157.  doi: [10.21105/joss.04157](https://doi.org/10.21105/joss.04157).

## Why

The main objective of this package is to avoid to interface directly with MPI or MPI-based libraries when prototyping
and debugging distributed parallel codes. MPI-based applications are executed as in batch mode with commands like `mpiexec -np 4 julia input.jl`, which break the Julia development workflow. In particular, one starts a fresh Julia session at each run, making difficult to reuse compiled code between runs. In addition, packages like `Revise` and `Debugger` are also difficult to use in combination with MPI computations on several processes.

To overcome these limitations, PartitionedArrays considers a data-oriented programming model that allows one to write distributed algorithms in a generic way, independent from the message passing back-end used to run them.  MPI is one of the possible back-ends available in PartitionedArrays, used to deploy large computations on computational clusters. However, one can also use other back-ends that are able to run on standard serial Julia sessions, which allows one to use the standard Julia workflow to develop and debug complex codes in an effective way.


