# Usage

Distributed linear algebra frameworks are the backbone for efficient parallel
codes in data analytics, scientific computing and machine learning. The central
idea is that vectors and matrices can be partitioned into potentially
overlapping chunks which are distributed across a set of workers on which we
define the usual operations like products and norms.

## Basic example

In this section we take a look on solving the finite difference discretization
of a Laplace problem in 1D over the domain [0,1]. As a reminder, the Laplace
problem states to find function u(x) such that Δu(x) = 0 for all x ∈ [0,1].
Without boundary conditions the problem is not well-posed, hence we introduce
the Dirichlet condition u(0) = 1.

Applying the finite difference method with length 0.25 we discretize the problem
into linear system with 5 unkowns (u₁,...,u₅), which we call degrees of freedom:
```math
\frac{1}{4}
\begin{pmatrix}
1 &  0 &  0 &  0 &  0 \\
0 & -2 &  1 &  0 &  0 \\
0 &  1 & -2 &  1 &  0 \\
0 &  0 &  1 & -2 &  1 \\
0 &  0 &  0 &  1 & -1
\end{pmatrix}
\begin{pmatrix}
u₁ \\
u₂ \\
u₃ \\
u₄ \\
u₅
\end{pmatrix}
=
\begin{pmatrix}
 1 \\
-1 \\
 0 \\
 0 \\
 0
\end{pmatrix}
```

A detailed derivation can be found in standard numerical analysis lecture notes and books e.g. [these](https://people.sc.fsu.edu/~jburkardt/classes/math2071_2020/poisson_steady_1d/poisson_steady_1d.pdf). The linear system is then solved with
conjugate gradients.

### Commented Code

To distribute the problem across two workers we have do choose a partitioning.
Here we arbitrarily assign the first 3 columns and rows to worker 1 and the
remaining 2 rows and columns to worker 2. The commented code for this problems
is given as

```julia
using PartitionedArrays, SparseArrays, IterativeSolvers

# We want a partitioning into 2 pieces
np = 2

# We chose the sequential backend to handle the task sequentially to simplify debugging
backend = SequentialBackend()

# This the the execution environment
prun(backend,np) do parts
    # Generate partitioning of rows and columns
    neighbors, row_partitioning, col_partitioning = map_parts(parts) do part
        if part == 1
            # Note that the entry in row 3 column 4 is visible to the first worker
            Int32[2], IndexSet(part, [1,2,3], Int32[1,1,1]), IndexSet(part, [1,2,3,4], Int32[1,1,1,2])
        else
            Int32[1], IndexSet(part, [3,4,5], Int32[1,2,2]), IndexSet(part, [3,4,5], Int32[1,2,2])
        end
    end

    # We create information exchangers to manage the synchronization of visible
    # shared portions of the sparse matrix and the actual row/col
    global_number_of_dofs = 5
    row_exchanger = Exchanger(row_partitioning,neighbors)
    rows = PRange(global_number_of_dofs,row_partitioning,row_exchanger)

    col_exchanger = Exchanger(col_partitioning,neighbors)
    cols = PRange(global_number_of_dofs,col_partitioning,col_exchanger)

    # Here we create the sparse matrix entries in COO format in their worker-local numbering
    I, J, V = map_parts(parts) do part
        if part == 1
            return [ 1, 1, 2, 2, 2, 3, 3, 3], [ 1, 2, 1, 2, 3, 2, 3, 4], Float64[1, 0, 0,-2, 1, 1,-1, 0]
        else
            return [ 1, 1, 2, 2, 2, 3, 3], [ 1, 2, 1, 2, 3, 2, 3], Float64[-1, 1, 1,-2, 1, 1,-1]
        end
    end
    A = PSparseMatrix(I, J, V, rows, cols, ids=:local)

    # Trigger sync between the processes
    assemble!(A)

    # Construct the right hand side
    b = PVector{Float64}(undef, A.rows)
    map_parts(parts,local_view(b, b.rows)) do part, b_local
        if part == 1
            b_local .= [1.0, -1.0, 0.0]
        else
            # Note that the first entry is shared with worker 1
            b_local .= [0.0, 0.0, 0.0]
        end
    end

    # Solve problem with cg
    u = IterativeSolvers.cg(A,b)
end
```

Now changing the backend to the MPI backend we can solve the problem in parallel.
This just requires to change the line
```julia
backend = SequentialBackend()
```
to
```julia
backend = MPIBackend()
```
and including MPI. Now launching the script with MPI makes the run parallel

```sh
$ mpirun -n 2 julia my-script.jl
```

### Note on Local Matrices

It should be noted that the local matrices are constructed as if they were
locally assembled on a process without knowledge of the remaining processes.

```
     Global Matrix
   P1  P1  P1  P2  P2
P1  1   0   0   0   0
P1  0  -2   1   0   0
P1  0   1  -2   1   0
P2  0   0   1  -2   1
P2  0   0   0   1  -1

           =

   Process 1 Portion
   P1  P1  P1  P2  P2
P1  1   0   0   0   0
P1  0  -2   1   0   0
P1  0   1  -1   0   0
P2  x   x   x   x   x
P2  x   x   x   x   x

          +

   Process 2 Portion
   P1  P1  P1  P2  P2
P1  x   x   x   x   x
P1  x   x   x   x   x
P1  0   0  -1   1   0
P2  0   0   1  -2   1
P2  0   0   0   1  -1
```

## Advanced example

A more complex example can be found in te package [PartitionedPoisson.jl](https://github.com/fverdugo/PartitionedPoisson.jl),
which describes the assembly of the finite difference discretization of a
Poisson problem in 3D.
