# # Jacobi method
#
# In this tutorial, you'll learn how to implement a parallel version of the one-dimensional Jacobi method using PartitionedArrays. Before you start, please make sure to have installed the following packages: 
# ```julia
# using Pkg
# Pkg.add("PartitionedArrays")
# Pkg.add("MPI")
# ```

# ## Learning Outcomes 
# In this notebook, you will learn: 
#
# - How to parallelize the one-dimensional Jacobi method
# - How create a block partition with ghost cells
# - How to run functions in parallel using `map`
# - How to update ghost cells using `consistent!`
# - The debugging vs. MPI execution mode
# - How to execute the parallel Julia code with MPI
#

# ## The Jacobi method for the Laplace equation
#
#
# The [Jacobi method](https://en.wikipedia.org/wiki/Jacobi_method) is a numerical tool to solve systems of linear algebraic equations. One of the main applications of the Jacobi method is to solve the equations resulting from boundary value problems (BVPs). I.e., given the values at the boundary (of a grid), we are interested in finding the interior values that fulfill a certain equation.
#
# A sketch of the discretization of the one-dimensional Laplace equation with boundary conditions is given in the figure below. A possible application of the 1D Laplace equation is e.g. to simulate the temperature of a thin bar where both ends of the bar are kept at a constant temperature. 

# ![discretization](assets/jacobi-discretization.png)

# When solving the Laplace equation in 1D, the Jacobi method leads to the following iterative scheme: The entry $i$ of vector $u$ at iteration $t+1$ is computed as:
#
# $u^{t+1}_i = \dfrac{u^t_{i-1}+u^t_{i+1}}{2}$

# ## Sequential version
# The following code implements the iterative scheme above for boundary conditions -1 and 1 on a grid with $n$ interior points and `niter` number of iterations. 

function jacobi(n,niters) 
    u = zeros(n+2) 
    u[1] = -1 
    u[end] = 1 
    u_new = copy(u) 
    for t in 1:niters 
        for i in 2:(n+1) 
            u_new[i] = 0.5*(u[i-1]+u[i+1]) 
        end 
        u, u_new = u_new, u 
    end 
    u 
end 

jacobi(10,100)

# !!! note
#     In this version of the Jacobi method, we return after a given number of iterations. Other stopping criteria are 
#     possible. For instance, iterate until the difference between u and u_new is below a tolerance. 


# ### Extracting parallelism
# Consider the two nested loops of the Jacobi function and to analyze where parallelism can be exploited: 
#
# ```julia
# for t in 1:nsteps
#     for i in 2:(n+1)
#         u_new[i] = 0.5*(u[i-1]+u[i+1])
#     end
#     u, u_new = u_new, u
# end
# ```
#
# - The outer loop cannot be parallelized. The value of `u` at step `t+1` depends on the value at the previous step `t`.
# - The inner loop can be parallelized.
#
# #### Partitioning scheme
# We chose block partitioning to distribute data `u` over several processes. The image below illustrates the partitioning with 3 processes. 

# ![partition](assets/jacobi-partition.png)

# #### Data dependencies
# Recall the Jacobi update:
#
# `u_new[i] = 0.5*(u[i-1]+u[i+1])`
#
# Thus, in order to update the local entries in `u_new`, we also need remote entries of vector `u` located in neighboring processes. Figure below shows the entries of `u` needed to update the local entries of `u_new` in a particular process (CPU 2).

# ![data-dependencies](assets/jacobi-data-dependencies.png)

# #### Ghost (aka halo) cells
#
# A usual way of implementing the Jacobi method and related algorithms is using so-called ghost cells. Ghost cells represent the missing data dependencies in the data owned by each process. After importing the appropriate values from the neighbor processes one can perform the usual sequential Jacobi update locally in the processes.

# ![ghost-cells](assets/jacobi-ghost-cell-update.png)

# Thus, the algorithm is usually implemented following two main phases at each iteration Jacobi:
#
# 1. Fill the ghost entries with communications
# 2. Do the Jacobi update sequentially at each process

# ## Parallel version
# Next, we will implement a parallelized version of Jacobi method using partitioned arrays. The parallel function will take the number of processes $p$ as an additional argument. 
# ```julia
# function jacobi_par(n,niters,p)
#   # TODO
# end
# ```

using PartitionedArrays

# Define the grid size `n` and the number of iterations `niters`. We also specify the number of processors as 3.

n = 10
niters = 100
p = 3;

# The following line creates an array of Julia type `LinearIndices`. This array holds linear indices of a specified range and shape ([documentation](https://docs.julialang.org/en/v1/base/arrays/#Base.LinearIndices)). 

ranks = LinearIndices((p,))

# ### Debug Mode
# While developing the parallel Jacobi method, we can make use of PartitionedArrays debug mode to test parallel code before running it in MPI. When running the code in parallel using MPI, the data type `MPIArray` is used to hold the partitioned data. This array type is not as flexible as standard Julia arrays and many operations are not allowed for `MPIArray` for performance reasons. For instance, it is not permitted to index arbitrary entries. 
#
# Essentially, in debug mode one uses the data structure `DebugArray`, which is a wrapper of a standard Julia array and can therefore be used in sequential debugging sessions. This allows for easier development of parallel code, since debugging on multiple running instances can be challenging. Additionally, `DebugArray` emulates the limitations of `MPIArray`, which enables the user to detect possible MPI-related errors while debugging the code in sequential. For more information on debug and MPI mode, see the [Usage](https://www.francescverdugo.com/PartitionedArrays.jl/dev/usage/) section of the documentation. 

ranks = DebugArray(LinearIndices((p,)))

# To demonstrate that `DebugArray` emulates the limitations of `MPIArray`, run the following code. It is expected to throw an error, since indexing is not permitted. 
try 
    ranks[1]
catch e
    println(e)
end


# ### Partition the data
#
# Next, we create a distributed partition of the data. Using PartitionedArrays.jl method `uniform_partition`, one can generate a block partition with roughly equal block sizes. It is also possible to create multi-dimensional partitions and to create ghost cells with `uniform_partition`. For more information on the function, view the [documentation](https://www.francescverdugo.com/PartitionedArrays.jl/dev/reference/partition/#PartitionedArrays.uniform_partition).

# The following line divides the `n=10` grid points into `p=3` approximately equally sized blocks and assigns the corresponding row indices to the ranks. 

row_partition = uniform_partition(ranks,p,n)

# As discussed above, the Jacobi method requires the neighboring values $u_{i-1}$ and $u_{i+1}$ to update $u_i$. Therefore, some neighboring values that are stored on remote processes need to be communicated in each iteration. To store these neighbor values, we add ghost cells to the partition using `uniform_partition` with optional argument `ghost=true`. 

ghost = true
row_partition = uniform_partition(ranks,p,n,ghost)

# Note that rows 3, 4, 6, and 7 are now stored on more than one process. The `DebugArray` also keeps the information about which process is the owner of each row. It is possible to retrieve this information with function `local_to_owner`. The output is a `DebugArray` of the rank ids of the owner of each element. 

map(local_to_owner,row_partition)

# Likewise, it is possible to view which are the ghost cells in each partition: 

map(local_to_ghost, row_partition)

# And, which process is the owner of the local ghost cells: 

map(ghost_to_owner, row_partition)

# The following line initializes the data structure that will hold the solution $u$ and fill it with all zero values. 

u = pzeros(Float64,row_partition)

# Note that, like `DebugArray`, a `PVector` represents an array whose elements are distributed (i.e. partitioned) across processes, and indexing is disabled here as well. Therefore, the following examples are expected to raise an error. 

try 
    u[1]  
    u[end] 
catch e
    println(e)
end


# To view the local values of a partitioned vector, use method `partition` or `local_values`.  

partition(u)

# Partition returns a `DebugArray`, so again indexing, such as in the following examples, is not permitted. 

try 
    partition(u)[1][1]
    partition(u)[end][end]
catch e
    println(e)
end 


# ### Initialize boundary conditions
# The values of the partition are still all 0, so next we need to set the correct boundary conditions: $u(0) = -1$ and $u(L)= 1$.
#
#
# Since `PVector` is distributed, one process cannot access the values that are owned by other processes, so we need to find a different approach. Each process can set the boundary conditions locally. This is possible with the following piece of code. Using Julia function `map`, the function `set_bcs` is executed locally by each process on its locally available part of `partition(u)`. These local partitions are standard Julia `Vector`s and are allowed to be indexed. 

function set_bcs(my_u,rank)
    if rank == 1
       my_u[1] = 1
    end
    if rank == 3
       my_u[end] = -1
    end
end
map(set_bcs,partition(u),ranks)
partition(u)

# Using `map` we can apply the boundary conditions to each vector within the `DebugArray` individually. The result is that the local border cells (= ghost cells), which are not global borders, will be initialized with values `-1` and `1` as well. But this is not a problem since the ghost cells are overwritten with the values of the neighboring process in each iteration. 

map(partition(u)) do my_u
    my_u[1] = 1
    my_u[end] = -1
end
partition(u)

# Remember that to perform the Jacobi update, alternate writing to one data structure `u` and another `u_new` was required. Hence, we need to create a second data structure to hold a copy of our partition. Using Julia function `copy`, the new object has the same type and values as the original data structure `u`.

u_new = copy(u)
partition(u_new)

# ### Communication of ghost cell values
# The PartitionedArrays package provides method `consistent!` to update all ghost values of partition `u` with the values from the corresponding remote owners. Thus, the local values are made globally _consistent_. The function returns a `Task`, such that latency hiding is enabled (i.e. other computations can be performed between calling `consistent!` and `wait`). In the first iteration, calling `consistent!` effectively overwrites the initialization of the ghost values with values `-1` and `1`. 

t = consistent!(u)
wait(t)

partition(u)

# ### Updating grid values with Jacobi iteration

# After having updated the ghost cells, each process can perform the Jacobi update on its local data. To perform the update on each part of the data in parallel, we again use `map`. You can verify that the grid points are updated correctly by running one iteration of the Jacobi method on the partitioned vectors `u` and `u_new` with the following code: 

map(partition(u),partition(u_new)) do my_u, my_u_new
    my_n = length(my_u)
    for i in 2:(my_n-1)
        my_u_new[i] = 0.5*(my_u[i-1]+my_u[i+1]) 
    end 
end
partition(u_new)

# ### Final parallel implementation
# To conclude, we can combine the steps to a final parallel implementation: 

function jacobi_par(n,niters,p)
    ranks = DebugArray(LinearIndices((p,)))
    ghost = true
    row_partition = uniform_partition(ranks,p,n,ghost)
    u = pzeros(Float64,row_partition)
    map(partition(u)) do my_u
        my_u[1] = 1
        my_u[end] = -1
    end
    u_new = copy(u)
    for iter in 1:niters
        t = consistent!(u)
        wait(t)
        map(partition(u),partition(u_new)) do my_u, my_u_new
            my_n = length(my_u)
            for i in 2:(my_n-1)
                my_u_new[i] = 0.5*(my_u[i-1]+my_u[i+1]) 
            end 
        end
        u, u_new = u_new, u
    end
    u
end

u = jacobi_par(10,100,3)

partition(u)

# ## Parallel execution
# After having debugged the code in sequential, we just need to change a couple of code passages to execute the Jacobi method in parallel using MPI. First of all, include the Julia MPI API `MPI.jl`. 

using MPI

# In general, any Julia program can be executed using `MPI.jl` like so: 

run(`$(mpiexec()) -np 3 julia -e 'println("hi!")'`);

# The command `mpiexec` launches MPI and `-np` specifies the number of processes. Instead of passing the code, you can also copy the code in a file called `filename.jl` and launch the code with 
# ```
# run(`$(mpiexec()) -np 3 julia --project=. filename.jl`)
# ```
# ### The MPI mode
# Now we can call the main function, which calls the parallel Jacobi method, using `with_mpi(main)`. This expression calls function `main` "in MPI mode". Essentially, `with_mpi(main)` calls function `main` with function argument `distribute_with_mpi`. The function `distribute_with_mpi` in turn creates an `MPIArray` from a given collection and distributes its items over the ranks of the given MPI communicator `comm`. (If `comm` is not specified, the standard communicator `MPI.COMM_WORLD` is used.) 
# The difference to the debug mode is that now a real distributed `MPIArray` is used where before `DebugArray` was employed. To switch back to debug mode, simply replace `with_mpi` with `with_debug`. 
#
# Finally the whole syntax is copied in a Julia `quote` block and run with `mpiexec`. 

# ```julia
# code = quote
#    using PartitionedArrays
#    
#    function main(distribute)
#        function jacobi_par(n,niters,p)
#            ranks = distribute(LinearIndices((p,)))
#            ghost = true
#            row_partition = uniform_partition(ranks,p,n,ghost)
#            u = pzeros(Float64,row_partition)
#            map(partition(u)) do my_u
#                my_u[1] = 1
#                my_u[end] = -1
#            end
#            u_new = copy(u)
#            for iter in 1:niters
#                t = consistent!(u)
#                wait(t)
#                map(partition(u),partition(u_new)) do my_u, my_u_new
#                    my_n = length(my_u)
#                    for i in 2:(my_n-1)
#                        my_u_new[i] = 0.5*(my_u[i-1]+my_u[i+1]) 
#                    end 
#               end
#               u, u_new = u_new, u
#            end
#            u
#        end
#    u = jacobi_par(10,100,3)
#    display(partition(u))
#    end # main
#       
#    with_mpi(main) 
#    
#    end # quote
#
# run(`$(mpiexec()) -np 3  julia --project=. -e $code`);
# ```


