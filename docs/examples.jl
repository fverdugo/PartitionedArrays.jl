# # Examples
#
# !!! note
#     The following examples are run with the native Julia arrays for demo purposes.
#     Substituting `LinearIndices((np,))` by `distribute_with_mpi(LinearIndices((np,)))`
#     will convert them
#     to distributed drivers. To learn how to run the examples with MPI, see the [Usage](@ref) section.
#
# ## Hello, world!

using PartitionedArrays
np = 4
ranks = LinearIndices((np,))
map(ranks) do rank
    println("Hello, world! I am proc $rank of $np.")
end;

# ## Collective communication
#
# The first rank generates an array of random integers in `1:30` and scatters it over all ranks. Each rank
# counts the number of even items in its part. Finally, the partial sums are reduced in the first rank.

# The first rank generates the data to send.
using PartitionedArrays
np = 4
load = 3
n = load*np
ranks = LinearIndices((np,))
a_snd = map(ranks) do rank
    if rank == 1
        a = rand(1:30,n)
        [ a[(1:load).+(i-1)*load] for i in 1:np ]
    else
        [ Int[] ]
    end
end

# Note that only the first entry contains meaningful data in previous output.
a_rcv = scatter(a_snd,source=1)

# After the scatter, all the parts have received their chunk. Now, we can count in parallel.
b_snd = map(ai->count(isodd,ai),a_rcv)

# Finally we reduce the partial sums.
b_rcv = reduction(+,b_snd,init=0,destination=1)

# Only the destination rank will receive the correct result.

# ## Point-to-point communication
#
# Each rank generates some message (in this case an integer 10 times the current rank id).
# Each rank sends this data to the next rank. The last one sends it to the first, closing the circle.
# After repeating this exchange a number of times equal to the number of ranks, we check
# that we ended up with the original message.

# First, each rank generates the ids of the neighbor to send data to.
using PartitionedArrays
np = 3
ranks = LinearIndices((np,))
neigs_snd = map(ranks) do rank
    if rank != np
        [rank + 1]
    else
        [1]
    end
end

# Now, generate the data we want to send
data_snd = map(ranks) do rank
    [10*rank]
end


# Prepare, the point-to-point communication graph
graph = ExchangeGraph(neigs_snd)

# Do the first exchange, and wait for the result to arrive
data_rcv = exchange(data_snd,graph) |> fetch

# Do the second exchange and wait for the result to arrive
map(copy!,data_snd,data_rcv)
exchange!(data_rcv,data_snd,graph) |> fetch

# Do the last exchange
map(copy!,data_snd,data_rcv)
exchange!(data_rcv,data_snd,graph) |> fetch

# Check that we got the initial message
map(ranks,data_rcv) do rank,data_rcv
    @assert data_rcv == [10*rank]
end;

# ## Distributed sparse linear solve
#
# Solve the following linear system by distributing it over several parts.
#
# ```math
# \begin{pmatrix}
# 1 &  0 &  0 &  0 &  0 \\
# -1 & 2 & -1 &  0 &  0 \\
# 0 & -1 & 2 & -1 &  0 \\
# 0 &  0 & -1 & 2 & -1 \\
# 0 &  0 &  0 &  0 &  1
# \end{pmatrix}
# \begin{pmatrix}
# u₁ \\
# u₂ \\
# u₃ \\
# u₄ \\
# u₅
# \end{pmatrix}
# =
# \begin{pmatrix}
#  1 \\
#  0 \\
#  0 \\
#  0 \\
# -1
# \end{pmatrix}
# ```

# First generate the row partition
using PartitionedArrays
using IterativeSolvers
using LinearAlgebra
np = 3
n = 5
ranks = LinearIndices((np,))
row_partition = uniform_partition(ranks,n)

# Compute the rhs vector
IV = map(row_partition) do row_indices
    I,V = Int[], Float64[]
    for global_row in local_to_global(row_indices)
        if global_row == 1
            v = 1.0
        elseif global_row == n
            v = -1.0
        else
            continue
        end
        push!(I,global_row)
        push!(V,v)
    end
    I,V
end
I,V = tuple_of_arrays(IV)
b = old_pvector!(I,V,row_partition) |> fetch


# Compute the system matrix
IJV = map(row_partition) do row_indices
    I,J,V = Int[], Int[], Float64[]
    for global_row in local_to_global(row_indices)
        if global_row in (1,n)
            push!(I,global_row)
            push!(J,global_row)
            push!(V,1.0)
        else
            push!(I,global_row)
            push!(J,global_row-1)
            push!(V,-1.0)
            push!(I,global_row)
            push!(J,global_row)
            push!(V,2.0)
            push!(I,global_row)
            push!(J,global_row+1)
            push!(V,-1.0)
        end
    end
    I,J,V
end
I,J,V = tuple_of_arrays(IJV)
col_partition = row_partition
A = old_psparse!(I,J,V,row_partition,col_partition) |> fetch

# Generate an initial guess that fulfills
# the boundary conditions.
# Solve and check the result
x = similar(b,axes(A,2))
x .= b
IterativeSolvers.cg!(x,A,b)
r = A*x - b
norm(r)







