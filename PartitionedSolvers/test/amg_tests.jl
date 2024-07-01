module AMGTests

using PartitionedArrays
using PartitionedArrays: laplace_matrix
using PartitionedSolvers
using LinearAlgebra
using Test
using IterativeSolvers: cg!
using SparseArrays

# Test strength graph computation 
# First with Psparse matrix 
nrows = 18
ngrid = 9
nrowgrid = 3
ncols = nrows 
p = 4
ranks = DebugArray(LinearIndices((p,)))
row_partition = uniform_partition(ranks, nrows)

IJV = map(row_partition) do row_indices
    I, J, V = Int[], Int[], Float64[]
    for i in local_to_global(row_indices)
        push!(V, 9)
        push!(I, i)
        push!(J, i)

        grid_point = 0 
        if (i % 2) == 1
            # 1st dimension
            push!(V, -1)
            push!(I, i)
            push!(J, i+1)
            grid_point = div((i+1),2)
        else
            # 2nd dimension
            push!(V, -1)
            push!(I, i)
            push!(J, i-1) 
            grid_point = div(i,2)
        end
        north = grid_point - nrowgrid
        south = grid_point + nrowgrid
        east = grid_point - 1
        west = grid_point + 1
        if north >= 1
            push!(V, -1)
            push!(I, i)
            push!(J, 2*north - 1)
            push!(V, -1)
            push!(I, i)
            push!(J, 2*north)
        end
        if south <= ngrid
            push!(V, -1)
            push!(I, i)
            push!(J, 2*south -1)
            push!(V, -1)
            push!(I, i)
            push!(J, 2*south)
        end
        if grid_point % nrowgrid != 1
            push!(V, -1)
            push!(I, i)
            push!(J, 2*east-1)
            push!(V, -1)
            push!(I, i)
            push!(J, 2*east)
        end
        if grid_point % nrowgrid != 0
            push!(V, -1)
            push!(I, i)
            push!(J, 2*west-1)
            push!(V, -1)
            push!(I, i)
            push!(J, 2*west)
        end
    end
    I,J,V
end

I,J,V = tuple_of_arrays(IJV)
col_partition = row_partition
A = psparse(I,J,V,row_partition, col_partition) |> fetch 
# TODO: implement and test psparse matrix 
#R = PartitionedSolvers.strength_graph(A, 2)

# Now with CSC sparse matrix 
# Test sample matrix with block size 2
theta = 0.02
A = centralize(A)

# build solution
I = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 
    6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9]
J = [1, 2, 4,
     1, 2, 3, 5, 
     2, 3, 6, 
     1, 4, 5, 7, 
     2, 4, 5, 6, 8,
     3, 5, 6, 9,
     4, 7, 8, 
     5, 7, 8, 9, 
     6, 8, 9]
V = ones(length(I))

solution = sparse(I, J, V, ngrid, ngrid)
R = PartitionedSolvers.strength_graph(A, 2, theta=theta)
@test solution ≈ R 

# Another test with 3 dims
M = rand([-2.0, -1, 1, 2], (3, 3))
M = sparse(M)
A = blockdiag(M, M, M)
solution = sparse([1, 2, 3], [1, 2, 3], fill(1.0, 3), 3, 3)
R = PartitionedSolvers.strength_graph(A, 3, theta=theta)
@test solution ≈ R 

# Test with minimal matrix size
R = PartitionedSolvers.strength_graph(M, 3, theta=theta)
solution = sparse([1], [1], 1.0, 1, 1)
@test solution ≈ R 


# First with a sequential matrix
nodes_per_dir = (100,100)
A = laplace_matrix(nodes_per_dir)
using Random
Random.seed!(1)
x = rand(size(A,2))
b = A*x
y = similar(x)
y .= 0

solver = amg()
S = setup(solver,y,A,b)
solve!(y,S,b)
update!(S,2*A)
solve!(y,S,b)
finalize!(S)

amg_statistics(S) |> display

# Non-default options

level_params = amg_level_params(;
    pre_smoother = jacobi(;iters=10,omega=2/3),
    cycle = v_cycle,
   )

fine_params = amg_fine_params(;
    level_params,
    n_fine_levels=5)

coarse_params = (;
    coarse_solver = lu_solver(),
    coarse_size = 15,
   )

solver = amg(;fine_params,coarse_params)

# Now with a nullspace

B = default_nullspace(A)
S = setup(solver,y,A,b;nullspace=B)
solve!(y,S,b)
update!(S,2*A;nullspace=B)
solve!(y,S,b)
finalize!(S)

# Now as a preconditioner

level_params = amg_level_params(;
   pre_smoother = gauss_seidel(;iters=1),
   )

fine_params = amg_fine_params(;level_params)

Pl = setup(amg(;fine_params),y,A,b;nullspace=B)
y .= 0
cg!(y,A,b;Pl,verbose=true)

# Now in parallel

parts_per_dir = (2,2)
np = prod(parts_per_dir)
parts = DebugArray(LinearIndices((np,)))

nodes_per_dir = (100,100)
A = laplace_matrix(nodes_per_dir,parts_per_dir,parts)
x = pones(partition(axes(A,2)))
b = A*x

y = similar(x)
y .= 0

solver = amg()
S = setup(solver,y,A,b)
amg_statistics(S) |> display
solve!(y,S,b)
update!(S,2*A)
solve!(y,S,b)
finalize!(S)

# Now with a nullspace

B = default_nullspace(A)
solver = amg()
S = setup(solver,y,A,b;nullspace=B)
solve!(y,S,b)
update!(S,2*A)
solve!(y,S,b)
finalize!(S)

level_params = amg_level_params(;
    pre_smoother = jacobi(;iters=1,omega=2/3),
    coarsening = smoothed_aggregation(;repartition_threshold=10000000)
   )

fine_params = amg_fine_params(;
    level_params,
    n_fine_levels=5)

solver = amg(;fine_params)

Pl = setup(solver,y,A,b;nullspace=B)
y .= 0
cg!(y,A,b;Pl,verbose=true)

nodes_per_dir = (40,40,40)
parts_per_dir = (2,2,1)
nparts = prod(parts_per_dir)
parts = LinearIndices((nparts,))
A = laplace_matrix(nodes_per_dir,parts_per_dir,parts)
x_exact = pones(partition(axes(A,2)))
b = A*x_exact
x = similar(b,axes(A,2))
x .= 0
Pl = setup(amg(),x,A,b)
_, history = cg!(x,A,b;Pl,log=true)
display(history)
end