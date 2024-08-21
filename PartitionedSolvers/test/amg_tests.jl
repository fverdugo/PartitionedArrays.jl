module AMGTests

using PartitionedArrays
using PartitionedSolvers
using LinearAlgebra
using IterativeSolvers
using IterativeSolvers: cg!
using Test
using SparseArrays


# Test strength graph computation 
# First with Psparse matrix and known solution
ndofs = 18
nnodes = 9
nrows = 3
p = 4
ranks = DebugArray(LinearIndices((p,)))
row_partition = uniform_partition(ranks, ndofs)

IJV = map(row_partition) do row_indices
    I, J, V = Int[], Int[], Float64[]
    for i in local_to_global(row_indices)
        push!(V, 9)
        push!(I, i)
        push!(J, i)

        node = 0 
        if (i % 2) == 1
            # 1st dimension
            push!(V, -1)
            push!(I, i)
            push!(J, i+1)
            node = div((i+1),2)
        else
            # 2nd dimension
            push!(V, -1)
            push!(I, i)
            push!(J, i-1) 
            node = div(i,2)
        end
        north = node - nrows
        south = node + nrows
        east = node - 1
        west = node + 1
        if north >= 1
            push!(V, -1)
            push!(I, i)
            push!(J, 2*north - 1)
            push!(V, -1)
            push!(I, i)
            push!(J, 2*north)
        end
        if south <= nnodes
            push!(V, -1)
            push!(I, i)
            push!(J, 2*south -1)
            push!(V, -1)
            push!(I, i)
            push!(J, 2*south)
        end
        if node % nrows != 1
            push!(V, -1)
            push!(I, i)
            push!(J, 2*east-1)
            push!(V, -1)
            push!(I, i)
            push!(J, 2*east)
        end
        if node % nrows != 0
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
A = psparse(I,J,V,row_partition, row_partition) |> fetch 

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
solution = sparse(I, J, V, nnodes, nnodes)

epsilon = 0.02

# Test with CSC sparse matrix 
# Test sample matrix with known solution
A_seq = centralize(A)
G_seq = PartitionedSolvers.strength_graph(A_seq, 2, epsilon=epsilon)
@test solution ≈ G_seq

# Another test with blocksize 3
M = rand([-2.0, -1, 1, 2], (3, 3))
M = sparse(M)
A = blockdiag(M, M, M)
solution = sparse([1, 2, 3], [1, 2, 3], fill(1.0, 3), 3, 3)
G = PartitionedSolvers.strength_graph(A, 3, epsilon=epsilon)
@test solution ≈ G 

# Test with minimal matrix size
G = PartitionedSolvers.strength_graph(M, 3, epsilon=epsilon)
solution = sparse([1], [1], 1.0, 1, 1)
@test solution ≈ G 

# Create Test matrix for Linear Elasticity
parts_per_dir = (2,2,2)
block_size = length(parts_per_dir) 
p = prod(parts_per_dir)
ranks = DebugArray(LinearIndices((p,)))
nodes_per_dir = map(i->block_size*i,parts_per_dir)
args = PartitionedArrays.linear_elasticity_fem(nodes_per_dir,parts_per_dir,ranks)
A_dist = psparse(args...) |> fetch
A_seq = centralize(A_dist)

# Test strength graph with sequential and parallel linear elasticity matrix
G_seq = PartitionedSolvers.strength_graph(A_seq, block_size, epsilon=epsilon)
G_dist = PartitionedSolvers.strength_graph(A_dist, block_size, epsilon=epsilon)
G_dist_centralized = centralize(G_dist)
diff = G_seq - G_dist_centralized
diff_nnz = nnz(diff)
println("Test strength graph: \n$(round(diff_nnz/(G_seq.m^2); digits=3))% of total entries are different between sequential and parallel G.") 
println("G_dist has $(round(1-nnz(G_dist_centralized)/nnz(G_seq);digits=2))% fewer nnz entries than G_seq.")
println("(G_dist nnz entries: $(nnz(G_dist_centralized)), G_seq nnz entries: $(nnz(G_seq)))")
@test isa(G_dist, PSparseMatrix)

# Test sequential collect nodes in aggregate 
diagG = dense_diag(centralize(G_dist))
node_to_aggregate_seq, node_aggregates_seq = PartitionedSolvers.aggregate(centralize(G_dist),diagG;epsilon)
aggregate_to_nodes_seq = PartitionedSolvers.collect_nodes_in_aggregate(node_to_aggregate_seq, node_aggregates_seq)
@test length(aggregate_to_nodes_seq.data) == length(node_to_aggregate_seq)
for i_agg in node_aggregates_seq
    pini = aggregate_to_nodes_seq.ptrs[i_agg]
    pend = aggregate_to_nodes_seq.ptrs[i_agg+1]-1
    nodes = aggregate_to_nodes_seq.data[pini:pend]
    @test all(node_to_aggregate_seq[nodes] .== i_agg)
end

# Test parallel collect_nodes_in_aggregate
diagG = dense_diag(G_dist)
node_to_aggregate_dist, node_aggregates_dist = PartitionedSolvers.aggregate(G_dist,diagG;epsilon)
aggregate_to_nodes_dist = PartitionedSolvers.collect_nodes_in_aggregate(node_to_aggregate_dist, node_aggregates_dist)
map(partition(aggregate_to_nodes_dist), partition(node_to_aggregate_dist), partition(node_aggregates_dist)) do my_aggregate_to_nodes, my_node_to_aggregate, my_aggregates
    @test length(my_aggregate_to_nodes.data) == length(my_node_to_aggregate)
    global_to_local_aggregate = global_to_local(my_aggregates) 
    local_aggregates = global_to_local_aggregate[my_aggregates]
    own_node_to_local_aggregate = map(my_node_to_aggregate) do global_aggregate
        global_to_local_aggregate[global_aggregate]
    end
    for i_agg in local_aggregates
        pini = my_aggregate_to_nodes.ptrs[i_agg]
        pend = my_aggregate_to_nodes.ptrs[i_agg+1]-1
        nodes = my_aggregate_to_nodes.data[pini:pend]
        @test all(own_node_to_local_aggregate[nodes] .== i_agg)
    end
end

# Create nullspace vectors for tentative prolongator
x_dist = node_coordinates_unit_cube(nodes_per_dir,parts_per_dir,ranks)
B_dist = nullspace_linear_elasticity(x_dist, partition(axes(A_dist,2)))
B_seq = [collect(b) for b in B_dist]

# Test tentative prolongator with sequential linear elasticity matrix
Pc, Bc = PartitionedSolvers.tentative_prolongator_with_block_size(aggregate_to_nodes_seq,B_seq, block_size) 
@test Pc * stack(Bc) ≈ stack(B_seq)

# Test tentative prolongator with parallel matrix 
Pc, Bc = PartitionedSolvers.tentative_prolongator_with_block_size(aggregate_to_nodes_dist,B_dist, block_size) 
n_B = length(B_dist)
for i in 1:n_B
    @test isa(Bc[i], PVector)
end
Bc_matrix = zeros(size(Pc,2), length(Bc))
for (i,b) in enumerate(Bc)
    Bc_matrix[:,i] = collect(b)
end
B_matrix = zeros(size(Pc,1), length(Bc))
for (i,b) in enumerate(B_dist)
    B_matrix[:,i] = collect(b)
end
@test centralize(Pc) * Bc_matrix ≈ B_matrix  


# Test spectral radius sequential & parallel 
diagA = dense_diag(A_dist)
invD = 1 ./ diagA
Dinv = PartitionedArrays.sparse_diag_matrix(invD,(axes(A_dist,1),axes(A_dist,1)))
M = Dinv * A_dist
exp = eigmax(Array(centralize(M)))
cols = axes(M, 2) 
x0 = prand(partition(cols))
x0_seq = collect(x0)
l_dist, x = PartitionedSolvers.spectral_radius(M, x0, 10)
l_seq, x = PartitionedSolvers.spectral_radius(centralize(M), x0_seq, 10)
@test l_dist ≈ l_seq 
@test abs((l_dist-exp)/exp) < 2*10^-1


# Test amg 
# first with a sequential matrix
nodes_per_dir = (100,100)
A = PartitionedArrays.laplace_matrix(nodes_per_dir)
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
    pre_smoother = PartitionedSolvers.jacobi(;iters=10,omega=2/3),
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
   pre_smoother = PartitionedSolvers.gauss_seidel(;iters=1),
   )

fine_params = amg_fine_params(;level_params)

Pl = setup(amg(;fine_params),y,A,b;nullspace=B)
y .= 0
cg!(y,A,b;Pl,verbose=true)

solver = linear_solver(IterativeSolvers.cg;Pl=amg(;fine_params),verbose=true)
S = setup(solver,y,A,b)
solve!(y,S,b)
update!(S,2*A)
solve!(y,S,b)



# Now for linear elasticity (sequential)

parts_per_dir = (2,2,2)
block_size = length(parts_per_dir) 
p = prod(parts_per_dir)
ranks = DebugArray(LinearIndices((p,)))
nodes_per_dir = map(i->block_size*i,parts_per_dir)
args = PartitionedArrays.linear_elasticity_fem(nodes_per_dir,parts_per_dir,ranks)
A_dist = psparse(args...) |> fetch 
A =  centralize(A_dist)

x = node_coordinates_unit_cube(nodes_per_dir,parts_per_dir, ranks)
B_dist = nullspace_linear_elasticity(x, partition(axes(A_dist,2)))
B = [collect(b) for b in B_dist]
x_exact = rand(size(A,2))
b = A*x_exact   
y = similar(x_exact)
y .= 0

level_params = amg_level_params_linear_elasticity(block_size)
fine_params = amg_fine_params(;level_params)
solver = amg(;fine_params)

Pl = setup(solver,y,A,b;nullspace=B)
cg!(y,A,b;Pl,verbose=true)
println("Linear Elasticity norm of error: $(norm(y-x_exact))")
@test y ≈ x_exact



# Now in parallel

parts_per_dir = (2,2)
np = prod(parts_per_dir)
parts = DebugArray(LinearIndices((np,)))

nodes_per_dir = (100,100)
A = PartitionedArrays.laplace_matrix(nodes_per_dir,parts_per_dir,parts)
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
    pre_smoother = PartitionedSolvers.jacobi(;iters=1,omega=2/3),
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
np = prod(parts_per_dir)
parts = LinearIndices((np,))
A = PartitionedArrays.laplace_matrix(nodes_per_dir,parts_per_dir,parts)
x_exact = pones(partition(axes(A,2)))
b = A*x_exact
x = similar(b,axes(A,2))
x .= 0
Pl = setup(amg(),x,A,b)
_, history = cg!(x,A,b;Pl,log=true)
display(history)


# Now for linear elasticity 

parts_per_dir = (2,2,2)
block_size = length(parts_per_dir) 
p = prod(parts_per_dir)
ranks = DebugArray(LinearIndices((p,)))
nodes_per_dir = map(i->block_size*i,parts_per_dir)
args = linear_elasticity_fem(nodes_per_dir,parts_per_dir,ranks)
A = psparse(args...) |> fetch 
x_exact = pones(partition(axes(A,2)))
x = node_coordinates_unit_cube(nodes_per_dir,parts_per_dir, ranks)
B = nullspace_linear_elasticity(x, partition(axes(A,2)))
b = A*x_exact   
y = similar(b,axes(A,2))
y .= 0

level_params = amg_level_params_linear_elasticity(block_size)
fine_params = amg_fine_params(;level_params)
solver = amg(;fine_params)

Pl = setup(solver,y,A,b;nullspace=B) 
cg!(y,A,b;Pl,verbose=true)
@test y ≈ x_exact

end