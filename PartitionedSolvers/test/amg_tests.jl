module AMGTests

# just for debugging
using Pkg
Pkg.activate("PartitionedSolvers")
using PartitionedArrays
using PartitionedArrays: laplace_matrix
using PartitionedSolvers
using LinearAlgebra
using Test
using IterativeSolvers: cg!
using SparseArrays

# Test strength graph computation 
# First with Psparse matrix 
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
col_partition = row_partition
A = psparse(I,J,V,row_partition, col_partition) |> fetch 
# TODO: implement and test psparse matrix 


# Now with CSC sparse matrix 
# Test sample matrix with block size 2
epsilon = 0.02
A_test = centralize(A)

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
G_test = PartitionedSolvers.strength_graph(A_test, 2, epsilon=epsilon)
@test solution ≈ G_test

# Another test with 3 dims
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

# Test tentative prolongator 
function random_nullspace(ndofs, n_B)
    B = Array{Array{Float64, 1}, 1}(undef, n_B)
    for i = 1:n_B
        B[i] = rand(ndofs)
    end
    B
end

# Test tentative prolongator with laplace matrix
nodes_per_dir = (90,90)
block_size = 3
A = laplace_matrix(nodes_per_dir)
G = PartitionedSolvers.strength_graph(A, block_size, epsilon=epsilon)
diagG = dense_diag(G)
B = random_nullspace(size(A, 1), block_size)
node_to_aggregate, node_aggregates = PartitionedSolvers.aggregate(G,diagG;epsilon)
aggregate_to_nodes_old = PartitionedSolvers.collect_nodes_in_aggregate(node_to_aggregate, node_aggregates)
aggregate_to_nodes = PartitionedSolvers.remove_singleton_aggregates(aggregate_to_nodes_old)
# Assert there are no singleton aggregates
@assert length(aggregate_to_nodes_old) == length(aggregate_to_nodes)
Pc, Bc = PartitionedSolvers.tentative_prolongator_with_block_size(aggregate_to_nodes,B, block_size) 
@test Pc * stack(Bc) ≈ stack(B)


# Test spectral radius estimation 
n_its = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
n_trials = 10
errors_powm = zeros(n_trials, length(n_its))
time_powm = zeros(n_trials,length(n_its))
errors_spectrad = zeros(n_trials,length(n_its))
time_spectrad = zeros(n_trials,length(n_its))
using IterativeSolvers
using Plots
using LinearMaps
using Statistics
A = laplace_matrix((100,100))
diagA = dense_diag(A)
invD = 1 ./ diagA
x0 = rand(size(A,2))
M = invD .* A
Fmap = LinearMap(M)
exp = 2.0
for (i, n_it) in enumerate(n_its)
    for t in 1:n_trials
        # Power method
        tic = time_ns()
        λ, x = powm!(Fmap, x0, maxiter = n_it)
        toc = time_ns()
        time_powm[t,i] = toc - tic
        errors_powm[t,i] = abs((λ-exp)/exp)
        
        # own implementation
        tic = time_ns()
        ρ = PartitionedSolvers.spectral_radius(M,x0, n_it)
        toc = time_ns()
        time_spectrad[t,i] = toc-tic
        errors_spectrad[t,i] = abs((ρ-exp)/exp)
    end
end
avg_time_powm = median(time_powm, dims=1)
avg_time_spectrad = median(time_spectrad, dims=1)
avg_error_powm = median(errors_powm, dims=1)
avg_error_spectrad = median(errors_spectrad, dims=1)
avg_time_powm = avg_time_powm ./ 10^9
avg_time_spectrad = avg_time_spectrad ./ 10^9
p1 = plot(n_its, [avg_error_powm' avg_error_spectrad'], label = ["powm" "spectrad"], 
    marker=[:c :x], ms=2)
plot!(p1, xlabel="#iterations", ylabel="avg rel error", xticks = n_its)
p2 = plot(n_its, [avg_time_powm' avg_time_spectrad'], label = ["powm" "spectrad"], 
    marker=[:c :x], ms=2)
plot!(p2, xlabel="#iterations", ylabel="avg runtime (s)", xticks = n_its)
p = plot(p1, p2, layout=(2,1), suptitle="Convergence of powm and spectral_radius")
savefig(p, "C:/Users/gelie/Home/ComputationalScience/GSoC/powm_convergence.png")    
display(p)

# First with a sequential matrix
#= nodes_per_dir = (100,100)
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

solver = linear_solver(IterativeSolvers.cg;Pl=amg(;fine_params),verbose=true)
S = setup(solver,y,A,b)
solve!(y,S,b)
update!(S,2*A)
solve!(y,S,b)

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
display(history) =#
end