module AMGTests

# just for debugging
using Pkg
Pkg.activate("PartitionedSolvers")
using PartitionedArrays
using PartitionedArrays: laplace_matrix
using PartitionedSolvers
using LinearAlgebra
using IterativeSolvers
using IterativeSolvers: cg!
using Plots
using Statistics
using Test
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
block_size = 3
parts_per_dir = (2,2,2)
p = prod(parts_per_dir)
ranks = DebugArray(LinearIndices((p,)))
nodes_per_dir = map(i->block_size*i,parts_per_dir)
args = laplacian_fdm(nodes_per_dir,parts_per_dir,ranks)
A_dist = psparse(args...) |> fetch 
A = centralize(A_dist)
println("dims A: $(size(A))")
G = PartitionedSolvers.strength_graph(A, block_size, epsilon=epsilon)
diagG = dense_diag(G)
B = random_nullspace(size(A, 1), block_size)
node_to_aggregate, node_aggregates = PartitionedSolvers.aggregate(G,diagG;epsilon)
aggregate_to_nodes = PartitionedSolvers.collect_nodes_in_aggregate(node_to_aggregate, node_aggregates)
Pc, Bc = PartitionedSolvers.tentative_prolongator_with_block_size(aggregate_to_nodes,B, block_size) 
@test Pc * stack(Bc) ≈ stack(B)

# Test spectral radius sequential & parallel 
diagA = dense_diag(A_dist)
invD = 1 ./ diagA
Dinv = PartitionedArrays.sparse_diag_matrix(invD,(axes(A_dist,1),axes(A_dist,1)))
M = Dinv * A_dist
exp = eigmax(Array(centralize(M)))
cols = axes(M, 2) 
x0 = prand(partition(cols))
x0_seq = collect(x0)
l, x = PartitionedSolvers.spectral_radius(M, x0, 10)
lseq, x = PartitionedSolvers.spectral_radius(centralize(M), x0_seq, 10)
@test l ≈ lseq 
@test abs((l-exp)/exp) < 2*10^-1

# Test spectral radius estimation 
#= maxiter = 5000
nnodes = [8, 16, 32]
exp = [1.9396926207859075, 1.9829730996838997, 1.9954719225730886]
msizes = zeros(Int, length(nnodes))
parts_per_dir = (2,2,2)
p = prod(parts_per_dir)
ranks = DebugArray(LinearIndices((p,)))
n_trials = 3
A_dims = zeros(length(nnodes))
errors_powm = zeros(length(nnodes), n_trials, maxiter)
time_powm = zeros(length(nnodes), n_trials)
errors_spectrad = zeros(length(nnodes), n_trials, maxiter)
time_spectrad = zeros(length(nnodes), n_trials)

for (n_i, n) in enumerate(nnodes)
    nodes_per_dir = (n, n, n)
    args = laplacian_fdm(nodes_per_dir,parts_per_dir,ranks)
    A = psparse(args...) |> fetch |> centralize
    println("dims A: $(size(A))")
    msizes[n_i] = size(A,1)
    diagA = dense_diag(A)
    invD = 1 ./ diagA
    M = invD .* A
    #M_dense = Array(M)
    #exp[n_i] = max(abs(eigmax(M_dense)), abs(eigmin(M_dense)))
    for t in 1:n_trials
        x0 = rand(size(A,2))
        lprev, x = powm!(M, x0, maxiter=1)
        tic = time_ns()
        for i in 1:maxiter
            # Power method
            l, x = powm!(M, x, maxiter = 1)
            errors_powm[n_i, t, i] = abs(l-lprev) 
            lprev = l
        end
        toc = time_ns()
        time_powm[n_i,t] = (toc - tic)/10^9
        ρprev, x = PartitionedSolvers.spectral_radius(M,x0, 1)
        tic = time_ns()
        for i in 1:maxiter
            # own implementation
            ρ, x = PartitionedSolvers.spectral_radius(M,x, 1)
            errors_spectrad[n_i, t, i] = abs(ρ-ρprev)
            ρprev = ρ
        end
        toc = time_ns()
        time_spectrad[n_i, t] = (toc-tic)/10^9
    end
end
avg_time_powm = median(time_powm, dims=2)
avg_time_spectrad = median(time_spectrad, dims=2)
avg_error_powm = median(errors_powm, dims=2)
avg_error_spectrad = median(errors_spectrad, dims=2)

p1 = plot()
plot!(p1, [1], [0], color="black", label="powm")
plot!(p1, [1], [0], color="black", ls=:dash, label="spectrad")
for (n_i, n) in enumerate(msizes)
    plot!(p1, 1:maxiter, avg_error_powm[n_i,:,:]', label = "($(n),$(n))", color= n_i)
    plot!(p1, 1:maxiter, avg_error_spectrad[n_i,:,:]', label = "", color=n_i, ls=:dash)
end

p2 = plot(msizes, [avg_time_powm avg_time_spectrad], label = ["powm" "spectrad"], color="black", marker=[:c :x])
yticks=[10^-16, 10^-14, 10^-12, 10^-10, 10^-8, 10^-6, 10^-4, 10^-2, 10^0]
xticks=[10^0, 10^1, 10^2, 10^3]
plot!(p1, xlabel="#iterations (k)", ylabel="|λ(k) - exp|", legend=:outertopleft, xscale=:log,
    xticks=xticks, yscale=:log, ylim=(10^-16, 1), yticks=yticks)
plot!(p2, xlabel="size of matrix", ylabel="avg runtime (s)", xscale=:log10)
p = plot(p1, p2, layout=(2,1), suptitle="Convergence of powm and spectral_radius")
savefig(p, "C:/Users/gelie/Home/ComputationalScience/GSoC/powm_l-lprev_k$(maxiter)_m$(msizes[end]).png")    
display(p)  =#

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