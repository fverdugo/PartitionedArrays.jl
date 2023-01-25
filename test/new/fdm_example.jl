using LinearAlgebra
using SparseArrays
using PartitionedArrays
using Test
using IterativeSolvers

function fdm_example(distribute)

    # Select number of ranks in each direction and distribute
    # them according the given backend
    parts_per_dir = (2,2,2)
    n_ranks = prod(parts_per_dir)
    rank = distribute(LinearIndices((n_ranks,)))
    t = PTimer(rank,verbose=true)

    # Data of the method
    u(x) = x[1]+x[2]
    f(x) = zero(x[1])
    length_in_x = 2.0
    length_per_dir = (length_in_x,length_in_x,length_in_x)
    nodes_in_x = 9
    nodes_per_dir = (nodes_in_x,nodes_in_x,nodes_in_x)
    n = prod(nodes_per_dir)
    h = length_in_x/(nodes_in_x-1)
    points = [(0,0,0),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
    coeffs = [-6,1,1,1,1,1,1]/(h^2)
    stencil = [ (coeff,CartesianIndex(point)) for (coeff,point) in zip(coeffs,points) ]
    function is_boundary_node(node_1d,nodes_1d)
        node_1d==1||node_1d==nodes_1d
    end

    # Chose the partition of the rows of the linear system
    # Each point in the grid will lead to an equation, including the boundary ones.
    # We use an auxiliary identity block to impose conditions at the boundary.
    # Note that we are not using ghost layer in this partition.
    row_partition = uniform_partition(rank,parts_per_dir,nodes_per_dir)

    # We don't need the ghost layer for the rhs
    # So, it can be allocated right now.
    b = PVector(undef,row_partition)

    # We don't need the ghost layer for the exact solution
    # So, it can be allocated right now.
    x̂ = similar(b)

    # Loop over local (==owned) rows, fill the coo-vectors, rhs, and the exact solution
    # In this case, we always touch local rows, but arbitrary cols.
    # Thus, row ids can be readily stored in local numbering so that we do not need to convert
    # them later.
    function coo_vectors!(local_to_global_row,local_b,local_x̂)
        linear_to_cartesian = CartesianIndices(nodes_per_dir)
        cartesian_to_linear = LinearIndices(linear_to_cartesian)
        I = Int[]
        J = Int[]
        V = Float64[]
        for local_row in 1:length(local_to_global_row)
            global_row = local_to_global_row[local_row]
            cartesian_row = linear_to_cartesian[global_row]
            node_coordinates = (Tuple(cartesian_row) .- 1) .* h
            local_x̂[local_row] = u(node_coordinates)
            boundary = any(map(is_boundary_node,Tuple(cartesian_row),nodes_per_dir))
            if boundary
                push!(I,global_row)
                push!(J,global_row)
                push!(V,one(eltype(V)))
                local_b[local_row] = u(node_coordinates)
            else
                for (v,dcj) in stencil
                    cartesian_col = cartesian_row + dcj
                    global_col = cartesian_to_linear[cartesian_col]
                    push!(I,global_row)
                    push!(J,global_col)
                    push!(V,-v)
                end
                local_b[local_row] = f(node_coordinates)
            end
        end
        I,J,V
    end
    tic!(t)
    IJV = map(coo_vectors!,row_partition,get_local_values(b),get_local_values(x̂))
    toc!(t,"IJV")
    I,J,V = tuple_of_arrays(IJV)

    # Build the PSparseMatrix from the coo-vectors
    # and the data distribution described by rows and cols.
    tic!(t)
    tentative_col_partition = row_partition
    A = psparse!(I,J,V,row_partition,tentative_col_partition,discover_rows=false) |> fetch
    toc!(t,"A")
    cols = axes(A,2)

    # The initial guess needs the ghost layer (that why we take cols)
    # in other to perform the product A*x in the cg solver.
    # We also need to set the boundary values
    function initial_guess!(own_to_global_col,own_x0)
        linear_to_cartesian = CartesianIndices(nodes_per_dir)
        for own_col in 1:length(own_to_global_col)
            global_col = own_to_global_col[own_col]
            cartesian_col = linear_to_cartesian[global_col]
            boundary = any(map(is_boundary_node,Tuple(cartesian_col),nodes_per_dir))
            if boundary
                node_coordinates = (Tuple(cartesian_col) .- 1) .* h
                own_x0[own_col] = u(node_coordinates)
            end
        end

    end
    x0 = pzeros(partition(cols))
    map(initial_guess!,get_own_to_global(cols),get_own_values(x0))

    # When this call returns, x has the correct answer only in the owned values.
    # The values at ghost ids can be recovered with consistent!(x) |> wait
    x = copy(x0)
    tic!(t)
    IterativeSolvers.cg!(x,A,b,verbose=i_am_main(rank))
    toc!(t,"solve")

    # This compares owned values, so we don't need to consistent!(x) |> wait
    @test norm(x-x̂) < 1.0e-5
    toc!(t,"norm")

    display(t)

end

