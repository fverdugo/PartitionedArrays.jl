using LinearAlgebra
using SparseArrays
using PartitionedArrays
using Test
using IterativeSolvers

function is_boundary_node(node_1d,nodes_1d)
    node_1d==1||node_1d==nodes_1d
end

u(x) = x[1]+x[2]

function setup_params()
    parts_in_x = 2
    length_in_x = 2.0
    cells_in_x = 10
    parts_per_dir = (parts_in_x,parts_in_x)
    length_per_dir = (length_in_x,length_in_x)
    cells_per_dir = (cells_in_x,cells_in_x)
    nodes_per_dir = cells_per_dir .+ 1
    h = maximum(length_per_dir./cells_per_dir)
    Ae = (h^2/6)*[
                4.0  -1.0  -1.0  -2.0
                -1.0   4.0  -2.0  -1.0
                -1.0  -2.0   4.0  -1.0
                -2.0  -1.0  -1.0   4.0
               ]
    (;parts_per_dir,length_per_dir,cells_per_dir,nodes_per_dir,u,Ae,h)
end

function setup_grid(cell_indices)
    params = setup_params()
    D = length(params.cells_per_dir)
    linear_to_cartesian_global_cell = CartesianIndices(params.cells_per_dir)
    linear_to_cartesian_global_node = CartesianIndices(params.nodes_per_dir)
    cartesian_to_linear_global_node = LinearIndices(linear_to_cartesian_global_node)
    linear_to_cartesian_element_node = CartesianIndices(ntuple(i->2,Val(D)))
    cartesian_to_linear_element_node = LinearIndices(linear_to_cartesian_element_node)
    cartesian_offset = CartesianIndex(ntuple(i->1,Val(D)))
    rank = part_id(cell_indices)
    local_to_global_cell = local_to_global(cell_indices)
    local_cell_to_owner = local_to_owner(cell_indices)
    first_global_cell = first(local_to_global_cell)
    last_global_cell = last(local_to_global_cell)
    first_cartesian_global_cell = linear_to_cartesian_global_cell[first_global_cell]
    last_cartesian_global_cell = linear_to_cartesian_global_cell[last_global_cell]
    global_cell_ranges = map((a,b)->a:b,Tuple(first_cartesian_global_cell),Tuple(last_cartesian_global_cell))
    global_node_ranges = map((a,b)->a:b,Tuple(first_cartesian_global_cell),Tuple(last_cartesian_global_cell+cartesian_offset))
    local_to_global_cartesian_cell = CartesianIndices(global_cell_ranges)
    local_to_global_cartesian_node = CartesianIndices(global_node_ranges)
    linear_to_cartesian_local_cell = CartesianIndices(local_to_global_cartesian_cell)
    cartesian_to_linear_local_node = LinearIndices(local_to_global_cartesian_node)
    cartesian_to_linear_local_cell = LinearIndices(local_to_global_cartesian_cell)
    (;
     params,
     cell_indices,
     local_to_global_cartesian_cell,
     local_to_global_cartesian_node,
     linear_to_cartesian_local_cell,
     linear_to_cartesian_element_node,
     cartesian_to_linear_local_node,
     cartesian_to_linear_local_cell,
     cartesian_to_linear_element_node,
     cartesian_offset)
end

function setup_space(grid)
    local_node_to_dof = ones(Int,length(grid.local_to_global_cartesian_node))
    for cartesian_local_cell in grid.linear_to_cartesian_local_cell
        cartesian_global_cell = grid.local_to_global_cartesian_cell[cartesian_local_cell]
        for cartesian_element_node in grid.linear_to_cartesian_element_node
            cartesian_global_node = cartesian_global_cell + (cartesian_element_node - grid.cartesian_offset)
            boundary = any(map(is_boundary_node,Tuple(cartesian_global_node),grid.params.nodes_per_dir))
            if boundary
                cartesian_local_node = cartesian_local_cell + (cartesian_element_node - grid.cartesian_offset)
                local_node = grid.cartesian_to_linear_local_node[cartesian_local_node]
                local_node_to_dof[local_node] = 0
            end
        end
    end
    local_dof_to_node = findall(i->i==1,local_node_to_dof)
    n_local_dofs = length(local_dof_to_node)
    local_node_to_dof[local_dof_to_node] = 1:n_local_dofs
    local_dof_to_owner = zeros(Int32,n_local_dofs)
    local_cell_to_owner = local_to_owner(grid.cell_indices)
    rank = part_id(grid.cell_indices)
    for cartesian_local_cell in grid.linear_to_cartesian_local_cell
        local_cell = grid.cartesian_to_linear_local_cell[cartesian_local_cell]
        owner = local_cell_to_owner[local_cell]
        for cartesian_element_node in grid.linear_to_cartesian_element_node
            cartesian_local_node = cartesian_local_cell + (cartesian_element_node - grid.cartesian_offset)
            local_node = grid.cartesian_to_linear_local_node[cartesian_local_node]
            local_dof = local_node_to_dof[local_node]
            if local_dof > 0
                local_dof_to_owner[local_dof] = max(local_dof_to_owner[local_dof],owner)
            end
        end
    end
    local_to_global_cell = local_to_global(grid.cell_indices)
    permutation = zeros(Int32,n_local_dofs)
    n_element_nodes = length(grid.linear_to_cartesian_element_node)
    n_local_cells = length(local_to_global_cell)
    ptrs = zeros(Int,n_local_cells+1)
    ptrs[2:end] .= n_element_nodes
    length_to_ptrs!(ptrs)
    data = zeros(Int,n_element_nodes*n_local_cells)
    local_cell_to_global_dofs = JaggedArray(data,ptrs)
    n_own_dofs = count(i->i==rank,local_dof_to_owner)
    n_own_dofs, (;local_node_to_dof,local_dof_to_owner,permutation,local_cell_to_global_dofs)
end

function setup_cell_dofs(grid,space,tentative_dof_indices)
    local_to_global_cell = local_to_global(grid.cell_indices)
    local_cell_to_owner = local_to_owner(grid.cell_indices)
    own_to_global_dof = own_to_global(tentative_dof_indices)
    dof_offset = first(own_to_global_dof) - 1
    n_local_dofs = length(space.local_dof_to_owner)
    rank = part_id(grid.cell_indices)
    own_dof_to_local_dof = findall(i->i==rank,space.local_dof_to_owner)
    n_own_dofs = length(own_dof_to_local_dof)
    space.permutation[own_dof_to_local_dof] = 1:n_own_dofs
    for cartesian_local_cell in grid.linear_to_cartesian_local_cell
        local_cell = grid.cartesian_to_linear_local_cell[cartesian_local_cell]
        for (element_node,cartesian_element_node) in enumerate(grid.linear_to_cartesian_element_node)
            cartesian_local_node = cartesian_local_cell + (cartesian_element_node - grid.cartesian_offset)
            local_node = grid.cartesian_to_linear_local_node[cartesian_local_node]
            local_dof = space.local_node_to_dof[local_node]
            if local_dof > 0
                own_dof = space.permutation[local_dof]
                if own_dof > 0
                    global_dof = own_dof + dof_offset
                    space.local_cell_to_global_dofs[local_cell][element_node] = global_dof
                end
            end
        end
    end
    space.local_cell_to_global_dofs
end

function finish_cell_dofs(grid,space,tentative_dof_indices)
    n_local_dofs = length(space.local_dof_to_owner)
    local_to_global_dof = zeros(Int,n_local_dofs)
    for cartesian_local_cell in grid.linear_to_cartesian_local_cell
        local_cell = grid.cartesian_to_linear_local_cell[cartesian_local_cell]
        for (element_node,cartesian_element_node) in enumerate(grid.linear_to_cartesian_element_node)
            cartesian_local_node = cartesian_local_cell + (cartesian_element_node - grid.cartesian_offset)
            local_node = grid.cartesian_to_linear_local_node[cartesian_local_node]
            local_dof = space.local_node_to_dof[local_node]
            global_dof = space.local_cell_to_global_dofs[local_cell][element_node]
            if local_dof > 0 && global_dof > 0
                local_to_global_dof[local_dof] = global_dof
            end
        end
    end
    for cartesian_local_cell in grid.linear_to_cartesian_local_cell
        local_cell = grid.cartesian_to_linear_local_cell[cartesian_local_cell]
        for (element_node,cartesian_element_node) in enumerate(grid.linear_to_cartesian_element_node)
            cartesian_local_node = cartesian_local_cell + (cartesian_element_node - grid.cartesian_offset)
            local_node = grid.cartesian_to_linear_local_node[cartesian_local_node]
            local_dof = space.local_node_to_dof[local_node]
            if local_dof > 0
                global_dof = local_to_global_dof[local_dof]
                @assert global_dof != 0
                space.local_cell_to_global_dofs[local_cell][element_node] = global_dof
            end
        end
    end
end

function setup_IJV(space,grid)
    params = grid.params
    Ae = params.Ae
    I = Int[]
    J = Int[]
    V = Float64[]
    local_cell_to_owner = local_to_owner(grid.cell_indices)
    rank = part_id(grid.cell_indices)
    n_local_cells = length(local_cell_to_owner)
    for local_cell in 1:n_local_cells
        if local_cell_to_owner[local_cell] != rank
            continue
        end
        global_dofs = space.local_cell_to_global_dofs[local_cell]
        for (element_row,global_row) in enumerate(global_dofs)
            if global_row <= 0
                continue
            end
            for (element_col,global_col) in enumerate(global_dofs)
                if global_col <= 0
                    continue
                end
                push!(I,global_row)
                push!(J,global_col)
                push!(V,Ae[element_row,element_col])
            end
        end
    end
    I,J,V
end

function setup_b(space,grid)
    params = grid.params
    h = params.h
    Ae = params.Ae
    I = Int[]
    V = Float64[]
    local_cell_to_owner = local_to_owner(grid.cell_indices)
    rank = part_id(grid.cell_indices)
    ue = zeros(size(Ae,2))
    ge = similar(ue)
    for cartesian_local_cell in grid.linear_to_cartesian_local_cell
        local_cell = grid.cartesian_to_linear_local_cell[cartesian_local_cell]
        if local_cell_to_owner[local_cell] != rank
            continue
        end
        fill!(ue,0.0)
        for (element_node,cartesian_element_node) in enumerate(grid.linear_to_cartesian_element_node)
            cartesian_local_node = cartesian_local_cell + (cartesian_element_node - grid.cartesian_offset)
            local_node = grid.cartesian_to_linear_local_node[cartesian_local_node]
            local_dof = space.local_node_to_dof[local_node]
            if local_dof <= 0
                cartesian_global_node = grid.local_to_global_cartesian_node[local_node]
                node_coordinates = (Tuple(cartesian_global_node) .- 1) .* h
                ue[element_node] = params.u(node_coordinates)
            end
        end
        mul!(ge,Ae,ue)
        global_dofs = space.local_cell_to_global_dofs[local_cell]
        for (element_row,global_row) in enumerate(global_dofs)
            if global_row > 0
                push!(I,global_row)
                push!(V,-ge[element_row])
            end
        end
    end
    I,V
end

function setup_exact_solution(values,space,grid,col_indices)
    params = grid.params
    h = maximum(params.length_per_dir./params.cells_per_dir)
    global_to_local_col = global_to_local(col_indices)
    local_cell_to_owner = local_to_owner(grid.cell_indices)
    rank = part_id(grid.cell_indices)
    for cartesian_local_cell in grid.linear_to_cartesian_local_cell
        local_cell = grid.cartesian_to_linear_local_cell[cartesian_local_cell]
        if local_cell_to_owner[local_cell] != rank
            continue
        end
        for (element_node,cartesian_element_node) in enumerate(grid.linear_to_cartesian_element_node)
            cartesian_local_node = cartesian_local_cell + (cartesian_element_node - grid.cartesian_offset)
            local_node = grid.cartesian_to_linear_local_node[cartesian_local_node]
            local_dof = space.local_node_to_dof[local_node]
            if local_dof > 0
                cartesian_global_node = grid.local_to_global_cartesian_node[local_node]
                node_coordinates = (Tuple(cartesian_global_node) .- 1) .* h
                global_dof = space.local_cell_to_global_dofs[local_cell][element_node]
                local_col = global_to_local_col[global_dof]
                values[local_col] = params.u(node_coordinates)
            end
        end
    end
end

function fem_example(distribute)
    params = setup_params()
    n_ranks = prod(params.parts_per_dir)
    rank = distribute(LinearIndices((n_ranks,)))
    t = PTimer(rank,verbose=true)
    ghost_per_dir = (true,true)
    cell_partition = uniform_partition(rank,params.parts_per_dir,params.cells_per_dir,ghost_per_dir)
    grid = map(setup_grid,cell_partition)
    n_own_dofs, space = map(setup_space,grid) |> tuple_of_arrays
    n_global_dofs = sum(n_own_dofs)
    dof_partition = variable_partition(n_own_dofs,n_global_dofs)
    local_cell_to_global_dofs = map(setup_cell_dofs,grid,space,dof_partition)
    cell_to_global_dofs = PVector(local_cell_to_global_dofs,cell_partition)
    consistent!(cell_to_global_dofs) |> wait
    map(finish_cell_dofs,grid,space,dof_partition)
    I,J,V = map(setup_IJV,space,grid) |> tuple_of_arrays
    t = psparse(I,J,V,dof_partition,dof_partition)
    II,VV = map(setup_b,space,grid) |> tuple_of_arrays
    A = fetch(t)
    b = pvector(II,VV,dof_partition) |> fetch
    x = IterativeSolvers.cg(A,b,verbose=i_am_main(rank))
    x̂ = similar(x)
    col_partition = partition(axes(A,2))
    map(setup_exact_solution,local_values(x̂),space,grid,col_partition)
    @test norm(x-x̂) < 1.0e-5

    ## re-assembly
    A,cacheA = psparse(I,J,V,dof_partition,dof_partition;reuse=true) |> fetch
    b,cacheb = pvector(II,VV,dof_partition;reuse=true) |> fetch
    psparse!(A,V,cacheA) |> wait
    pvector!(b,VV,cacheb) |> wait
    display((A,cacheA))

    x = IterativeSolvers.cg(A,b,verbose=i_am_main(rank))
    @test norm(x-x̂) < 1.0e-5

    # Simultaneous generation of matrix and vector
    A,b = psystem(I,J,V,II,VV,dof_partition,dof_partition) |> fetch
    x = IterativeSolvers.cg(A,b,verbose=i_am_main(rank))
    @test norm(x-x̂) < 1.0e-5

    # Simultaneous generation of matrix and vector
    # giving extra info and allowing for reassembly
    # This one is potentially the most efficient
    # with the only caveat that the computation
    # of II and VV is not overlapped with the assembly
    # of the matrix (not a big problem since I,J,V,II,VV are usually
    # computed in the same cell-wise loop)
    neighbors = assembly_graph(cell_partition)
    A,b,cache = psystem(
        I,J,V,II,VV,dof_partition,dof_partition;
        assembly_neighbors_options_rows = (;neighbors),
        assembly_neighbors_options_cols = (;neighbors),
        reuse = true) |> fetch
    x = IterativeSolvers.cg(A,b,verbose=i_am_main(rank))
    @test norm(x-x̂) < 1.0e-5

    ## Re assembly with different values
    V2 = map(i->2*i,V)
    VV2 = map(i->2*i,VV)
    #display(partition(b))
    psystem!(A,b,V2,VV2,cache) |> wait
    #display(partition(b))
    x = IterativeSolvers.cg(A,b,verbose=i_am_main(rank))
    @test norm(x-x̂) < 1.0e-5

    # Sub-assembled variant
    A,b = psystem(I,J,V,II,VV,dof_partition,dof_partition;assemble=false) |> fetch
    assemble!(b) |> wait
    x = IterativeSolvers.cg(A,b,verbose=i_am_main(rank))
    col_partition = partition(axes(A,2))
    x̂ = similar(x)
    map(setup_exact_solution,local_values(x̂),space,grid,col_partition)
    @test norm(x-x̂) < 1.0e-5

end

