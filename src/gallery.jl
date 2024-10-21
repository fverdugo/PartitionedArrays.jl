
"""
    laplacian_fdm(
            nodes_per_dir,
            parts_per_dir,
            parts;
            index_type = Int64,
            value_type = Float64)

Document me!
"""
function laplacian_fdm(
        nodes_per_dir,
        parts_per_dir,
        parts;
        index_type::Type{Ti} = Int64,
        value_type::Type{Tv} = Float64,
    ) where {Ti,Tv}
    function neig_node(cartesian_node_i,d,i,cartesian_node_to_node)
        function is_boundary_node(node_1d,nodes_1d)
            !(node_1d in 1:nodes_1d)
        end
        D = length(nodes_per_dir)
        inc = ntuple(k->( k==d ? i : zero(i)),Val{D}())
        cartesian_node_j = CartesianIndex(Tuple(cartesian_node_i) .+ inc)
        boundary = any(map(is_boundary_node,Tuple(cartesian_node_j),nodes_per_dir))
        T = eltype(cartesian_node_to_node)
        if boundary
            return zero(T)
        end
        node_j = cartesian_node_to_node[cartesian_node_j]
        node_j
    end
    function setup(nodes,::Type{index_type},::Type{value_type}) where {index_type,value_type}
        D = length(nodes_per_dir)
        α = value_type(prod(i->(i+1),nodes_per_dir))
        node_to_cartesian_node = CartesianIndices(nodes_per_dir)
        cartesian_node_to_node = LinearIndices(nodes_per_dir)
        first_cartesian_node = node_to_cartesian_node[first(nodes)]
        last_cartesian_node = node_to_cartesian_node[last(nodes)]
        ranges = map(:,Tuple(first_cartesian_node),Tuple(last_cartesian_node))
        cartesian_nodes = CartesianIndices(ranges)
        nnz = 0
        for cartesian_node_i in cartesian_nodes
            nnz+=1
            for d in 1:D
                for i in (-1,1)
                    node_j = neig_node(cartesian_node_i,d,i,cartesian_node_to_node)
                    if node_j == 0
                        continue
                    end
                    nnz+=1
                end
            end
        end
        myI = zeros(index_type,nnz)
        myJ = zeros(index_type,nnz)
        myV = zeros(value_type,nnz)
        t = 0
        for cartesian_node_i in cartesian_nodes
            t += 1
            node_i = cartesian_node_to_node[cartesian_node_i]
            myI[t] = node_i
            myJ[t] = node_i
            myV[t] = α*2*D
            for d in 1:D
                for i in (-1,1)
                    node_j = neig_node(cartesian_node_i,d,i,cartesian_node_to_node)
                    if node_j == 0
                        continue
                    end
                    t += 1
                    myI[t] = node_i
                    myJ[t] = node_j
                    myV[t] = -α
                end
            end
        end
        myI,myJ,myV
    end
    node_partition = uniform_partition(parts,parts_per_dir,nodes_per_dir)
    I,J,V = map(node_partition) do nodes
        setup(nodes,Ti,Tv)
    end |> tuple_of_arrays
    I,J,V,node_partition,node_partition
end

"""
    laplacian_fem(
            nodes_per_dir,
            parts_per_dir,
            parts;
            index_type = Int64,
            value_type = Float64)

Document me!
"""
function laplacian_fem(
        nodes_per_dir, # free (== interior) nodes
        parts_per_dir,
        parts;
        index_type::Type{Ti} = Int64,
        value_type::Type{Tv} = Float64,
    ) where {Ti,Tv}

    cells_per_dir = nodes_per_dir .+ 1

    function is_boundary_node(node_1d,nodes_1d)
        !(node_1d in 1:nodes_1d)
    end
    function ref_matrix(cartesian_local_nodes,h_per_dir,::Type{value_type}) where value_type
        D = ndims(cartesian_local_nodes)
        gp_1d = value_type[-sqrt(3)/3,sqrt(3)/3]
        sf_1d = zeros(value_type,length(gp_1d),2)
        sf_1d[:,1] = 0.5 .* (1 .- gp_1d)
        sf_1d[:,2] = 0.5 .* (gp_1d .+ 1)
        sg_1d = zeros(value_type,length(gp_1d),2)
        sg_1d[:,1] .= - 0.5
        sg_1d[:,2] .=  0.5
        cartesian_points = CartesianIndices(ntuple(d->length(gp_1d),Val{D}()))
        cartesian_local_node_to_local_node = LinearIndices(cartesian_local_nodes)
        cartesian_point_to_point = LinearIndices(cartesian_points)
        n = 2^D
        sg = Matrix{NTuple{D,value_type}}(undef,n,length(gp_1d)^D)
        for cartesian_local_node in cartesian_local_nodes
            local_node = cartesian_local_node_to_local_node[cartesian_local_node]
            local_node_tuple = Tuple(cartesian_local_node)
            for cartesian_point in cartesian_points
                point = cartesian_point_to_point[cartesian_point]
                point_tuple = Tuple(cartesian_point)
                v = ntuple(Val{D}()) do d
                    prod(1:D) do i
                        if i == d
                            (2/h_per_dir[d])*sg_1d[local_node_tuple[d],point_tuple[d]]
                        else
                            sf_1d[local_node_tuple[i],point_tuple[i]]
                        end
                    end
                end
                sg[local_node,point] = v
            end
        end
        Aref = zeros(value_type,n,n)
        dV = prod(h_per_dir)/(2^D)
        for i in 1:n
            for j in 1:n
                for k in 1:n
                    Aref[i,j] += dV*dot(sg[k,i],sg[k,j])
                end
            end
        end
        Aref
    end
    function setup(cells,::Type{index_type},::Type{value_type}) where {index_type,value_type}
        D = length(nodes_per_dir)
        h_per_dir = map(i->1/(i+1),nodes_per_dir)
        ttt = ntuple(d->2,Val{D}())
        cartesian_local_nodes = CartesianIndices(ttt)
        Aref = ref_matrix(cartesian_local_nodes,h_per_dir,value_type)#ones(value_type,2^D,2^D)
        cell_to_cartesian_cell = CartesianIndices(cells_per_dir)
        first_cartesian_cell = cell_to_cartesian_cell[first(cells)]
        last_cartesian_cell = cell_to_cartesian_cell[last(cells)]
        ranges = map(:,Tuple(first_cartesian_cell),Tuple(last_cartesian_cell))
        cartesian_cells = CartesianIndices(ranges)
        offset = CartesianIndex(ttt)
        cartesian_local_node_to_local_node = LinearIndices(cartesian_local_nodes)
        cartesian_node_to_node = LinearIndices(nodes_per_dir)
        nnz = 0 
        for cartesian_cell in cartesian_cells
            for cartesian_local_node_i in cartesian_local_nodes
                local_node_i = cartesian_local_node_to_local_node[cartesian_local_node_i]
                # This is ugly to support Julia 1.6 (idem below)
                cartesian_node_i = CartesianIndex(Tuple(cartesian_cell) .+ Tuple(cartesian_local_node_i) .- Tuple(offset))
                boundary = any(map(is_boundary_node,Tuple(cartesian_node_i),nodes_per_dir))
                if boundary
                    continue
                end
                node_i = cartesian_node_to_node[cartesian_node_i]
                for cartesian_local_node_j in cartesian_local_nodes
                    local_node_j = cartesian_local_node_to_local_node[cartesian_local_node_j]
                    cartesian_node_j = CartesianIndex(Tuple(cartesian_cell) .+ Tuple(cartesian_local_node_j) .- Tuple(offset))
                    boundary = any(map(is_boundary_node,Tuple(cartesian_node_j),nodes_per_dir))
                    if boundary
                        continue
                    end
                    node_j = cartesian_node_to_node[cartesian_node_j]
                    nnz += 1
                end
            end
        end
        myI = zeros(index_type,nnz)
        myJ = zeros(index_type,nnz)
        myV = zeros(value_type,nnz)
        k = 0 
        for cartesian_cell in cartesian_cells
            for cartesian_local_node_i in cartesian_local_nodes
                local_node_i = cartesian_local_node_to_local_node[cartesian_local_node_i]
                cartesian_node_i = CartesianIndex(Tuple(cartesian_cell) .+ Tuple(cartesian_local_node_i) .- Tuple(offset))
                boundary = any(map(is_boundary_node,Tuple(cartesian_node_i),nodes_per_dir))
                if boundary
                    continue
                end
                node_i = cartesian_node_to_node[cartesian_node_i]
                for cartesian_local_node_j in cartesian_local_nodes
                    local_node_j = cartesian_local_node_to_local_node[cartesian_local_node_j]
                    cartesian_node_j = CartesianIndex(Tuple(cartesian_cell) .+ Tuple(cartesian_local_node_j) .- Tuple(offset))
                    boundary = any(map(is_boundary_node,Tuple(cartesian_node_j),nodes_per_dir))
                    if boundary
                        continue
                    end
                    node_j = cartesian_node_to_node[cartesian_node_j]
                    k += 1
                    myI[k] = node_i
                    myJ[k] = node_j
                    myV[k] = Aref[local_node_i,local_node_j]
                end
            end
        end
        myI,myJ,myV
    end
    node_partition = uniform_partition(parts,parts_per_dir,nodes_per_dir)
    cell_partition = uniform_partition(parts,parts_per_dir,cells_per_dir)
    I,J,V = map(cell_partition) do cells
        setup(cells,Ti,Tv)
    end |> tuple_of_arrays
    I,J,V,node_partition,node_partition
end

function linear_elasticity_fem(
        nodes_per_dir, # free (== interior) nodes
        parts_per_dir,
        parts,
        ;
        E = 1,
        ν = 0.25,
        index_type::Type{Ti} = Int64,
        value_type::Type{Tv} = Float64,
    ) where {Ti,Tv}

    cells_per_dir = nodes_per_dir .+ 1

    function is_boundary_node(node_1d,nodes_1d)
        !(node_1d in 1:nodes_1d)
    end
    function ref_matrix(cartesian_local_nodes,h_per_dir,::Type{value_type}) where value_type
        D = ndims(cartesian_local_nodes)
        gp_1d = value_type[-sqrt(3)/3,sqrt(3)/3]
        sf_1d = zeros(value_type,length(gp_1d),2)
        sf_1d[:,1] = 0.5 .* (1 .- gp_1d)
        sf_1d[:,2] = 0.5 .* (gp_1d .+ 1)
        sg_1d = zeros(value_type,length(gp_1d),2)
        sg_1d[:,1] .= - 0.5
        sg_1d[:,2] .=  0.5
        cartesian_points = CartesianIndices(ntuple(d->length(gp_1d),Val{D}()))
        cartesian_local_node_to_local_node = LinearIndices(cartesian_local_nodes)
        cartesian_point_to_point = LinearIndices(cartesian_points)
        n = 2^D
        sg = Matrix{NTuple{D,value_type}}(undef,n,length(gp_1d)^D)
        for cartesian_local_node in cartesian_local_nodes
            local_node = cartesian_local_node_to_local_node[cartesian_local_node]
            local_node_tuple = Tuple(cartesian_local_node)
            for cartesian_point in cartesian_points
                point = cartesian_point_to_point[cartesian_point]
                point_tuple = Tuple(cartesian_point)
                v = ntuple(Val{D}()) do d
                    prod(1:D) do i
                        if i == d
                            (2/h_per_dir[d])*sg_1d[local_node_tuple[d],point_tuple[d]]
                        else
                            sf_1d[local_node_tuple[i],point_tuple[i]]
                        end
                    end
                end
                sg[local_node,point] = v
            end
        end
        Aref = zeros(value_type,n*D,n*D)
        dV = prod(h_per_dir)/(2^D)
        ε_i = zeros(value_type,D,D)
        ε_j = zeros(value_type,D,D)
        λ = (E*ν)/((1+ν)*(1-2*ν))
        μ = E/(2*(1+ν))
        for i in 1:n
            for j in 1:n
                for ci in 1:D
                    for cj in 1:D
                        idof = (i-1)*D+ci
                        jdof = (j-1)*D+cj
                        ε_i .= 0
                        ε_j .= 0
                        for k in 1:n
                            ε_i[ci,:] = collect(sg[k,i])
                            ε_j[cj,:] = collect(sg[k,j])
                            ε_i .= 0.5 .* ( ε_i .+ transpose(ε_i))
                            ε_j .= 0.5 .* ( ε_j .+ transpose(ε_j))
                            σ_j = λ*tr(ε_j)*one(ε_j) + 2*μ*ε_j
                            Aref[idof,jdof] += tr(ε_i*σ_j)
                        end
                    end
                end
            end
        end
        Aref
    end
    function setup(cells,::Type{index_type},::Type{value_type}) where {index_type,value_type}
        D = length(nodes_per_dir)
        h_per_dir = map(i->1/(i+1),nodes_per_dir)
        ttt = ntuple(d->2,Val{D}())
        cartesian_local_nodes = CartesianIndices(ttt)
        Aref = ref_matrix(cartesian_local_nodes,h_per_dir,value_type)#ones(value_type,2^D,2^D)
        cell_to_cartesian_cell = CartesianIndices(cells_per_dir)
        first_cartesian_cell = cell_to_cartesian_cell[first(cells)]
        last_cartesian_cell = cell_to_cartesian_cell[last(cells)]
        ranges = map(:,Tuple(first_cartesian_cell),Tuple(last_cartesian_cell))
        cartesian_cells = CartesianIndices(ranges)
        offset = CartesianIndex(ttt)
        cartesian_local_node_to_local_node = LinearIndices(cartesian_local_nodes)
        cartesian_node_to_node = LinearIndices(nodes_per_dir)
        nnz = 0 
        for cartesian_cell in cartesian_cells
            for cartesian_local_node_i in cartesian_local_nodes
                local_node_i = cartesian_local_node_to_local_node[cartesian_local_node_i]
                # This is ugly to support Julia 1.6 (idem below)
                cartesian_node_i = CartesianIndex(Tuple(cartesian_cell) .+ Tuple(cartesian_local_node_i) .- Tuple(offset))
                boundary = any(map(is_boundary_node,Tuple(cartesian_node_i),nodes_per_dir))
                if boundary
                    continue
                end
                node_i = cartesian_node_to_node[cartesian_node_i]
                for cartesian_local_node_j in cartesian_local_nodes
                    local_node_j = cartesian_local_node_to_local_node[cartesian_local_node_j]
                    cartesian_node_j = CartesianIndex(Tuple(cartesian_cell) .+ Tuple(cartesian_local_node_j) .- Tuple(offset))
                    boundary = any(map(is_boundary_node,Tuple(cartesian_node_j),nodes_per_dir))
                    if boundary
                        continue
                    end
                    node_j = cartesian_node_to_node[cartesian_node_j]
                    nnz += D*D
                end
            end
        end
        myI = zeros(index_type,nnz)
        myJ = zeros(index_type,nnz)
        myV = zeros(value_type,nnz)
        k = 0 
        for cartesian_cell in cartesian_cells
            for cartesian_local_node_i in cartesian_local_nodes
                local_node_i = cartesian_local_node_to_local_node[cartesian_local_node_i]
                cartesian_node_i = CartesianIndex(Tuple(cartesian_cell) .+ Tuple(cartesian_local_node_i) .- Tuple(offset))
                boundary = any(map(is_boundary_node,Tuple(cartesian_node_i),nodes_per_dir))
                if boundary
                    continue
                end
                node_i = cartesian_node_to_node[cartesian_node_i]
                for cartesian_local_node_j in cartesian_local_nodes
                    local_node_j = cartesian_local_node_to_local_node[cartesian_local_node_j]
                    cartesian_node_j = CartesianIndex(Tuple(cartesian_cell) .+ Tuple(cartesian_local_node_j) .- Tuple(offset))
                    boundary = any(map(is_boundary_node,Tuple(cartesian_node_j),nodes_per_dir))
                    if boundary
                        continue
                    end
                    node_j = cartesian_node_to_node[cartesian_node_j]
                    for ci in 1:D
                        for cj in 1:D
                            dof_i = (node_i-1)*D + ci
                            dof_j = (node_j-1)*D + cj
                            local_dof_i = (local_node_i-1)*D + ci
                            local_dof_j = (local_node_j-1)*D + cj
                            k += 1
                            myI[k] = dof_i
                            myJ[k] = dof_j
                            myV[k] = Aref[local_dof_i,local_dof_j]
                        end
                    end
                end
            end
        end
        myI,myJ,myV
    end
    node_partition = uniform_partition(parts,parts_per_dir,nodes_per_dir)
    dof_partition = node_to_dof_partition(node_partition,length(nodes_per_dir))
    cell_partition = uniform_partition(parts,parts_per_dir,cells_per_dir)
    I,J,V = map(cell_partition) do cells
        setup(cells,Ti,Tv)
    end |> tuple_of_arrays
    I,J,V,dof_partition,dof_partition
end

function node_to_dof_partition(node_partition,D)
    global_node_to_owner = global_to_owner(node_partition)
    dof_partition = map(node_partition) do mynodes
        @assert ghost_length(mynodes) == 0
        own_to_global_node = own_to_global(mynodes)
        n_own_nodes = length(own_to_global_node)
        own_to_global_dof = zeros(Int,D*n_own_nodes)
        for own_node in 1:n_own_nodes
            for ci in 1:D
                own_dof = (own_node-1)*D+ci
                global_node = own_to_global_node[own_node]
                global_dof = (global_node-1)*D+ci
                own_to_global_dof[own_dof] = global_dof
            end
        end
        n_global_dofs = global_length(mynodes)*D
        owner = part_id(mynodes)
        own_dofs = OwnIndices(n_global_dofs,owner,own_to_global_dof)
        ghost_dofs = GhostIndices(n_global_dofs,Int[],Int32[])
        global_dof_to_owner = global_dof -> begin
            global_node = div(global_dof-1,D)+1 
            global_node_to_owner[global_node]
        end
        mydofs = OwnAndGhostIndices(own_dofs,ghost_dofs,global_dof_to_owner)
        mydofs
    end
    dof_partition
end

function node_coordinates_unit_cube(
        nodes_per_dir, # free (== interior) nodes
        parts_per_dir,
        parts,
        ;
        split_format = Val(false),
        value_type::Type{Tv} = Float64,) where Tv

    function setup!(own_x,mynodes)
        D = length(nodes_per_dir)
        h_per_dir = map(i->1/(i+1),nodes_per_dir)
        global_node_to_cartesian_global_node = CartesianIndices(nodes_per_dir)
        n_own_nodes = own_length(mynodes)
        own_to_global_node = own_to_global(mynodes)
        for own_node in 1:n_own_nodes
            global_node = own_to_global_node[own_node]
            cartesian_global_node = global_node_to_cartesian_global_node[global_node]
            xi = Tuple(cartesian_global_node)
            own_x[own_node] = h_per_dir .* xi
        end
    end
    node_partition = uniform_partition(parts,parts_per_dir,nodes_per_dir)
    T = SVector{length(nodes_per_dir),Tv}
    x = pzeros(T,node_partition;split_format)
    foreach(setup!,own_values(x),node_partition)
    x
end

function near_nullspace_linear_elasticity(a...;b...)
    @warn "near_nullspace_linear_elasticity is deprecated, use nullspace_linear_elasticity instead"
    nullspace_linear_elasticity(a...;b...)
end

function nullspace_linear_elasticity(x,
        row_partition = node_to_dof_partition(partition(axes(x,1)),length(eltype(x)))
    )
    T = eltype(x)
    D = length(T)
    Tv = eltype(T)
    if D == 1
        nb = 1
    elseif D==2
        nb=3
    elseif D == 3
        nb = 6
    else
        error("case not implemented")
    end
    dof_partition = row_partition
    split_format = Val(eltype(partition(x)) <: SplitVector)
    B = [ pzeros(Tv,dof_partition;split_format) for _ in 1:nb ]
    nullspace_linear_elasticity!(B,x)
end

function nullspace_linear_elasticity!(B,x)
    D = length(eltype(x))
    if D == 1
        foreach(own_values(B[1])) do own_b
            fill!(own_b,1)
        end
    elseif D==2
        foreach(own_values(B[1]),own_values(B[2]),own_values(x)) do own_b1,own_b2,own_x
            T = eltype(own_b1)
            n_own_nodes = length(own_x)
            for own_node in 1:n_own_nodes
                dof_x1 = (own_node-1)*2 + 1
                dof_x2 = (own_node-1)*2 + 2
                #
                own_b1[dof_x1] = one(T)
                own_b1[dof_x2] = zero(T)
                #
                own_b2[dof_x1] = zero(T)
                own_b2[dof_x2] = one(T)
            end
        end
        foreach(own_values(B[3]),own_values(x)) do own_b3,own_x
            T = eltype(own_b3)
            n_own_nodes = length(own_x)
            for own_node in 1:n_own_nodes
                x1,x2 = own_x[own_node]
                dof_x1 = (own_node-1)*2 + 1
                dof_x2 = (own_node-1)*2 + 2
                #
                own_b3[dof_x1] = -x2
                own_b3[dof_x2] = x1
            end
        end
    elseif D == 3
        foreach(own_values(B[1]),own_values(B[2]),own_values(B[3]),own_values(x)) do own_b1,own_b2,own_b3,own_x
            T = eltype(own_b1)
            n_own_nodes = length(own_x)
            for own_node in 1:n_own_nodes
                dof_x1 = (own_node-1)*3 + 1
                dof_x2 = (own_node-1)*3 + 2
                dof_x3 = (own_node-1)*3 + 3
                #
                own_b1[dof_x1] = one(T)
                own_b1[dof_x2] = zero(T)
                own_b1[dof_x3] = zero(T)
                #
                own_b2[dof_x1] = zero(T)
                own_b2[dof_x2] = one(T)
                own_b2[dof_x3] = zero(T)
                #
                own_b3[dof_x1] = zero(T)
                own_b3[dof_x2] = zero(T)
                own_b3[dof_x3] = one(T)
            end
        end
        foreach(own_values(B[4]),own_values(B[5]),own_values(B[6]),own_values(x)) do own_b4,own_b5,own_b6,own_x
            T = eltype(own_b4)
            n_own_nodes = length(own_x)
            for own_node in 1:n_own_nodes
                x1,x2,x3 = own_x[own_node]
                dof_x1 = (own_node-1)*3 + 1
                dof_x2 = (own_node-1)*3 + 2
                dof_x3 = (own_node-1)*3 + 3
                #
                own_b4[dof_x1] = -x2
                own_b4[dof_x2] = x1
                own_b4[dof_x3] = zero(T)
                #
                own_b5[dof_x1] = zero(T)
                own_b5[dof_x2] = -x3
                own_b5[dof_x3] = x2
                #
                own_b6[dof_x1] = x3
                own_b6[dof_x2] = zero(T)
                own_b6[dof_x3] = -x1
            end
        end
    else
        error("case not implemented")
    end
    B
end


function prolongator(T,
                     nodes_per_dir,
                     parts_per_dir,
                     parts;
                     index_type::Type{Ti} = Int64,
                     value_type::Type{Tv} = Float64) where {Ti,Tv}
    # Improved version of aggregate function not using inefficient direct JaggedRange indexing causing many view allocations
    # Also uses a generic function to obtain pointer arrays and index arrays to work with both CSC and CSR.
    # If the problem is symmetric neighbours dont change by this.
    function aggregate(A,diagA=dense_diag(A);epsilon=0)
        # This one is algorithm 5.1 from
        # "Algebraic multigrid by smoothed aggregation for second and fourth order elliptic problems"
        epsi = epsilon
        typeof_strength = eltype(A.nzval)

        nnodes = size(A,1)
        pending = Ti(0)
        isolated = Ti(-1)
        
        node_to_aggregate = fill(pending,nnodes)
        node_to_old_aggregate = similar(node_to_aggregate)

        node_to_neigs = jagged_array(index_array(A),pointer_array(A))
        neigs = node_to_neigs.data
        node_to_vals = jagged_array(A.nzval,pointer_array(A))
        vals = node_to_vals.data
        strongly_connected = (node,ineig) -> begin
            neig = neigs[ineig]
            aii = diagA[node]
            ajj = diagA[neig]
            aij = vals[ineig]
            abs(aij) > epsi*sqrt(aii*ajj)
        end
        coupling_strength = (node,ineig) -> begin
            abs(vals[ineig])
        end

        # Initialization
        for node in 1:nnodes
            neig_range = jagged_range(node_to_neigs,node)
            isolated_node = count(i->neigs[i]!=node,neig_range) == 0
            if isolated_node
                node_to_aggregate[node] = isolated
            end
        end
        # Step 1
        aggregate = Ti(0)
        for node in 1:nnodes
            if node_to_aggregate[node] != pending
                continue
            end
            neig_range = jagged_range(node_to_neigs,node)
            all_pending = true
            for ineig in neig_range
                neig = neigs[ineig]
                if neig == node || !strongly_connected(node,ineig)
                    continue
                end
                all_pending &= (node_to_aggregate[neig] == pending)
            end
            if !all_pending
                continue
            end
            aggregate += Ti(1)
            node_to_aggregate[node] = aggregate
            for ineig in neig_range
                neig = neigs[ineig]
                if neig == node || !strongly_connected(node,ineig)
                    continue
                end
                node_to_aggregate[neig] = aggregate
            end
        end
        # Step 2
        copy!(node_to_old_aggregate,node_to_aggregate)
        for node in 1:nnodes
            if node_to_aggregate[node] != pending
                continue
            end
            strength = zero(typeof_strength)
            neig_range = jagged_range(node_to_neigs, node)
            for ineig in neig_range
                neig = neigs[ineig]
                if neig == node || !strongly_connected(node,ineig)
                    continue
                end
                neig_aggregate = node_to_old_aggregate[neig]
                if neig_aggregate != pending && neig_aggregate != isolated
                    neig_strength = coupling_strength(node,ineig)
                    if neig_strength > strength
                        strength = neig_strength
                        node_to_aggregate[node] = neig_aggregate
                    end
                end
            end
        end

        # Step 3
        for node in 1:nnodes
            if node_to_aggregate[node] != pending
                continue
            end
            aggregate += Ti(1)
            node_to_aggregate[node] = aggregate
            # neigs = node_to_neigs[node]
            neig_range = jagged_range(node_to_neigs, node)
            for ineig in neig_range
                neig = neigs[ineig]
                if neig == node || !strongly_connected(node,ineig)
                    continue
                end
                neig_aggregate = node_to_old_aggregate[neig]
                if neig_aggregate == pending || neig_aggregate == isolated
                    node_to_aggregate[neig] = aggregate
                end
            end
        end
        naggregates = aggregate

        if nnodes == 1
            node_to_aggregate .= 1
            naggregates = 1
        end
        node_to_aggregate, 1:naggregates
    end

    function aggregate(A::PSparseMatrix,diagA=dense_diag(A);kwargs...)
        # This is the vanilla "uncoupled" strategy from "Parallel Smoothed Aggregation Multigrid : Aggregation Strategies on Massively Parallel Machines"
        # TODO: implement other more advanced strategies
        @assert A.assembled
        node_to_aggregate_data, local_ranges = map((A,diagA)->aggregate(A,diagA;kwargs...),own_own_values(A),own_values(diagA)) |> tuple_of_arrays
        nown = map(length,local_ranges)
        n_aggregates = sum(nown)
        nparts = length(nown)
        aggregate_partition = variable_partition(nown,n_aggregates)
        node_partition = partition(axes(A,1))
        map(map_own_to_global!,node_to_aggregate_data,aggregate_partition)
        node_to_aggregate = PVector(node_to_aggregate_data,node_partition)
        node_to_aggregate, PRange(aggregate_partition)
    end

    function constant_prolongator(T,node_to_aggregate::PVector,aggregates::PRange,n_nullspace_vecs)
        if n_nullspace_vecs != 1
            error("case not implemented yet")
        end
        function setup_triplets(node_to_aggregate,nodes)
            myI = UnitRange{Ti}(1:local_length(nodes))
            myJ = node_to_aggregate
            myV = ones(length(node_to_aggregate))
            (myI,myJ,myV)
        end
        node_partition = partition(axes(node_to_aggregate,1))
        I,J,V = map(setup_triplets,partition(node_to_aggregate),node_partition) |> tuple_of_arrays
        aggregate_partition = partition(aggregates)
        J_owner = find_owner(aggregate_partition,J)
        aggregate_partition = map(union_ghost,aggregate_partition,J,J_owner)
        map(map_global_to_local!,J,aggregate_partition)
        P0 = psparse(T,I,J,V,node_partition,aggregate_partition;assembled=true,indices=:local) |> fetch
        P0
    end
end


