
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


