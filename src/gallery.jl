
function laplace_matrix_fdm(
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

function laplace_matrix_fem(
        nodes_per_dir,
        parts_per_dir,
        parts;
        index_type::Type{Ti} = Int64,
        value_type::Type{Tv} = Float64,
    ) where {Ti,Tv}

    cells_per_dir = nodes_per_dir .+ 1

    function is_boundary_node(node_1d,nodes_1d)
        !(node_1d in 1:nodes_1d)
    end
    function ref_matrix(D,::Type{value_type}) where value_type
        if D == 1
            value_type[1 -1; -1 1]
        elseif D == 2
            a = 2/3
            b = -1/3
            c = -1/6
            value_type[a c c b; c a b c; c b a c; b c c a]
        else
            error()
        end
    end
    function setup(cells,::Type{index_type},::Type{value_type}) where {index_type,value_type}
        D = length(nodes_per_dir)
        α = value_type(prod(i->(i+1),nodes_per_dir))
        Aref = α*ref_matrix(D,value_type)#ones(value_type,2^D,2^D)
        cell_to_cartesian_cell = CartesianIndices(cells_per_dir)
        first_cartesian_cell = cell_to_cartesian_cell[first(cells)]
        last_cartesian_cell = cell_to_cartesian_cell[last(cells)]
        ranges = map(:,Tuple(first_cartesian_cell),Tuple(last_cartesian_cell))
        cartesian_cells = CartesianIndices(ranges)
        ttt = ntuple(d->2,Val{D}())
        cartesian_local_nodes = CartesianIndices(ttt)
        offset = CartesianIndex(ttt)
        cartesian_local_node_to_local_node = LinearIndices(cartesian_local_nodes)
        cartesian_node_to_node = LinearIndices(nodes_per_dir)
        nnz = 0 
        for cartesian_cell in cartesian_cells
            for cartesian_local_node_i in cartesian_local_nodes
                local_node_i = cartesian_local_node_to_local_node[cartesian_local_node_i]
                cartesian_node_i = cartesian_cell .+ cartesian_local_node_i .- offset
                boundary = any(map(is_boundary_node,Tuple(cartesian_node_i),nodes_per_dir))
                if boundary
                    continue
                end
                node_i = cartesian_node_to_node[cartesian_node_i]
                for cartesian_local_node_j in cartesian_local_nodes
                    local_node_j = cartesian_local_node_to_local_node[cartesian_local_node_j]
                    cartesian_node_j = cartesian_cell .+ cartesian_local_node_j .- offset
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
                cartesian_node_i = cartesian_cell .+ cartesian_local_node_i .- offset
                boundary = any(map(is_boundary_node,Tuple(cartesian_node_i),nodes_per_dir))
                if boundary
                    continue
                end
                node_i = cartesian_node_to_node[cartesian_node_i]
                for cartesian_local_node_j in cartesian_local_nodes
                    local_node_j = cartesian_local_node_to_local_node[cartesian_local_node_j]
                    cartesian_node_j = cartesian_cell .+ cartesian_local_node_j .- offset
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


