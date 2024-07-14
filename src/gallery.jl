
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

