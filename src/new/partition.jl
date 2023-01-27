"""
    uniform_partition(ranks,np,n[,ghost[,periodic]])

Generate an `N` dimensional
block partition with a (roughly) constant block size.

# Arguments
- `ranks`: Array containing the distribution of ranks.
-  `np::NTuple{N}`: Number of parts per direction.
-  `n::NTuple{N}`: Number of global indices per direction.
-  `ghost::NTuple{N}=ntuple(i->false,N)`: Use or not ghost indices per direction.
-  `periodic::NTuple{N}=ntuple(i->false,N)`: Use or not periodic boundaries per direction.

For convenience, one can also provide scalar inputs instead tuples
to create 1D block partitions.

# Examples

2D partition of 4x4 indices into 2x2 parts with ghost

    julia> using PartitionedArrays
    
    julia> rank = LinearIndices((4,));
    
    julia> pr = uniform_partition(rank,(2,2),(4,4),(true,true))
    1:1:16
    
    julia> local_to_global(pr)
    4-element Vector{PartitionedArrays.BlockPartitionLocalToGlobal{2, Vector{Int32}}}:
     [1, 2, 3, 5, 6, 7, 9, 10, 11]
     [2, 3, 4, 6, 7, 8, 10, 11, 12]
     [5, 6, 7, 9, 10, 11, 13, 14, 15]
     [6, 7, 8, 10, 11, 12, 14, 15, 16]

"""
function uniform_partition(rank,np,n,args...)
    @assert prod(np) == length(rank)
    indices = map(rank) do rank
        block_with_constant_size(rank,np,n,args...)
    end
    if length(args) == 0
        map(indices) do indices
            cache = assembly_cache(indices)
            copy!(cache,empty_assembly_cache())
        end
    else
        assembly_neighbors(indices;symmetric=true)
    end
    indices
end

"""
    uniform_partition(ranks,n::Integer[,ghost::Bool[,periodic::Bool]])

Generate an  1d dimensional
block partition with a (roughly) constant block size by inferring the number of parts to use from `ranks`.

# Arguments
- `ranks`: Array containing the distribution of ranks. The number of parts is taken as `length(ranks)`.
-  `n`: Number of global indices.
-  `ghost`: Use or not ghost indices.
-  `periodic`: Use or not periodic boundaries.
"""
function uniform_partition(rank,n::Integer)
    uniform_partition(rank,length(rank),n)
end

function uniform_partition(rank,n::Integer,ghost::Bool,periodic::Bool=false)
    uniform_partition(rank,length(rank),n,ghost,periodic)
end

function uniform_partition(rank,np::Integer,n::Integer)
    uniform_partition(rank,(np,),(n,))
end

function uniform_partition(rank,np::Integer,n::Integer,ghost::Bool,periodic::Bool=false)
    uniform_partition(rank,(np,),(n,),(ghost,),(periodic,))
end

function block_with_constant_size(rank,np,n)
    N = length(n)
    p = CartesianIndices(np)[rank]
    ghost = GhostIndices(prod(n))
    LocalIndicesWithConstantBlockSize(p,np,n,ghost)
end

function block_with_constant_size(rank,np,n,ghost,periodic=map(i->false,ghost))
    N = length(n)
    p = CartesianIndices(np)[rank]
    own_ranges = map(local_range,Tuple(p),np,n)
    local_ranges = map(local_range,Tuple(p),np,n,ghost,periodic)
    owners = map(Tuple(p),own_ranges,local_ranges) do p,or,lr
        owners = zeros(Int32,length(lr))
        for i in 1:length(lr)
            if lr[i] in or
                owners[i] = p
            end
        end
        if owners[1] == 0
            owners[1] = p-1
        end
        if owners[end] == 0
            owners[end] = p+1
        end
        owners
    end
    n_ghost = 0
    cis = CartesianIndices(map(length,local_ranges))
    predicate(p,i,owners) = owners[i] == p
    for ci in cis
        flags = map(predicate,Tuple(p),Tuple(ci),owners)
        if !all(flags)
            n_ghost += 1
        end
    end
    ghost_to_global = zeros(Int,n_ghost)
    ghost_to_owner = zeros(Int32,n_ghost)
    n_local = prod(map(length,local_ranges))
    perm = zeros(Int32,n_local)
    i_ghost = 0
    i_own = 0
    n_own = prod(map(length,own_ranges))
    lis = CircularArray(LinearIndices(n))
    local_cis = CartesianIndices(local_ranges)
    owner_lis = CircularArray(LinearIndices(np))
    for (i,ci) in enumerate(cis)
        flags = map(predicate,Tuple(p),Tuple(ci),owners)
        if !all(flags)
            i_ghost += 1
            ghost_to_global[i_ghost] = lis[local_cis[i]]
            o = map(getindex,owners,Tuple(ci))
            o_ci = CartesianIndex(o)
            ghost_to_owner[i_ghost] = owner_lis[o_ci]
            perm[i] = i_ghost + n_own
        else
            i_own += 1
            perm[i] = i_own
        end
    end
    ghostids = GhostIndices(prod(n),ghost_to_global,ghost_to_owner)
    ids = LocalIndicesWithConstantBlockSize(p,np,n,ghostids)
    PermutedLocalIndices(ids,perm)
end

"""
    variable_partition(n_own,n_global[;start])

Build a 1D variable-size block partition.

# Arguments

-  `n_own::AbstractArray{<:Integer}`: Array containing the block size for each part.
-  `n_global::Integer`: Number of global indices. It should be equal to `sum(n_own)`.
-  `start::AbstractArray{Int}=scan(+,n_own,type=:exclusive,init=1)`: First global index in each part.

We ask the user to provide `n_global` and (optionally) `start` since discovering them requires communications.

# Examples

    julia> using PartitionedArrays
    
    julia> rank = LinearIndices((4,));
    
    julia> n_own = [3,2,2,3];
    
    julia> pr = variable_partition(n_own,sum(n_own))
    1:1:10
    
    julia> own_to_global(pr)
    4-element Vector{PartitionedArrays.BlockPartitionOwnToGlobal{1}}:
     [1, 2, 3]
     [4, 5]
     [6, 7]
     [8, 9, 10]

"""
function variable_partition(
    n_own,
    n_global,
    ghost=false,
    periodic=false;
    start=scan(+,n_own,type=:exclusive,init=one(eltype(n_own))))
    rank = linear_indices(n_own)
    if ghost == true || periodic == true
        error("This case is not yet implemented.")
    end
    n_parts = length(n_own)
    indices = map(rank,n_own,start) do rank,n_own,start
        p = CartesianIndex((rank,))
        np = (n_parts,)
        n = (n_global,)
        ranges = ((1:n_own).+(start-1),)
        ghost = GhostIndices(n_global)
        indices = LocalIndicesWithVariableBlockSize(p,np,n,ranges,ghost)
        # This should be changed when including ghost
        cache = assembly_cache(indices)
        copy!(cache,empty_assembly_cache())
        indices
    end
    indices
end

function local_range(p,np,n,ghost=false,periodic=false)
    l = n ÷ np
    offset = l * (p-1)
    rem = n % np
    if rem >= (np-p+1)
        l = l + 1
        offset = offset + p - (np-rem) - 1
    end
    start = 1+offset
    stop = l+offset
    if ghost && np != 1
        if periodic || p!=1
            start -= 1
        end
        if periodic || p!=np
            stop += 1
        end
    end
    start:stop
end

function boundary_owner(p,np,n,ghost=false,periodic=false)
    start = p
    stop = p
    if ghost && np != 1
        if periodic || p!=1
            start -= 1
        end
        if periodic || p!=np
            stop += 1
        end
    end
    (start,p,stop)
end

