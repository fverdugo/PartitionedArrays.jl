
"""
    local_range(p, n, np, ghost=false, periodic=false)

Return the local range of indices in the component number `p`
of a uniform partition of indices `1:n` into `np` parts.
If `ghost==true` then include a layer of
"ghost" entries. If `periodic == true` the ghost layer is created assuming
periodic boundaries in the range  `1:n`. In this case, the first ghost
index is `0` for `p==1` and the last ghost index is `n+1`  for `p==np`

# Examples

## Without ghost entries

    julia> using PartitionedArrays
    
    julia> local_range(1,10,3)
    1:3

    julia> local_range(2,10,3)
    4:6

    julia> local_range(3,10,3)
    7:10

## With ghost entries

    julia> using PartitionedArrays
    
    julia> local_range(1,10,3,true)
    1:4

    julia> local_range(2,10,3,true)
    3:7

    julia> local_range(3,10,3,true)
    6:10

## With periodic boundaries

    julia> using PartitionedArrays
    
    julia> local_range(1,10,3,true,true)
    0:4

    julia> local_range(2,10,3,true,true)
    3:7

    julia> local_range(3,10,3,true,true)
    6:11
"""
function local_range(p,n,np,ghost=false,periodic=false)

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

"""
    struct UniformBlockPartition{N}

Array-like type representing a uniform block partition of `N` dimensions.

# Properties
-  `np::NTuple{N,Int}`: Number of parts in each direction
-  `n::NTuple{N,Int}`: Number of items partitioned in each direction

# Supertype hierarchy

    UniformBlockPartition{N} <: AbstractArray{T,N}

where `T=CartesianIndices{N,NTuple{N,UnitRange{Int64}}}`

"""
struct UniformBlockPartition{N} <: AbstractArray{CartesianIndices{N,NTuple{N,UnitRange{Int64}}},N}
    np::NTuple{N,Int}
    n::NTuple{N,Int}
    @doc """
        UniformBlockPartition(np::NTuple{N,Int},n::NTuple{N,Int}) where N

Build a uniform block partition resulting from spliting `n` items per direction
into `np` parts per direction. The result is an array-like object `a` such that
`a[p]` contains the Cartesian indices associated with the block in part `p`.  

# Examples

    julia> using PartitionedArrays
    
    julia> partition = UniformBlockPartition((3,),(10,))
    3-element UniformBlockPartition{1}:
     CartesianIndices((1:3,))
     CartesianIndices((4:6,))
     CartesianIndices((7:10,))
      
    julia> partition = UniformBlockPartition((3,2),(10,10))
    3×2 UniformBlockPartition{2}:
     CartesianIndices((1:3, 1:5))   CartesianIndices((1:3, 6:10))
     CartesianIndices((4:6, 1:5))   CartesianIndices((4:6, 6:10))
     CartesianIndices((7:10, 1:5))  CartesianIndices((7:10, 6:10))
    """
    function UniformBlockPartition(np::NTuple{N,Int},n::NTuple{N,Int}) where N
        new{N}(np,n)
    end
end

Base.size(a::UniformBlockPartition) = a.np
Base.IndexStyle(::Type{<:UniformBlockPartition}) = IndexCartesian()
function Base.getindex(a::UniformBlockPartition{N},i::Vararg{Int,N}) where N
    CartesianIndices(map(local_range,i,a.n,a.np))
end

"""
    struct BlockPartition{N}

Array-like type representing a (possibly non-uniform) block partition of `N` dimensions.

# Properties
- `start::NTuple{N,Vector{Int}}`: `start[d][i]:(s[d][i+1]-1)` is the range of indices corresponding to the `i`-th block in direction `d`. `length(start[d])`  is the number of blocks in direction `d` plus one, being `s[d][end]-1` the number of items partitioned in direction `d`.

# Supertype hierarchy

    BlockPartition{N} <: AbstractArray{T,N}

where `T=CartesianIndices{N,NTuple{N,UnitRange{Int64}}}`

"""
struct BlockPartition{N} <: AbstractArray{CartesianIndices{N,NTuple{N,UnitRange{Int64}}},N}
    start::NTuple{N,Vector{Int}}
    @doc """
        BlockPartition(start::NTuple{N,Vector{Int}}) where N

Create a (possibly non-uniform) block partition of `N` dimensions. `start` encodes the range
of the blocks in each direction. I.e., `start[d][i]:(s[d][i+1]-1)` is the range of indices
corresponding to the `i`-th block in direction `d`. `length(start[d])`  is the number
of blocks in direction `d` plus one, being `s[d][end]-1` the number of items partitioned
in direction `d`. The result is an array-like object `a` such that `a[p]` contains the Cartesian indices associated with the block in part `p`.

# Examples

    julia> using PartitionedArrays
    
    julia> BlockPartition(([1,3,7,11],))
    3-element BlockPartition{1}:
     CartesianIndices((1:2,))
     CartesianIndices((3:6,))
     CartesianIndices((7:10,))

    """
    BlockPartition(start::NTuple{N,Vector{Int}}) where N = new{N}(start)
end

Base.size(a::BlockPartition) = map(i->length(i)-1,a.start)
Base.IndexStyle(::Type{<:BlockPartition}) = IndexCartesian()
function Base.getindex(a::BlockPartition{N},i::Vararg{Int,N}) where N
    ranges = map(a.start,i) do start,i
        start[i]:(start[i+1]-1)
    end
    CartesianIndices(ranges)
end

#function block_start(blocks,n=sum(blocks))
#    init = one(eltype(blocks))
#    type = :exclusive
#    start = collect(scan(+,blocks;type,init))
#    push!(start,n+one(typeof(n)))
#    start
#end


