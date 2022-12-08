
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


