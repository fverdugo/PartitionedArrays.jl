
"""
    struct PRange{A,B}

`PRange` (partitioned range) is a type representing a range of indices `1:n_global`
distributed into several parts. The indices in the range `1:n_global` are called the
*global* indices. Each global index is *owned* by one part and only one part.
The set of indices owned by a part are called the *own* indices of this part.
Each part contains a second set of indices called the *ghost* indices. 
The set of ghost indices in a given part is an arbitrary subset
of the global indices that are owned by other parts. The union of the own and ghost
indices is referred to as the *local* indices of this part.
The sets of own, ghost, and local indices are stored using vector-like containers,
which equips them with a certain order. Thus, the `i`-th own index
in a part is the one being stored at index `i` in the array that contains
the own indices in this part.
The same rationale applies for ghost and local indices.

# Properties
- `indices::A`: Array-like object with `length(indices)` equal to the number of parts in the partitioned range.


The item `indices[i]` is an object that contains information about the own, ghost, and local indices of part number `i`.
`typeof(indices[i])` is a type that
implements the methods of the [`AbstractLocalIndices`](@ref) interface. Use this
interface to access the underlying information about own, ghost, and local indices.

# Supertype hierarchy

    PRange{A} <: AbstractUnitRange{Int}

"""
struct PRange{A} <: AbstractUnitRange{Int}
    partition::A
    @doc """
        PRange(n_global,indices)

    Build an instance of [`PRange`](@ref) from the underlying properties
    `n_global` and `indices`.

    # Examples
   
        julia> using PartitionedArrays
        
        julia> rank = LinearIndices((2,));
        
        julia> indices = map(rank) do rank
                   if rank == 1
                       LocalIndices(8,1,[1,2,3,4,5],Int32[1,1,1,1,2])
                   else
                       LocalIndices(8,2,[4,5,6,7,8],Int32[1,2,2,2,2])
                   end
               end;
        
        julia> pr = PRange(8,indices)
        1:1:8
        
        julia> local_to_global(pr)
        2-element Vector{Vector{Int64}}:
         [1, 2, 3, 4, 5]
         [4, 5, 6, 7, 8]
    """
    function PRange(indices)
        A = typeof(indices)
        new{A}(indices)
    end
end
partition(a::PRange) = a.partition
Base.first(a::PRange) = 1
Base.last(a::PRange) = getany(map(global_length,partition(a)))
function Base.show(io::IO,k::MIME"text/plain",data::PRange)
    np = length(partition(data))
    map_main(partition(data)) do indices
        println(io,"1:$(global_length(indices)) partitioned into $(np) parts")
    end
end

function matching_local_indices(a::PRange,b::PRange)
    partition(a) === partition(b) && return true
    c = map(matching_local_indices,partition(a),partition(b))
    reduce(&,c,init=true)
end

function matching_own_indices(a::PRange,b::PRange)
    partition(a) === partition(b) && return true
    c = map(matching_own_indices,partition(a),partition(b))
    reduce(&,c,init=true)
end

function matching_ghost_indices(a::PRange,b::PRange)
    partition(a) === partition(b) && return true
    c = map(matching_ghost_indices,partition(a),partition(b))
    reduce(&,c,init=true)
end

##prange(f,args...) = PRange(f(args...))

"""
    global_length(pr::PRange)

Equivalent to `map(global_length,pr.indices)`.
"""
global_length(pr::PRange) = map(local_length,partition(pr))

"""
    local_length(pr::PRange)

Equivalent to `map(local_length,pr.indices)`.
"""
local_length(pr::PRange) = map(local_length,partition(pr))

"""
    own_length(pr::PRange)

Equivalent to `map(own_length,pr.indices)`.
"""
own_length(pr::PRange) = map(own_length,partition(pr))

"""
    local_to_global(pr::PRange)

Equivalent to `map(local_to_global,pr.indices)`.
"""
local_to_global(pr::PRange) = map(local_to_global,partition(pr))

"""
    own_to_global(pr::PRange)

Equivalent to `map(own_to_global,pr.indices)`.
"""
own_to_global(pr::PRange) = map(own_to_global,partition(pr))

"""
    ghost_to_global(pr::PRange)

Equivalent to `map(ghost_to_global,pr.indices)`.
"""
ghost_to_global(pr::PRange) = map(ghost_to_global,partition(pr))

"""
    local_to_owner(pr::PRange)

Equivalent to `map(local_to_owner,pr.indices)`.
"""
local_to_owner(pr::PRange) = map(local_to_owner,partition(pr))

"""
    own_to_owner(pr::PRange)

Equivalent to `map(own_to_owner,pr.indices)`.
"""
own_to_owner(pr::PRange) = map(own_to_owner,partition(pr))

"""
    ghost_to_owner(pr::PRange)

Equivalent to `map(ghost_to_owner,pr.indices)`.
"""
ghost_to_owner(pr::PRange) = map(ghost_to_owner,partition(pr))

"""
    global_to_local(pr::PRange)

Equivalent to `map(global_to_local,pr.indices)`.
"""
global_to_local(pr::PRange) = map(global_to_local,partition(pr))

"""
    global_to_own(pr::PRange)

Equivalent to `map(global_to_own,pr.indices)`.
"""
global_to_own(pr::PRange) = map(global_to_own,partition(pr))

"""
    global_to_ghost(pr::PRange)

Equivalent to `map(global_to_ghost,pr.indices)`.
"""
global_to_ghost(pr::PRange) = map(global_to_ghost,partition(pr))

"""
    own_to_local(pr::PRange)

Equivalent to `map(own_to_local,pr.indices)`.
"""
own_to_local(pr::PRange) = map(own_to_local,partition(pr))

"""
    ghost_to_local(pr::PRange)

Equivalent to `map(ghost_to_local,pr.indices)`.
"""
ghost_to_local(pr::PRange) = map(ghost_to_local,partition(pr))

"""
    local_to_own(pr::PRange)

Equivalent to `map(local_to_own,pr.indices)`.
"""
local_to_own(pr::PRange) = map(local_to_own,partition(pr))

"""
    local_to_ghost(pr::PRange)

Equivalent to `map(local_to_ghost,pr.indices)`.
"""
local_to_ghost(pr::PRange) = map(local_to_ghost,partition(pr))

