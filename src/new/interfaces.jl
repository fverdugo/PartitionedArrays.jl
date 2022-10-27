
const SCALAR_INDEXING = Ref(:warn)

function scalar_indexing(b)
    @assert b in (:warn,:error,:allow)
    SCALAR_INDEXING[] = b
    b
end

function scalar_indexing_error(a)
    if SCALAR_INDEXING[] === :warn
        @warn "Scalar indexing on $(nameof(typeof(a))) is discuraged for performance reasons."
    elseif SCALAR_INDEXING[] === :error
        error("Scalar indexing on $(nameof(typeof(a))) is not allowed for performance reasons.")
    end
    nothing
end

"""
    linear_indices(backend,shape)

Generate an array equal to `LinearIndices(shape)`, but whose type
can depend on the given back-end `backend`.
Return `LinearIndices(shape)` by default.
"""
function linear_indices(backend,shape)
    LinearIndices(shape)
end

"""
    cartesian_indices(backend,shape)

Generate an array equal to `CartesianIndices(shape)`, but whose type
can depend on the given back-end `backend`.
Return `CartesianIndices(shape)` by default.
"""
function cartesian_indices(backend,shape)
    CartesianIndices(shape)
end

"""
    linear_indices(a)

Generate an array equal to `LinearIndices(a)`, but whose type
and can depend on `a`. Return `LinearIndices(a)` by default.
"""
linear_indices(a) = LinearIndices(a)

"""
    cartesian_indices(a)

Generate an array equal to `CartesianIndices(a)`, but whose type
and can depend on `a`. Return `CartesianIndices(a)` by default.
"""
cartesian_indices(a) = CartesianIndices(a)

"""
    with_backend(f,backend)

Run all initialization steps required by the back-end `backend`,
call `f(backend)`, and finally run all finalization steps required for
the back-end `backend`. Return the result of `f(backend)`.

This is the safest way of running `f(backned)`.
"""
function with_backend(f,backend)
    f(backend)
end

"""
    unpack(a)

Convert the array of tuples `a` into a tuple of arrays.

# Examples

    julia> using PartitionedArrays

    julia> a = [(1,2),(3,4),(5,6)]
    3-element Vector{Tuple{Int64, Int64}}:
     (1, 2)
     (3, 4)
     (5, 6)

    julia> b,c = unpack(a)
    ([1, 3, 5], [2, 4, 6])

"""
function unpack(a)
  if eltype(a) <: Tuple{<:Any}
    x = map(first,a)
    (x,)
  else
    x, y = unpack_first_and_tail(a)
    (x,unpack(y)...)
  end
end

function unpack_first_and_tail(a)
  x = map(first,a)
  y = map(Base.tail,a)
  x, y
end

"""
    map_one(f,args...;kwargs...)

Like `map(f,args...)` but only apply `f` to one component of the arrays
in `args` (the first component by default). Set the remaining entries to `nothing` by default.

# Optional key-word arguments

- `source = 1`: The linear index of the component to map
- `otherwise = (args...)->nothing`: The function to apply when mapping indices different from `source`.

# Examples

    julia> using PartitionedArrays

    julia> a = [1,2,3,4]
    4-element Vector{Int64}:
     1
     2
     3
     4

    julia> map_one(-,a,source=2)
    4-element Vector{Union{Nothing, Int64}}:
       nothing
     -2
       nothing
       nothing
"""
function map_one(f,args...;otherwise=(args...)->nothing,source=1)
    rank = linear_indices(first(args))
    map(rank,args...) do rank,args...
        if rank == source
            f(args...)
        else
            otherwise(args...)
        end
    end
end

"""
    gather!(rcv,snd;destination=1)

Copy the elements of array `snd` into the first component of the array `rcv` and return `rcv`.
Another component different from the
first one can be used to store the result by setting the optional key-word argument `destination`.

The result array `rcv` can be allocated with the helper function [`allocate_gather`](@ref).

# Examples

    julia> using PartitionedArrays

    julia> snd = collect(1:3)
    3-element Vector{Int64}:
     1
     2
     3

    julia> rcv = allocate_gather(snd,destination=3)
    3-element Vector{Vector{Int64}}:
     []
     []
     [140002225987696, 140002162818736, 140002162818752]

    julia> gather!(rcv,snd,destination=3)
    3-element Vector{Vector{Int64}}:
     []
     []
     [1, 2, 3]
"""
function gather!(rcv,snd;destination=1)
    @assert size(rcv) == size(snd)
    @assert size(rcv[destination]) == size(snd)
    for i in eachindex(snd)
        rcv[destination][i] = snd[i]
    end
    rcv
end

"""
    allocate_gather(snd;destination=1)

Allocate an array to be used in the first argument of [`gather!`](@ref).
"""
function allocate_gather(snd;destination=1)
    s = size(snd)
    T = eltype(snd)
    N = ndims(snd)
    f = (snd)->Array{T,N}(undef,s)
    g = (snd)->Array{T,N}(undef,ntuple(i->0,N))
    rcv = map_one(f,snd;otherwise=g,source=destination)
    rcv
end

function allocate_gather(snd::AbstractVector{<:AbstractVector};destination=1)
    l = map(length,snd)
    l_dest = gather(l;destination)
    function f(l,snd)
        ptrs = exclusive_scan!(pushfirst!(l,zero(eltype(l))))
        ndata = ptrs[end]-1
        data = Vector{eltype(snd)}(undef,ndata)
        JaggedArray{eltype(snd),Int32}(data,ptrs)
    end
    function g(l,snd)
        ptrs = Vector{Int32}(undef,1)
        data = Vector{eltype(snd)}(undef,0)
        JaggedArray(data,ptrs)
    end
    rcv = map_one(f,l_dest,snd;otherwise=g,source=destination)
    rcv
end

"""
    gather(snd;destination=1)

Return an array whose first entry contains a copy of the elements of array `snd`.
Another component different from the
first one can be used to store the result by setting the optional key-word argument `destination`.

# Examples

    julia> using PartitionedArrays

    julia> snd = collect(1:3)
    3-element Vector{Int64}:
     1
     2
     3

    julia> gather(snd,destination=3)
    3-element Vector{Vector{Int64}}:
     []
     []
     [1, 2, 3]
"""
function gather(snd;destination=1)
  rcv = allocate_gather(snd;destination)
  gather!(rcv,snd;destination)
  rcv
end

