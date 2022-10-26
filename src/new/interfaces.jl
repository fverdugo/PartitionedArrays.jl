
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
Return `LinearIndices(a)` by default.
"""
function linear_indices(backend,shape)
    LinearIndices(shape)
end

"""
    cartesian_indices(backend,shape)

Generate an array equal to `CartesianIndices(shape)`, but whose type
can depend on the given back-end `backend`.
Return `CartesianIndices(a)` by default.
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

# Example

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
    map_first(f,args...)

Like `map(f,args...)` but only apply `f` to the first entries of the arrays
in `args`. Set the remaining entries to `nothing`.

# Examples

    julia> using PartitionedArrays

    julia> a = [1,2,3,4]
    4-element Vector{Int64}:
     1
     2
     3
     4
    
    julia> map_first(-,a)
    4-element Vector{Union{Nothing, Int64}}:
     -1
       nothing
       nothing
       nothing
    
"""
function map_first(f,args...;init=(args...)->nothing,destination=1)
    rank = linear_indices(first(args))
    map(rank,args...) do rank,args...
        if rank == destination
            f(args...)
        else
            init(args...)
        end
    end
end

function gather!(rcv,snd;destination=1)
  @assert size(rcv) == size(snd)
  @assert size(rcv[destination]) == size(snd)
  for i in eachindex(snd)
      rcv[destination][i] = snd[i]
  end
  rcv
end

function allocate_gather(snd;destination=1)
  s = size(snd)
  T = eltype(snd)
  N = ndims(snd)
  f = (snd)->Array{T,N}(undef,s)
  g = (snd)->Array{T,N}(undef,ntuple(i->0,N))
  rcv = map_first(f,snd;init=g,destination)
  rcv
end

function gather(snd;destination=1)
  rcv = allocate_gather(snd;destination)
  gather!(rcv,snd;destination)
  rcv
end

