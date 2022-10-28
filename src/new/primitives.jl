
const SCALAR_INDEXING = Ref(:error)

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

- `index = 1`: The linear index of the component to map
- `otherwise = (args...)->nothing`: The function to apply when mapping indices different from `index`.

# Examples

    julia> using PartitionedArrays

    julia> a = [1,2,3,4]
    4-element Vector{Int64}:
     1
     2
     3
     4

    julia> map_one(-,a,index=2)
    4-element Vector{Union{Nothing, Int64}}:
       nothing
     -2
       nothing
       nothing
"""
function map_one(f,args...;otherwise=(args...)->nothing,index=1)
    if isa(index,Integer)
        rank = linear_indices(first(args))
        map(rank,args...) do rank,args...
            if rank == index
                f(args...)
            else
                otherwise(args...)
            end
        end
    else
      @assert index === :all
      map(f,args...)
    end
end

"""
    map_one!(f,dest,args...;kwargs...)

Like [`map_one`](@ref) but store the result in-place in `dest`.
"""
function map_one!(f,args...;otherwise=(args...)->nothing,index=1)
    if isa(index,Integer)
        rank = linear_indices(first(args))
        map!(args...,rank) do x...
            rank = x[end]
            args = Base.front(x)
            if rank == index
                f(args...)
            else
                otherwise(args...)
            end
        end
    else
      @assert index === :all
      map!(f,args...)
    end
end

"""
    gather!(rcv,snd;destination=1)

Copy the elements of array `snd` into the first component of the array `rcv` and return `rcv`.
Another component different from the
first one can be used to store the result by setting the optional key-word argument `destination`.
Setting `destination=:all`, will store the result in all entries of `rcv` resulting in a "gather all"
operation.

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
    if isa(destination,Integer)
        @assert size(rcv[destination]) == size(snd)
        for i in eachindex(snd)
            rcv[destination][i] = snd[i]
        end
    else
        @assert destination === :all
        for j in eachindex(rcv)
            for i in eachindex(snd)
                rcv[j][i] = snd[i]
            end
        end
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
    if isa(destination,Integer)
        g = (snd)->Array{T,N}(undef,ntuple(i->0,N))
        rcv = map_one(f,snd;otherwise=g,index=destination)
    else
        @assert destination === :all
        rcv = map(f,snd)
    end
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
    if isa(destination,Integer)
        function g(l,snd)
            ptrs = Vector{Int32}(undef,1)
            data = Vector{eltype(snd)}(undef,0)
            JaggedArray(data,ptrs)
        end
        rcv = map_one(f,l_dest,snd;otherwise=g,index=destination)
    else
        @assert destination === :all
        rcv = map(f,l_dest,snd)
    end
    rcv
end

"""
    gather(snd;destination=1)

Return an array whose first entry contains a copy of the elements of array `snd`.
Another component different from the
first one can be used to store the result by setting the optional key-word argument `destination`.
Setting `destination=:all`, will store the result in all entries of `rcv` resulting in a "gather all"
operation.

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

    julia> gather(snd,destination=:all)
    3-element Vector{Vector{Int64}}:
     [1, 2, 3]
     [1, 2, 3]
     [1, 2, 3]
"""
function gather(snd;destination=1)
  rcv = allocate_gather(snd;destination)
  gather!(rcv,snd;destination)
  rcv
end

