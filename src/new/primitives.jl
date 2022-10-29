
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
    loop(f,args...;index=:all)

Call `f` element-wise on the arrays in `args`, but return `nothing`
instead of the result. The optional key-word argument `index`
can be set to a linear index if we just want to visit a single
index in the loop.
"""
function loop(f,args...;index=:all)
    map_one(f,args...;index)
    nothing
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

Copy the elements of array `snd` into the first component of the array of arrays `rcv` and return `rcv`.
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
        @assert length(rcv[destination]) == length(snd)
        for i in 1:length(snd)
            rcv[destination][i] = copy(snd[i])
        end
    else
        @assert destination === :all
        for j in eachindex(rcv)
            for i in 1:length(snd)
                rcv[j][i] = copy(snd[i])
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
    n = length(snd)
    T = eltype(snd)
    f = (snd)->Vector{T}(undef,n)
    if isa(destination,Integer)
        g = (snd)->Vector{T}(undef,0)
        rcv = map_one(f,snd;otherwise=g,index=destination)
    else
        @assert destination === :all
        rcv = map(f,snd)
    end
    rcv
end

function allocate_gather(snd::AbstractArray{<:AbstractVector};destination=1)
    l = map(length,snd)
    l_dest = gather(l;destination)
    function f(l,snd)
        ptrs = prefix_sum!(pushfirst!(l,one(eltype(l))))
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

Return an array whose first entry contains a copy of the elements of array `snd` collected in a vector.
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

"""
    allocate_scatter(snd;source=1)

Allocate an array to be used in the first argument of [`scatter!`](@ref).
"""
function allocate_scatter(snd;source=1)
    @assert source !== :all "Scatter all not implemented"
    T = eltype(eltype(snd))
    similar(snd,T)
end

function allocate_scatter(snd::AbstractArray{<:JaggedArray};source=1)
    @assert source !== :all "Scatter all not implemented"
    T = eltype(eltype(eltype(snd)))
    similar(snd,Vector{T})
end

"""
    scatter!(rcv,snd;source=1)

In-place version of [`scatter`](@ref). The destination array `rcv`
can be generated with the helper function [`allocate_scatter`](@ref).
"""
function scatter!(rcv,snd;source=1)
    @assert source !== :all "Scatter all not implemented"
    @assert length(snd[source]) == length(rcv)
    for i in 1:length(snd)
        rcv[i] = copy(snd[source][i])
    end
    rcv
end

"""
    scatter(snd;source=1)

Copy the items in the collection `snd[source]` into an array of the same
size and container type as `snd`.
This function requires `length(snd[source]) == length(snd)`.

# Examples

    julia> using PartitionedArrays

    julia> a = [Int[],[1,2,3],Int[]]
    3-element Vector{Vector{Int64}}:
     []
     [1, 2, 3]
     []

    julia> scatter(a,source=2)
    3-element Vector{Int64}:
     1
     2
     3

"""
function scatter(snd;source=1)
    @assert source !== :all "Scatter all not implemented"
    rcv = allocate_scatter(snd;source)
    scatter!(rcv,snd;source)
    rcv
end

"""
    emit!(rcv,snd;source=1)

In-place version of [`emit`](@ref). The destination array `rcv`
can be generated with the helper function [`allocate_emit`](@ref).
"""
function emit!(rcv,snd;source=1)
    @assert source !== :all "Emit all not implemented"
    for i in eachindex(rcv)
        rcv[i] = copy(snd[source])
    end
    rcv
end

"""
    allocate_emit(snd;source=1)

Allocate an array to be used in the first argument of [`emit!`](@ref).
"""
function allocate_emit(snd;source=1)
    @assert source !== :all "Scatter all not implemented"
    similar(snd)
end

"""
    emit(snd;source=1)

Copy `snd[source]` into a new array of the same size and type as `snd`.

# Examples

    julia> using PartitionedArrays

    julia> a = [0,0,2,0]
    4-element Vector{Int64}:
     0
     0
     2
     0

    julia> emit(a,source=3)
    4-element Vector{Int64}:
     2
     2
     2
     2
"""
function emit(snd;source=1)
    @assert source !== :all "Scatter all not implemented"
    rcv = allocate_emit(snd;source)
    emit!(rcv,snd;source)
    rcv
end

"""
    inclusive_scan!(op,b,a;init=nothing,destination=1)

In-place version of [`inclusive_scan`](@ref).
The result `b` can be allocated with [allocate_gather](@ref).
"""
function inclusive_scan!(op,b,a;init=nothing,destination=1)
    gather!(b,a;destination)
    map_one(b;index=destination) do b
        if init !== nothing
            b[1] = op(b[1],init)
        end
        n = length(b)
        for i in 1:(n-1)
            b[i+1] = op(b[i+1],b[i])
        end
    end
    b
end

"""
    inclusive_scan(op,a;init=nothing,destination=1)

Do the inclusive scan of the values in `a`. The result is stored
in the entry `destination` of the resulting array. If `init` is provided,
it will be added to all items in the result.

# Examples

    julia> using PartitionedArrays

    julia> a = [2,4,1,3]
    4-element Vector{Int64}:
     2
     4
     1
     3
    
    julia> inclusive_scan(+,a,init=10,destination=3)
    4-element Vector{Vector{Int64}}:
     []
     []
     [12, 16, 17, 20]
     []
    
    julia> inclusive_scan(+,a,init=10,destination=:all)
    4-element Vector{Vector{Int64}}:
     [12, 16, 17, 20]
     [12, 16, 17, 20]
     [12, 16, 17, 20]
     [12, 16, 17, 20]
"""
function inclusive_scan(op,a;init=nothing,destination=1)
    b = allocate_gather(a;destination)
    inclusive_scan!(op,b,a;init,destination)
end

"""
    exclusive_scan!(op,b,a;init,destination=1)

In-place version of [`exclusive_scan`](@ref).
The result `b` can be allocated with [allocate_gather](@ref).
"""
function exclusive_scan!(op,b,a;init,destination=1)
    inclusive_scan!(op,b,a;init,destination)
    map_one(b;index=destination) do b
        right_shift!(b)
        b[1] = init
    end
    b
end

"""
    exclusive_scan(op,a;init,destination=1)

Do the inclusive scan of the values in `a`. The result is stored
in the entry `destination` of the resulting array.
The value `init` should be provided. It will be added to all items in the result
except for the first one that will set to `init`.

# Examples

    julia> using PartitionedArrays

    julia> a = [2,4,1,3]
    4-element Vector{Int64}:
     2
     4
     1
     3
    
    julia> exclusive_scan(+,a,init=10,destination=3)
    4-element Vector{Vector{Int64}}:
     []
     []
     [10, 12, 16, 17]
     []
    
    julia> exclusive_scan(+,a,init=10,destination=:all)
    4-element Vector{Vector{Int64}}:
     [10, 12, 16, 17]
     [10, 12, 16, 17]
     [10, 12, 16, 17]
     [10, 12, 16, 17]
"""
function exclusive_scan(op,a;init,destination=1)
    b = allocate_gather(a;destination)
    exclusive_scan!(op,b,a;init,destination)
end

