
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
    T = eltype(snd)
    gather_impl(snd,destination,T)
end

function gather_impl(snd,destination,::Type{T}) where T
    rcv = allocate_gather(snd;destination)
    gather!(rcv,snd;destination)
    rcv
end

"""
    allocate_gather(snd;destination=1)

Allocate an array to be used in the first argument of [`gather!`](@ref).
"""
function allocate_gather(snd;destination=1)
    T = eltype(snd)
    allocate_gather_impl(snd,destination,T)
end

function allocate_gather_impl(snd,destination,::Type{T}) where T
    n = length(snd)
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

function allocate_gather_impl(snd,destination,::Type{T}) where T<:AbstractVector
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
    T = eltype(snd)
    gather_impl!(rcv,snd,destination,T)
end

function gather_impl!(rcv,snd,destination,::Type{T}) where T
    if isa(destination,Integer)
        @assert length(rcv[destination]) == length(snd)
        for i in 1:length(snd)
            rcv[destination][i] = snd[i]
        end
    else
        @assert destination === :all
        for j in eachindex(rcv)
            for i in 1:length(snd)
                rcv[j][i] = snd[i]
            end
        end
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
    T = eltype(eltype(snd))
    scatter_impl(snd,source,T)
end

function scatter_impl(snd,source,::Type{T}) where T
    @assert source !== :all "All to all not implemented"
    rcv = allocate_scatter(snd;source)
    scatter!(rcv,snd;source)
    rcv
end

"""
    allocate_scatter(snd;source=1)

Allocate an array to be used in the first argument of [`scatter!`](@ref).
"""
function allocate_scatter(snd;source=1)
    @assert source !== :all "All to all not implemented"
    T = eltype(eltype(snd))
    allocate_scatter_impl(snd,source,T)
end

function allocate_scatter_impl(snd,source,::Type{T}) where T
    similar(snd,T)
end

function allocate_scatter_impl(snd,source,::Type{T}) where T <:AbstractVector
    counts = map(snd) do snd
        map(length,snd)
    end
    counts_scat = scatter(counts;source)
    S = eltype(T)
    map(counts_scat) do count
        Vector{S}(undef,count)
    end
end

"""
    scatter!(rcv,snd;source=1)

In-place version of [`scatter`](@ref). The destination array `rcv`
can be generated with the helper function [`allocate_scatter`](@ref).
"""
function scatter!(rcv,snd;source=1)
    T = eltype(eltype(snd))
    scatter_impl!(rcv,snd,source,T)
end

function scatter_impl!(rcv,snd,source,::Type{T}) where T
    @assert source !== :all "Scatter all not implemented"
    @assert length(snd[source]) == length(rcv)
    for i in 1:length(snd)
        rcv[i] = snd[source][i]
    end
    rcv
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
    T = eltype(snd)
    emit_impl(snd,source,T)
end

function emit_impl(snd,source,::Type{T}) where T
    @assert source !== :all "All to all not implemented"
    rcv = allocate_emit(snd;source)
    emit!(rcv,snd;source)
    rcv
end

"""
    allocate_emit(snd;source=1)

Allocate an array to be used in the first argument of [`emit!`](@ref).
"""
function allocate_emit(snd;source=1)
    T = eltype(snd)
    allocate_emit_impl(snd,source,T)
end

function allocate_emit_impl(snd,source,::Type{T}) where T
    @assert source !== :all "Scatter all not implemented"
    similar(snd)
end

function allocate_emit_impl(snd,source,::Type{T}) where T<:AbstractVector
    @assert source !== :all "Scatter all not implemented"
    n = map(length,snd)
    n_all = emit(n;source)
    S = eltype(T)
    map(n_all) do n
        Vector{S}(undef,n)
    end
end

"""
    emit!(rcv,snd;source=1)

In-place version of [`emit`](@ref). The destination array `rcv`
can be generated with the helper function [`allocate_emit`](@ref).
"""
function emit!(rcv,snd;source=1)
    T = eltype(snd)
    emit_impl!(rcv,snd,source,T)
end

function emit_impl!(rcv,snd,source,::Type{T}) where T
    @assert source !== :all "Emit all not implemented"
    for i in eachindex(rcv)
        rcv[i] = snd[source]
    end
    rcv
end

"""
    scan(op,a;init,type)

Return the scan of the values in `a` for the operation `op`.
Use `type=:inclusive` or `type=:exclusive` to use an inclusive or exclusive scan.
`init` will be added to all items in the result. Additionally, for exclusive scans,
the first item in the result will be set to `init`.

# Examples

    julia> using PartitionedArrays

    julia> a = [2,4,1,3]
    4-element Vector{Int64}:
     2
     4
     1
     3

    julia> scan(+,a,type=:inclusive,init=0)
    4-element Vector{Int64}:
      2
      6
      7
     10
    
    julia> scan(+,a,type=:exclusive,init=0)
    4-element Vector{Int64}:
     0
     2
     6
     7

    julia> scan(+,a,type=:exclusive,init=1)
    4-element Vector{Int64}:
     1
     3
     7
     8
"""
function scan(op,a;init,type)
    @assert type in (:inclusive,:exclusive)
    b = gather(a)
    map(b) do b
        n = length(b)
        if init !== nothing && n > 0
            b[1] = op(b[1],init)
        end
        for i in 1:(n-1)
            b[i+1] = op(b[i+1],b[i])
        end
        if type === :exclusive && n > 0
            right_shift!(b)
            b[1] = init
        end
    end
    scatter(b)
end

"""
    reduction(op,a;init,destination=1)

Reduce the values in array `a` according with operation
`op` and the initial value `init` and store the result in
a new array of the same size as `a` at index `destination`.

# Examples

    julia> using PartitionedArrays
    
    julia> a = [1,3,2,4]
    4-element Vector{Int64}:
     1
     3
     2
     4
    
    julia> reduction(+,a;init=0,destination=2)
    4-element Vector{Int64}:
      0
     10
      0
      0
"""
function reduction(op,a;init,destination=1)
  b = gather(a;destination)
  map(i->reduce(op,i;init=init),b)
end

