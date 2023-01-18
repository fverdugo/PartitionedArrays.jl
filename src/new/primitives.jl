
const SCALAR_INDEXING_ACTION = Ref(:error)

function scalar_indexing_action(a)
    if SCALAR_INDEXING_ACTION[] === :warn
        @warn "Scalar indexing on $(nameof(typeof(a))) is discuraged for performance reasons."
    elseif SCALAR_INDEXING_ACTION[] === :error
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

const MAIN = 1

"""
    map_main(f,args...;kwargs...)

Like `map(f,args...)` but only apply `f` to one component of the arrays
in `args` (the first component by default). Set the remaining entries to `nothing` by default.

# Optional key-word arguments

- `main = MAIN`: The linear index of the component to map
- `otherwise = (args...)->nothing`: The function to apply when mapping indices different from `main`.

# Examples

    julia> using PartitionedArrays

    julia> a = [1,2,3,4]
    4-element Vector{Int64}:
     1
     2
     3
     4

    julia> map_main(-,a,main=2)
    4-element Vector{Union{Nothing, Int64}}:
       nothing
     -2
       nothing
       nothing
"""
function map_main(f,args...;otherwise=(args...)->nothing,main=MAIN)
    if isa(main,Integer)
        rank = linear_indices(first(args))
        map(rank,args...) do rank,args...
            if rank == main
                f(args...)
            else
                otherwise(args...)
            end
        end
    else
      @assert main === :all
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
        rcv = map_main(f,snd;otherwise=g,main=destination)
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
        rcv = map_main(f,l_dest,snd;otherwise=g,main=destination)
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
    b = similar(a)
    scan!(op,b,a;init,type)
end

"""
    scan!(op,b,a;init,type)

In-place version of [`scan`](@ref) on the result `b`.
"""
function scan!(op,b,a;init,type)
    @assert type in (:inclusive,:exclusive)
    c = gather(a)
    map(c) do c
        n = length(c)
        if init !== nothing && n > 0
            c[1] = op(c[1],init)
        end
        for i in 1:(n-1)
            c[i+1] = op(c[i+1],c[i])
        end
        if type === :exclusive && n > 0
            right_shift!(c)
            c[1] = init
        end
    end
    scatter!(b,c)
    b
end

"""
    reduction(op,a;destination=1[,init])

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
function reduction(op,a;kwargs...)
    b = similar(a)
    reduction!(op,b,a;kwargs...)
end

"""
    reduction!(op,b,a;destination=1[,init])

In-place version of [`reduction`](@ref) on the result `b`.
"""
function reduction!(op,b,a;destination=1,kwargs...)
  c = gather(a;destination)
  map!(i->reduce(op,i;kwargs...),b,c)
  b
end

"""
    struct ExchangeGraph{A}

Type representing a directed graph to be used in exchanges,
see function [`exchange!`](@ref) and [`exchange`](@ref).

# Properties
    snd::A
    rcv::A

`snd[i]` contains a list of the outgoing neighbors of node `i`.
`rcv[i]` contains a list of the incomming neighbors of node `i`.
`A` is a vector-like container type.

# Supertype hierarchy
    ExchangeGraph <: Any

This type has enough information to implement the `AbstractGraph` interface
of the  `Graphs.jl` package. However, we do not really need the functionallity
of this package here.
"""
struct ExchangeGraph{A}
    snd::A
    rcv::A
end
Base.reverse(g::ExchangeGraph) = ExchangeGraph(g.rcv,g.snd)

function Base.show(io::IO,k::MIME"text/plain",data::ExchangeGraph)
    println(io,typeof(data)," with $(length(data.snd)) nodes")
end

"""
    ExchangeGraph(snd;neighbors=nothing,symmetric=false)

Create an `ExchangeGraph` object only from the lists of outgoing 
neighbors in `snd`. If `symmetric==true`, then the incoming neighbors
are set to `snd`. Otherwise, the optional `neighbors` is considered.
 `neighbors` is also an `ExchangeGraph`
that contains a super set of the outgoing and incoming neighbors
associated with `snd`. It is used to find the incoming neighbors `rcv`
efficiently.
"""
function ExchangeGraph(snd;neighbors=nothing,symmetric=false)
    if symmetric
        ExchangeGraph(snd,snd)
    else
        ExchangeGraph_impl(snd,neighbors)
    end
end

# Discover snd parts from rcv assuming that snd is a subset of neighbors
function ExchangeGraph_impl(snd_ids,neighbors::ExchangeGraph)
    rank = linear_indices(snd_ids)
    # Tell the neighbors whether I want to send to them
    data_snd = map(rank,neighbors.snd,snd_ids) do rank, neighbors_snd, snd_ids
        dict_rcv = Dict(( n=>Int(-1) for n in neighbors_snd))
        for i in snd_ids
            dict_rcv[i] = rank
        end
        [ dict_rcv[n] for n in neighbors_snd ]
    end
    data_rcv = exchange_fetch(data_snd,neighbors)
    # build rcv_ids
    rcv_ids = map(data_rcv) do data_rcv
        k = findall(j->j>0,data_rcv)
        data_rcv[k]
    end
    ExchangeGraph(snd_ids,rcv_ids)
end

# If neighbors not provided, we need to gather in main
function ExchangeGraph_impl(snd_ids,neighbors::Nothing)
    discover_rcv_neighbors_action()
    snd_ids_main = gather(snd_ids)
    rcv_ids_main = map(snd_ids_main) do snd_ids_main
        snd = JaggedArray(snd_ids_main)
        I = Int32[]
        J = Int32[]
        np = length(snd)
        for p in 1:np
            kini = snd.ptrs[p]
            kend = snd.ptrs[p+1]-1
            for k in kini:kend
                push!(I,p)
                push!(J,snd.data[k])
            end
        end
        adjmat = sparse(I,J,I,np,np)
        ptrs = similar(snd.ptrs)
        fill!(ptrs,zero(eltype(ptrs)))
        for (i,j,_) in nziterator(adjmat)
            ptrs[j+1] += 1
        end
        length_to_ptrs!(ptrs)
        ndata = ptrs[end]-1
        data = similar(snd.data,eltype(snd.data),ndata)
        for (i,j,_) in nziterator(adjmat)
            data[ptrs[j]] = i
            ptrs[j] += 1
        end
        rewind_ptrs!(ptrs)
        rcv = JaggedArray(data,ptrs)
    end
    rcv_ids = scatter(rcv_ids_main)
    ExchangeGraph(snd_ids,rcv_ids)
end

const DISCOVER_RCV_NEIGHBORS_ACTION = Ref(:allow)

function discover_rcv_neighbors_action()
    DISCOVER_RCV_NEIGHBORS_ACTION[] === :allow && return nothing
    msg =
    """
    [PartitionedArrays.jl] Using a non-scalable implementation
    to discover the incoming neighbours of a `ExchangeGraph`.
    This might cause trouble when running the code at medium/large scales.
    You can avoid this using the Exchanger constructor with a superset of
    the actual receivers/senders
    """
    if DISCOVER_RCV_NEIGHBORS_ACTION[] === :error
        error(msg)
    elseif DISCOVER_RCV_NEIGHBORS_ACTION[] === :warn
        @warn msg
    end
    nothing
end

function is_consistent(graph::ExchangeGraph)
    snd = graph.snd
    rcv = graph.rcv
    @assert length(rcv) == length(snd)
    for part in 1:length(rcv)
        for i in rcv[part]
            length(findall(k->k==part,snd[i])) == 1 || return false
        end
        for i in snd[part]
            length(findall(k->k==part,rcv[i])) == 1 || return false
        end
    end
    true
end

"""
    exchange(snd,graph::ExchangeGraph) -> Task

Send the data in `snd` according the directed graph `graph`.
This function returns immediately
and returns a task that produces the result, allowing for latency hiding.
Use [`fetch`](@ref) to wait and get the result.
The object `snd` and `rcv=fetch(exchange(snd,graph))`
are array of vectors. The  value `snd[i][j]` is sent
to node `graph.snd[i][j]`. The value `rcv[i][j]` is the one
received from  node `graph.rcv[i][j]`.

# Examples

    julia> using PartitionedArrays
    
    julia> snd_ids = [[3,4],[1,3],[1,4],[2]]
    4-element Vector{Vector{Int64}}:
     [3, 4]
     [1, 3]
     [1, 4]
     [2]
    
    julia> graph = ExchangeGraph(snd_ids)
    ExchangeGraph{Vector{Vector{Int64}}} with 4 nodes
    
    
    julia> snd = [[10,10],[20,20],[30,30],[40]]
    4-element Vector{Vector{Int64}}:
     [10, 10]
     [20, 20]
     [30, 30]
     [40]
    
    julia> t = exchange(snd,graph)
    Task (done) @0x00007fe9e3850460
    
    julia> rcv = fetch(t)
    4-element Vector{Vector{Int64}}:
     [20, 30]
     [40]
     [10, 20]
     [10, 30]
"""
function exchange(snd,graph::ExchangeGraph)
    rcv = allocate_exchange(snd,graph)
    task = exchange!(rcv,snd,graph)
    task
end

"""
    exchange_fetch(snd,graph::ExchangeGraph)

Equivalent to `fetch(exchange(snd,graph))`, but
it can consider more optimizations.
"""
function exchange_fetch(snd,graph::ExchangeGraph)
    fetch(exchange(snd,graph))
end

"""
    allocate_exchange(snd,graph::ExchangeGraph)

Allocate the result to be used in the first argument
of [`exchange`](@ref).
"""
function allocate_exchange(snd,graph::ExchangeGraph)
    T = eltype(eltype(snd))
    allocate_exchange_impl(snd,graph,T)
end

function allocate_exchange_impl(snd,graph,::Type{T}) where T
    rcv = map(snd,graph.rcv) do snd,rcv_ids
        similar(snd,eltype(snd),length(rcv_ids))
    end
    rcv
end

function allocate_exchange_impl(snd,graph,::Type{T}) where T<:AbstractVector
    n_snd = map(snd) do snd
        map(length,snd)
    end
    n_rcv = exchange_fetch(n_snd,graph)
    S = eltype(eltype(eltype(snd)))
    rcv = map(n_rcv) do n_rcv
        ptrs = zeros(Int32,length(n_rcv)+1)
        ptrs[2:end] = n_rcv
        length_to_ptrs!(ptrs)
        n_data = ptrs[end]-1
        data = Vector{S}(undef,n_data)
        JaggedArray(data,ptrs)
    end
    rcv
end

"""
    exchange!(rcv,snd,graph::ExchangeGraph) -> Task

In-place and asynchronous version of [`exchange`](@ref). This function
returns immediately and returns a task that produces `rcv` with the updated values.
Use [`fetch`](@ref) to get the updated version of `rcv`.
The input `rcv` can be allocated with [`allocate_exchange`](@ref).

# Examples

    julia> using PartitionedArrays
    
    julia> snd = [[10,10],[20,20],[30,30],[40]]
    4-element Vector{Vector{Int64}}:
     [10, 10]
     [20, 20]
     [30, 30]
     [40]
    
    julia> graph = ExchangeGraph(snd_ids)
    ExchangeGraph{Vector{Vector{Int64}}} with 4 nodes
    
    
    julia> rcv = allocate_exchange(snd,graph)
    4-element Vector{Vector{Int64}}:
     [140477703842832, 140480236018064]
     [0]
     [140477373823312, 140477373823344]
     [140477691773008, 140480035219440]


    julia> t = exchange!(rcv,snd,graph)
    Task (done) @0x00007fe9e73c1cd0
    
    julia> rcv = fetch(t)
    4-element Vector{Vector{Int64}}:
     [20, 30]
     [40]
     [10, 20]
     [10, 30]
"""
function exchange!(rcv,snd,graph::ExchangeGraph)
    T = eltype(eltype(snd))
    exchange_impl!(rcv,snd,graph,T)
end

"""
    exchange_fetch!(rcv,snd,graph::ExchangeGraph)

Equivalent to `fetch(exchange_fetch!(rcv,snd,graph))`, but
it can consider more optimizations.
"""
function exchange_fetch!(rcv,snd,graph::ExchangeGraph)
    fetch(exchange!(rcv,snd,graph))
end

function exchange_impl!(rcv,snd,graph,::Type{T}) where T
    @assert is_consistent(graph)
    snd_ids = graph.snd
    rcv_ids = graph.rcv
    @assert length(rcv_ids) == length(rcv)
    @assert length(rcv_ids) == length(snd)
    for rcv_id in 1:length(rcv_ids)
        for (i, snd_id) in enumerate(rcv_ids[rcv_id])
            j = first(findall(k->k==rcv_id,snd_ids[snd_id]))
            rcv[rcv_id][i] = snd[snd_id][j]
        end
    end
    @async rcv
end

function exchange_impl!(rcv,snd,graph,::Type{T}) where T<:AbstractVector
    @assert is_consistent(graph)
    @assert eltype(rcv) <: JaggedArray
    snd_ids = graph.snd
    rcv_ids = graph.rcv
    @assert length(rcv_ids) == length(rcv)
    @assert length(rcv_ids) == length(snd)
    for rcv_id in 1:length(rcv_ids)
        for (i, snd_id) in enumerate(rcv_ids[rcv_id])
            snd_snd_id = JaggedArray(snd[snd_id])
            j = first(findall(k->k==rcv_id,snd_ids[snd_id]))
            ptrs_rcv = rcv[rcv_id].ptrs
            ptrs_snd = snd_snd_id.ptrs
            @assert ptrs_rcv[i+1]-ptrs_rcv[i] == ptrs_snd[j+1]-ptrs_snd[j]
            for p in 1:(ptrs_rcv[i+1]-ptrs_rcv[i])
                p_rcv = p+ptrs_rcv[i]-1
                p_snd = p+ptrs_snd[j]-1
                rcv[rcv_id].data[p_rcv] = snd_snd_id.data[p_snd]
            end
        end
    end
    @async rcv
end

