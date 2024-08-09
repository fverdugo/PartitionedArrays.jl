
"""
    with_debug(f)

Call `f(DebugArray)`.
"""
function with_debug(f)
    f(DebugArray)
end

# Auxiliary array type
# This new array type is not strictly needed
# but it is useful for testing purposes since
# it mimics the warning and errors one would get
# when using the MPI backend
"""
    struct DebugArray{T,N}

Data structure that emulates the limitations of [`MPIArray`](@ref), but that can be
used on a standard sequential (a.k.a. serial) Julia session. This struct implements
the Julia array interface.
Like for [`MPIArray`](@ref), using `setindex!` and `getindex`
on `DebugArray` is disabled
since this will not be efficient in actual parallel runs (communication cost).

# Properties

The fields of this struct are private.

# Supertype hierarchy

    DebugArray{T,N} <: AbstractArray{T,N}
"""
struct DebugArray{T,N} <: AbstractArray{T,N}
    items::Array{T,N}
    function DebugArray{T,N}(a) where {T,N}
      new{T,N}(convert(Array{T,N},a))
    end
    @doc """
        DebugArray(a)

    Create a `DebugArray{T,N}` data object from the items in collection
    `a`, where `T=eltype(a)` and `N=ndims(a)` .
    If `a::Array{T,N}`, then the result takes ownership of the input.
    Otherwise, a copy of the input is created.
    """
    function DebugArray(a)
      T = eltype(a)
      N = ndims(a)
      new{T,N}(convert(Array{T,N},a))
    end
end

Base.size(a::DebugArray) = size(a.items)
Base.IndexStyle(::Type{<:DebugArray}) = IndexLinear()
function Base.getindex(a::DebugArray,i::Int)
    scalar_indexing_action(a)
    a.items[i]
end
function Base.setindex!(a::DebugArray,v,i::Int)
    error("DebugArray is inmutable for performance reasons")
    #scalar_indexing_action(a)
    #a.items[i] = v
    #v
end
linear_indices(a::DebugArray) = DebugArray(collect(LinearIndices(a)))
cartesian_indices(a::DebugArray) = DebugArray(collect(CartesianIndices(a)))
function Base.show(io::IO,k::MIME"text/plain",data::DebugArray)
    header = ""
    if ndims(data) == 1
        header *= "$(length(data))-element"
    else
        for n in 1:ndims(data)
            if n!=1
                header *= "×"
            end
            header *= "$(size(data,n))"
        end
    end
    header *= " $(typeof(data)):"
    println(io,header)
    for i in CartesianIndices(data.items)
        index = "["
        for (j,t) in enumerate(Tuple(i))
            if j != 1
                index *=","
            end
            index *= "$t"
        end
        index *= "]"
        println(io,"$index = $(data.items[i])")
    end
end
function Base.show(io::IO,data::DebugArray)
    print(io,"DebugArray(…)")
end
getany(a::DebugArray) = getany(a.items)

function Base.similar(a::DebugArray,::Type{T},dims::Dims) where T
    error("DebugArray is inmutable for performance reasons")
    #DebugArray(similar(a.items,T,dims))
end

function Base.copyto!(b::DebugArray,a::DebugArray)
    error("DebugArray is inmutable for performance reasons")
    #copyto!(b.items,a.items)
    #b
end

function Base.map(f,args::DebugArray...)
    DebugArray(map(f,map(i->i.items,args)...))
end

function Base.foreach(f,args::DebugArray...)
    foreach(f,map(i->i.items,args)...)
    nothing
end

function Base.map!(f,r::DebugArray,args::DebugArray...)
    error("DebugArray is inmutable for performance reasons")
    #map!(f,r.items,map(i->i.items,args)...)
    #r
end

function Base.all(a::DebugArray)
    reduce(&,a;init=true)
end
function Base.all(p::Function,a::DebugArray)
    b = map(p,a)
    all(b)
end

#function allocate_gather_impl(snd::DebugArray,destination)
#    rcv = allocate_gather_impl(snd.items,destination)
#    DebugArray(rcv)
#end

function gather_impl!(
    rcv::DebugArray, snd::DebugArray,
    destination)
    gather_impl!(rcv.items,snd.items,destination)
end

#function setup_scatter_impl(snd::DebugArray,source)
#    setup_scatter_impl(snd.items,source)
#end

function scatter_impl(snd::DebugArray,source)
    rcv = scatter_impl(snd.items,source)
    DebugArray(rcv)
end

function scatter_impl!(rcv::DebugArray,snd::DebugArray,source,::Type{T}) where T
    error("In place scatter only for vectors")
end

function scatter_impl!(rcv::DebugArray,snd::DebugArray,source,::Type{T}) where T<:AbstractVector
    scatter_impl!(rcv.items,snd.items,source)
    rcv
end

#function setup_multicast_impl(snd::DebugArray,source)
#    setup_multicast_impl(snd.items,source)
#end

function multicast_impl(snd::DebugArray,source)
    rcv = multicast_impl(snd.items,source)
    DebugArray(rcv)
end

function multicast_impl!(rcv::DebugArray,snd::DebugArray,source,::Type{T}) where T
    error("In place multicast only for vectors")
end

function multicast_impl!(rcv::DebugArray,snd::DebugArray,source,::Type{T}) where T<:AbstractVector
    multicast_impl!(rcv.items,snd.items,source)
    rcv
end

#function setup_scatter_impl(op,a::DebugArray,init,type)
#    setup_scatter_impl(op,a.items,init,type)
#end

function scan_impl(op,a::DebugArray,init,type)
    b = scan_impl(op,a.items,init,type)
    DebugArray(b)
end

#function setup_reduction_impl(op,a::DebugArray,destination)
#    setup_reduction_impl(op,a.items,destination)
#end

function reduction_impl(op,a::DebugArray,destination;kwargs...)
    b = reduction_impl(op,a.items,destination;kwargs...)
    DebugArray(b)
end

#function gather_impl!(
#    rcv::DebugArray, snd::DebugArray,
#    destination, ::Type{T}) where T <: AbstractVector
#    gather_impl!(rcv.items,snd.items,destination,T)
#end

#function scatter_impl!(
#    rcv::DebugArray,snd::DebugArray,
#    source,::Type{T}) where T
#    scatter_impl!(rcv.items,snd.items,source,T)
#end
#
#function scatter_impl!(
#    rcv::DebugArray,snd::DebugArray,
#    source,::Type{T}) where T<:AbstractVector
#    scatter_impl!(rcv.items,snd.items,source,T)
#end
#
#function multicast_impl!(
#    rcv::DebugArray,snd::DebugArray,
#    source,::Type{T}) where T
#    multicast_impl!(rcv.items,snd.items,source,T)
#end
#
#function multicast_impl!(
#    rcv::DebugArray,snd::DebugArray,
#    source,::Type{T}) where T<:AbstractVector
#    multicast_impl!(rcv.items,snd.items,source,T)
#end

Base.reduce(op,a::DebugArray;kwargs...) = reduce(op,a.items;kwargs...)
Base.sum(a::DebugArray) = reduce(+,a)
Base.collect(a::DebugArray) = collect(a.items)

function is_consistent(graph::ExchangeGraph{<:DebugArray})
    g = ExchangeGraph(graph.snd.items,graph.rcv.items)
    is_consistent(g)
end

function allocate_exchange_impl(
    snd::DebugArray,graph::ExchangeGraph{<:DebugArray})
    graph2 = ExchangeGraph(graph.snd.items,graph.rcv.items)
    rcv = allocate_exchange_impl(snd.items,graph2)
    DebugArray(rcv)
end

function setup_exchange_impl(
    rcv::DebugArray,snd::DebugArray,graph::ExchangeGraph{<:DebugArray})
    graph2 = ExchangeGraph(graph.snd.items,graph.rcv.items)
    setup_exchange_impl(rcv.items,snd.items,graph2)
end

function exchange_impl!(
    rcv::DebugArray,snd::DebugArray,graph::ExchangeGraph{<:DebugArray},setup)
    graph2 = ExchangeGraph(graph.snd.items,graph.rcv.items)
    exchange_impl!(rcv.items,snd.items,graph2,setup)
    @fake_async rcv
end

#function exchange_impl!(
#    rcv::DebugArray,
#    snd::DebugArray,
#    graph::ExchangeGraph{<:DebugArray},
#    setup,
#    ::Type{T}) where T
#    g = ExchangeGraph(graph.snd.items,graph.rcv.items)
#    @fake_async begin
#        yield() # This is to make more likely to have errors if we don't wait
#        exchange_impl!(rcv.items,snd.items,g,setup,T) |> wait
#        rcv
#    end
#end
#
#function exchange_impl!(
#    rcv::DebugArray,
#    snd::DebugArray,
#    graph::ExchangeGraph{<:DebugArray},
#    setup,
#    ::Type{T}) where T <: AbstractVector
#    g = ExchangeGraph(graph.snd.items,graph.rcv.items)
#    @fake_async begin
#        yield() # This is to make more likely to have errors if we don't wait
#        exchange_impl!(rcv.items,snd.items,g,setup,T) |> wait
#        rcv
#    end
#end

