
"""
    with_debug_data(f)

Call `f(DebugData)`.
"""
function with_debug_data(f)
    f(DebugData)
end

# Auxiliary array type
# This new array type is not strictly needed
# but it is useful for testing purposes since
# it mimics the warning and errors one would get
# when using the MPI backend
"""
    struct DebugData{T,N}

Data structure that emulates the behavior of [`MPIData`](@ref), but that can be
used on a standard sequential (a.k.a. serial) Julia session. This struct implements
the Julia array interface.
However, using [`setindex!`](@ref) and [`getindex!`](@ref) is strongly discouraged
since this will not be efficient in actual parallel runs (communication cost).

# Properties

The fields of this struct are private.

# Supertype hierarchy

    DebugData{T,N} <: AbstractArray{T,N}
"""
struct DebugData{T,N} <: AbstractArray{T,N}
    items::Array{T,N}
    @doc """
        DebugData{T,N}(a) where {T,N}

    Create a `DebugData{T,N}` data object from the items in collection
    `a`. If `a::Array{T,N}`, then the result takes ownership of the input.
    Otherwise, a copy of the input is created.
    """
    function DebugData{T,N}(a) where {T,N}
      new{T,N}(convert(Array{T,N},a))
    end
    @doc """
        DebugData(a)

    Create a `DebugData{T,N}` data object from the items in collection
    `a`, where `T=eltype(a)` and `N=ndims(a)` .
    If `a::Array{T,N}`, then the result takes ownership of the input.
    Otherwise, a copy of the input is created.
    """
    function DebugData(a)
      T = eltype(a)
      N = ndims(a)
      new{T,N}(convert(Array{T,N},a))
    end
end

Base.size(a::DebugData) = size(a.items)
Base.IndexStyle(::Type{<:DebugData}) = IndexLinear()
function Base.getindex(a::DebugData,i::Int)
    scalar_indexing_action(a)
    a.items[i]
end
function Base.setindex!(a::DebugData,v,i::Int)
    scalar_indexing_action(a)
    a.items[i] = v
    v
end
linear_indices(a::DebugData) = DebugData(collect(LinearIndices(a)))
cartesian_indices(a::DebugData) = DebugData(collect(CartesianIndices(a)))
function Base.show(io::IO,k::MIME"text/plain",data::DebugData)
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
getany(a::DebugData) = getany(a.items)

function Base.similar(a::DebugData,::Type{T},dims::Dims) where T
  DebugData(similar(a.items,T,dims))
end

function Base.copyto!(b::DebugData,a::DebugData)
    copyto!(b.items,a.items)
    b
end

function Base.map(f,args::DebugData...)
    DebugData(map(f,map(i->i.items,args)...))
end

function Base.map!(f,r::DebugData,args::DebugData...)
    map!(f,r.items,map(i->i.items,args)...)
    r
end

function gather_impl!(
    rcv::DebugData, snd::DebugData,
    destination, ::Type{T}) where T
    gather_impl!(rcv.items,snd.items,destination,T)
end

function gather_impl!(
    rcv::DebugData, snd::DebugData,
    destination, ::Type{T}) where T <: AbstractVector
    gather_impl!(rcv.items,snd.items,destination,T)
end

function scatter_impl!(
    rcv::DebugData,snd::DebugData,
    source,::Type{T}) where T
    scatter_impl!(rcv.items,snd.items,source,T)
end

function scatter_impl!(
    rcv::DebugData,snd::DebugData,
    source,::Type{T}) where T<:AbstractVector
    scatter_impl!(rcv.items,snd.items,source,T)
end

function emit_impl!(
    rcv::DebugData,snd::DebugData,
    source,::Type{T}) where T
    emit_impl!(rcv.items,snd.items,source,T)
end

function emit_impl!(
    rcv::DebugData,snd::DebugData,
    source,::Type{T}) where T<:AbstractVector
    emit_impl!(rcv.items,snd.items,source,T)
end

Base.reduce(op,a::DebugData;kwargs...) = reduce(op,a.items;kwargs...)
Base.sum(a::DebugData) = reduce(+,a)
Base.collect(a::DebugData) = collect(a.items)

function is_consistent(graph::ExchangeGraph{<:DebugData})
    g = ExchangeGraph(graph.snd.items,graph.rcv.items)
    is_consistent(g)
end

function exchange_impl!(
    rcv::DebugData,
    snd::DebugData,
    graph::ExchangeGraph{<:DebugData},
    ::Type{T}) where T
    g = ExchangeGraph(graph.snd.items,graph.rcv.items)
    t = exchange_impl!(rcv.items,snd.items,g,T)
    rcv = DebugData(fetch(t))
    @async rcv
end

function exchange_impl!(
    rcv::DebugData,
    snd::DebugData,
    graph::ExchangeGraph{<:DebugData},
    ::Type{T}) where T <: AbstractVector
    g = ExchangeGraph(graph.snd.items,graph.rcv.items)
    t = exchange_impl!(rcv.items,snd.items,g,T)
    rcv = DebugData(fetch(t))
    @async rcv
end

