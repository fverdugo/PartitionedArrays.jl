
"""
    with_debug(f)

Call `f(DebugArray)`.
"""
function with_debug(f,args...;kwargs...)
    f(DebugArray,args...;kwargs...)
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
    scalar_indexing_action(a)
    a.items[i] = v
    v
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
getany(a::DebugArray) = getany(a.items)

function Base.similar(a::DebugArray,::Type{T},dims::Dims) where T
  DebugArray(similar(a.items,T,dims))
end

function Base.copyto!(b::DebugArray,a::DebugArray)
    copyto!(b.items,a.items)
    b
end

function Base.map(f,args::DebugArray...)
    DebugArray(map(f,map(i->i.items,args)...))
end

function Base.map!(f,r::DebugArray,args::DebugArray...)
    map!(f,r.items,map(i->i.items,args)...)
    r
end

function gather_impl!(
    rcv::DebugArray, snd::DebugArray,
    destination, ::Type{T}) where T
    gather_impl!(rcv.items,snd.items,destination,T)
end

function gather_impl!(
    rcv::DebugArray, snd::DebugArray,
    destination, ::Type{T}) where T <: AbstractVector
    gather_impl!(rcv.items,snd.items,destination,T)
end

function scatter_impl!(
    rcv::DebugArray,snd::DebugArray,
    source,::Type{T}) where T
    scatter_impl!(rcv.items,snd.items,source,T)
end

function scatter_impl!(
    rcv::DebugArray,snd::DebugArray,
    source,::Type{T}) where T<:AbstractVector
    scatter_impl!(rcv.items,snd.items,source,T)
end

function emit_impl!(
    rcv::DebugArray,snd::DebugArray,
    source,::Type{T}) where T
    emit_impl!(rcv.items,snd.items,source,T)
end

function emit_impl!(
    rcv::DebugArray,snd::DebugArray,
    source,::Type{T}) where T<:AbstractVector
    emit_impl!(rcv.items,snd.items,source,T)
end

Base.reduce(op,a::DebugArray;kwargs...) = reduce(op,a.items;kwargs...)
Base.sum(a::DebugArray) = reduce(+,a)
Base.collect(a::DebugArray) = collect(a.items)

function is_consistent(graph::ExchangeGraph{<:DebugArray})
    g = ExchangeGraph(graph.snd.items,graph.rcv.items)
    is_consistent(g)
end

function exchange_impl!(
    rcv::DebugArray,
    snd::DebugArray,
    graph::ExchangeGraph{<:DebugArray},
    ::Type{T}) where T
    g = ExchangeGraph(graph.snd.items,graph.rcv.items)
    @async begin
        sleep(0.2) # This is to make more likely to have errors if we don't wait
        exchange_impl!(rcv.items,snd.items,g,T) |> wait
        rcv
    end
end

function exchange_impl!(
    rcv::DebugArray,
    snd::DebugArray,
    graph::ExchangeGraph{<:DebugArray},
    ::Type{T}) where T <: AbstractVector
    g = ExchangeGraph(graph.snd.items,graph.rcv.items)
    @async begin
        sleep(0.2) # This is to make more likely to have errors if we don't wait
        exchange_impl!(rcv.items,snd.items,g,T) |> wait
        rcv
    end
end

