
"""
    with_sequential_data(f)

Call `f(SequentialData)`.
"""
function with_sequential_data(f)
    f(SequentialData)
end

# Auxiliary array type
# This new array type is not strictly needed
# but it is useful for testing purposes since
# it mimics the warning and errors one would get
# when using the MPI backend
"""
    struct SequentialData{T,N}

Data structure that emulates the behavior of [`MPIData`](@ref), but that can be
used on a standard sequential (a.k.a. serial) Julia session. This struct implements
the Julia array interface, but it is immutable. I.e., [`setindex!`](@ref) is not
available for this struct.

# Properties

The fields of this struct are private.

# Supertype hierarchy

    SequentialData{T,N} <: AbstractArray{T,N}
"""
struct SequentialData{T,N} <: AbstractArray{T,N}
    items::Array{T,N}
    @doc """
        SequentialData{T,N}(a) where {T,N}

    Create a `SequentialData{T,N}` data object from the items in collection
    `a`. If `a::Array{T,N}`, then the result takes ownership of the input.
    Otherwise, a copy of the input is created.
    """
    function SequentialData{T,N}(a) where {T,N}
      new{T,N}(convert(Array{T,N},a))
    end
    @doc """
        SequentialData(a)

    Create a `SequentialData{T,N}` data object from the items in collection
    `a`, where `T=eltype(a)` and `N=ndims(a)` .
    If `a::Array{T,N}`, then the result takes ownership of the input.
    Otherwise, a copy of the input is created.
    """
    function SequentialData(a)
      T = eltype(a)
      N = ndims(a)
      new{T,N}(convert(Array{T,N},a))
    end
end

Base.size(a::SequentialData) = size(a.items)
Base.IndexStyle(::Type{<:SequentialData}) = IndexLinear()
function Base.getindex(a::SequentialData,i::Int)
    scalar_indexing_error(a)
    a.items[i]
end
function Base.setindex!(a::SequentialData,v,i::Int)
    msg = "setindex! is not allowed on SequentialData to emulate the behaviour of MPIData."
    error(msg)
end
linear_indices(a::SequentialData) = SequentialData(collect(LinearIndices(a)))
cartesian_indices(a::SequentialData) = SequentialData(collect(CartesianIndices(a)))
function Base.show(io::IO,k::MIME"text/plain",data::SequentialData)
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

function Base.similar(a::SequentialData,::Type{T},dims::Dims) where T
  SequentialData(similar(a.items,T,dims))
end

function Base.map(f,args::SequentialData...)
    SequentialData(map(f,map(i->i.items,args)...))
end

function gather_impl!(
    rcv::SequentialData, snd::SequentialData,
    destination, ::Type{T}) where T
    gather_impl!(rcv.items,snd.items,destination,T)
end

function gather_impl!(
    rcv::SequentialData, snd::SequentialData,
    destination, ::Type{T}) where T <: AbstractVector
    gather_impl!(rcv.items,snd.items,destination,T)
end

function scatter_impl!(
    rcv::SequentialData,snd::SequentialData,
    source,::Type{T}) where T
    scatter_impl!(rcv.items,snd.items,source,T)
    msg = "scatter! cannot be used when scatering scalars. Use scatter instead."
    error(msg)
end

function scatter_impl!(
    rcv::SequentialData,snd::SequentialData,
    source,::Type{T}) where T<:AbstractVector
    scatter_impl!(rcv.items,snd.items,source,T)
end

function scatter_impl(
    snd::SequentialData,source,::Type{T}) where T
    items = scatter_impl(snd.items,source,T)
    SequentialData(items)
end

function scatter_impl(
    snd::SequentialData,source,::Type{T}) where T<:AbstractVector
    items = scatter_impl(snd.items,source,T)
    SequentialData(items)
end

function emit_impl!(
    rcv::SequentialData,snd::SequentialData,
    source,::Type{T}) where T
    msg = "emit! cannot be used when sending scalars. Use scatter instead."
    error(msg)
end

function emit_impl!(
    rcv::SequentialData,snd::SequentialData,
    source,::Type{T}) where T<:AbstractVector
    emit_impl!(rcv.items,snd.items,source,T)
end

function emit_impl(
    snd::SequentialData,source,::Type{T}) where T
    items = emit_impl(snd.items,source,T)
    SequentialData(items)
end

function emit_impl(
    snd::SequentialData,source,::Type{T}) where T<:AbstractVector
    items = emit_impl(snd.items,source,T)
    SequentialData(items)
end

