
# Auxiliary array type
# This new array type is not strictly needed
# but it is useful for testing purposes since
# it mimics the warning and errors one would get
# when using the MPI backend
struct SequentialArray{T,N} <: AbstractArray{T,N}
    items::Array{T,N}
    function SequentialArray{T,N}(a) where {T,N}
      new{T,N}(convert(Array{T,N},a))
    end
    function SequentialArray(a)
      T = eltype(a)
      N = ndims(a)
      new{T,N}(convert(Array{T,N},a))
    end
end

Base.size(a::SequentialArray) = size(a.items)
Base.IndexStyle(::Type{<:SequentialArray}) = IndexLinear()
function Base.getindex(a::SequentialArray,i::Int)
    scalar_indexing_error(a)
    a.items[i]
end
function Base.setindex!(a::SequentialArray,v,i::Int)
    scalar_indexing_error(a)
    a.items[i] = v
end
linear_indices(a::SequentialArray) = SequentialArray(collect(LinearIndices(a)))
cartesian_indices(a::SequentialArray) = SequentialArray(collect(CartesianIndices(a)))
function Base.show(io::IO,k::MIME"text/plain",data::SequentialArray)
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

function Base.similar(a::SequentialArray,::Type{T},dims::Dims) where T
  SequentialArray(similar(a.items,T,dims))
end

function Base.map(f,args::SequentialArray...)
    SequentialArray(map(f,map(i->i.items,args)...))
end

function Base.map!(f,args::SequentialArray...)
    SequentialArray(map!(f,map(i->i.items,args)...))
end

function gather!(rcv::SequentialArray,snd::SequentialArray;destination=1)
    gather!(rcv.items,snd.items;destination)
    rcv
end

function scatter!(rcv::SequentialArray,snd::SequentialArray;source=1)
    scatter!(rcv.items,snd.items;source)
    rcv
end

function emit!(rcv::SequentialArray,snd::SequentialArray;source=1)
    emit!(rcv.items,snd.items;source)
    rcv
end



