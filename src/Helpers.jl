"""
    @abstractmethod

Macro used in generic functions that must be overloaded by derived types.
"""
macro abstractmethod(message="This function belongs to an interface definition and cannot be used.")
  quote
    error($(esc(message)))
  end
end

"""
    @notimplemented
    @notimplemented "Error message"

Macro used to raise an error, when something is not implemented.
"""
macro notimplemented(message="This function is not yet implemented")
  quote
    error($(esc(message)))
  end
end

"""
    @notimplementedif condition
    @notimplementedif condition "Error message"

Macro used to raise an error if the `condition` is true
"""
macro notimplementedif(condition,message="This function is not yet implemented")
  quote
    if $(esc(condition))
      @notimplemented $(esc(message))
    end
  end
end

"""
    @unreachable
    @unreachable "Error message"

Macro used to make sure that a line of code is never reached.
"""
macro unreachable(message="This line of code cannot be reached")
  quote
    error($(esc(message)))
  end
end

"""
  @check condition
  @check condition "Error message"

Macro used to make sure that condition is fulfilled, like `@assert`
but the check gets deactivated when running Julia with --boundscheck=no
"""
macro check(test,msg="A check failed")
  quote
    @boundscheck @assert $(esc(test)) $(esc(msg))
  end
end

struct Table{T} <: AbstractVector{Vector{T}}
  data::Vector{T}
  ptrs::Vector{Int32}
end

get_data(a::Table) = a.data
get_ptrs(a::Table) = a.ptrs

Base.size(a::Table) = (length(a.ptrs)-1,)
Base.IndexStyle(::Type{<:Table}) = IndexLinear()
function Base.getindex(a::Table{T},i::Integer) where T
  pini = a.ptrs[i]
  pend = a.ptrs[i+1]-1
  ps = pini:pend
  v = zeros(T,length(ps))
  for (i,p) in enumerate(ps)
    v[i] = a.data[p]
  end
  v
end

function Table(a::AbstractArray{<:AbstractArray})
  data, ptrs = generate_data_and_ptrs(a)
  Table(data,ptrs)
end

function empty_table(a::Table)
  ptrs = similar(a.ptrs,eltype(a.ptrs),0+1)
  data = similar(a.data,eltype(a.data),0)
  Table(data,ptrs)
end

function generate_data_and_ptrs(vv::AbstractArray{<:AbstractArray{T}}) where T
  ptrs = Vector{Int32}(undef,length(vv)+1)
  _generate_data_and_ptrs_fill_ptrs!(ptrs,vv)
  length_to_ptrs!(ptrs)
  ndata = ptrs[end]-1
  data = Vector{T}(undef,ndata)
  _generate_data_and_ptrs_fill_data!(data,vv)
  (data, ptrs)
end

function _generate_data_and_ptrs_fill_ptrs!(ptrs,vv)
  k = 1
  for i in eachindex(vv)
    v = vv[i]
    ptrs[k+1] = length(v)
    k += 1
  end
end

function _generate_data_and_ptrs_fill_data!(data,vv)
  k = 1
  for i in eachindex(vv)
    v = vv[i]
    for vi in v
      data[k] = vi
      k += 1
    end
  end
end

function length_to_ptrs!(ptrs::AbstractArray{<:Integer})
  ptrs[1] = 1
  @inbounds for i in 1:(length(ptrs)-1)
    ptrs[i+1] += ptrs[i]
  end
end

function counts_to_ptrs(counts)
  n = length(counts)
  ptrs = Vector{Int32}(undef,n+1)
  @inbounds for i in 1:n
    ptrs[i+1] = counts[i]
  end
  length_to_ptrs!(ptrs)
  ptrs
end

function ptrs_to_counts(ptrs)
  counts = similar(ptrs,eltype(ptrs),length(ptrs)-1)
  @inbounds for i in 1:length(counts)
    counts[i] = ptrs[i+1]-ptrs[i]
  end
  counts
end

function rewind_ptrs!(ptrs::AbstractVector{<:Integer})
  @inbounds for i in (length(ptrs)-1):-1:1
    ptrs[i+1] = ptrs[i]
  end
  ptrs[1] = 1
end

