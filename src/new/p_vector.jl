
abstract type AbstractLocalVector{T} end

function get_local_values end

function get_own_values end

function get_ghost_values end

function similar_local_vector end

struct LocalVector{T} <: AbstractLocalVector{T}
    local_values::Vector{T}
end

function similar_local_vector(::Type{<:LocalVector},::Type{T},indices) where T
    n_local = get_n_local(indices)
    local_values = Vector{T}(undef,n_local)
    LocalVector(local_values)
end

function get_local_values(values::LocalVector,indices)
    values.local_values
end

function get_own_values(values::LocalVector,indices)
    own_to_local = get_own_to_local(indices)
    view(values.local_values,own_to_local)
end

function get_ghost_values(values::LocalVector,indices)
    ghost_to_local = get_ghost_to_local(indices)
    view(values.local_values,ghost_to_local)
end

struct OwnAndGhostVectors{T} <: AbstractLocalVector{T}
    own_values::Vector{T}
    ghost_values::Vector{T}
end

function similar_local_vector(::Type{<:OwnAndGhostVectors},::Type{T},indices) where T
    n_own = get_n_own(indices)
    n_ghost = get_n_ghost(indices)
    own_values = Vector{T}(undef,n_own)
    ghost_values = Vector{T}(undef,n_ghost)
    OwnAndGhostVectors(own_values,ghost_values)
end

function get_local_values(values::LocalVector,indices)
    perm = get_permutation(indices)
    LocalToValue(values.own_values,values.ghost_values,perm)
end

function get_own_values(values::LocalVector,indices)
    values.own_values
end

function get_ghost_values(values::LocalVector,indices)
    values.ghost_values
end

struct LocalToValue{T,C} <: AbstractVector{T}
    own_values::Vector{T}
    ghost_values::Vector{T}
    perm::C
end
Base.IndexStyle(::Type{<:LocalToValue}) = IndexLinear()
Base.size(a::LocalToValue) = (length(a.own_values)+length(a.ghost_values),)
function Base.getindex(a::LocalToValue,local_id::Int)
    n_own = length(a.own_values)
    j = a.perm[local_id]
    if j > n_own
        a.ghost_values[j-n_own]
    else
        a.own_values[j]
    end
end

struct PVector{T,A,B} <: AbstractVector{T}
    values::A
    rows::B
end

function get_local_values(a::PVector)
    map(get_local_values,a.values,rows.indices)
end

function get_own_values(a::PVector)
    map(get_own_values,a.values,rows.indices)
end

function get_ghost_values(a::PVector)
    map(get_ghost_values,a.values,rows.indices)
end

Base.size(a::PVector) = (length(a.rows),)
Base.axes(a::PVector) = (a.rows,)
Base.IndexStyle(::Type{<:PVector}) = IndexLinear()
function Base.getindex(a::PVector,gid::Int)
    scalar_indexing_action(a)
end

function Base.similar(a::PVector)
    similar(a,eltype(a),axes(a))
end

function Base.similar(a::PVector,::Type{T}) where T
    similar(a,T,axes(a))
end

function Base.similar(a::PVector,::Type{T},axes::Tuple) where T
    error("Not enough information to build a PVector")
end

function Base.similar(a::PVector,::Type{T},axes::Tuple{<:PRange}) where T
    rows = axes[1]
    values = map(a.values,rows.indices) do values, indices
        similar_local_vector(typeof(values),T,indices)
    end
    PVector(values,rows)
end

function Base.similar(::Type{<:PVector},axes::Tuple)
    error("Not enough information to build a PVector")
end

function Base.similar(P::Type{<:PVector{A}},axes::Tuple{<:PRange}) where A
  T = eltype(P)
  V = eltype(A)
  rows = axes[1]
  values = map(rows.indices) do indices
      similar_local_vector(V,T,indices)
  end
  PVector(values,rows)
end


