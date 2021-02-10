
compresscoo(a::AbstractSparseMatrix,args...) = compresscoo(typeof(a),args...)

function compresscoo(
  ::Type{<:AbstractSparseMatrix},
  I::AbstractVector,
  J::AbstractVector,
  V::AbstractVector,
  n::Integer,
  m::Integer,
  combine=+)
  @notimplemented
end

function compresscoo(
  ::Type{SparseMatrixCSC{Tv,Ti}},
  I::AbstractVector,
  J::AbstractVector,
  V::AbstractVector,
  m::Integer,
  n::Integer,
  combine=+) where {Tv,Ti}

  sparse(
    EltypeVector(Ti,I),
    EltypeVector(Ti,J),
    EltypeVector(Tv,V),
    m,n,combine)
end

struct EltypeVector{T,V} <: AbstractVector{T}
  parent::V
  function EltypeVector(::Type{T},parent::V) where {T,V<:AbstractVector}
    new{T,V}(parent)
  end
end
EltypeVector(::Type{T},parent::AbstractVector{T}) where T = parent
Base.size(v::EltypeVector) = size(v.parent)
Base.axes(v::EltypeVector) = axes(v.parent)
Base.@propagate_inbounds Base.getindex(v::EltypeVector{T},i::Integer) where T = convert(T,v.parent[i])
Base.@propagate_inbounds Base.setindex!(v::EltypeVector,w,i::Integer) = (v.parent[i] = w)
Base.IndexStyle(::Type{<:EltypeVector{T,V}}) where {T,V} = IndexStyle(V)

function nzindex(A::AbstractSparseMatrix, i0::Integer, i1::Integer)
  @notimplemented
end

function nzindex(A::SparseArrays.AbstractSparseMatrixCSC, i0::Integer, i1::Integer)
    if !(1 <= i0 <= size(A, 1) && 1 <= i1 <= size(A, 2)); throw(BoundsError()); end
    ptrs = SparseArrays.getcolptr(A)
    r1 = Int(ptrs[i1])
    r2 = Int(ptrs[i1+1]-1)
    (r1 > r2) && return -1
    r1 = searchsortedfirst(rowvals(A), i0, r1, r2, Base.Order.Forward)
    ((r1 > r2) || (rowvals(A)[r1] != i0)) ? -1 : r1
end

nziterator(a::AbstractSparseMatrix) = @notimplemented

nziterator(a::SparseArrays.AbstractSparseMatrixCSC) = NZIteratorCSC(a)

struct NZIteratorCSC{A}
  matrix::A
end

Base.length(a::NZIteratorCSC) = nnz(a.matrix)
Base.eltype(::Type{<:NZIteratorCSC{A}}) where A = Tuple{Int,Int,eltype(A)}
Base.eltype(::T) where T <: NZIteratorCSC = eltype(T)
Base.IteratorSize(::Type{<:NZIteratorCSC}) = Base.HasLength()
Base.IteratorEltype(::Type{<:NZIteratorCSC}) = Base.HasEltype()

@inline function Base.iterate(a::NZIteratorCSC)
  if nnz(a.matrix) == 0
    return nothing
  end
  col = 0
  ptrs = SparseArrays.getcolptr(a.matrix)
  knext = nothing
  while knext === nothing
    col += 1
    ks = ptrs[col]:(ptrs[col+1]-1)
    knext = iterate(ks)
  end
  k, kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = nonzeros(a.matrix)[k]
  (i,j,v), (col,kstate)
end

@inline function Base.iterate(a::NZIteratorCSC,state)
  ptrs = SparseArrays.getcolptr(a.matrix)
  col, kstate = state
  ks = ptrs[col]:(ptrs[col+1]-1)
  knext = iterate(ks,kstate)
  if knext === nothing
    while knext === nothing
      if col == size(a.matrix,2)
        return nothing
      end
      col += 1
      ks = ptrs[col]:(ptrs[col+1]-1)
      knext = iterate(ks)
    end
  end
  k, kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = nonzeros(a.matrix)[k]
  (i,j,v), (col,kstate)
end

struct SubSparseMatrix{T,A,B,C} <: AbstractMatrix{T}
  parent::A
  indices::B
  inv_indices::C
  flag::Tuple{Int,Int}
  function SubSparseMatrix(
    parent::AbstractSparseMatrix{T},
    indices::Tuple,
    inv_indices::Tuple,
    flag::Tuple{Int,Int}) where T

    A = typeof(parent)
    B = typeof(indices)
    C = typeof(inv_indices)
    new{T,A,B,C}(parent,indices,inv_indices,flag)
  end
end

Base.size(a::SubSparseMatrix) = map(length,a.indices)
Base.IndexStyle(::Type{<:SubSparseMatrix}) = IndexCartesian()
function Base.getindex(a::SubSparseMatrix,i::Integer,j::Integer)
  I = a.indices[1][i]
  J = a.indices[2][j]
  a.parent[I,J]
end

function LinearAlgebra.mul!(
  C::AbstractVector,
  A::SubSparseMatrix,
  B::AbstractVector,
  α::Number,
  β::Number)

  @notimplemented
end

function LinearAlgebra.mul!(
  C::AbstractVector,
  A::SubSparseMatrix{T,<:SparseArrays.AbstractSparseMatrixCSC} where T,
  B::AbstractVector,
  α::Number,
  β::Number)

  size(A, 2) == size(B, 1) || throw(DimensionMismatch())
  size(A, 1) == size(C, 1) || throw(DimensionMismatch())
  size(B, 2) == size(C, 2) || throw(DimensionMismatch())
  if β != 1
      β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
  end
  rows, cols = A.indices
  invrows, invcols = A.inv_indices
  rflag, cflag = A.flag
  Ap = A.parent
  nzv = nonzeros(Ap)
  rv = rowvals(Ap)
  colptrs = SparseArrays.getcolptr(Ap)
  for (j,J) in enumerate(cols)
      αxj = B[j] * α
      for p = colptrs[J]:(colptrs[J+1]-1)
        I = rv[p]
        i = invrows[I]*rflag
        if i>0
          C[i] += nzv[p]*αxj
        end
      end
  end
  C
end

