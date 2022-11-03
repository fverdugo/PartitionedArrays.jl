
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
  knext = nothing
  while knext === nothing
    col += 1
    ks = nzrange(a.matrix,col)
    knext = iterate(ks)
  end
  k, kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = nonzeros(a.matrix)[k]
  (i,j,v), (col,kstate)
end

@inline function Base.iterate(a::NZIteratorCSC,state)
  col, kstate = state
  ks = nzrange(a.matrix,col)
  knext = iterate(ks,kstate)
  if knext === nothing
    while knext === nothing
      if col == size(a.matrix,2)
        return nothing
      end
      col += 1
      ks = nzrange(a.matrix,col)
      knext = iterate(ks)
    end
  end
  k, kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = nonzeros(a.matrix)[k]
  (i,j,v), (col,kstate)
end
