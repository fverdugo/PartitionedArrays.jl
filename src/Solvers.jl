mutable struct AdditiveSchwarz{A,B,C,D}
  problems::A
  solvers::B
  rhs::C
  matrix::D
end

function _increase_overlap(a::PSparseMatrix)
  @check oids_are_equal(a.rows,a.cols)
  I,J,V = map_parts(collectcoo,a.values)
  to_gids!(I,a.rows)
  to_gids!(J,a.cols)
  rows = a.cols
  exchange!(I,J,V,rows)
  cols = add_gids(rows,J)
  exchanger = empty_exchanger(rows.partition)
  T = eltype(a.values)
  PSparseMatrix(
    (args...)->compresscoo(T,args...),
    I,J,V,rows,cols,exchanger,ids=:global)
end

function _increase_overlap(a::PSparseMatrix,width::Integer)
  if width == 0
    return a
  else
    b = _increase_overlap(a)
    return _increase_overlap(b,width-1)
  end
end

function AdditiveSchwarzConfig(A::PSparseMatrix)
  @check oids_are_equal(A.rows,A.cols)
  I,J,V = map_parts(A.values,A.rows.partition,A.cols.partition) do A,rows,cols
    n = 0
    for (i,j,v) in nziterator(A)
      if haskey(rows.gid_to_lid,cols.lid_to_gid[j])
        n+=1
      end
    end
    I = zeros(Int32,n)
    J = zeros(Int32,n)
    V = zeros(eltype(A),n)
    n = 0
    for (i,j,v) in nziterator(A)
      gj = cols.lid_to_gid[j]
      if haskey(rows.gid_to_lid,gj)
        n+=1
        I[n] = i
        J[n] = rows.gid_to_lid[gj]
        V[n] = v
      end
    end
    I,J,V
  end
  T = eltype(A.values)
  problems = PSparseMatrix(
    (args...)->compresscoo(T,args...),
    I,J,V,A.rows,A.rows,ids=:local)
  rhs = PVector{eltype(A)}(undef,A.rows)
  problems, rhs
end

function AdditiveSchwarz(setup,A::PSparseMatrix,width::Integer)
  B = _increase_overlap(A,width)
  problems, rhs = AdditiveSchwarzConfig(B)
  solvers = map_parts(setup,problems.values)
  AdditiveSchwarz(problems,solvers,rhs,A)
end

function AdditiveSchwarz!(setup!,P::AdditiveSchwarz,A::PSparseMatrix)
  @check oids_are_equal(P.problems.rows,A.rows)
  map_parts(
    P.problems.values,
    P.problems.rows.partition,
    P.problems.cols.partition,
    A.values,
    A.rows.partition,
    A.cols.partition) do P,rowsP,colsP,A,rows,cols
    nz = nonzeros(P)
    for (i,j,v) in nziterator(A)
      oi = rows.lid_to_ohid[i]
      if oi>0
        gj = cols.lid_to_gid[j]
        if haskey(colsP.gid_to_lid,gj)
          li = rowsP.oid_to_lid[oi]
          lj = colsP.gid_to_lid[gj]
          n = nzindex(P,li,lj)
          nz[n] = v
        end
      end
    end
  end
  exchange!(P.problems)
  map_parts(setup!,P.solvers,P.problems.values)
  P.matrix = A
  P
end

function LinearAlgebra.ldiv!(y::PVector,P::AdditiveSchwarz,b::PVector)
  @check oids_are_equal(P.problems.rows,b.rows)
  @check oids_are_equal(P.problems.cols,y.rows)
  P.rhs .= b
  exchange!(P.rhs)
  map_parts(ldiv!,P.solvers,P.rhs.values)
  y .= P.rhs
end

function LinearAlgebra.ldiv!(P::AdditiveSchwarz,b::PVector)
  ldiv!(b,P,b)
end

function Base.:\(P::AdditiveSchwarz,b::PVector)
  y = similar(b,eltype(b),P.matrix.cols)
  ldiv!(y,P,b)
  y
end

