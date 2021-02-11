struct AdditiveSchwarz{A,B,C,D,E}
  problems::A
  solvers::B
  rhs::C
  rows::D
  cols::E
end

function AdditiveSchwarzConfig(A::PSparseMatrix)
  map_parts(A.values,A.rows.partition,A.cols.partition) do A,rows,cols
    part = rows.part
    n = 0
    for (i,j,v) in nziterator(A)
      if rows.lid_to_part[i] == part && cols.lid_to_part[j] == part
        n+=1
      end
    end
    I = zeros(Int32,n)
    J = zeros(Int32,n)
    V = zeros(eltype(A),n)
    n = 0
    for (i,j,v) in nziterator(A)
      if rows.lid_to_part[i] == part && cols.lid_to_part[j] == part
        n+=1
        I[n] = rows.lid_to_ohid[i]
        J[n] = rows.lid_to_ohid[j]
        V[n] = v
      end
    end
    B = compresscoo(A,I,J,V,num_oids(rows),num_oids(cols))
    B, zeros(eltype(A),num_oids(rows))
  end
end

function AdditiveSchwarz(setup,A::PSparseMatrix)
  problems, rhs = AdditiveSchwarzConfig(A)
  solvers = map_parts(setup,problems)
  AdditiveSchwarz(problems,solvers,rhs,A.rows,A.cols)
end

function AdditiveSchwarz!(setup!,P::AdditiveSchwarz,A::PSparseMatrix)
  map_parts(
    P.problems,
    P.solvers,
    A.values,
    A.rows.partition,
    A.cols.partition) do p,s,A,rows,cols

    part = rows.part
    n = 0
    nz = nonzeros(p)
    for (i,j,v) in nziterator(A)
      if rows.lid_to_part[i] == part && cols.lid_to_part[j] == part
        n+=1
        nz[n] = v
      end
    end
    setup!(s,p)
    nothing
  end
  P
end

function LinearAlgebra.ldiv!(y::PVector,P::AdditiveSchwarz,b::PVector)
  @check oids_are_equal(P.rows,b.rows)
  @check oids_are_equal(P.cols,y.rows)
  map_parts(
    y.owned_values,P.solvers,P.rhs,b.owned_values) do y,s,r,b
    r .= b
    ldiv!(s,r)
    y .= r
    nothing
  end
  y
end

function LinearAlgebra.ldiv!(P::AdditiveSchwarz,b::PVector)
  ldiv!(b,P,b)
end

function Base.:\(P::AdditiveSchwarz,b::PVector)
  y = similar(b,eltype(b),P.cols)
  ldiv!(y,P,b)
  y
end

