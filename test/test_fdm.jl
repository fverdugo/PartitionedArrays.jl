
using LinearAlgebra
using SparseArrays
using DistributedDataDraft
using Test
using IterativeSolvers

function test_fdm(parts)

  lx = 2.0
  ls = (lx,lx,lx)
  nx = 10
  ns = (nx,nx,nx)
  n = prod(ns)
  h = lx/nx
  points = [(0,0,0),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
  coeffs = [-6,1,1,1,1,1,1]/(h^2)
  stencil = [ (coeff,CartesianIndex(point)) for (coeff,point) in zip(coeffs,points) ]

  #lx = 2.0
  #ls = (lx,lx)
  #nx = 4
  #ns = (nx,nx)
  #n = prod(ns)
  #h = lx/nx
  #points = [(0,0),(-1,0),(1,0),(0,-1),(0,1)]
  #coeffs = [-4,1,1,1,1]/(h^2)
  #stencil = [ (coeff,CartesianIndex(point)) for (coeff,point) in zip(coeffs,points) ]

  # TODO create an empty Exchanger
  # TODO allow to pass ns to have a better partition
  rows = DistributedRange(parts,n)
  
  I,J,V = map_parts(rows.lids) do rows
    cis = CartesianIndices(ns)
    lis = LinearIndices(cis)
    I = Int[]
    J = Int[]
    V = Float64[]
    for lid in rows.oid_to_lid
      i = rows.lid_to_gid[lid]
      ci = cis[i]
      for (v,dcj) in stencil
        cj = ci + dcj
        if all(s->(1<=s&&s<=nx),Tuple(cj))
          j = lis[cj]
          push!(I,i)
          push!(J,j)
          push!(V,-v)
        end
      end
    end
    I,J,V
  end

  add_gid!(rows,I)
  add_gid!(rows,J)
  # TODO do not create an Exchanger if not needed
  A = DistributedSparseMatrix(I,J,V,rows,rows;ids=:global)

  #display(rows.lids)

  #display(A.values)

  b = DistributedVector{eltype(A)}(undef,rows)

  f(x) = x[1]+x[2]

  map_parts(b.values,rows.lids) do b,rows
    cis = CartesianIndices(ns)
    #b .= 0
    for lid in rows.oid_to_lid
      i = rows.lid_to_gid[lid]
      ci = cis[i]
      xi = (Tuple(ci) .- 1) .* h
      b[lid] = f(xi)
    end
    b
  end

  P = Jacobi(A)

  x = IterativeSolvers.cg(A,b,verbose=i_am_master(parts))
  x = IterativeSolvers.cg(A,b,verbose=i_am_master(parts),Pl=P)
  exchange!(x)

  #display(P.diaginv)

  #display(b.values)
  #display(x.values)

end
