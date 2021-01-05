
using LinearAlgebra
using SparseArrays
using DistributedDataDraft
using Test
using IterativeSolvers

function test_fdm(parts)

  u(x) = x[1]+x[2]
  f(x) = zero(x[1])

  lx = 2.0
  ls = (lx,lx,lx)
  nx = 10
  ns = (nx,nx,nx)
  n = prod(ns)
  h = lx/(nx-1)
  points = [(0,0,0),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
  coeffs = [-6,1,1,1,1,1,1]/(h^2)
  stencil = [ (coeff,CartesianIndex(point)) for (coeff,point) in zip(coeffs,points) ]

  #lx = 2.0
  #ls = (lx,lx)
  #nx = 10
  #ns = (nx,nx)
  #n = prod(ns)
  #h = lx/(nx-1)
  #points = [(0,0),(-1,0),(1,0),(0,-1),(0,1)]
  #coeffs = [-4,1,1,1,1]/(h^2)
  #stencil = [ (coeff,CartesianIndex(point)) for (coeff,point) in zip(coeffs,points) ]

  if ndims(parts) == length(ns)
    rows = DistributedRange(parts,ns)
  else
    rows = DistributedRange(parts,n)
  end
  
  I,J,V = map_parts(rows.lids) do rows
    cis = CartesianIndices(ns)
    lis = LinearIndices(cis)
    I = Int[]
    J = Int[]
    V = Float64[]
    for lid in rows.oid_to_lid
      i = rows.lid_to_gid[lid]
      ci = cis[i]
      boundary = any(s->(1==s||s==nx),Tuple(ci))
      if boundary
        push!(I,i)
        push!(J,i)
        push!(V,one(eltype(V)))
      else
        for (v,dcj) in stencil
          cj = ci + dcj
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

  # TODO a way of building the vector along the matrix
  b = DistributedVector{eltype(A)}(undef,rows)

  map_parts(b.values,rows.lids) do b,rows
    cis = CartesianIndices(ns)
    for lid in rows.oid_to_lid
      i = rows.lid_to_gid[lid]
      ci = cis[i]
      xi = (Tuple(ci) .- 1) .* h
      boundary = any(s->(1==s||s==nx),Tuple(ci))
      if boundary
        b[lid] = u(xi)
      else
        b[lid] = f(xi)
      end
    end
  end
  #exchange!(b)

  x̂ = similar(b)
  map_parts(x̂.values,rows.lids) do x̂,rows
    cis = CartesianIndices(ns)
    for lid in rows.oid_to_lid
      i = rows.lid_to_gid[lid]
      ci = cis[i]
      xi = (Tuple(ci) .- 1) .* h
      x̂[lid] = u(xi)
    end
  end

  x = copy(b)
  IterativeSolvers.cg!(x,A,b,verbose=i_am_master(parts))

  @test norm(x-x̂) < 1.0e-5

  #display(b.values)
  #display(x.values)

end
