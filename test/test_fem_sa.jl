using LinearAlgebra
using SparseArrays
using PartitionedArrays
using Test
using IterativeSolvers

function test_fem_sa(parts)

  u(x) = 1.0 #x[1]+x[2]
  lx = 2.0
  ls = (lx,lx)
  nx = 10
  ns = (nx,nx)
  n = prod(ns)
  h = lx/nx
  D = length(ns)
  Ae = (h/6)*[
    4.0  -1.0  -1.0  -2.0
   -1.0   4.0  -2.0  -1.0
   -1.0  -2.0   4.0  -1.0
   -2.0  -1.0  -1.0   4.0
  ]

  cis_gcells = CartesianIndices(ns)
  cis_gnodes = CartesianIndices(ns.+1)
  cis_enodes = CartesianIndices(ntuple(i->2,Val(D)))
  lis_enodes = LinearIndices(cis_enodes)
  lis_gnodes = LinearIndices(cis_gnodes)
  c1 = CartesianIndex(ntuple(i->1,Val(D)))

  cells = PRange(parts,ns)

  # Loop over owned cells, and fill the coo-vectors
  # Note that during the process we will touch remote rows and cols
  # This will be fixed later with an assembly.
  I,J,V = map_parts(cells.partition) do cells
    I = Int[]
    J = Int[]
    V = Float64[]
    for ocell in cells.oid_to_lid
      gcell = cells.lid_to_gid[ocell]
      ci_gcell = cis_gcells[gcell]
      for ci_erow in cis_enodes
        ci_grow = ci_gcell + (ci_erow - c1)
        grow = lis_gnodes[ci_grow]
        erow = lis_enodes[ci_erow]
        boundary = any(s->(1==s||s==(nx+1)),Tuple(ci_grow))
        if boundary
          push!(I,grow)
          push!(J,grow)
          push!(V,1)
        else
          for ci_ecol in cis_enodes
            ci_gcol = ci_gcell + (ci_ecol - c1)
            gcol = lis_gnodes[ci_gcol]
            ecol = lis_enodes[ci_ecol]
            push!(I,grow)
            push!(J,gcol)
            push!(V,Ae[erow,ecol])
          end
        end
      end
    end
    I,J,V
  end |> unpack

  # Create rows and cols without ghost layer
  rows = PRange(parts,ns.+1)
  cols = PRange(parts,ns.+1)

  # Add remote row gids to the rows ghost layer
  add_gids!(rows,I)

  # Start an assembly of the COO Vectors,
  # i.e., send and add the triplets (i,j,v) to the part that owns i.
  t = async_assemble!(I,J,V,rows)

  # Meanwhile we can fill the rhs.
  # Allocate it.
  b = PVector(0.0,rows)

  # Fill it.
  # we use `global_view` in order to be able to index
  # b with global ids from within the parts.
  # Note that we touch non owned rows.
  map_parts(global_view(b,b.rows),cells.partition) do b, cells
    for ocell in cells.oid_to_lid
      gcell = cells.lid_to_gid[ocell]
      ci_gcell = cis_gcells[gcell]
      for ci_erow in cis_enodes
        ci_grow = ci_gcell + (ci_erow - c1)
        grow = lis_gnodes[ci_grow]
        erow = lis_enodes[ci_erow]
        boundary = any(s->(1==s||s==(nx+1)),Tuple(ci_grow))
        if boundary
          x = (Tuple(ci_grow) .- 1) .* h
          b[grow] += u(x)
        end
      end
    end
  end

  # Wait the assembly of the coo vectors to finish
  map_parts(wait∘schedule,t)

  # Now we can add off processor col ids to the ghost layer of cols.
  add_gids!(cols,J)

  # Compress the coo vectors and build the matrix
  A = PSparseMatrix(I,J,V,rows,cols,ids=:global)

  # When filling b we have touched remote rows.
  # Send and add their contribution to the owner part
  # NOTE: calling this assembly before the previous one finishes can be
  # problematic since we can have wrongly matching snd/rcv in MPI.
  assemble!(b)

  # Setup initial and exact solution
  x = PVector(0.0,cols)
  x̂ = similar(x)
  map_parts(x.values,x̂.values,x.rows.partition) do x,x̂,rows
    for lid in rows.oid_to_lid
      grow = rows.lid_to_gid[lid]
      ci_grow = cis_gnodes[grow]
      xi = (Tuple(ci_grow) .- 1) .* h
      x̂[lid] = u(xi)
      boundary = any(s->(1==s||s==(nx+1)),Tuple(ci_grow))
      if boundary
        x[lid] = u(xi)
      end
    end
  end

  # Solve!
  IterativeSolvers.cg!(x,A,b,verbose=i_am_main(parts))

  @test norm(x-x̂) < 1.0e-5

end
