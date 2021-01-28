
using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using LinearAlgebra

import PartitionedArrays
const PArrays = PartitionedArrays

function bench(parts,n,title)

  u_2d(x) = x[1]+x[2]
  u_3d(x) = x[1]+x[2]+x[3]
  domain_2d = (0,1,0,1)
  domain_3d = (0,1,0,1,0,1)

  domain = length(n) == 3 ? domain_3d : domain_2d
  u = length(n) == 3 ? u_3d : u_2d
  order = 1

  t = PArrays.PTimer(parts)
  PArrays.tic!(t)

  # Partition of the Cartesian ids
  # with ghost layer
  cell_gcis = PArrays.PCartesianIndices(parts,n,PArrays.with_ghost)
  PArrays.toc!(t,"cell_gcis")

  # Local discrete models
  model = PArrays.map_parts(cell_gcis) do gcis
    cmin = first(gcis)
    cmax = last(gcis)
    desc = CartesianDescriptor(domain,n)
    CartesianDiscreteModel(desc,cmin,cmax)
  end
  PArrays.toc!(t,"model")

  # Partitioned range of cells
  # with ghost layer
  cell_range = PArrays.PRange(parts,n,PArrays.with_ghost)
  PArrays.toc!(t,"cell_range")

  # Local FE spaces
  U, V = PArrays.map_parts(model) do model
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe;dirichlet_tags="boundary")
    U = TrialFESpace(u,V)
    U, V
  end
  PArrays.toc!(t,"U, V")

  # Cell-wise local dofs
  cell_to_ldofs, nldofs = PArrays.map_parts(V) do V
    get_cell_dof_ids(V), num_free_dofs(V)
  end
  PArrays.toc!(t,"cell_to_ldofs, nldofs")

  # Find and count number owned dofs
  ldof_to_part, nodofs = PArrays.map_parts(
    cell_range.partition,cell_to_ldofs,nldofs) do partition,cell_to_ldofs,nldofs

    ldof_to_part = fill(Int32(0),nldofs)
    cache = array_cache(cell_to_ldofs)
    for cell in 1:length(cell_to_ldofs)
      owner = partition.lid_to_part[cell]
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      for ldof in ldofs
        if ldof>0
          ldof_to_part[ldof] = max(owner,ldof_to_part[ldof])
        end
      end
    end
    nodofs = count(p->p==partition.part,ldof_to_part)
    ldof_to_part, nodofs
  end
  PArrays.toc!(t,"ldof_to_part, nodofs")

  # Find the global range of owned dofs
  first_gdof = PArrays.xscan(+,nodofs,init=1)
  PArrays.toc!(t,"first_gdof")

  # Distribute gdofs to owned ones
  ldof_to_gdof = PArrays.map_parts(
    parts,first_gdof,ldof_to_part) do part,first_gdof,ldof_to_part

    offset = first_gdof-1
    ldof_to_gdof = Vector{Int}(undef,length(ldof_to_part))
    odof = 0
    gdof = 0
    for (ldof,owner) in enumerate(ldof_to_part)
      if owner == part
        odof += 1
        ldof_to_gdof[ldof] = odof
      else
        ldof_to_gdof[ldof] = gdof
      end
    end
    for (ldof,owner) in enumerate(ldof_to_part)
      if owner == part
        ldof_to_gdof[ldof] += offset
      end
    end
    ldof_to_gdof
  end
  PArrays.toc!(t,"ldof_to_gdof (owned)")

  # Create cell-wise global dofs
  cell_to_gdofs = PArrays.map_parts(
    parts,
    ldof_to_gdof,cell_to_ldofs,cell_range.partition) do part,
    ldof_to_gdof,cell_to_ldofs,partition

    cache = array_cache(cell_to_ldofs)
    ncells = length(cell_to_ldofs)
    ptrs = Vector{Int32}(undef,ncells+1)
    for cell in 1:ncells
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      ptrs[cell+1] = length(ldofs)
    end
    PArrays.length_to_ptrs!(ptrs)
    ndata = ptrs[end]-1
    data = Vector{Int}(undef,ndata)
    gdof = 0
    for cell in partition.oid_to_lid
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      p = ptrs[cell]-1
      for (i,ldof) in enumerate(ldofs)
        if ldof > 0 && ldof_to_gdof[ldof] != gdof
          data[i+p] = ldof_to_gdof[ldof]
        end
      end
    end
    PArrays.Table(data,ptrs)
  end
  PArrays.toc!(t,"cell_to_gdofs (owned)")

  # Exchange the global dofs
  PArrays.exchange!(cell_to_gdofs,cell_range.exchanger)
  PArrays.toc!(t,"cell_to_gdofs (ghost)")

  # Distribute global dof ids also to ghost
  PArrays.map_parts(
    parts,
    cell_to_ldofs,cell_to_gdofs,ldof_to_gdof,ldof_to_part,cell_range.partition) do part,
    cell_to_ldofs,cell_to_gdofs,ldof_to_gdof,ldof_to_part,partition

    gdof = 0
    cache = array_cache(cell_to_ldofs)
    for cell in partition.hid_to_lid
      ldofs = getindex!(cache,cell_to_ldofs,cell)
      p = cell_to_gdofs.ptrs[cell]-1
      for (i,ldof) in enumerate(ldofs)
        if ldof > 0 && ldof_to_part[ldof] == partition.lid_to_part[cell]
          ldof_to_gdof[ldof] = cell_to_gdofs.data[i+p]
        end
      end
    end
  end
  PArrays.toc!(t,"ldof_to_gdof (ghost)")

  # Setup Integration (only for owned cells)
  Ω, dΩ = PArrays.map_parts(cell_range.partition,model) do partition, model
    Ω = Triangulation(model,partition.oid_to_lid)
    dΩ = Measure(Ω,2*order)
    Ω, dΩ
  end
  PArrays.toc!(t,"Ω, dΩ")

  # Integrate the coo vectors
  I,J,C,vec = PArrays.map_parts(Ω,dΩ,U,V,ldof_to_gdof) do Ω,dΩ,U,V,ldof_to_gdof
    v = get_cell_shapefuns(V)
    u = get_cell_shapefuns_trial(U)
    cellmat = ∫( ∇(u)⋅∇(v) )dΩ
    cellvec = 0
    uhd = zero(U)
    matvecdata = collect_cell_matrix_and_vector(cellmat,cellvec,uhd)
    assem = SparseMatrixAssembler(U,V)
    ncoo = count_matrix_and_vector_nnz_coo(assem,matvecdata)
    I = zeros(Int,ncoo)
    J = zeros(Int,ncoo)
    C = zeros(Float64,ncoo)
    vec = zeros(Float64,num_free_dofs(V))
    fill_matrix_and_vector_coo_numeric!(I,J,C,vec,assem,matvecdata)
    for i in 1:length(I)
      I[i] = ldof_to_gdof[I[i]]
      J[i] = ldof_to_gdof[J[i]]
    end
    I,J,C,vec
  end
  PArrays.toc!(t,"I,J,C,vec")

  # Create the range for rows
  rows = PArrays.PRange(parts,nodofs)
  PArrays.toc!(t,"rows")

  # Add remote rows
  PArrays.add_gid!(rows,I)
  PArrays.toc!(t,"rows (add_gid!)")

  # Move values to the owner part
  # since we have integrated only over owned cells
  assemble!(I,J,C,rows)
  PArrays.toc!(t,"I,J,C (assemble!)")

  # Create the range for rows
  cols = copy(rows)
  PArrays.toc!(t,"cols")

  # Add remote cols
  PArrays.add_gid!(cols,J)
  PArrays.toc!(t,"cols (add_gid!)")

  # Create the sparse matrix
  # TODO do not build the exchanger
  A = PArrays.PSparseMatrix(I,J,C,rows,cols,ids=:global)
  PArrays.toc!(t,"A")

  # Number of global ids
  ngdofs = length(rows)
  PArrays.toc!(t,"ngdofs")

  # Setup dof partition
  dof_partition = map_parts(parts,ldof_to_gdof,ldof_to_part) do part,ldof_to_gdof,ldof_to_part
    PArrays.IndexSet(part,ngdofs,ldof_to_gdof,ldof_to_part)
  end
  PArrays.toc!(t,"dof_partition")

  # Setup dof range
  dofs = PArrays.PRange(ngdofs,dof_partition)
  PArrays.toc!(t,"dofs")

  # Rhs aligned with the FESpace
  dof_values = PArrays.PVector(vec,dofs)
  PArrays.toc!(t,"dof_values")

  # Allocate rhs aligned with the matrix
  b = PArrays.PVector(0.0,rows)
  PArrays.toc!(t,"b (allocate)")

  # Fill rhs
  PArrays.map_parts(
    b.values,dof_values.values,rows.partition,first_gdof) do b1, b2, p1, first_gdof
    offset = first_gdof - 1
    for i in 1:length(b1)
      gdof = p1.lid_to_gid[i]
      ldof = gdof-offset
      b1[i] = b2[ldof]
    end
  end
  PArrays.toc!(t,"b (fill)")

  # Import and add remote contributions
  PArrays.assemble!(b)
  PArrays.toc!(t,"b (assemble!)")

  x = PArrays.PVector(0.0,cols)
  PArrays.toc!(t,"x")

  c = similar(b)
  PArrays.toc!(t,"c")

  mul!(c,A,x)
  PArrays.toc!(t,"A*x")

  PArrays.exchange!(x)
  PArrays.toc!(t,"x (exchange!)")

  display(t)
  PArrays.print_timer(title,t)
end
