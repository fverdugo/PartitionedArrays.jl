using Gridap

u_2d(x) = x[1]+x[2]
u_3d(x) = x[1]+x[2]+x[3]
domain_2d = (0,1,0,1)
domain_3d = (0,1,0,1,0,1)
cells_2d = (4,4)
cells_3d = (4,4,4)

for cells in (cells_2d,cells_3d)
  domain = length(cells) == 3 ? domain_3d : domain_2d
  u = length(cells) == 3 ? u_3d : u_2d
  model = CartesianDiscreteModel(domain,cells)
  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)
  a(u,v) = ∫( ∇(u)⋅∇(v) )dΩ
  l(v) = 0
  op = AffineFEOperator(a,l,U,V)
  uh = solve(op)
  sum(∫( abs2(u-uh) )dΩ)
end

