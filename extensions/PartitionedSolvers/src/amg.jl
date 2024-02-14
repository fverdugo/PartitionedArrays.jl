
function aggregate(A,diagA=diag(A);epsilon)
    # TODO It assumes CSC format for the moment

    # This one is algorithm 5.1 from
    # "Algebraic multigrid by smoothed aggregation for second and fourth order elliptic problems"

    epsi = epsilon
    typeof_aggregate = Int32
    typeof_strength = eltype(A.nzval)

    nnodes = size(A,1)
    pending = typeof_aggregate(0)
    isolated = typeof_aggregate(-1)
    
    node_to_aggregate = fill(pending,nnodes)
    node_to_old_aggregate = similar(node_to_aggregate)

    node_to_neigs = jagged_array(A.rowval,A.colptr)
    node_to_vals = jagged_array(A.nzval,A.colptr)
    strongly_connected = (node,ineig) -> begin
        neig = node_to_neigs[node][ineig]
        aii = diagA[node]
        ajj = diagA[neig]
        aij = node_to_vals[node][ineig]
        abs(aij) > epsi*sqrt(aii*ajj)
    end
    coupling_strength = (node,ineig) -> begin
        abs(node_to_vals[node][ineig])
    end

    # Initialization
    for node in 1:nnodes
        neigs = node_to_neigs[node]
        isolated_node = count(i->i!=node,neigs) == 0
        if isolated_node
            node_to_aggregate[node] = isolated
        end
    end

    # Step 1
    aggregate = typeof_aggregate(0)
    for node in 1:nnodes
        if node_to_aggregate[node] != pending
            continue
        end
        neigs = node_to_neigs[node]
        nneigs = length(neigs)
        all_pending = true
        for ineig in 1:nneigs
            neig = neigs[ineig]
            if neig == node || !strongly_connected(node,ineig)
                continue
            end
            all_pending &= (node_to_aggregate[neig] == pending)
        end
        if !all_pending
            continue
        end
        aggregate += typeof_aggregate(1)
        node_to_aggregate[node] = aggregate
        for ineig in 1:nneigs
            neig = neigs[ineig]
            if neig == node || !strongly_connected(node,ineig)
                continue
            end
            node_to_aggregate[neig] = aggregate
        end
    end

    # Step 2
    copy!(node_to_old_aggregate,node_to_aggregate)
    for node in 1:nnodes
        if node_to_aggregate[node] != pending
            continue
        end
        strength = zero(typeof_strength)
        neigs = node_to_neigs[node]
        nneigs = length(neigs)
        for ineig in 1:nneigs
            neig = neigs[ineig]
            if neig == node || !strongly_connected(node,ineig)
                continue
            end
            neig_aggregate = node_to_old_aggregate[neig]
            if neig_aggregate != pending && neig_aggregate != isolated
                neig_strength = coupling_strength(node,ineig)
                if neig_strength > strength
                    strength = neig_strength
                    node_to_aggregate[node] = neig_aggregate
                end
            end
        end
    end

    # Step 3
    for node in 1:nnodes
        if node_to_aggregate[node] != pending
            continue
        end
        aggregate += typeof_aggregate(1)
        node_to_aggregate[node] = aggregate
        neigs = node_to_neigs[node]
        nneigs = length(neigs)
        for ineig in 1:nneigs
            neig = neigs[ineig]
            if neig == node || !strongly_connected(node,ineig)
                continue
            end
            neig_aggregate = node_to_old_aggregate[neig]
            if neig_aggregate == pending || neig_aggregate == isolated
                node_to_aggregate[neig] = aggregate
            end
        end
    end
    naggregates = aggregate

    node_to_aggregate, 1:naggregates
end

function aggregate(A::PSparseMatrix,diagA=diag(A);kwargs...)
    # This is the vanilla "uncoupled" strategy from "Parallel Smoothed Aggregation Multigrid : Aggregation Strategies on Massively Parallel Machines"
    # TODO: implement other more advanced strategies
    @assert A.assembled
    node_to_aggregate_data, local_ranges = map((A,diagA)->aggregate(A,diagA;kwargs...),own_own_values(A),own_values(diagA)) |> tuple_of_arrays
    nown = map(length,local_ranges)
    n = sum(nown)
    aggregate_partition = variable_partition(nown,n)
    node_partition = partition(axes(A,1))
    map(map_own_to_global!,node_to_aggregate_data,aggregate_partition)
    node_to_aggregate = PVector(node_to_aggregate_data,node_partition)
    node_to_aggregate, PRange(aggregate_partition)
end

function tentative_prolongator(node_to_aggregate,aggregates)
    typeof_aggregate = eltype(node_to_aggregate)
    nnodes = length(node_to_aggregate)
    pending = typeof_aggregate(0)
    isolated = typeof_aggregate(-1)
    nnodes = length(node_to_aggregate)
    naggregates = length(aggregates)
    aggregate_to_nodes_ptrs = zeros(Int,naggregates+1)
    for node in 1:nnodes
        agg = node_to_aggregate[node]
        if agg == pending
            continue
        end
        aggregate_to_nodes_ptrs[agg+1] += 1
    end
    length_to_ptrs!(aggregate_to_nodes_ptrs)
    ndata = aggregate_to_nodes_ptrs[end]-1
    aggregate_to_nodes_data = zeros(Int,ndata)
    for node in 1:nnodes
        agg = node_to_aggregate[node]
        if agg == pending
            continue
        end
        p = aggregate_to_nodes_ptrs[agg]
        aggregate_to_nodes_data[p] = node
        aggregate_to_nodes_ptrs[agg] += 1
    end
    rewind_ptrs!(aggregate_to_nodes_ptrs)

    P0 = SparseMatrixCSC(
        nnodes,
        naggregates,
        aggregate_to_nodes_ptrs,
        aggregate_to_nodes_data,
        ones(ndata))
    P0
end

function tentative_prolongator(node_to_aggregate::PVector,aggregates::PRange)
    function setup_triplets(node_to_aggregate,nodes)
        myI = 1:local_length(nodes)
        myJ = node_to_aggregate
        myV = ones(length(node_to_aggregate))
        (myI,myJ,myV)
    end
    node_partition = partition(axes(node_to_aggregate,1))
    I,J,V = map(setup_triplets,partition(node_to_aggregate),node_partition) |> tuple_of_arrays
    aggregate_partition = partition(aggregates)
    J_owner = find_owner(aggregate_partition,J)
    aggregate_partition = map(union_ghost,aggregate_partition,J,J_owner)
    map(map_global_to_local!,J,aggregate_partition)
    P0 = psparse(I,J,V,node_partition,aggregate_partition;assembled=true,indices=:local) |> fetch
    P0
end

function smoothed_prolongator(A,P0,diagA=diag(A);omega)
    Dinv = sparse_diag(1 ./ diagA,(axes(A,1),axes(A,1)))
    P = (I-omega*Dinv*A)*P0
    P
end

function smoothed_aggregation(;epsilon,omega)
    function coarsen(problem)
        A = matrix(problem)
        B = nullspace(problem)
        diagA = diag(A)
        node_to_aggregate, aggregates = aggregate(A,diagA;epsilon)
        P0 = tentative_prolongator(node_to_aggregate, aggregates)
        P = smoothed_prolongator(A,P0,diagA;omega)
        R = transpose(P)
        Ac,cache = rap(R,A,P;reuse=true)
        #TODO enhance the partition of Ac,R,P
        #TODO null space for vector-valued problems
        Bc = default_nullspace(Ac)
        # TODO enhance partitioning for Ac,R,P
        b = rhs(problem)
        bc = similar(b,axes(Ac,1))
        coarse_problem = linear_problem(Ac,bc,Bc)
        coarse_problem,R,P,cache
    end
    function coarsen!(problem,coarse_problem,R,P,cache)
        A = matrix(problem)
        Ac = matrix(coarse_problem)
        rap!(Ac,R,A,P,cache)
        coarse_problem,R,P,cache
    end
    (coarsen, coarsen!)
end

function default_amg_fine_params()
    #TODO more resonable defaults?
    pre_smoother = jacobi(;maxiters=10,omega=2/3)
    coarsening = smoothed_aggregation(;epsilon=0,omega=1)
    cycle = v_cycle()
    pos_smoother = pre_smoother
    level_params = (;pre_smoother,pos_smoother,coarsening,cycle)
    nfine = 3
    fine_params = fill(level_params,nfine)
    fine_params
end

function default_amg_coarse_params()
    coarse_solver = lu_solver()
    coarse_size = 10
    coarse_params = (;coarse_solver,coarse_size)
    coarse_params
end

function amg(;
        fine_params=default_amg_fine_params(),
        coarse_params=default_amg_coarse_params(),
    )
    AMG(fine_params,coarse_params)
end

struct AMG{A,B} <: AbstractLinearSolver
    fine_params::A
    coarse_params::B
end

setup(amg::AMG) = (problem,x) -> amg_setup(amg,problem,x)

function amg_setup(amg,problem,x)
    fine_params = amg.fine_params
    coarse_params = amg.coarse_params
    (;coarse_solver,coarse_size) = coarse_params
    done = false
    fine_levels =  map(fine_params) do fine_level
        if done
            return nothing
        end
        (;pre_smoother,pos_smoother,coarsening,cycle) = fine_level
        pre_setup = setup(pre_smoother)(problem,x)
        pos_setup = setup(pos_smoother)(problem,x)
        coarsen, _ = coarsening
        coarse_problem,R,P,coarse_problem_setup = coarsen(problem)
        Ac = matrix(coarse_problem)
        nc = size(Ac,1)
        if nc <= coarse_size
            done = true
        end
        r = similar(rhs(problem))
        e = similar(x)
        ec = similar(e,axes(Ac,2))
        level_setup = (;R,P,r,e,ec,coarse_problem,pre_setup,pos_setup,coarse_problem_setup)
        x = ec
        problem = coarse_problem
        level_setup
    end
    n_fine_levels = count(i->i!==nothing,fine_levels)
    nlevels = n_fine_levels+1
    coarse_solver_setup = setup(coarse_solver)(problem,x)
    coarse_level = (;coarse_solver_setup)
    (;nlevels,fine_levels,coarse_level)
end

function use!(amg::AMG)
    function use_amg!(problem,x,setup)
        level=1
        amg_cycle!(amg,x,problem,setup,level)
        x
    end
end

function amg_cycle!(amg,x,problem,setup,level)
    if level == setup.nlevels
        coarse_solver = amg.coarse_params.coarse_solver
        coarse_solver_setup = setup.coarse_level.coarse_solver_setup
        return use!(coarse_solver)(problem,x,coarse_solver_setup)
    end
    level_params = amg.fine_params[level]
    level_setup = setup.fine_levels[level]
    (;pre_smoother,pos_smoother,cycle) = level_params
    (;R,P,r,e,ec,coarse_problem,pre_setup,pos_setup) = level_setup
    use!(pre_smoother)(problem,x,pre_setup)
    A = matrix(problem)
    b = rhs(problem)
    mul!(r,A,x)
    r .= r .- b
    rc = rhs(coarse_problem)
    mul!(rc,R,r)
    fill!(ec,zero(eltype(ec)))
    cycle(amg,ec,coarse_problem,setup,level+1)
    mul!(e,P,ec)
    x .-= e
    use!(pos_smoother)(problem,x,pos_setup)
    x
end

function v_cycle()
    amg_cycle!
end

function w_cycle()
    function cycle(args...)
        amg_cycle!(args...)
        amg_cycle!(args...)
    end
end

setup!(amg::AMG) = (problem,x,setup) -> amg_setup!(amg,problem,x,setup)

function amg_setup!(amg,problem,x,setup)
    nlevels = setup.nlevels
    for level in 1:(nlevels-1)
        level_params = amg.fine_params[level]
        level_setup = setup.fine_levels[level]
        (;coarsening,pre_smoother,pos_smoother) = level_params
        _, coarsen! = coarsening
        (;R,P,r,e,ec,coarse_problem,pre_setup,pos_setup,coarse_problem_setup) = level_setup
        setup!(pre_smoother)(problem,x,pre_setup)
        setup!(pos_smoother)(problem,x,pos_setup)
        coarsen!(problem,coarse_problem,R,P,coarse_problem_setup)
        problem = coarse_problem
        x = ec
    end
    coarse_solver = amg.coarse_params.coarse_solver
    coarse_solver_setup = setup.coarse_level.coarse_solver_setup
    setup!(coarse_solver)(problem,x,coarse_solver_setup)
    setup
end

finalize!(amg::AMG) = (setup) -> amg_finalize!(amg,setup)

function amg_finalize!(amg,setup)
    nlevels = setup.nlevels
    for level in 1:(nlevels-1)
        level_params = amg.fine_params[level]
        level_setup = setup.fine_levels[level]
        (;pre_smoother,pos_smoother) = level_params
        (;pre_setup,pos_setup) = level_setup
        finalize!(pre_smoother)(pre_setup)
        finalize!(pos_smoother)(pos_setup)
    end
    coarse_solver = amg.coarse_params.coarse_solver
    coarse_solver_setup = setup.coarse_level.coarse_solver_setup
    finalize!(coarse_solver)(coarse_solver_setup)
    nothing
end

