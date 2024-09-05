using PartitionedArrays
using Test
using LinearAlgebra
using Random
using Distances

function p_vector_tests(distribute)

    np = 4
    rank = distribute(LinearIndices((np,)))
    row_partition = uniform_partition(rank,(2,2),(6,6))

    a1 = PVector(undef,row_partition)
    @test isa(axes(a1,1),PRange)
    a2 = pvector(inds->zeros(Int,length(inds)),row_partition)
    a3 = PVector{OwnAndGhostVectors{Vector{Int}}}(undef,row_partition) # TODO deprecated
    a4 = split_format(a2)
    a5 = similar(a4)
    display(a5)
    split_format!(a5,a2)
    for a in [a1,a2,a3,a4,a5]
        b = similar(a)
        b = similar(a,Int)
        b = similar(a,Int,axes(a,1))
        b = similar(typeof(a),axes(a,1))
        copy!(b,a)
        b = copy(a)
        fill!(b,5.)
        rows = axes(a,1)
        @test length(a) == length(rows)
        @test partition(axes(b,1)) === partition(rows)
        assemble!(a) |> wait
        consistent!(a) |> wait
    end

    a = pfill(4,row_partition)
    a = pfill(4,row_partition;split_format=true)
    a = pzeros(row_partition)
    a = pzeros(row_partition;split_format=true)
    a = pones(row_partition)
    a = pones(row_partition;split_format=true)
    a = prand(row_partition)
    a = prand(row_partition;split_format=true)
    a = prandn(row_partition)
    assemble!(a) |> wait
    consistent!(a) |> wait
    a = prandn(row_partition;split_format=true)
    assemble!(a) |> wait
    consistent!(a) |> wait

    aa = prand(1:10,row_partition)
    ab = split_format(aa)
    @test aa == ab
    ab = similar(ab)
    split_format!(ab,aa)
    @test aa == ab

    @test a == copy(a)

    ac = pvector_from_split_blocks(own_values(aa),ghost_values(aa),row_partition)
    @test aa == ac

    n = 10
    I,V = map(rank) do rank
        Random.seed!(rank)
        I = rand(1:n,5)
        Random.seed!(2*rank)
        V = rand(1:2,5)
        I,V
    end |> tuple_of_arrays

    row_partition = uniform_partition(rank,n)
    a = old_pvector!(I,V,row_partition) |> fetch

    @test any(i->i>n,a) == false
    @test all(i->i<n,a)
    @test minimum(a) <= maximum(a)

    b = 2*a
    b = a*2
    b = a/2
    c = a .+ a
    c = a .+ b .+ a
    c = a - b
    c = a + b

    r = reduce(+,a)
    @test sum(a) == r
    @test norm(a) > 0
    @test sqrt(a⋅a) ≈ norm(a)
    @test euclidean(a,a) + 1 ≈ 1

    n = 10
    parts = rank
    row_partition = map(parts) do part
        if part == 1
            LocalIndices(n,part,[1,2,3,5,7,8],Int32[1,1,1,2,3,3])
        elseif part == 2
            LocalIndices(n,part,[2,4,5,10],Int32[1,2,2,4])
        elseif part == 3
            LocalIndices(n,part,[6,7,8,5,4,10],Int32[3,3,3,2,2,4])
        else
            LocalIndices(n,part,[1,3,7,9,10],Int32[1,1,3,4,4])
        end
    end
    v = pzeros(row_partition)
    map(parts,partition(v),row_partition) do part, values, indices
        local_index_to_owner = local_to_owner(indices)
        for lid in 1:length(local_index_to_owner)
            owner = local_index_to_owner[lid]
            if owner == part
                values[lid] = 10*part
            end
        end
    end
    consistent!(v) |> wait

    map(partition(v),row_partition) do values, indices
        local_index_to_owner = local_to_owner(indices)
        for lid in 1:length(local_index_to_owner)
            owner = local_index_to_owner[lid]
            @test values[lid] == 10*owner
        end
    end

    map(local_values(v)) do values
        fill!(values,10.0)
    end

    assemble!(v) |> wait
    map(parts,local_values(v)) do part,values
        if part == 1
            @test values == [20.0, 20.0, 20.0, 0.0, 0.0, 0.0]
        elseif part == 2
            @test values == [0.0, 20.0, 30.0, 0.0]
        elseif part == 3
            @test values == [10.0, 30.0, 20.0, 0.0, 0.0, 0.0]
        else
            @test values == [0.0, 0.0, 0.0, 10.0, 30.0]
        end
    end
    @test collect(v) == [20.0, 20.0, 20.0, 20.0, 30.0, 10.0, 30.0, 20.0, 10.0, 30.0]

    n = 10

    gids = map(parts) do part
        if part == 1
            [1,4,6]
        elseif part == 2
            [3,1,2,8]
        elseif part == 3
            [1,9,6]
        else
            [3,2,8,10]
        end
    end
    row_partition = uniform_partition(parts,n)
    values = map(copy,gids)
    v = old_pvector!(gids,values,row_partition) |> fetch
    u = 2*v
    map(partition(u),partition(v)) do u,v
        @test u == 2*v
    end
    u = v + u
    map(local_values(u),local_values(v)) do u,v
        @test u == 3*v
    end
    @test any(i->i>4,v) == true
    @test any(i->i>17,v) == false
    @test all(i->i<17,v) == true
    @test all(i->i<4,v) == false
    @test maximum(v) == 16
    @test minimum(v) == 0
    @test maximum(i->i-1,v) == 15
    @test minimum(i->i-1,v) == -1

    w = copy(v)
    rmul!(w,-1)
    @test all(i->i==0,v+w)

    @test w == w
    @test w != v

    @test sqeuclidean(w,v) ≈ (norm(w-v))^2
    @test euclidean(w,v) ≈ norm(w-v)

    w = similar(v)
    w = zero(v)
    @test isa(w,PVector)
    @test norm(w) == 0
    @test sum(w) == 0

    w = v .- u
    map(local_values(w),local_values(u),local_values(v)) do w,u,v
        @test w == v - u
    end
    @test isa(w,PVector)
    w =  1 .+ v
    @test isa(w,PVector)
    w =  v .+ 1
    @test isa(w,PVector)
    w =  v .+ w .- u
    @test isa(w,PVector)
    w =  v .+ 1 .- u
    @test isa(w,PVector)

    w .= v .- u
    w .= v .- 1 .- u
    w .= u
    map(local_values(w),local_values(u)) do w,u
        @test w == u
    end

    w = v .- u
    @test isa(w,PVector)
    w =  v .+ w .- u
    @test isa(w,PVector)
    w =  v .+ 1 .- u
    @test isa(w,PVector)
    display(w)

    w .= v .- u
    w .= v .- 1 .- u
    w .= u
    map(own_values(w),own_values(u)) do w,u
        @test w == u
    end

    α = 0.2
    v .= w
    w .=  (1.0/α) .* w
    @. v = (1.0/α) * v
    map(local_values(w),local_values(v)) do w,v
      @test w == v
    end

    v .= w
    w .=  w .* (1.0/α)
    @. v =  v * (1.0/α)
    map(local_values(w),local_values(v)) do w,v
      @test w == v
    end

    @test isa(renumber(w),PVector)

end
