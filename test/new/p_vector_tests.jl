using PartitionedArrays
using Test
using LinearAlgebra
using Random
using Distances

function p_vector_tests(distribute)

    np = 4
    rank = distribute(LinearIndices((np,)))
    rows = uniform_partition(rank,(2,2),(6,6))

    a1 = pvector(rows)
    a2 = pvector(inds->zeros(Int,length(inds)),rows)
    a3 = PVector{OwnAndGhostValues{Vector{Int}}}(undef,rows)
    for a in [a1,a2,a3]
        b = similar(a)
        b = similar(a,Int)
        b = similar(a,Int,rows)
        b = similar(typeof(a),rows)
        copy!(b,a)
        b = copy(a)
        fill!(b,5.)
        @test length(a) == length(rows)
        @test a.rows === rows
        @test b.rows === rows
    end

    a = pfill(4,rows)
    a = pzeros(rows)
    a = pones(rows)
    a = prand(rows)
    a = prandn(rows)
    assemble!(a) |> wait
    consistent!(a) |> wait

    @test a == copy(a)

    n = 10
    I,V = map(rank) do rank
        Random.seed!(rank)
        I = rand(1:n,5)
        Random.seed!(2*rank)
        V = rand(1:2,5)
        I,V
    end |> unpack

    rows = uniform_partition(rank,n)
    a = pvector!(I,V,rows) |> fetch

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

end
