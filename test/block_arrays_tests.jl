using PartitionedArrays
using Test
using LinearAlgebra
using Random
using Distances
using BlockArrays

function block_arrays_tests(distribute)

    np = 4
    rank = distribute(LinearIndices((np,)))
    row_partition = uniform_partition(rank,(2,2),(6,6))

    a1 = pones(row_partition,split_format=true)
    a2 = pzeros(row_partition,split_format=true)
    a = mortar([a1,a2])
    display(a)
    rows = axes(a,1)
    display(rows)
    partition(rows)
    local_block_ranges(rows)
    own_block_ranges(rows)
    ghost_block_ranges(rows)

    @test a[Block(1)] == a1
    @test a[Block(2)] == a2
    display(a[Block(1)])
    @test view(a,Block(1)) === a1
    @test view(a,Block(2)) === a2

    partition(a)
    #map(display,own_values(a))
    local_values(a)
    own_values(a)
    ghost_values(a)

    b = similar(a)
    b = similar(a,Int)
    b = similar(a,Int,axes(a,1))
    b = similar(typeof(a),axes(a,1))
    copy!(b,a)
    b = copy(a)
    @test typeof(b) == typeof(a)
    fill!(b,5.)
    rows = axes(a,1)
    @test length(a) == length(rows)
    assemble!(a) |> wait
    consistent!(a) |> wait

    rmul!(a,-1)
    fill!(a,0)
    @test any(i->i>0,a) == false
    @test all(i->i==0,a) == true
    @test minimum(a) <= maximum(a)

    b = 2*a
    b = a*2
    b = a/2
    c = a .+ a
    c = a .+ b .+ a
    @test isa(c,BlockPVector)
    c = a - b
    c = a + b

    fill!(a,1)
    r = reduce(+,a)
    @test sum(a) == r
    @test norm(a) > 0
    @test sqrt(a⋅a) ≈ norm(a)
    euclidean(a,a)
    @test euclidean(a,a) + 1 ≈ 1

    u = a
    v = b
    w =  1 .+ v
    @test isa(w,BlockPVector)
    w =  v .+ 1
    @test isa(w,BlockPVector)
    w =  v .+ w .- u
    @test isa(w,BlockPVector)
    w =  v .+ 1 .- u
    @test isa(w,BlockPVector)
    w .= u

end

