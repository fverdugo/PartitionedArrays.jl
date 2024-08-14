using PartitionedArrays
using Test
using LinearAlgebra
using Random
using Distances
using BlockArrays
using SparseArrays

function block_arrays_tests(distribute)
    block_arrays_tests(distribute,false)
    block_arrays_tests(distribute,true)
end

function block_arrays_tests(distribute,split_format)

    np = 4
    rank = distribute(LinearIndices((np,)))
    row_partition = uniform_partition(rank,(2,2),(6,6))

    a1 = pones(row_partition;split_format)
    a2 = pzeros(row_partition;split_format)
    a = mortar([a1,a2])
    display(a)
    collect(a)
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

    nodes_per_dir = (4,4)
    parts_per_dir = (2,2)
    args = laplacian_fem(nodes_per_dir,parts_per_dir,rank)
    A11 = psparse(args...) |> fetch
    x1 = pones(axes(A11,2);split_format)
    assemble!(x1) |> wait
    consistent!(x1) |> wait

    A = mortar(fill(A11,(2,2)))
    display(A)
    @show A
    #display(centralize(A))
    own_own_values(A)
    own_ghost_values(A)
    ghost_own_values(A)
    ghost_ghost_values(A)
    B = copy(A)
    copy!(B,A)
    copyto!(B,A)
    nnz(A)
    ax = axes(A,2)
    axb = blocks(ax)
    x = similar(a,axes(A,2))
    fill!(x,1)
    assemble!(x) |> wait
    consistent!(x) |> wait
    b = similar(x,axes(A,1))
    mul!(b,A,x)
    b = A*x
    mul!(b,A,x)
    B = 2*A
    B = A*2
    B = +A
    B = -A
    C = B+A
    D = B-A

end

