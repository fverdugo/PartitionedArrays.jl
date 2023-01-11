using PartitionedArrays
using Test

function p_vector_tests(distribute)

    rank = distribute(LinearIndices((4,)))
    rows = PRange(ConstantBlockSize(),rank,(2,2),(6,6))

    a1 = PVector(undef,rows)
    a2 = PVector{Vector{Int}}(undef,rows)
    a3 = PVector{OwnAndGhostValues{Vector{Float64}}}(undef,rows)
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

end
