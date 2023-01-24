using PartitionedArrays
using Test
using LinearAlgebra
using Random
using Distances
using SparseArrays
using IterativeSolvers

function p_sparse_matrix_tests(distribute)

    np = 4
    rank = distribute(LinearIndices((np,)))
    n = 10
    rows = prange(uniform_partition,rank,n)
    I,J,V = map(rank) do rank
        if rank == 1
            I = [1,2,6,4]
            J = [1,4,2,7]
        elseif rank == 2
            I = [3,2]
            J = [4,7]
        elseif rank == 3
            I = [3,7,8,2]
            J = [7,3,8,6]
        else
            I = [6,9,10]
            J = [1,10,9]
        end
        I,J,fill(Float64(rank),length(J))
    end |> unpack

    rows = prange(uniform_partition,rank,n)
    cols = rows
    A = psparse!(I,J,V,rows,cols) |> fetch


    n = 10
    parts = rank
    rows = prange(uniform_partition,parts,n)
    cols = rows

    values = map(rows.indices,cols.indices) do rows, cols
        i = collect(1:length(rows))
        j = i
        v = fill(2.0,length(i))
        a=sparse(i,j,v,length(rows),length(cols))
        a
    end

    A = PSparseMatrix(values,rows,cols)
    A = PSparseMatrix(values,rows,cols,A.assembler)
    x = pfill(3.0,cols)
    b = similar(x,rows)
    mul!(b,A,x)
    map(get_own_values(b)) do values
        @test all( values .== 6 )
    end

    consistent!(b) |> wait
    map(b.values) do values
      @test all( values .== 6 )
    end

    LinearAlgebra.fillstored!(A,1.0)
    fill!(x,3.0)
    mul!(b,A,x)
    consistent!(b) |> wait
    map(b.values) do values
        @test all( values .== 3 )
    end

    I,J,V = map(parts) do part
        if part == 1
            [1,2,1,2,2], [2,6,1,2,1], [1.0,2.0,30.0,10.0,1.0]
        elseif part == 2
            [3,3,4,6], [3,9,4,2], [10.0,2.0,30.0,2.0]
        elseif part == 3
            [5,5,6,7], [5,6,6,7], [10.0,2.0,30.0,1.0]
        else
            [9,9,8,10,6], [9,3,8,10,5], [10.0,2.0,30.0,50.0,2.0]
        end
    end |> unpack

    A = psparse!(I,J,V,rows,cols) |> fetch
    x = pfill(1.5,axes(A,2))

    x = pones(A.cols)
    y = A*x
    dy = y - y

    x = IterativeSolvers.cg(A,y)
    r = A*x-y
    @test norm(r) < 1.0e-9

    x = pfill(0.0,A.cols)
    IterativeSolvers.cg!(x,A,y)
    r = A*x-y
    @test norm(r) < 1.0e-9
    fill!(x,0.0)

    x = A\y
    @test isa(x,PVector)
    r = A*x-y
    @test norm(r) < 1.0e-9

    factors = lu(A)
    x .= 0
    ldiv!(x,factors,y)
    r = A*x-y
    @test norm(r) < 1.0e-9

    lu!(factors,A)
    x .= 0
    ldiv!(x,factors,y)
    r = A*x-y
    @test norm(r) < 1.0e-9

end

