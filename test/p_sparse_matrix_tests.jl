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
    row_partition = uniform_partition(rank,n)
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
    end |> tuple_of_arrays

    row_partition = uniform_partition(rank,n)
    col_partition = row_partition
    A = psparse!(I,J,V,row_partition,col_partition) |> fetch


    n = 10
    parts = rank
    row_partition = uniform_partition(parts,n)
    col_partition = row_partition

    values = map(row_partition,col_partition) do rows, cols
        i = collect(1:length(rows))
        j = i
        v = fill(2.0,length(i))
        a=sparse(i,j,v,length(rows),length(cols))
        a
    end

    A = PSparseMatrix(values,row_partition,col_partition)
    x = pfill(3.0,col_partition)
    b = similar(x,axes(A,1))
    mul!(b,A,x)
    map(own_values(b)) do values
        @test all( values .== 6 )
    end

    consistent!(b) |> wait
    map(partition(b)) do values
      @test all( values .== 6 )
    end

    _A = similar(A)
    _A = similar(A,eltype(A),axes(A))
    #_A = similar(typeof(A),axes(A)) # This should work, but fails down the line in SparseArrays.jl
    copy!(_A,A)

    LinearAlgebra.fillstored!(A,1.0)
    fill!(x,3.0)
    mul!(b,A,x)
    consistent!(b) |> wait
    map(partition(b)) do values
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
    end |> tuple_of_arrays

    A = psparse!(I,J,V,row_partition,col_partition) |> fetch
    assemble!(A) |> wait
    x = pfill(1.5,partition(axes(A,2)))

    x = pones(partition(axes(A,2)))
    y = A*x
    dy = y - y

    x = IterativeSolvers.cg(A,y)
    r = A*x-y
    @test norm(r) < 1.0e-9

    x = pfill(0.0,partition(axes(A,2)))
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
    map(i->fill!(i,100),ghost_values(r))
    @test norm(r) < 1.0e-9
    display(A)

end

