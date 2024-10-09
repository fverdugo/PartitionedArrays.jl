using PartitionedArrays
using Test
using LinearAlgebra
using Random
using Distances
using SparseArrays
using IterativeSolvers

function p_sparse_matrix_tests(distribute)

    #np = 4
    #rank = distribute(LinearIndices((np,)))
    #n = 10
    #row_partition = uniform_partition(rank,n)
    #I,J,V = map(rank) do rank
    #    if rank == 1
    #        I = [1,2,6,4]
    #        J = [1,4,2,7]
    #    elseif rank == 2
    #        I = [3,2]
    #        J = [4,7]
    #    elseif rank == 3
    #        I = [3,7,8,2]
    #        J = [7,3,8,6]
    #    else
    #        I = [6,9,10]
    #        J = [1,10,9]
    #    end
    #    I,J,fill(Float64(rank),length(J))
    #end |> tuple_of_arrays

    #row_partition = uniform_partition(rank,n)
    #col_partition = row_partition
    #A = old_psparse!(I,J,V,row_partition,col_partition) |> fetch


    #n = 10
    #parts = rank
    #row_partition = uniform_partition(parts,n)
    #col_partition = row_partition

    #values = map(row_partition,col_partition) do rows, cols
    #    i = collect(1:length(rows))
    #    j = i
    #    v = fill(2.0,length(i))
    #    a=sparse(i,j,v,length(rows),length(cols))
    #    a
    #end

    #A = OldPSparseMatrix(values,row_partition,col_partition)
    #x = pfill(3.0,col_partition)
    #b = similar(x,axes(A,1))
    #mul!(b,A,x)
    #map(own_values(b)) do values
    #    @test all( values .== 6 )
    #end

    #consistent!(b) |> wait
    #map(partition(b)) do values
    #  @test all( values .== 6 )
    #end

    #_A = similar(A)
    #_A = similar(A,eltype(A),axes(A))
    ##_A = similar(typeof(A),axes(A)) # This should work, but fails down the line in SparseArrays.jl
    #copy!(_A,A)

    #LinearAlgebra.fillstored!(A,1.0)
    #fill!(x,3.0)
    #mul!(b,A,x)
    #consistent!(b) |> wait
    #map(partition(b)) do values
    #    @test all( values .== 3 )
    #end

    #I,J,V = map(parts) do part
    #    if part == 1
    #        [1,2,1,2,2], [2,6,1,2,1], [1.0,2.0,30.0,10.0,1.0]
    #    elseif part == 2
    #        [3,3,4,6], [3,9,4,2], [10.0,2.0,30.0,2.0]
    #    elseif part == 3
    #        [5,5,6,7], [5,6,6,7], [10.0,2.0,30.0,1.0]
    #    else
    #        [9,9,8,10,6], [9,3,8,10,5], [10.0,2.0,30.0,50.0,2.0]
    #    end
    #end |> tuple_of_arrays

    #A = old_psparse!(I,J,V,row_partition,col_partition) |> fetch
    #assemble!(A) |> wait
    #x = pfill(1.5,partition(axes(A,2)))

    #x = pones(partition(axes(A,2)))
    #y = A*x
    #dy = y - y

    #x = IterativeSolvers.cg(A,y)
    #r = A*x-y
    #@test norm(r) < 1.0e-9

    #x = pfill(0.0,partition(axes(A,2)))
    #IterativeSolvers.cg!(x,A,y)
    #r = A*x-y
    #@test norm(r) < 1.0e-9
    #fill!(x,0.0)

    #x = A\y
    #@test isa(x,PVector)
    #r = A*x-y
    #@test norm(r) < 1.0e-9

    #factors = lu(A)
    #x .= 0
    #ldiv!(x,factors,y)
    #r = A*x-y
    #@test norm(r) < 1.0e-9

    #lu!(factors,A)
    #x .= 0
    #ldiv!(x,factors,y)
    #r = A*x-y
    #map(i->fill!(i,100),ghost_values(r))
    #@test norm(r) < 1.0e-9
    #display(A)

    #B = copy(A)
    #@test reduce(&, map((a,b) -> nnz(a) == nnz(b),partition(A),partition(B)))
    #@test reduce(&, map((a,b) -> rowvals(a) == rowvals(b),partition(A),partition(B)))

    # New stuff

    np = 4
    rank = distribute(LinearIndices((np,)))
    n = 10
    parts = rank
    row_partition = uniform_partition(parts,n)
    col_partition = row_partition

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

    A = psparse(I,J,V,row_partition,col_partition,split_format=false,assemble=false) |> fetch
    B = split_format(A)
    B, cache = split_format(A,reuse=true)
    split_format!(B,A,cache)
    C = assemble(B) |> fetch
    C,cache = assemble(B,reuse=true) |> fetch
    assemble!(C,B,cache) |> wait
    display(C)

    A = psparse(I,J,V,row_partition,col_partition,split_format=true,assemble=false) |> fetch
    A = psparse(I,J,V,row_partition,col_partition,split_format=true,assemble=true) |> fetch
    A = psparse(I,J,V,row_partition,col_partition) |> fetch
    centralize(A) |> display
    B = A*A
    @test centralize(B) == centralize(A)*centralize(A)

    copy_I = deepcopy(I)
    copy_J = deepcopy(J)
    copy_V = deepcopy(V)
    new_A, _ = PartitionedArrays.psparse_yung_sheng!(sparse, copy_I, copy_J, copy_V, row_partition, col_partition) |> fetch
    centralize(new_A) |> display
    new_B = new_A * new_A
    @test centralize(new_B) == centralize(new_A) * centralize(new_A)

    # TODO Assembly in non-split_format format not yet implemented
    #A = psparse(I,J,V,row_partition,col_partition,split_format=false,assemble=true) |> fetch
    
    A,cache = psparse(I,J,V,row_partition,col_partition,reuse=true) |> fetch
    psparse!(A,V,cache) |> wait

    copy_I = deepcopy(I)
    copy_J = deepcopy(J)
    copy_V = deepcopy(V)
    new_A, new_cache = PartitionedArrays.psparse_yung_sheng!(sparse, copy_I, copy_J, copy_V, row_partition, col_partition) |> fetch
    copy_V = deepcopy(V)
    PartitionedArrays.psparse_yung_sheng!(new_A, copy_V, new_cache) |> wait

    A_fa = psparse(I,J,V,row_partition,col_partition) |> fetch
    rows_co = partition(axes(A_fa,2))
    A_co = consistent(A_fa,rows_co) |> fetch
    A_co,cache = consistent(A_fa,rows_co;reuse=true) |> fetch
    consistent!(A_co,A_fa,cache) |> wait

    copy_I = deepcopy(I)
    copy_J = deepcopy(J)
    copy_V = deepcopy(V)
    new_A_fa, _ = PartitionedArrays.psparse_yung_sheng!(sparse, copy_I, copy_J, copy_V, row_partition, col_partition) |> fetch
    new_rows_co = partition(axes(new_A_fa, 2))
    new_A_co = consistent(new_A_fa, new_rows_co) |> fetch
    new_A_co, new_cache = consistent(new_A_fa, new_rows_co; reuse=true) |> fetch
    consistent!(new_A_co, new_A_fa, new_cache) |> wait

    n = 10
    parts = rank
    row_partition = uniform_partition(parts,n)
    col_partition = row_partition

    I,J,V = map(row_partition,col_partition) do rows, cols
        i = collect(own_to_global(rows))
        j = copy(i)
        v = fill(2.0,length(i))
        i,j,v
    end |> tuple_of_arrays

    A = psparse(I,J,V,row_partition,col_partition) |> fetch
    x = pfill(3.0,axes(A,2);split_format=true)
    b = similar(x,axes(A,1))
    mul!(b,A,x)
    map(own_values(b)) do values
        @test all( values .== 6 )
    end
    consistent!(b) |> wait
    map(partition(b)) do values
      @test all( values .== 6 )
    end

    A = psparse_from_split_blocks(own_own_values(A),own_ghost_values(A),row_partition,col_partition)
    x = pfill(3.0,axes(A,2);split_format=true)
    b = similar(x,axes(A,1))
    mul!(b,A,x)
    map(own_values(b)) do values
        @test all( values .== 6 )
    end
    consistent!(b) |> wait
    map(partition(b)) do values
      @test all( values .== 6 )
    end

    A = psparse(I,J,V,row_partition,col_partition) |> fetch
    x = pfill(3.0,axes(A,2))
    b = similar(x,axes(A,1))
    mul!(b,A,x)
    map(own_values(b)) do values
        @test all( values .== 6 )
    end
    consistent!(b) |> wait
    map(partition(b)) do values
      @test all( values .== 6 )
    end

    copy_I = deepcopy(I)
    copy_J = deepcopy(J)
    copy_V = deepcopy(V)
    new_A, _ = PartitionedArrays.psparse_yung_sheng!(sparse, copy_I, copy_J, copy_V, row_partition, col_partition) |> fetch
    new_x = pfill(3.0, axes(new_A, 2); split_format=true)
    new_b = similar(new_x, axes(new_A, 1))
    mul!(new_b, new_A, new_x)
    map(own_values(new_b)) do values
        @test all(values .== 6)
    end
    consistent!(new_b) |> wait
    map(partition(new_b)) do values
        @test all(values .== 6)
    end

    copy_I = deepcopy(I)
    copy_J = deepcopy(J)
    copy_V = deepcopy(V)
    new_A, _ = PartitionedArrays.psparse_yung_sheng!(sparse, copy_I, copy_J, copy_V, row_partition, col_partition) |> fetch
    new_x = pfill(3.0, axes(new_A, 2))
    new_b = similar(new_x, axes(new_A, 1))
    mul!(new_b, new_A, new_x)
    map(own_values(new_b)) do values
        @test all(values .== 6)
    end
    consistent!(new_b) |> wait
    map(partition(new_b)) do values
        @test all(values .== 6)
    end

    _A = similar(A)
    _A = similar(A,eltype(A))
    copy!(_A,A)
    _A = copy(A)

    LinearAlgebra.fillstored!(A,1.0)
    fill!(x,3.0)
    mul!(b,A,x)
    consistent!(b) |> wait
    map(partition(b)) do values
        @test all( values .== 3 )
    end

    _new_A = similar(new_A)
    _new_A = similar(new_A, eltype(new_A))
    copy!(_new_A, new_A)
    _new_A = copy(new_A)

    LinearAlgebra.fillstored!(new_A, 1.0)
    fill!(new_x, 3.0)
    mul!(new_b, new_A, new_x)
    consistent!(new_b) |> wait
    map(partition(new_b)) do values
        @test all(values .== 3)
    end
    
    I,J,V = map(parts) do part
        if part == 1
            [1,2,1,2,2], [2,6,1,2,1], [1.0,2.0,30.0,10.0,1.0]
        elseif part == 2
            [3,3,4,6,0], [3,9,4,2,0], [10.0,2.0,30.0,2.0,2.0]
        elseif part == 3
            [5,5,6,7], [5,6,6,7], [10.0,2.0,30.0,1.0]
        else
            [9,9,8,10,6,-1], [9,3,8,10,5,1], [10.0,2.0,30.0,50.0,2.0,1.0]
        end
    end |> tuple_of_arrays

    A = psparse(SparseMatrixCSC{Float64,Int32},I,J,V,row_partition,col_partition) |> fetch
    A = psparse(I,J,V,row_partition,col_partition) |> fetch
    x = pones(partition(axes(A,2)))
    y = A*x
    @test isa(y,PVector)
    dy = y - y

    copy_I = deepcopy(I)
    copy_J = deepcopy(J)
    copy_V = deepcopy(V)
    new_A, _ = PartitionedArrays.psparse_yung_sheng!(sparse, copy_I, copy_J, copy_V, row_partition, col_partition) |> fetch
    new_x = pones(partition(axes(new_A, 2)))
    new_y = new_A * new_x
    @test isa(new_y, PVector)
    new_dy = new_y - new_y

    x = IterativeSolvers.cg(A,y)
    r = A*x-y
    @test norm(r) < 1.0e-9

    new_x = IterativeSolvers.cg(new_A, new_y)
    new_r = new_A * new_x - new_y
    @test norm(new_r) < 1.0e-9

    x = pfill(0.0,partition(axes(A,2)))
    IterativeSolvers.cg!(x,A,y)
    r = A*x-y
    @test norm(r) < 1.0e-9
    fill!(x,0.0)

    new_x = pfill(0.0, partition(axes(new_A, 2)))
    IterativeSolvers.cg!(new_x, new_A, new_y)
    new_r = new_A * new_x - new_y
    @test norm(new_r) < 1.0e-9
    fill!(new_x, 0.0)

    x = A\y
    @test isa(x,PVector)
    r = A*x-y
    @test norm(r) < 1.0e-9

    new_x = new_A \ new_y
    @test isa(new_x, PVector)
    new_r = new_A * new_x - new_y
    @test norm(new_r) < 1.0e-9

    factors = lu(A)
    x .= 0
    ldiv!(x,factors,y)
    r = A*x-y
    @test norm(r) < 1.0e-9

    new_factors = lu(new_A)
    new_x .= 0
    ldiv!(new_x, new_factors, new_y)
    new_r = new_A * new_x - new_y
    @test norm(new_r) < 1.0e-9

    lu!(factors,A)
    x .= 0
    ldiv!(x,factors,y)
    r = A*x-y
    map(i->fill!(i,100),ghost_values(r))
    @test norm(r) < 1.0e-9

    lu!(new_factors, new_A)
    new_x .= 0
    ldiv!(new_x, new_factors, new_y)
    new_r = new_A * new_x - new_y
    map(i -> fill!(i, 100), ghost_values(new_r))
    @test norm(new_r) < 1.0e-9

    rows_trivial = trivial_partition(parts,n)
    cols_trivial = rows_trivial
    values = map(collect∘local_to_global,rows_trivial)
    w0 = PVector(values,rows_trivial)
    values = map(collect∘local_to_global,row_partition)
    v = PVector(values,row_partition)
    v0 = copy(v)
    w = repartition(v,rows_trivial) |> fetch
    @test w == w0
    repartition!(w,v) |> wait
    @test w == w0
    w, cache = repartition(v,rows_trivial;reuse=true) |> fetch
    repartition!(w,v,cache) |> wait
    @test w == w0
    repartition!(v,w,cache;reversed=true) |> wait
    @test v == v0

    B = repartition(A,rows_trivial,cols_trivial) |> fetch
    B,cache = repartition(A,rows_trivial,cols_trivial;reuse=true) |> fetch
    repartition!(B,A,cache)

    B,w = repartition(A,v,rows_trivial,cols_trivial) |> fetch
    B,w,cache = repartition(A,v,rows_trivial,cols_trivial,reuse=true) |> fetch
    repartition!(B,w,A,v,cache) |> wait

    new_B = repartition(new_A, rows_trivial, cols_trivial) |> fetch
    new_B, new_cache = repartition(new_A, rows_trivial, cols_trivial; reuse=true) |> fetch
    repartition!(new_B, new_A, new_cache)

    new_B, new_w = repartition(new_A, v, rows_trivial, cols_trivial) |> fetch
    new_B, new_w, new_cache = repartition(new_A, v, rows_trivial, cols_trivial, reuse=true) |> fetch
    repartition!(new_B, new_w, new_A, v, new_cache) |> wait

    I2 = map(copy,I)
    V2 = map(copy,I)
    rows = row_partition
    cols = col_partition
    v = pvector(I2,V2,rows) |> fetch
    v,cache = pvector(I2,V2,rows;reuse=true) |> fetch
    pvector!(v,V,cache) |> wait

    v = pvector(I2,V2,rows;split_format=true) |> fetch
    v,cache = pvector(I2,V2,rows;reuse=true,split_format=true) |> fetch
    pvector!(v,V,cache) |> wait

    v = pvector(I2,V2,rows;assemble=false) |> fetch
    w = assemble(v) |> fetch
    w = assemble(v,rows) |> fetch
    w,cache = assemble(v,reuse=true) |> fetch
    assemble!(w,v,cache) |> wait

    A_cols = partition(axes(A,2))
    u = consistent(w,A_cols) |> fetch
    u,cache = consistent(w,A_cols;reuse=true) |> fetch
    consistent!(u,w,cache) |> wait

    new_A_cols = partition(axes(new_A, 2))
    u = consistent(w, new_A_cols) |> fetch
    u, cache = consistent(w, A_cols; reuse=true) |> fetch
    consistent!(u, w, cache) |> wait

    A,b = psystem(I,J,V,I2,V2,rows,cols) |> fetch
    A,b,cache = psystem(I,J,V,I2,V2,rows,cols,reuse=true) |> fetch
    psystem!(A,b,V,V2,cache) |> wait

    display((A,A))
    display((b,b))

    LinearAlgebra.fillstored!(A,3)
    B = 2*A
    @test eltype(partition(B)) == eltype(partition(A))
    B = A*2
    @test eltype(partition(B)) == eltype(partition(A))
    B = +A
    @test eltype(partition(B)) == eltype(partition(A))
    B = -A
    @test eltype(partition(B)) == eltype(partition(A))
    C = B+A
    @test eltype(partition(C)) == eltype(partition(A))
    C = B-A
    @test eltype(partition(C)) == eltype(partition(A))

    nodes_per_dir = (5,5)
    parts_per_dir = (2,2)
    A = PartitionedArrays.laplace_matrix(nodes_per_dir)
    A = PartitionedArrays.laplace_matrix(nodes_per_dir,parts_per_dir,parts)
    d = dense_diag(A)
    dense_diag!(d,A)

    nodes_per_dir = (5,5)
    parts_per_dir = (2,2)
    A = PartitionedArrays.laplace_matrix(nodes_per_dir,parts_per_dir,parts)
    A_seq = centralize(A)
    Z = 2*A
    Z_seq = centralize(Z)

    B = Z*A
    @test centralize(B) ≈ Z_seq*A_seq

    B = spmm(Z,A)
    @test centralize(B) ≈ Z_seq*A_seq
    B,cacheB = spmm(Z,A;reuse=true)
    map(partition(A)) do A
        nonzeros(A.blocks.own_own) .*= 4
        nonzeros(A.blocks.own_ghost) .*= 4
    end
    A_seq = centralize(A)
    spmm!(B,Z,A,cacheB)
    @test centralize(B) ≈ Z_seq*(A_seq)

    B = transpose(Z)*A
    @test centralize(B) ≈ transpose(Z_seq)*A_seq

    B = spmtm(Z,A)
    B,cacheB = spmtm(Z,A;reuse=true)
    @test centralize(B) ≈ transpose(Z_seq)*A_seq
    map(partition(A)) do A
        nonzeros(A.blocks.own_own) .*= 4
        nonzeros(A.blocks.own_ghost) .*= 4
    end
    A_seq = centralize(A)
    spmtm!(B,Z,A,cacheB)
    @test centralize(B) ≈ transpose(Z_seq)*A_seq

    C = rap(transpose(A),Z,A)
    @test centralize(C) ≈ transpose(A_seq)*Z_seq*A_seq
    C,cacheC = rap(transpose(A),Z,A;reuse=true)
    @test centralize(C) ≈ transpose(A_seq)*Z_seq*A_seq
    map(partition(A)) do A
        nonzeros(A.blocks.own_own) .*= 4
        nonzeros(A.blocks.own_ghost) .*= 4
    end
    A_seq = centralize(A)
    rap!(C,transpose(A),Z,A,cacheC)
    @test centralize(C) ≈ transpose(A_seq)*Z_seq*A_seq

    r = pzeros(partition(axes(A,2)))
    x = pones(partition(axes(A,1)))
    mul!(r,transpose(A),x)

    B = LinearAlgebra.I-A

    @test isa(renumber(A),PSparseMatrix)
    
end
