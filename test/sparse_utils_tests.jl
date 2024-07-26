module SparseUtilsTests

using SparseArrays
using SparseMatricesCSR
using PartitionedArrays
using LinearAlgebra
using Test

function test_mat(T)

  Tv = eltype(T)
  Ti = indextype(T)

  I = Ti[1,2,5,4,1]
  J = Ti[3,6,1,1,3]
  V = Tv[4,5,3,2,5]
  m = 7
  n = 6
  
  A = sparse(I,J,V,m,n)
  B = compresscoo(A,I,J,V,m,n)
  @test typeof(B) == typeof(A)
  @test A == B
  
  B = compresscoo(SparseMatrixCSC{Float64,Int},I,J,V,m,n)
  @test typeof(B) == SparseMatrixCSC{Float64,Int}
  @test A == B
  
  B = compresscoo(T,I,J,V,m,n)
  @test typeof(B) == T
  @test A == B

  b1 = ones(Tv,size(B,1))
  b2 = ones(Tv,size(B,1))
  x = collect(Tv,1:size(B,2))
  mul!(b1,B,x)
  spmv!(b2,B,x)
  @test norm(b1-b2)/norm(b1) + 1 ≈ 1

  b1 = ones(Tv,size(B,2))
  b2 = ones(Tv,size(B,2))
  x = collect(Tv,1:size(B,1))
  mul!(b1,transpose(B),x)
  spmtv!(b2,B,x)
  @test norm(b1-b2)/norm(b1) + 1 ≈ 1

  
  i,j,v = findnz(B)
  for (k,(ki,kj,kv)) in enumerate(nziterator(B))
    @test i[k] == ki
    @test j[k] == kj
    @test v[k] == kv
    @test nzindex(B,ki,kj) == k
  end
  
  rows = [4,2,3]
  inv_rows = zeros(Int,size(B,1))
  inv_rows[rows] = 1:length(rows)
  cols = [6,2,5,1]
  inv_cols = zeros(Int,size(B,2))
  inv_cols[cols] = 1:length(cols)
  indices = (rows,cols)
  inv_indices = (inv_rows,inv_cols)
  C = PartitionedArrays.SubSparseMatrix(B,indices,inv_indices)
  
  x = rand(Tv,size(C,2))
  y = similar(x,Tv,size(C,1))
  mul!(y,C,x)
  @test y ≈ A[rows,cols]*x
  @test C*x ≈ A[rows,cols]*x

  I = Ti[1,2,5,4,1]
  J = Ti[3,6,1,1,3]
  V = Tv[4,5,3,2,5]
  m = 7
  n = 6
  A = sparse_matrix(I,J,V,m,n)
  A,Acache = sparse_matrix(I,J,V,m,n;reuse=true)
  sparse_matrix!(A,V,Acache)

  I = Ti[-1]
  J = Ti[-1]
  V = Tv[-1]
  m = 7
  n = 6
  A = sparse_matrix(I,J,V,m,n)
  A,Acache = sparse_matrix(I,J,V,m,n;reuse=true)
  sparse_matrix!(A,V,Acache)

  I = Ti[-1]
  J = Ti[-1]
  V = Tv[-1]
  m = 0
  n = 0
  A = sparse_matrix(I,J,V,m,n)
  A,Acache = sparse_matrix(I,J,V,m,n;reuse=true)
  sparse_matrix!(A,V,Acache)

  I = Ti[]
  J = Ti[]
  V = Tv[]
  m = 0
  n = 0
  A = sparse_matrix(I,J,V,m,n)
  A,Acache = sparse_matrix(I,J,V,m,n;reuse=true)
  sparse_matrix!(A,V,Acache)

  d = dense_diag(A)
  PartitionedArrays.sparse_diag_matrix(d,axes(A))

end

test_mat(SparseMatrixCSC{Float64,Int})
test_mat(SparseMatrixCSC{Float32,Int32})
test_mat(SparseMatrixCSR{1,Float64,Int})
test_mat(SparseMatrixCSR{1,Float32,Int32})
test_mat(SparseMatrixCSR{0,Float64,Int})
test_mat(SparseMatrixCSR{0,Float32,Int32})

end # module
