module SparseUtilsTests

using SparseArrays
using PartitionedArrays
using Test

I = [1,2,5,4,1]
J = [3,6,1,1,3]
V = Float64[4,5,3,2,5]
m = 7
n = 6

A = sparse(I,J,V,m,n)
B = compresscoo(A,I,J,V,m,n)
@test typeof(B) == typeof(A)
@test A == B

B = compresscoo(SparseMatrixCSC{Int,Int32},I,J,V,m,n)
@test typeof(B) == SparseMatrixCSC{Int,Int32}
@test A == B

end # module
