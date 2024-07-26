
using PartitionedArrays
using Test
using SparseMatricesCSR

function gallery_tests(distribute)
    gallery_tests(distribute,(4,))
    gallery_tests(distribute,(2,2))
    gallery_tests(distribute,(2,1,2))
end

function gallery_tests(distribute,parts_per_dir)
    p = prod(parts_per_dir)
    ranks = distribute(LinearIndices((p,)))
    nodes_per_dir = map(i->2*i,parts_per_dir)
    args = laplacian_fdm(nodes_per_dir,parts_per_dir,ranks)
    A = psparse(args...) |> fetch
    A |> centralize |> display
    y = A*pones(axes(A,2))
    @test isa(y,PVector)
    A = psparse(args...;assembled=true) |> fetch
    y = A*pones(axes(A,2))
    @test isa(y,PVector)
    args = laplacian_fem(nodes_per_dir,parts_per_dir,ranks)
    A = psparse(args...) |> fetch
    A |> centralize |> display
    Y = A*pones(axes(A,2))
    @test isa(y,PVector)
    A = psparse(sparsecsr,args...) |> fetch
    A |> centralize |> display
    Y = A*pones(axes(A,2))
    @test isa(y,PVector)
    A = psparse(SparseMatrixCSR{1,Float64,Int32},args...) |> fetch
    A |> centralize |> display
    Y = A*pones(axes(A,2))
    @test isa(y,PVector)
end


