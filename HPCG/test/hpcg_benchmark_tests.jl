using HPCG
using PartitionedArrays
using PartitionedSolvers
using LinearAlgebra
using Primes
using DataStructures
using Dates
using Statistics
using DelimitedFiles
using SparseMatricesCSR
using Test
import Base: iterate

function hpcg_benchmark_tests(distribute)
    np = 4
    nx = ny = nz = 16
    l = 4
    ranks = distribute(LinearIndices((np,)))

    I, J, V, b, I_b = build_matrix(32, 32, 16, 32, 32, 16, 1, 1, 1)
    A_seq = sparsecsr(I, J, V)

    pA, pb = build_p_matrix(ranks, nx, ny, nz, 32, 32, 16, 2, 2, 1)


    #	@test isequal(A_seq, centralize(pA)) # centralize does not work with csr?

    @test isequal(b, collect(pb))

    # convergence test
    nx = ny = nz = 32
    S, geom = pc_setup(np, ranks, l, nx, ny, nz)
    ref_timing_data = zeros(Float64, 10)
    x = similar(S.x[l])
    x .= 0
    b = S.r[l]
    x, ref_timing_data, normr0, normr, iters = ref_cg!(x, S.A_vec[l], b, ref_timing_data, maxiter = 50, tolerance = 0.0, Pl = S)

    ref_tol = normr / normr0
    # expected tolerance = 2.877476184683206e-13
    @test ref_tol < 1.0E-12

    hpcg_benchmark(distribute, np, nx, ny, nz; total_runtime = 10, output_type = "none")
end

