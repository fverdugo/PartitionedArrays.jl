


I, J, V, b, I_b = build_matrix(32, 32, 16, 32, 32, 16, 0, 0, 0)
A_seq = sparse(I, J, V)

pA, pb = build_p_matrix(4, 16, 16, 16, 32, 32, 16, 2, 2, 1)

@test isequal(A_seq, centralize(pA))

@test isequal(b, collect(pb))

# # test sequential processing
# hpcg_benchmark()

# # test parallel processing
# hpcg_benchmark()