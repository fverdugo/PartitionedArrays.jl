"""
    Special mul function for hpcg without latency hiding.
    The version with latency hiding uses a slower csr mul.
"""

function mul_no_lat!(c::PVector, a::PSparseMatrix, b::PVector)
    @boundscheck @assert PartitionedArrays.matching_own_indices(axes(c, 1), axes(a, 1))
    @boundscheck @assert PartitionedArrays.matching_own_indices(axes(a, 2), axes(b, 1))
    @boundscheck @assert PartitionedArrays.matching_ghost_indices(axes(a, 2), axes(b, 1))
    if !a.assembled
        @boundscheck @assert PartitionedArrays.matching_ghost_indices(axes(a, 1), axes(c, 1))
        return mul!(c, a, b, 1, 0)
    end
    consistent!(b) |> wait
    foreach(spmv!, own_values(c), partition(a), partition(b))
    c
end

