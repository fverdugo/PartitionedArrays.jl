
function get_own_ghost_values end

function get_ghost_own_values end

function allocate_local_values(a,::Type{T},indices_rows,indices_cols) where T
    m = get_n_local(indices_rows)
    n = get_n_local(indices_cols)
    similar(a,T,m,n)
end

function allocate_local_values(::Type{V},indices_rows,indices_cols) where V
    m = get_n_local(indices_rows)
    n = get_n_local(indices_cols)
    similar(V,m,n)
end

function get_local_values(values,indices_rows,indices_cols)
    values
end

function get_own_values(values,indices_rows,indices_cols)
    subindices = (get_own_to_local(indices_rows),get_own_to_local(indices_cols))
    subindices_inv = (get_local_to_own(indices_rows),get_local_to_own(indices_cols))
    SubSparseMatrix(values,subindices,subindices_inv)
end

function get_ghost_values(values,indices_rows,indices_cols)
    subindices = (get_ghost_to_local(indices_rows),get_ghost_to_local(indices_cols))
    subindices_inv = (get_local_to_ghost(indices_rows),get_local_to_ghost(indices_cols))
    SubSparseMatrix(values,subindices,subindices_inv)
end

function get_own_ghost_values(values,indices_rows,indices_cols)
    subindices = (get_own_to_local(indices_rows),get_ghost_to_local(indices_cols))
    subindices_inv = (get_local_to_own(indices_rows),get_local_to_ghost(indices_cols))
    SubSparseMatrix(values,subindices,subindices_inv)
end

function get_ghost_own_values(values,indices_rows,indices_cols)
    subindices = (get_ghost_to_local(indices_rows),get_own_to_local(indices_cols))
    subindices_inv = (get_local_to_ghost(indices_rows),get_local_to_own(indices_cols))
    SubSparseMatrix(values,subindices,subindices_inv)
end

struct PSparseMatrix{V,A,B,C,D,E,T} <: AbstractMatrix{T}
    values::A
    rows::B
    cols::C
    assembler::D
    buffers::E
    @doc """
    """
    function PSparseMatrix(
            values,
            rows::PRange,
            cols::PRange,
            assembler=sparse_matrix_assembler(values,rows,cols),
            buffers=assembly_buffers(values,assembler))
        V = eltype(values)
        T = eltype(V)
        A = typeof(values)
        B = typeof(rows)
        C = typeof(cols)
        D = typeof(assembler)
        E = typeof(buffers)
        new{V,A,B,C,D,E,T}(values,rows,cols,assembler,buffers)
    end
end

function PSparseMatrix{V}(::UndefInitializer,rows::PRange,cols::PRange) where V
    values = map(rows.indices,cols.indices) do row_indices, col_indices
        allocate_local_values(V,row_indices,col_indices)
    end
    PSparseMatrix(values,rows,cols)
end


Base.size(a::PSparseMatrix) = (length(a.rows),length(a.cols))
Base.axes(a::PSparseMatrix) = (a.rows,a.cols)
Base.IndexStyle(::Type{<:PSparseMatrix}) = IndexCartesian()
function Base.getindex(a::PSparseMatrix,gi::Int,gj::Int)
    scalar_indexing_action(a)
end
function Base.setindex!(a::PSparseMatrix,v,gi::Int,gj::Int)
    scalar_indexing_action(a)
end

function Base.show(io::IO,k::MIME"text/plain",data::PSparseMatrix)
    println(io,typeof(data)," on $(length(data.values)) parts")
end

function get_local_values(a::PSparseMatrix)
    map(get_local_values,a.values,a.rows.indices,a.cols.indices)
end

function get_own_values(a::PSparseMatrix)
    map(get_own_values,a.values,a.rows.indices,a.cols.indices)
end

function get_ghost_values(a::PSparseMatrix)
    map(get_ghost_values,a.values,a.rows.indices,a.cols.indices)
end

function get_own_ghost_values(a::PSparseMatrix)
    map(get_own_ghost_values,a.values,a.rows.indices,a.cols.indices)
end

function get_ghost_own_values(a::PSparseMatrix)
    map(get_ghost_own_values,a.values,a.rows.indices,a.cols.indices)
end

function Base.similar(a::PSparseMatrix,::Type{T},inds::Tuple{<:PRange,<:PRange}) where T
    rows,cols = inds
    values = map(a.values,rows.indices,col.indices) do values, row_indices, col_indices
        allocate_local_values(values,T,row_indices,col_indices)
    end
    PSparseMatrix(values,rows,cols)
end

function Base.similar(::Type{<:PSparseMatrix{V}},inds::Tuple{<:PRange,<:PRange}) where V
    rows,cols = inds
    values = map(a.values,rows.indices,col.indices) do values, row_indices, col_indices
        allocate_local_values(V,row_indices,col_indices)
    end
    PSparseMatrix(values,rows,cols)
end

function LinearAlgebra.fillstored!(a::PSparseMatrix,v)
  map(a.values) do values
    LinearAlgebra.fillstored!(values,v)
  end
  a
end

function Base.:*(a::Number,b::PSparseMatrix)
  values = map(b.values) do values
    a*values
  end
  PSparseMatrix(values,b.rows,b.cols,b.assembler)
end

function Base.:*(b::PSparseMatrix,a::Number)
  a*b
end

function Base.:*(a::PSparseMatrix{Ta},b::PVector{Tb}) where {Ta,Tb}
  T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
  c = PVector{Vector{T}}(undef,a.rows)
  mul!(c,a,b)
  c
end

for op in (:+,:-)
  @eval begin
    function Base.$op(a::PSparseMatrix)
      values = map(a.values) do a
        $op(a)
      end
      PSparseMatrix(values,a.rows,a.cols,b.assembler)
    end
  end
end

function LinearAlgebra.mul!(c::PVector,a::PSparseMatrix,b::PVector,α::Number,β::Number)
  @boundscheck @assert matching_own_indices(c.rows,a.rows)
  @boundscheck @assert matching_own_indices(a.cols,b.rows)
  @boundscheck @assert matching_ghost_indices(a.cols,b.rows)
  # Start the exchange
  t = consistent!(b)
  # Meanwhile, process the owned blocks
  map(get_own_values(c),get_own_values(a),get_own_values(b)) do co,aoo,bo
    if β != 1
        β != 0 ? rmul!(co, β) : fill!(co,zero(eltype(co)))
    end
    mul!(co,aoo,bo,α,1)
  end
  # Wait for the exchange to finish
  wait(t)
  # process the ghost block
  map(get_own_values(c),get_own_ghost_values(a),get_ghost_values(b)) do co,aoh,bh
    mul!(co,aoh,bh,α,1)
  end
  c
end

# this one could be also used for sparse vectors
struct SparseAssembler{A,B}
    neighbors::A
    local_nzindices::B
end
function Base.show(io::IO,k::MIME"text/plain",data::SparseAssembler)
    println(io,nameof(typeof(data))," partitioned in $(length(data.neighbors.snd)) parts")
end
Base.reverse(a::SparseAssembler) = SparseAssembler(reverse(a.neighbors),reverse(a.local_indices))

function sparse_matrix_assembler(values,rows::PRange,cols::PRange)
    function setup_snd(part,parts_snd,row_indices,col_indices,values)
        local_to_owner = get_local_to_owner(row_indices)
        local_to_global_row = get_local_to_global(row_indices)
        local_to_global_col = get_local_to_global(col_indices)
        owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
        ptrs = zeros(Int32,length(parts_snd)+1)
        for (li,lj,v) in nziterator(values)
            owner = local_to_owner[li]
            if owner != part
                ptrs[owner_to_i[owner]+1] +=1
            end
        end
        length_to_ptrs!(ptrs)
        k_snd_data = zeros(Int,ptrs[end]-1)
        gi_snd_data = zeros(Int,ptrs[end]-1)
        gj_snd_data = zeros(Int,ptrs[end]-1)
        for (k,(li,lj,v)) in enumerate(nziterator(values))
            owner = local_to_owner[li]
            if owner != part
                p = ptrs[owner_to_i[owner]]
                k_snd_data[p] = k
                gi_snd_data[p] = local_to_global_row[li]
                gj_snd_data[p] = local_to_global_col[lj]
                ptrs[owner_to_i[owner]] += 1
            end
        end
        rewind_ptrs!(ptrs)
        k_snd = JaggedArray(k_snd_data,ptrs)
        gi_snd = JaggedArray(gi_snd_data,ptrs)
        gj_snd = JaggedArray(gj_snd_data,ptrs)
        k_snd, gi_snd, gj_snd
    end
    function setup_rcv(part,row_indices,col_indices,gi_rcv,gj_rcv,values)
        global_to_local_row = get_global_to_local(row_indices)
        global_to_local_col = get_global_to_local(col_indices)
        ptrs = gi_rcv.ptrs
        k_rcv_data = zeros(Int,ptrs[end]-1)
        for p in 1:length(gi_rcv.data)
            gi = gi_rcv.data[p]
            gj = gj_rcv.data[p]
            li = global_to_local_row[gi]
            lj = global_to_local_col[gj]
            k = nzindex(values,li,lj)
            @boundscheck @assert k > 0 "The sparsity pattern of the ghost layer is inconsistent"
            k_rcv_data[p] = k
        end
        k_rcv = JaggedArray(k_rcv_data,ptrs)
        k_rcv
    end
    part = linear_indices(rows.indices)
    parts_snd = rows.assembler.neighbors.snd
    parts_rcv = rows.assembler.neighbors.rcv
    k_snd, gi_snd, gj_snd = map(setup_snd,part,parts_snd,rows.indices,cols.indices,values) |> unpack
    gi_rcv = exchange_fetch(gi_snd,rows.assembler.neighbors)
    gj_rcv = exchange_fetch(gj_snd,rows.assembler.neighbors)
    k_rcv = map(setup_rcv,part,rows.indices,cols.indices,gi_rcv,gj_rcv,values)
    SparseAssembler(rows.assembler.neighbors,AssemblyLocalIndices(k_snd,k_rcv))
end

function assembly_buffers(values,assembler::SparseAssembler)
    default_assembler = Assembler(assembler.neighbors,assembler.local_nzindices)
    assembly_buffers(values,default_assembler)
end

function assemble!(f,values,assembler::SparseAssembler,buffers)
    nzvals = map(nonzeros,values)
    default_assembler = Assembler(assembler.neighbors,assembler.local_nzindices)
    assemble!(nzvals,default_assembler,buffers)
end

function assemble!(a::PSparseMatrix)
    assemble!(+,a)
end

function assemble!(o,a::PSparseMatrix)
    t = assemble!(o,a.values,a.assembler,a.buffers)
    @async begin
        wait(t)
        map(get_ghost_values(a)) do a
            fillstored!(a,zero(eltype(a)))
        end
        a
    end
end

function assemble_coo!(I,J,V,rows)
    function setup_snd(part,parts_snd,row_lids,coo_values)
        global_to_local = get_global_to_local(row_lids)
        local_to_owner = get_local_to_owner(row_lids)
        owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
        ptrs = zeros(Int32,length(parts_snd)+1)
        k_gi, k_gj, k_v = coo_values
        for k in 1:length(k_gi)
            gi = k_gi[k]
            li = global_to_local[gi]
            owner = local_to_owner[li]
            if owner != part
                ptrs[owner_to_i[owner]+1] +=1
            end
        end
        length_to_ptrs!(ptrs)
        gi_snd_data = zeros(eltype(k_gi),ptrs[end]-1)
        gj_snd_data = zeros(eltype(k_gj),ptrs[end]-1)
        v_snd_data = zeros(eltype(k_v),ptrs[end]-1)
        for k in 1:length(k_gi)
            gi = k_gi[k]
            li = global_to_local[gi]
            owner = local_to_owner[li]
            if owner != part
                gj = k_gj[k]
                v = k_v[k]
                p = ptrs[owner_to_i[owner]]
                gi_snd_data[p] = gi
                gj_snd_data[p] = gj
                v_snd_data[p] = v
                k_v[k] = zero(v)
                ptrs[owner_to_i[owner]] += 1
            end
        end
        rewind_ptrs!(ptrs)
        gi_snd = JaggedArray(gi_snd_data,ptrs)
        gj_snd = JaggedArray(gj_snd_data,ptrs)
        v_snd = JaggedArray(v_snd_data,ptrs)
        gi_snd, gj_snd, v_snd
    end
    function setup_rcv(gi_rcv,gj_rcv,v_rcv,coo_values)
        k_gi, k_gj, k_v = coo_values
        current_n = length(k_gi)
        new_n = current_n + length(gi_rcv.data)
        resize!(k_gi,new_n)
        resize!(k_gj,new_n)
        resize!(k_v,new_n)
        for p in 1:length(gi_rcv.data)
          k_gi[current_n+p] = gi_rcv.data[p]
          k_gj[current_n+p] = gj_rcv.data[p]
          k_v[current_n+p] = v_rcv.data[p]
        end
    end
    part = linear_indices(rows.indices)
    neighbors = rows.assembler.neighbors
    parts_snd = neighbors.snd
    parts_rcv = neighbors.rcv
    coo_values = map(tuple,I,J,V)
    gi_snd, gj_snd, v_snd = map(setup_snd,part,parts_snd,rows.indices,coo_values) |> unpack
    t1 = exchange(gi_snd,neighbors)
    t2 = exchange(gj_snd,neighbors)
    t3 = exchange(v_snd,neighbors)
    @async begin
        gi_rcv = fetch(t1)
        gj_rcv = fetch(t2)
        v_rcv = fetch(t3)
        map(setup_rcv,gi_rcv,gj_rcv,v_rcv,coo_values)
        I,J,V
    end
end

function psparse!(f,I,J,V,rows,cols;kwargs...)
    rows = union_ghost(rows,I)
    t = assemble_coo!(I,J,V,rows)
    @async begin
        wait(t)
        cols = union_ghost(cols,J)
        to_local!(I,rows)
        to_local!(J,cols)
        values = map(f,I,J,V,rows.indices,cols.indices)
        PSparseMatrix(values,rows,cols)
    end
end

function psparse!(I,J,V,rows,cols;kwargs...)
    psparse!(default_local_values,I,J,V,rows,cols;kwargs...)
end

function psparse(f,rows,cols)
    values = map(f,rows.indices,cols.indices)
    PSparseMatrix(values,rows,cols)
end

function psparse(rows,cols)
    psparse(default_local_values,rows,cols)
end

function default_local_values(row_indices,col_indices)
    m = get_n_local(row_indices)
    n = get_n_local(col_indices)
    sparse(Int32[],Int32[],Float64[],m,n)
end

function default_local_values(I,J,V,row_indices,col_indices)
    m = get_n_local(row_indices)
    n = get_n_local(col_indices)
    sparse(I,J,V,m,n)
end

# Misc functions that could be removed if IterativeSolvers was implemented in terms
# of axes(A,d) instead of size(A,d)
function IterativeSolvers.zerox(A::PSparseMatrix,b::PVector)
  T = IterativeSolvers.Adivtype(A, b)
  x = similar(b, T, axes(A, 2))
  fill!(x, zero(T))
  return x
end
