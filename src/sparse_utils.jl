
"""
    indextype(a)

Return the element type of the vector
used to store the row or column indices in the sparse matrix `a`. 
"""
function indextype end

indextype(a::AbstractSparseMatrix) = indextype(typeof(a))
indextype(a::Type{SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti} = Ti
indextype(a::Type{SparseMatrixCSR{Bi,Tv,Ti}}) where {Bi,Tv,Ti} = Ti

"""
    for (i,j,v) in nziterator(a)
    ...
    end

Iterate over the non zero entries of `a` returning the corresponding
row `i`, column `j` and value `v`.
"""
function nziterator end

nziterator(a::SparseArrays.AbstractSparseMatrixCSC) = NZIteratorCSC(a)

struct NZIteratorCSC{A}
    matrix::A
end

Base.length(a::NZIteratorCSC) = nnz(a.matrix)
Base.eltype(::Type{<:NZIteratorCSC{A}}) where A = Tuple{Int,Int,eltype(A)}
Base.eltype(::T) where T <: NZIteratorCSC = eltype(T)
Base.IteratorSize(::Type{<:NZIteratorCSC}) = Base.HasLength()
Base.IteratorEltype(::Type{<:NZIteratorCSC}) = Base.HasEltype()

@inline function Base.iterate(a::NZIteratorCSC)
    if nnz(a.matrix) == 0
        return nothing
    end
    col = 0
    knext = nothing
    while knext === nothing
        col += 1
        ks = nzrange(a.matrix,col)
        knext = iterate(ks)
    end
    k, kstate = knext
    i = Int(rowvals(a.matrix)[k])
    j = col
    v = nonzeros(a.matrix)[k]
    (i,j,v), (col,kstate)
end

@inline function Base.iterate(a::NZIteratorCSC,state)
    col, kstate = state
    ks = nzrange(a.matrix,col)
    knext = iterate(ks,kstate)
    if knext === nothing
        while knext === nothing
            if col == size(a.matrix,2)
                return nothing
            end
            col += 1
            ks = nzrange(a.matrix,col)
            knext = iterate(ks)
        end
    end
    k, kstate = knext
    i = Int(rowvals(a.matrix)[k])
    j = col
    v = nonzeros(a.matrix)[k]
    (i,j,v), (col,kstate)
end

nziterator(a::SparseMatrixCSR) = NZIteratorCSR(a)

struct NZIteratorCSR{A}
    matrix::A
end

Base.length(a::NZIteratorCSR) = nnz(a.matrix)
Base.eltype(::Type{<:NZIteratorCSR{A}}) where A = Tuple{Int,Int,eltype(A)}
Base.eltype(::T) where T <: NZIteratorCSR = eltype(T)
Base.IteratorSize(::Type{<:NZIteratorCSR}) = Base.HasLength()
Base.IteratorEltype(::Type{<:NZIteratorCSR}) = Base.HasEltype()

@inline function Base.iterate(a::NZIteratorCSR)
    if nnz(a.matrix) == 0
        return nothing
    end
    row = 0
    ptrs = a.matrix.rowptr
    knext = nothing
    while knext === nothing
        row += 1
        ks = nzrange(a.matrix,row)
        knext = iterate(ks)
    end
    k, kstate = knext
    i = row
    j = Int(colvals(a.matrix)[k]+getoffset(a.matrix))
    v = nonzeros(a.matrix)[k]
    (i,j,v), (row,kstate)
end

@inline function Base.iterate(a::NZIteratorCSR,state)
    row, kstate = state
    ks = nzrange(a.matrix,row)
    knext = iterate(ks,kstate)
    if knext === nothing
        while knext === nothing
            if row == size(a.matrix,1)
                return nothing
            end
            row += 1
            ks = nzrange(a.matrix,row)
            knext = iterate(ks)
        end
    end
    k, kstate = knext
    i = row
    j = Int(colvals(a.matrix)[k]+getoffset(a.matrix))
    v = nonzeros(a.matrix)[k]
    (i,j,v), (row,kstate)
end

struct SubSparseMatrix{T,A,B,C} <: AbstractMatrix{T}
    parent::A
    indices::B
    inv_indices::C
    function SubSparseMatrix(
            parent::AbstractSparseMatrix{T},
            indices::Tuple,
            inv_indices::Tuple) where T

        A = typeof(parent)
        B = typeof(indices)
        C = typeof(inv_indices)
        new{T,A,B,C}(parent,indices,inv_indices)
    end
end

Base.size(a::SubSparseMatrix) = map(length,a.indices)
Base.IndexStyle(::Type{<:SubSparseMatrix}) = IndexCartesian()
function Base.getindex(a::SubSparseMatrix,i::Integer,j::Integer)
    I = a.indices[1][i]
    J = a.indices[2][j]
    a.parent[I,J]
end

function LinearAlgebra.mul!(
        C::AbstractVector,
        A::SubSparseMatrix{T,<:SparseArrays.AbstractSparseMatrixCSC} where T,
        B::AbstractVector,
        α::Number,
        β::Number)

    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    rows, cols = A.indices
    invrows, invcols = A.inv_indices
    Ap = A.parent
    nzv = nonzeros(Ap)
    rv = rowvals(Ap)
    for (j,J) in enumerate(cols)
        αxj = B[j] * α
        for p in nzrange(Ap,J)
            I = rv[p]
            i = invrows[I]
            if i>0
                C[i] += nzv[p]*αxj
            end
        end
    end
    C
end

function LinearAlgebra.mul!(
        C::AbstractVector,
        A::SubSparseMatrix{T,<:SparseMatrixCSR} where T,
        B::AbstractVector,
        α::Number,
        β::Number)

    size(A, 2) == size(B, 1) || throw(DimensionMismatch())
    size(A, 1) == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    rows, cols = A.indices
    invrows, invcols = A.inv_indices
    Ap = A.parent
    nzv = nonzeros(Ap)
    cv = colvals(Ap)
    o = getoffset(Ap)
    for (i,I) in enumerate(rows)
        for p in nzrange(Ap,I)
            J = cv[p]+o
            j = invcols[J]
            if j>0
                C[i] += nzv[p]*B[j]*α
            end
        end
    end
    C
end

function LinearAlgebra.fillstored!(A::SubSparseMatrix{T,<:SparseArrays.AbstractSparseMatrixCSC},v) where T
    rows, cols = A.indices
    invrows, invcols = A.inv_indices
    Ap = A.parent
    nzv = nonzeros(Ap)
    rv = rowvals(Ap)
    for (j,J) in enumerate(cols)
        for p in nzrange(Ap,J)
            I = rv[p]
            i = invrows[I]
            if i>0
                nzv[p]=v
            end
        end
    end
    A
end

function LinearAlgebra.fillstored!(A::SubSparseMatrix{T,<:SparseMatrixCSR},v) where T
    rows, cols = A.indices
    invrows, invcols = A.inv_indices
    Ap = A.parent
    nzv = nonzeros(Ap)
    cv = colvals(Ap)
    o = getoffset(Ap)
    for (i,I) in enumerate(rows)
        for p in nzrange(Ap,I)
            J = cv[p]+o
            j = invcols[J]
            if j>0
                nzv[p] = v
            end
        end
    end
    A
end

"""
    nzindex(a,i,j)

Return the position in `nonzeros(a)` that stores the non zero value of `a` at row `i`
and column `j`.
"""
function nzindex end

function nzindex(A::SparseArrays.AbstractSparseMatrixCSC, i0::Integer, i1::Integer)
    if !(1 <= i0 <= size(A, 1) && 1 <= i1 <= size(A, 2)); throw(BoundsError()); end
    ptrs = SparseArrays.getcolptr(A)
    r1 = Int(ptrs[i1])
    r2 = Int(ptrs[i1+1]-1)
    (r1 > r2) && return -1
    r1 = searchsortedfirst(rowvals(A), i0, r1, r2, Base.Order.Forward)
    ((r1 > r2) || (rowvals(A)[r1] != i0)) ? 0 : r1
end

function nzindex(A::SparseMatrixCSR, i0::Integer, i1::Integer)
  if !(1 <= i0 <= size(A, 1) && 1 <= i1 <= size(A, 2)); throw(BoundsError()); end
  o = getoffset(A)
  Bi = getBi(A)
  r1 = Int(A.rowptr[i0]+o)
  r2 = Int(A.rowptr[i0+1]-Bi)
  (r1 > r2) && return -1
  i1o = i1-o
  k = searchsortedfirst(colvals(A), i1o, r1, r2, Base.Order.Forward)
  ((k > r2) || (colvals(A)[k] != i1o)) ? 0 : k
end

# TODO remove and simply use sparse_matrix
"""
    compresscoo(T,args...)

Like `sparse(args...)`, but generates a sparse matrix of type `T`.
"""
function compresscoo end

compresscoo(a::AbstractSparseMatrix,args...) = compresscoo(typeof(a),args...)

function compresscoo(
  ::Type{SparseMatrixCSC{Tv,Ti}},
  I::AbstractVector,
  J::AbstractVector,
  V::AbstractVector,
  m::Integer,
  n::Integer;
  combine=+,
  skip=false,
  ) where {Tv,Ti}

    T = SparseMatrixCSRR{Tv,Ti}
    Acsrr = compresscoo(T,I,J,V,m,n;combine,skip)
    SparseMatrixCSC{Tv,Ti}(Acsrr)

  #sparse(
  #  EltypeVector(Ti,I),
  #  EltypeVector(Ti,J),
  #  EltypeVector(Tv,V),
  #  m,n,combine)
end


function compresscoo(
  ::Type{SparseMatrixCSR{Bi,Tv,Ti}},
  I::AbstractVector,
  J::AbstractVector,
  V::AbstractVector,
  m::Integer,
  n::Integer;
  combine=+,
  skip=false,
    ) where {Bi,Tv,Ti}

    #Tcsrr = SparseMatrixCSRR{Tv,Ti}
    #Atcsrr = compresscoo(T,J,I,V,n,m;combine,skip)
    #Atcsc = SparseMatrixCSC{Tv,Ti}(Atcsrr)
    #Atcsc |> transpose |> SparseMatrixCSR{Bi,Tv,Ti}

    # TODO
    if !skip
        I2 = I
        J2 = J
        V2 = V
    elseif m*n == 0
        I2 = eltype(I)[]
        J2 = eltype(I)[]
        V2 = eltype(V)[]
    else
        I2 = FilteredCooVector(one,I,J,I)
        J2 = FilteredCooVector(one,I,J,J)
        V2 = FilteredCooVector(zero,I,J,V)
    end

  sparsecsr(
    Val(Bi),
    EltypeVector(Ti,I2),
    EltypeVector(Ti,J2),
    EltypeVector(Tv,V2),
    m,n,combine)
end

struct EltypeVector{T,V} <: AbstractVector{T}
  parent::V
  function EltypeVector(::Type{T},parent::V) where {T,V<:AbstractVector}
    new{T,V}(parent)
  end
end
EltypeVector(::Type{T},parent::AbstractVector{T}) where T = parent
Base.size(v::EltypeVector) = size(v.parent)
Base.axes(v::EltypeVector) = axes(v.parent)
Base.@propagate_inbounds Base.getindex(v::EltypeVector{T},i::Integer) where T = convert(T,v.parent[i])
Base.@propagate_inbounds Base.setindex!(v::EltypeVector,w,i::Integer) = (v.parent[i] = w)
Base.IndexStyle(::Type{<:EltypeVector{T,V}}) where {T,V} = IndexStyle(V)

# TODO deprecated
function setcoofast!(A,V,K)
    sparse_matrix!(A,V,K)
end

struct FilteredCooVector{F,A,B,C,T} <: AbstractVector{T}
    f::F
    I::A
    J::B
    V::C
    function FilteredCooVector(f::F,I::A,J::B,V::C) where {F,A,B,C}
        T = eltype(C)
        new{F,A,B,C,T}(f,I,J,V)
    end
end
Base.size(a::FilteredCooVector) = size(a.V)
Base.IndexStyle(::Type{<:FilteredCooVector}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(a::FilteredCooVector,k::Int)
    i = a.I[k]
    j = a.J[k]
    v = a.V[k]
    if i < 1 || j < 1
        return a.f(v)
    end
    v
end

function sparse_matrix(I,J,V,m,n;kwargs...)
    Tv = eltype(V)
    Ti = eltype(I)
    sparse_matrix(SparseMatrixCSC{Tv,Ti},I,J,V,m,n;kwargs...)
end

function sparse_matrix(
    ::Type{T},I,J,V,m,n;reuse=Val(false),combine=+,skip=true) where T
    A = compresscoo(T,I,J,V,m,n;combine,skip)
    if val_parameter(reuse)
        K = precompute_nzindex(A,I,J)
        return A,K
    end
    A
    #f(args...) = compresscoo(T,args...)
    #sparse_matrix(f,I,J,V,m,n;kwargs...)
end

#function sparse_matrix(f,I,J,V,m,n;reuse=Val(false),skip=true)
#    if !skip
#        I2 = I
#        J2 = J
#        V2 = V
#    elseif m*n == 0
#        Ti = eltype(I)
#        Tv = eltype(V)
#        I2 = Ti[]
#        J2 = Ti[]
#        V2 = Tv[]
#    else
#        I2 = FilteredCooVector(one,I,J,I)
#        J2 = FilteredCooVector(one,I,J,J)
#        V2 = FilteredCooVector(zero,I,J,V)
#    end
#    A = f(I2,J2,V2,m,n)
#    if val_parameter(reuse)
#        K = precompute_nzindex(A,I,J)
#        return A,K
#    end
#    A
#end

# Variants for findnz() that only allocates memory for the conversion of the pointer array to an index array.
function findnz_minimal(A::SparseMatrixCSC)
    J = ptr_to_coo(A.colptr)
    rowvals(A),J,nonzeros(A)
end
function findnz_minimal(A::SparseMatrixCSR)
    I = ptr_to_coo(A.rowptr)
    I,colvals(A),nonzeros(A)
end

# Behaves like findnz, but without copying the values.
function find_indices(A::SparseMatrixCSC)
    I,J,_ = findnz_minimal(A)
    copy(I),J
end
function find_indices(A::SparseMatrixCSR)
    I,J,_ = findnz_minimal(A)
    I,copy(J)
end

# Could be optimized by a two-way merge-like method when A is a guaranteed submatrix of C.
function precompute_nzindex(C::AbstractSparseArray,A::AbstractSparseArray)
    I,J,_ = findnz_minimal(A)
    K = similar(I)
    K .= 0
    for (p,(i,j)) in enumerate(zip(I,J))
        if i < 1 || j < 1
            continue
        end
        K[p] = nzindex(C,i,j)
    end
    K
end

function precompute_nzindex(A,I,J)
    K = zeros(Int32,length(I))
    for (p,(i,j)) in enumerate(zip(I,J))
        if i < 1 || j < 1
            continue
        end
        K[p] = nzindex(A,i,j)
    end
    K
end

# Reuse I vector as K vector. 
function precompute_nzindex!(I,A,J)
    for (p,(i,j)) in enumerate(zip(I,J))
        if i < 1 || j < 1
            continue
        end
        I[p] = nzindex(A,i,j)
    end
    I
end

function sparse_matrix!(A,V,K;reset=true)
    if reset
        LinearAlgebra.fillstored!(A,0)
    end
    A_nz = nonzeros(A)
    for (k,v) in zip(K,V)
        if k < 1
            continue
        end
        A_nz[k] += v
    end
    A
end


# Notation
# csrr: csr with repeated and unsorted columns
# csru: csr with unsorted columns
# csc: csc with sorted columns

struct SparseMatrixCSRR{Tv,Ti,A}
    rowptr::Vector{Ti}
    colval::Vector{Ti}
    nzval::Vector{Tv}
    nrows::Int
    ncols::Int
    combine::A
end

function compresscoo(
        ::Type{SparseMatrixCSRR{Tv,Ti}},
        I::AbstractVector,
        J::AbstractVector,
        V::AbstractVector,
        m::Integer,
        n::Integer;
        combine=+,
        skip=false) where {Tv,Ti}

    nrows = m
    ncols = n
    rowptr = zeros(Ti,nrows+1)
    for (row,col) in zip(I,J)
        if !skip || ((row in 1:nrows) && (col in 1:ncols))
            rowptr[row+1] += Ti(1)
        end
    end
    length_to_ptrs!(rowptr)
    nnz = rowptr[end]-1
    colval = Vector{Ti}(undef,nnz)
    nzval = Vector{Tv}(undef,nnz)
    for (row,col,val) in zip(I,J,V)
        if !skip || ((row in 1:nrows) && (col in 1:ncols))
            p = rowptr[row]
            colval[p] = col
            nzval[p] = val
            rowptr[row] = p + Ti(1)
        end
    end
    rewind_ptrs!(rowptr)
    SparseMatrixCSRR(rowptr,colval,nzval,nrows,ncols,combine)
end

function SparseMatrixCSC{Tv,Ti}(a::SparseMatrixCSRR) where {Tv,Ti}
    colptr = Vector{Ti}(undef,a.ncols+1)
    work = Vector{Ti}(undef,a.ncols)
    cscnnz = csrr_to_csc_step_1(a.combine,colptr,a.rowptr,a.colval,a.nzval,work)
    rowval = Vector{Ti}(undef,cscnnz)
    nzvalcsc = Vector{Tv}(undef,cscnnz)
    csrr_to_csc_step_2(colptr,rowval,nzvalcsc,a.rowptr,a.colval,a.nzval)
    SparseMatrixCSC(a.nrows,a.ncols,colptr,rowval,nzvalcsc)
end

function csrr_to_csc_step_1(
        combine,
        colptrs::Vector{Ti},
        rowptrs::Vector{Tj},
        colvals::Vector{Tj},
        nzvalscsr::Vector{Tv},
        work::Vector{Tj},
    ) where {Ti,Tj,Tv}

    nrows = length(rowptrs)-1
    ncols = length(colptrs)-1
    if nrows == 0 || ncols == 0
        fill!(colptrs, Ti(1))
        return Tj(0)
    end
    # Convert csrr to csru by identifying repeated cols with array work.
    # At the same time, count number of unique rows in colptrs shifted by one.
    fill!(colptrs, Ti(0))
    fill!(work, Tj(0))
    writek = Tj(1)
    newcsrrowptri = Ti(1)
    origcsrrowptri = Tj(1)
    origcsrrowptrip1 = rowptrs[2]
    for i in 1:nrows
        for readk in origcsrrowptri:(origcsrrowptrip1-Tj(1))
            j = colvals[readk]
            if work[j] < newcsrrowptri
                work[j] = writek
                if writek != readk
                    colvals[writek] = j
                    nzvalscsr[writek] = nzvalscsr[readk]
                end
                writek += Tj(1)
                colptrs[j+1] += Ti(1)
            else
                klt = work[j]
                nzvalscsr[klt] = combine(nzvalscsr[klt], nzvalscsr[readk])
            end
        end
        newcsrrowptri = writek
        origcsrrowptri = origcsrrowptrip1
        origcsrrowptrip1 != writek && (rowptrs[i+1] = writek)
        i < nrows && (origcsrrowptrip1 = rowptrs[i+2])
    end
    # Convert colptrs from counts to ptrs shifted by one
    # (ptrs will be corrected below)
    countsum = Tj(1)
    colptrs[1] = Ti(1)
    for j in 2:(ncols+1)
        overwritten = colptrs[j]
        colptrs[j] = countsum
        countsum += overwritten
    end
    cscnnz = countsum - Tj(1)
    Tj(cscnnz)
end

function csrr_to_csc_step_2(
  colptrs::Vector{Ti},rowvals::Vector{Ti},nzvalscsc::Vector{Tv},
  rowptrs::Vector{Tj},colvals::Vector{Tj},nzvalscsr::Vector{Tv}) where {Ti,Tj,Tv}

  nrows = length(rowptrs)-1
  ncols = length(colptrs)-1
  if nrows == 0 || ncols == 0
    return nothing
  end
  # From csru to csc
  # Tracking write positions in colptrs corrects
  # the column pointers to the final value.
  for i in 1:nrows
    for csrk in rowptrs[i]:(rowptrs[i+1]-Tj(1))
      j = colvals[csrk]
      x = nzvalscsr[csrk]
      csck = colptrs[j+1]
      colptrs[j+1] = csck + Ti(1)
      rowvals[csck] = i
      nzvalscsc[csck] = x
    end
  end
  nothing
end

@inline function spmv!(b,A,x)
    mul!(b,A,x)
end

@inline function spmtv!(b,A,x)
    mul!(b,transpose(A),x)
end

function spmv!(b,A::SparseMatrixCSR{1},x)
    @boundscheck begin
        @assert length(b) == size(A,1)
        @assert length(x) == size(A,2)
    end
    spmv_csr!(b,x,A.rowptr,A.colval,A.nzval)
end

function spmtv!(b,A::SparseMatrixCSR{1},x)
    @boundscheck begin
        @assert length(b) == size(A,2)
        @assert length(x) == size(A,1)
    end
    spmv_csc!(b,x,A.rowptr,A.colval,A.nzval)
end

function spmv!(b,A::SparseMatrixCSC,x)
    @boundscheck begin
        @assert length(b) == size(A,1)
        @assert length(x) == size(A,2)
    end
    spmv_csc!(b,x,A.colptr,A.rowval,A.nzval)
end

function spmtv!(b,A::SparseMatrixCSC,x)
    @boundscheck begin
        @assert length(b) == size(A,2)
        @assert length(x) == size(A,1)
    end
    spmv_csr!(b,x,A.colptr,A.rowval,A.nzval)
end

function spmv_csr!(b,x,rowptr_A,colval_A,nzval_A)
    ncols = length(x)
    nrows = length(b)
    u = one(eltype(rowptr_A))
    z = zero(eltype(b))
    @inbounds for row in 1:nrows
        pini = rowptr_A[row]
        pend = rowptr_A[row+1]
        bi = z
        p = pini
        while p < pend
            aij = nzval_A[p]
            col = colval_A[p]
            xj = x[col]
            bi += aij*xj
            p += u
        end
        b[row] = bi
    end
    b
end

function spmv_csc!(b,x,colptr_A,rowval_A,nzval_A)
    ncols = length(x)
    nrows = length(b)
    u = one(eltype(colptr_A))
    z = zero(eltype(b))
    fill!(b,z)
    @inbounds for col in 1:ncols
        pini = colptr_A[col]
        pend = colptr_A[col+1]
        p = pini
        xj = x[col]
        while p < pend
            aij = nzval_A[p]
            row = rowval_A[p]
            b[row] += aij*xj
            p += u
        end
    end
    b
end

function expand_sparse_matrix_columns(A::SparseMatrixCSR{Bi,Tv,Ti} where {Tv, Ti}, n) where Bi
    p,q = size(A)
    @assert n >= q
    SparseMatrixCSR{Bi}(p,n,A.rowptr,A.colval,A.nzval)
end

function expand_sparse_matrix_columns(A::SparseMatrixCSC{Tv,Ti}, n) where {Tv,Ti}
    p,q = size(A)
    @assert n >= q
    new_colptr = similar(A.colptr,n+1)
    map!(identity,new_colptr,A.colptr)
    last_index = A.colptr[end]
    foreach(q+1:n+1) do i
        new_colptr[i] = last_index
    end
    SparseMatrixCSC{Tv,Ti}(p,n,new_colptr,A.rowval,A.nzval)
end

# Currently not implemented by the SparseMatricesCSR module
function Base.similar(A::SparseMatrixCSR{Bi}, m::Integer, n::Integer) where Bi
    SparseMatrixCSR{1}(m, n, ones(eltype(A.rowptr), m+1), eltype(A.colval)[], eltype(A.nzval)[])
end

# Currently not implemented by the SparseMatricesCSR module
function Base.similar(A::SparseMatrixCSR{Bi}) where Bi
    SparseMatrixCSR{Bi}(size(A)..., copy(A.rowptr), copy(colvals(A)), similar(nonzeros(A)))
end

# This method is implemented also by SparseMatricesCSR, but related methods aren't.
# function Base.copy(A::SparseMatrixCSR{Bi}) where Bi
#     SparseMatrixCSR{Bi}(size(A)..., copy(A.rowptr), copy(colvals(A)), copy(nonzeros(A)))
# end

# Currently not implemented by the SparseMatricesCSR module
function Base.copy(At::Transpose{Tv,SparseMatrixCSR{Bi,Tv,Ti}} where {Bi,Tv,Ti})
    Acsc_T = copy(transpose(ascsc(At.parent))) # materialize SparseMatrixCSC transpose
    ascsr(Acsc_T)
end

function SparseMatricesCSR.sparsecsr(A::SparseMatrixCSC)
    sparsecsr(findnz(A)..., size(A)...)
end

function SparseMatricesCSR.sparsecsr(At::Transpose)
    transpose(sparsecsr(At.parent))
end

function SparseMatricesCSR.sparsecsr(A::SparseMatrixCSR)
    A
end

function SparseMatricesCSR.sparsecsr(T::Type, A::SparseMatrixCSC)
    compresscoo(T,findnz(A)..., size(A)...)
end


function pointer_array(A::SparseMatrixCSR)
    A.rowptr
end

function pointer_array(A::SparseMatrixCSC)
    A.colptr
end

function index_array(A::SparseMatrixCSR)
    colvals(A)
end

function index_array(A::SparseMatrixCSC)
    rowvals(A)
end

function ptr_to_coo(ptr_array)
    K = zeros(Int32, (ptr_array[end]-1))
    for i in 1:(length(ptr_array)-1)
        for p in ptr_array[i]:ptr_array[i+1]-1
            K[p] = i
        end
    end
    K
end

function find_max_row_length(A::SparseMatrixCSR)
    max_rA = 0
    for i in 1:size(A,1)
        l = length(nzrange(A,i))
        max_rA = max_rA > l ? max_rA : l
    end
    max_rA
end

function find_max_col_length(A::SparseMatrixCSC)
    max_cA = 0
    for j in 1:size(A,2)
        l = length(nzrange(A,j))
        max_cA = max_cA > l ? max_cA : l
    end
    max_cA
end

# Lazily convert CSC matrix to CSR matrix, by interpreting columnpointers as row pointers, and colvals as rowvals,
# effectively transposing it in the process.
function ascsr(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    p,q = size(A)
    SparseMatrixCSR{1}(q,p,A.colptr,rowvals(A),nonzeros(A))
end

# Lazily convert CSR matrix to CSC matrix, by interpreting rowpointers as column pointers, and rowvals as colvals,
# effectively transposing it in the process.
function ascsc(A::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    p,q = size(A)
    SparseMatrixCSC{Tv,Ti}(q,p,A.rowptr,colvals(A),nonzeros(A))
end


function halfperm(A::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    q = size(A,2)
    JA,VA = colvals(A),nonzeros(A)
    IAt,JAt,VAt = similar(A.rowptr,q+1),similar(JA),similar(VA)
    halfperm!(IAt,JAt,VAt,A)
end

# transpose A into At using vectors IAt,JAt, and VAt
function halfperm!(IAt,JAt,VAt,A::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    JA,VA = colvals(A),nonzeros(A)
    p,q = size(A)
    count_occurrences!(IAt,JA)
    counts_to_ptrs!(IAt)
    shift_by_one!(IAt)
    for i in 1:p
        for jp in nzrange(A,i)
            j = JA[jp]
            jpt = IAt[j+1]
            JAt[jpt] = i
            VAt[jpt] = VA[jp]
            IAt[j+1] = jpt+1
        end
    end
    IAt[1] = 1
    SparseMatrixCSR{Bi}(q,p,IAt,JAt,VAt)
end

# retranspose At back into A
function halfperm!(A::SparseMatrixCSR{Bi,Tv,Ti},At::SparseMatrixCSR{Bi,Tv,Ti}) where {Bi,Tv,Ti}
    IA,JA,VA = A.rowptr,colvals(A),nonzeros(A)
    JAt,VAt = colvals(At),nonzeros(At)
    p,q = size(At)
    shift_by_one!(IA) # pointer to row 1 must be located at IA[2], row 2 at IA[3] etc.
    IA[1] = 1
    for i in 1:p
        for jpt in nzrange(At,i)
            j = JAt[jpt]
            jp = IA[j+1]
            JA[jp] = i
            VA[jp] = VAt[jpt]
            IA[j+1] = jp+1
        end
    end
    At
end

function halfperm!(A::SparseMatrixCSC,At::SparseMatrixCSC)
    halfperm!(ascsr(A),ascsr(At))
    A
end

function halfperm(A::SparseMatrixCSC)
    At = halfperm(ascsr(A))
    ascsc(At)
end

function count_occurrences!(v1::AbstractVector{<:Integer},v2::AbstractVector{<:Integer};set_zero=true)
    if set_zero
        v1 .= 0
    end
    foreach(i->v1[i]+=1,v2)
    v1
end

# shift all entries one element to the right in-place. Not circular.
function shift_by_one!(v)
    l = length(v)
    prev = v[1]
    tmp = prev
    for i in 1:l-1
        tmp = v[i+1]
        v[i+1] = prev
        prev = tmp
    end
end

function counts_to_ptrs!(v)
    l = length(v)
    v[1] += 1
    foreach(i->v[i]+=v[i-1],2:l)
    shift_by_one!(v)
    v[1] = 1
end