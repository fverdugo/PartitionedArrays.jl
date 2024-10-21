
struct BRange{A} <: AbstractUnitRange{Int}
  blocks::A
  blocklasts::Vector{Int}
  function BRange(blocks)
      nblocks = length(blocks)
      blocklasts = zeros(Int,nblocks)
      prev = 0
      for i in 1:nblocks
          prev += length(blocks[i])
          blocklasts[i] = prev
      end
      A = typeof(blocks)
      new{A}(blocks,blocklasts)
  end
end
BlockArrays.blocks(a::BRange) = a.blocks
BlockArrays.blocklasts(a::BRange) = a.blocklasts
BlockArrays.blockaxes(a::BRange) = (Block.(Base.OneTo(length(blocks(a)))),)
Base.getindex(a::BRange,k::Block{1}) = blocks(a)[first(k.n)]
function BlockArrays.findblock(b::BRange, k::Integer)
    @boundscheck k in b || throw(BoundsError(b,k))
    Block(searchsortedfirst(blocklasts(b), k))
end
Base.first(a::BRange) = 1
Base.last(a::BRange) = sum(length,blocks(a))

function Base.show(io::IO,k::MIME"text/plain",data::BRange)
    function tostr(a)
        "$a"
    end
    function tostr(a::PRange)
        np = length(partition(a))
        "PRange 1:$(length(a)) partitioned into $(np) parts"
    end
    nb = blocklength(data)
    println(io,"BRange $(first(data)):$(last(data)) with $nb blocks:")
    for ib in 1:nb
        t = ib != nb ? "├──" : "└──"
        println(io,"$(t) Block($ib) = $(tostr(blocks(data)[ib]))")
    end
end

function Base.show(io::IO,data::BRange)
    print(io,"BRange(…)")
end

function partition(a::BRange)
    ps = map(partition,blocks(a))
    ps2 = permute_nesting(ps)
    map(BVector,ps2)
end

struct BArray{A,T,N} <: BlockArrays.AbstractBlockArray{T,N}
    blocks::A
    function BArray(blocks)
      T = eltype(first(blocks))
      N = ndims(first(blocks))
      @assert all(block->eltype(block)==T,blocks)
      @assert all(block->ndims(block)==N,blocks)
      A = typeof(blocks)
      new{A,T,N}(blocks)
    end
end
const BVector{A,T} = BArray{A,T,1}
const BMatrix{A,T} = BArray{A,T,2}
function BVector(blocks)
    N = ndims(first(blocks))
    @assert N==1
    BArray(blocks)
end
function BMatrix(blocks)
    N = ndims(first(blocks))
    @assert N==2
    BArray(blocks)
end
BlockArrays.blocks(a::BArray) = a.blocks
BlockArrays.viewblock(a::BArray, i::Block) = blocks(a)[i.n...]
BlockArrays.blockaxes(a::BArray,d::Int) = Block.(Base.OneTo(size(blocks(a),d)))
BlockArrays.blockaxes(a::BArray) = map(i->blockaxes(a,i),ntuple(j->Int(j),ndims(a)))
function Base.axes(a::BArray,d::Int)
    N = ndims(a)
    I = ntuple(i-> (i == d) ? (:) : 1, Val(N))
    BRange(map(block->axes(block,d),blocks(a)[I...]))
end
Base.axes(a::BArray) = map(i->axes(a,i),ntuple(j->Int(j),ndims(a)))
Base.size(a::BArray) = map(length,axes(a))
Base.length(a::BArray) = sum(length,blocks(a))
Base.IndexStyle(::Type{<:BArray}) = IndexCartesian()
function Base.getindex(a::BArray{A,T,N} where {A,T},gid::Vararg{Int,N}) where N
    # This could be implemented as it makes sense when the blocks
    # are sequential arrays
    scalar_indexing_action(a)
end
function Base.setindex(a::BArray{A,T,N} where {A,T},v,gid::Vararg{Int,N}) where N
    # This could be implemented as it makes sense when the blocks
    # are sequential arrays
    scalar_indexing_action(a)
end
function Base.show(io::IO,k::MIME"text/plain",data::BArray)
    N = ndims(data)
    if N == 1
        at = "BVector"
    elseif N==2
        at = "BMatrix"
    else
        at = "BArray"
    end
    bs = blocksize(data) 
    bst = map(i->"$(i)×",collect(bs))
    bst[end] = string(bs[end])
    s = size(data)
    st = map(i->"$(i)×",collect(s))
    st[end] = string(s[end])
    println(io,"$(join(st))-element $at with $(join(bst)) blocks:")
    for cb in CartesianIndices(bs)
        ib = LinearIndices(bs)[cb]
        nb = prod(bs)
        t = ib != nb ? "├──" : "└──"
        tcb =Tuple(cb)
        b = map(i->"$i, ",collect(tcb))
        b[end] = string(tcb[end])
        println(io,"$(t) Block($(join(b))) = $(blocks(data)[cb])")
    end
end
function Base.show(io::IO,data::BArray)
    print(io,"BArray(…)")
end
function Base.show(io::IO,data::BVector)
    print(io,"BVector(…)")
end
function Base.show(io::IO,data::BMatrix)
    print(io,"BMatrix(…)")
end

function partition(a::BArray)
    ps = map(partition,blocks(a))
    ps2 = permute_nesting(ps)
    map(BArray,ps2)
end

function local_values(a::BVector)
    ps = map(local_values,blocks(a))
    ps2 = permute_nesting(ps)
    map(BVector,ps2)
end

function own_values(a::BVector)
    ps = map(own_values,blocks(a))
    ps2 = permute_nesting(ps)
    map(BVector,ps2)
end

function ghost_values(a::BVector)
    ps = map(ghost_values,blocks(a))
    ps2 = permute_nesting(ps)
    map(BVector,ps2)
end

function consistent!(a::BVector)
    ts = map(consistent!,blocks(a))
    @fake_async begin
        foreach(wait,ts)
        a
    end
end

function assemble!(a::BVector)
    ts = map(assemble!,blocks(a))
    @fake_async begin
        foreach(wait,ts)
        a
    end
end

function Base.similar(a::BVector,::Type{T},inds::Tuple{<:BRange}) where T
    r = first(inds)
    bs = map((ai,ri)->similar(ai,T,ri),blocks(a),blocks(r))
    BVector(bs)
end

function Base.copy(a::BArray)
    map(copy,blocks(a)) |> BArray
end

function Base.copy!(a::BArray,b::BArray)
    foreach(copy!,blocks(a),blocks(b))
    a
end

function Base.copyto!(a::BArray,b::BArray)
    foreach(copyto!,blocks(a),blocks(b))
    a
end

function Base.fill!(a::BArray,v)
    foreach(ai->fill!(ai,v),blocks(a))
    a
end

function Base.:(==)(a::BArray,b::BArray)
    all(map(==,blocks(a),blocks(b)))
end

function Base.any(f::Function,x::BArray)
    any(xi->any(f,xi),blocks(x))
end

function Base.all(f::Function,x::BArray)
    all(xi->all(f,xi),blocks(x))
end

Base.maximum(x::BArray) = maximum(identity,x)
function Base.maximum(f::Function,x::BArray)
    maximum(xi->maximum(f,xi),blocks(x))
end

Base.minimum(x::BArray) = minimum(identity,x)
function Base.minimum(f::Function,x::BArray)
    minimum(xi->minimum(f,xi),blocks(x))
end

function Base.collect(v::BVector)
    reduce(vcat,map(collect,blocks(v)))
end

function Base.:*(a::Number,b::BArray)
    bs = map(bi->a*bi,blocks(b))
    BArray(bs)
end

function Base.:*(b::BArray,a::Number)
    a*b
end

function Base.:/(b::BArray,a::Number)
    (1/a)*b
end

for op in (:+,:-)
    @eval begin
        function Base.$op(a::BArray)
            bs = map($op,blocks(a))
            BArray(bs)
        end
        function Base.$op(a::BArray,b::BArray)
            bs = map($op,blocks(a),blocks(b))
            BArray(bs)
        end
    end
end

function Base.reduce(op,a::BArray;neutral=neutral_element(op,eltype(a)),kwargs...)
    rs = map(ai->reduce(op,ai;neutral,kwargs...),blocks(a))
    reduce(op,rs;kwargs...)
end

function Base.sum(a::BArray)
  reduce(+,a,init=zero(eltype(a)))
end

function LinearAlgebra.dot(a::BVector,b::BVector)
    c = map(dot,blocks(a),blocks(b))
    sum(c)
end

function LinearAlgebra.rmul!(a::BArray,v::Number)
    map(blocks(a)) do l
        rmul!(l,v)
    end
    a
end

function LinearAlgebra.norm(a::BVector,p::Real=2)
    contibs = map(blocks(a)) do oid_to_value
        norm(oid_to_value,p)^p
    end
    reduce(+,contibs;init=zero(eltype(contibs)))^(1/p)
end

for M in Distances.metrics
    @eval begin
        function (d::$M)(a::BVector,b::BVector)
            s = distance_eval_body(d,a,b)
            Distances.eval_end(d,s)
        end
    end
end

function distance_eval_body(d,a::BVector,b::BVector)
    partials = map(blocks(a),blocks(b)) do ai,bi
        distance_eval_body(d,ai,bi)
    end
    s = reduce((i,j)->Distances.eval_reduce(d,i,j),
               partials,
               init=Distances.eval_start(d, a, b))
    s
end

struct BroadcastedBArray{A}
    blocks::A
end
BlockArrays.blocks(a::BroadcastedBArray) = a.blocks

function Base.broadcasted(f, args::Union{BArray,BroadcastedBArray}...)
    map( (bs...) -> Base.broadcasted(f,bs...) , map(blocks,args)...) |> BroadcastedBArray
end

function Base.broadcasted( f, a::Number, b::Union{BArray,BroadcastedBArray})
    map( bi -> Base.broadcasted(f,a,bi), blocks(b)) |> BroadcastedBArray
end

function Base.broadcasted( f, a::Union{BArray,BroadcastedBArray}, b::Number)
    map( ai -> Base.broadcasted(f,ai,b), blocks(a)) |> BroadcastedBArray
end

function Base.broadcasted(f,
                          a::Union{BArray,BroadcastedBArray},
                          b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
    Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
    f,
    a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
    b::Union{BArray,BroadcastedBArray})
    Base.broadcasted(f,Base.materialize(a),b)
 end

function Base.materialize(b::BroadcastedBArray)
    map(Base.materialize,blocks(b)) |> BArray
end

function Base.materialize!(a::BArray,b::BroadcastedBArray)
    foreach(Base.materialize!,blocks(a),blocks(b))
    a
end

function own_own_values(a::BMatrix)
    ps = map(own_own_values,blocks(a))
    ps2 = permute_nesting(ps)
    map(BMatrix,ps2)
end
function own_ghost_values(a::BMatrix)
    ps = map(own_ghost_values,blocks(a))
    ps2 = permute_nesting(ps)
    map(BMatrix,ps2)
end
function ghost_own_values(a::BMatrix)
    ps = map(ghost_own_values,blocks(a))
    ps2 = permute_nesting(ps)
    map(BMatrix,ps2)
end
function ghost_ghost_values(a::BMatrix)
    ps = map(ghost_ghost_values,blocks(a))
    ps2 = permute_nesting(ps)
    map(BMatrix,ps2)
end

function LinearAlgebra.fillstored!(a::BMatrix,v)
    foreach(ai->LinearAlgebra.fillstored!(ai,v),blocks(a))
    a
end

#function centralize(v::BMatrix)
#    # TODO not correct
#    reduce((ai...)->cat(ai...,dims=(1,2)),map(centralize,blocks(v)))
#end

function SparseArrays.nnz(a::BMatrix)
    ns = map(nnz,blocks(a))
    sum(ns)
end

# This function could be removed if IterativeSolvers was implemented in terms
# of axes(A,d) instead of size(A,d)
function IterativeSolvers.zerox(A::BMatrix,b::BVector)
    T = IterativeSolvers.Adivtype(A, b)
    x = similar(b, T, axes(A, 2))
    fill!(x, zero(T))
    return x
end

function Base.:*(a::BMatrix,b::BVector)
    Ta = eltype(a)
    Tb = eltype(b)
    T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
    c = similar(b,T,axes(a,1))
    mul!(c,a,b)
    c
end

function LinearAlgebra.mul!(b::BVector,A::BMatrix,x::BVector)
    mul!(b,A,x,1,0)
end

function LinearAlgebra.mul!(b::BVector,A::BMatrix,x::BVector,α::Number,β::Number)
    if β != 1
        β != 0 ? rmul!(b, β) : fill!(b,zero(eltype(b)))
    end
    bb = blocks(b)
    Ab = blocks(A)
    xb = blocks(x)
    o = one(eltype(b))
    for i in 1:blocksize(A,1)
        for j in 1:blocksize(A,2)
            mul!(bb[i],Ab[i,j],xb[j],α,o)
        end
    end
    b
end

