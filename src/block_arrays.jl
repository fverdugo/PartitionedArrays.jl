
function BlockArrays.mortar(blocks::Vector{<:PRange})
    BlockedPRange(blocks)
end

struct BlockedPRange{A} <: AbstractUnitRange{Int}
  blocks::A
  blocklasts::Vector{Int}
  function BlockedPRange(blocks)
      @assert all(block->isa(block,PRange),blocks)
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
BlockArrays.blocks(a::BlockedPRange) = a.blocks
BlockArrays.blocklasts(a::BlockedPRange) = a.blocklasts
BlockArrays.blockaxes(a::BlockedPRange) = (Block.(Base.OneTo(length(blocks(a)))),)
Base.getindex(a::BlockedPRange,k::Block{1}) = blocks(a)[first(k.n)]
function BlockArrays.findblock(b::BlockedPRange, k::Integer)
    @boundscheck k in b || throw(BoundsError(b,k))
    Block(searchsortedfirst(blocklasts(b), k))
end
Base.first(a::BlockedPRange) = 1
Base.last(a::BlockedPRange) = sum(length,blocks(a))

function PartitionedArrays.partition(a::BlockedPRange)
    local_block_ranges(a)
end

function local_block_ranges(a::BlockedPRange)
    ids = map(partition,blocks(a)) |> array_of_tuples
    map(ids) do myids
        map(local_length,myids) |> collect |> blockedrange
    end
end

function own_block_ranges(a::BlockedPRange)
    ids = map(partition,blocks(a)) |> array_of_tuples
    map(ids) do myids
        map(own_length,myids) |> collect |> blockedrange
    end
end

function ghost_block_ranges(a::BlockedPRange)
    ids = map(partition,blocks(a)) |> array_of_tuples
    map(ids) do myids
        map(ghost_length,myids) |> collect |> blockedrange
    end
end

# BlockPVector

function BlockArrays.mortar(blocks::Vector{<:PVector})
    BlockPVector(blocks)
end

struct BlockPVector{A,T} <: BlockArrays.AbstractBlockVector{T}
    blocks::A
    function BlockPVector(blocks)
      T = eltype(first(blocks))
      @assert all(block->isa(block,PVector),blocks)
      @assert all(block->eltype(block)==T,blocks)
      A = typeof(blocks)
      new{A,T}(blocks)
    end
end
BlockArrays.blocks(a::BlockPVector) = a.blocks
BlockArrays.viewblock(a::BlockPVector, i::Block) = blocks(a)[i.n...]
Base.axes(a::BlockPVector) = (BlockedPRange(map(block->axes(block,1),blocks(a))),)
Base.length(a::BlockPVector) = sum(length,blocks(a))
Base.IndexStyle(::Type{<:BlockPVector}) = IndexLinear()
function Base.getindex(a::BlockPVector,gid::Int)
    scalar_indexing_action(a)
end
function Base.setindex(a::BlockPVector,v,gid::Int)
    scalar_indexing_action(a)
end
function Base.show(io::IO,k::MIME"text/plain",data::BlockPVector)
    T = eltype(data)
    n = length(data)
    ps = partition(axes(data,1))
    np = length(ps)
    nb = blocklength(data)
    map_main(ps) do _
        println(io,"$nb-blocked $n-element BlockPVector with eltype $T partitioned into $np parts.")
    end
end
function Base.show(io::IO,data::BlockPVector)
    print(io,"BlockPVector(…)")
end

function partition(a::BlockPVector)
    local_values(a)
end

function local_values(a::BlockPVector)
    r = local_block_ranges(axes(a,1))
    v = map(local_values,blocks(a)) |> array_of_tuples
    map(v,r) do myv, myr
        mortar(collect(myv),(myr,))
    end
end

function own_values(a::BlockPVector)
    r = own_block_ranges(axes(a,1))
    v = map(own_values,blocks(a)) |> array_of_tuples
    map(v,r) do myv, myr
        mortar(collect(myv),(myr,))
    end
end

function ghost_values(a::BlockPVector)
    r = ghost_block_ranges(axes(a,1))
    v = map(ghost_values,blocks(a)) |> array_of_tuples
    map(v,r) do myv, myr
        mortar(collect(myv),(myr,))
    end
end

function consistent!(a::BlockPVector)
    ts = map(consistent!,blocks(a))
    @fake_async begin
        foreach(wait,ts)
        a
    end
end

function assemble!(a::BlockPVector)
    ts = map(assemble!,blocks(a))
    @fake_async begin
        foreach(wait,ts)
        a
    end
end

function Base.similar(a::BlockPVector,::Type{T},inds::Tuple{<:BlockedPRange}) where T
    r = first(inds)
    bs = map((ai,ri)->similar(ai,T,ri),blocks(a),blocks(r))
    BlockPVector(bs)
end

function Base.similar(::Type{<:BlockPVector{A}},inds::Tuple{<:BlockedPRange}) where A
    V = eltype(A)
    r = first(inds)
    bs = map(ri->similar(V,ri),blocks(r))
    BlockPVector(bs)
end

function BlockPVector(::UndefInitializer,r::BlockedPRange)
    bs = map(ri->PVector(undef,ri),blocks(r))
    BlockPVector(bs)
end

function BlockPVector{A}(::UndefInitializer,r::BlockedPRange) where A
    similar(BlockPVector{A},(r,))
end

function Base.copy!(a::BlockPVector,b::BlockPVector)
    foreach(copy!,blocks(a),blocks(b))
    a
end

function Base.copyto!(a::BlockPVector,b::BlockPVector)
    foreach(copyto!,blocks(a),blocks(b))
    a
end

function Base.fill!(a::BlockPVector,v)
    foreach(ai->fill!(ai,v),blocks(a))
end


function Base.:(==)(a::BlockPVector,b::BlockPVector)
    all(map(==,blocks(a),blocks(b)))
end

function Base.any(f::Function,x::BlockPVector)
    any(xi->any(f,xi),blocks(x))
end

function Base.all(f::Function,x::BlockPVector)
    all(xi->all(f,xi),blocks(x))
end

Base.maximum(x::BlockPVector) = maximum(identity,x)
function Base.maximum(f::Function,x::BlockPVector)
    maximum(xi->maximum(f,xi),blocks(x))
end

Base.minimum(x::BlockPVector) = minimum(identity,x)
function Base.minimum(f::Function,x::BlockPVector)
    minimum(xi->minimum(f,xi),blocks(x))
end

function Base.collect(v::BlockPVector)
    reduce(vcat,map(collect,blocks(v)))
end

function Base.:*(a::Number,b::BlockPVector)
    bs = map(bi->a*bi,blocks(b))
    BlockPVector(bs)
end

function Base.:*(b::BlockPVector,a::Number)
    a*b
end

function Base.:/(b::BlockPVector,a::Number)
    (1/a)*b
end

for op in (:+,:-)
    @eval begin
        function Base.$op(a::BlockPVector)
            bs = map(ai->$op(ai),blocks(a))
            BlockPVector(bs)
        end
        function Base.$op(a::BlockPVector,b::BlockPVector)
            $op.(a,b)
        end
    end
end

function Base.reduce(op,a::BlockPVector;neutral=neutral_element(op,eltype(a)),kwargs...)
    rs = map(ai->reduce(op,ai;neutral,kwargs...),blocks(a))
    reduce(op,rs;kwargs...)
end

function Base.sum(a::BlockPVector)
  reduce(+,a,init=zero(eltype(a)))
end

function LinearAlgebra.dot(a::BlockPVector,b::BlockPVector)
    c = map(dot,blocks(a),blocks(b))
    sum(c)
end

function LinearAlgebra.rmul!(a::BlockPVector,v::Number)
    map(blocks(a)) do l
        rmul!(l,v)
    end
    a
end

function LinearAlgebra.norm(a::BlockPVector,p::Real=2)
    contibs = map(blocks(a)) do oid_to_value
        norm(oid_to_value,p)^p
    end
    reduce(+,contibs;init=zero(eltype(contibs)))^(1/p)
end

for M in Distances.metrics
    @eval begin
        function (d::$M)(a::BlockPVector,b::BlockPVector)
            s = distance_eval_body(d,a,b)
            Distances.eval_end(d,s)
        end
    end
end

function distance_eval_body(d,a::BlockPVector,b::BlockPVector)
    partials = map(blocks(a),blocks(b)) do ai,bi
        distance_eval_body(d,ai,bi)
    end
    s = reduce((i,j)->Distances.eval_reduce(d,i,j),
               partials,
               init=Distances.eval_start(d, a, b))
    s
end

struct BroadcastedBlockPVector{A}
    blocks::A
end
BlockArrays.blocks(a::BroadcastedBlockPVector) = a.blocks

function Base.broadcasted(f, args::Union{BlockPVector,BroadcastedBlockPVector}...)
    map( (bs...) -> Base.broadcasted(f,bs...) , map(blocks,args)...) |> BroadcastedBlockPVector
end

function Base.broadcasted( f, a::Number, b::Union{BlockPVector,BroadcastedBlockPVector})
    map( bi -> Base.broadcasted(f,a,bi), blocks(b)) |> BroadcastedBlockPVector
end

function Base.broadcasted( f, a::Union{BlockPVector,BroadcastedBlockPVector}, b::Number)
    map( ai -> Base.broadcasted(f,ai,b), blocks(a)) |> BroadcastedBlockPVector
end

function Base.broadcasted(f,
                          a::Union{BlockPVector,BroadcastedBlockPVector},
                          b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
    Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
    f,
    a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
    b::Union{BlockPVector,BroadcastedBlockPVector})
    Base.broadcasted(f,Base.materialize(a),b)
 end

function Base.materialize(b::BroadcastedBlockPVector)
    map(Base.materialize,blocks(b)) |> BlockPVector
end

function Base.materialize!(a::BlockPVector,b::BroadcastedBlockPVector)
    foreach(Base.materialize!,blocks(a),blocks(b))
    a
end

