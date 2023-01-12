
function get_local_values end

function get_own_values end

function get_ghost_values end

function allocate_local_values(a,::Type{T},indices) where T
    similar(a,T,get_n_local(indices))
end

function allocate_local_values(::Type{V},indices) where V
    similar(V,get_n_local(indices))
end

function get_local_values(values,indices)
    values
end

function get_own_values(values,indices)
    own_to_local = get_own_to_local(indices)
    view(values,own_to_local)
end

function get_ghost_values(values,indices)
    ghost_to_local = get_ghost_to_local(indices)
    view(values,ghost_to_local)
end

"""
    struct OwnAndGhostValues{A,C,T}

Vector type that stores the local values of a [`PVector`](@ref) instance
using a vector of own values, a vector of ghost values, and a permutation.
This is not the default data layout of [`PVector`](@ref) (which is a plain vector),
but corresponds to the layout of distributed vectors in other packages, such as PETSc.
Use this type to avoid duplicating memory when passing data to these other packages.

# Properties

- `own_values::A`: The vector of own values.
- `ghost_values::A`: The vector of ghost values.
- `perm::C`: A permutation vector such that `vcat(own_values,ghost_values)[perm]` corresponds to the local values.

# Supertype hierarchy

    OwnAndGhostValues{A,C,T} <: AbstractVector{T}
"""
struct OwnAndGhostValues{A,C,T} <: AbstractVector{T}
    own_values::A
    ghost_values::A
    perm::C
    @doc """
        OwnAndGhostValues{A,C}(own_values,ghost_values,perm) where {A,C}
        OwnAndGhostValues{A}(own_values,ghost_values,perm) where {A}
        OwnAndGhostValues(own_values::A,ghost_values::A,perm) where A

    Build an instance of [`OwnAndGhostValues`](@ref) from the underlying fields.
    """
    function OwnAndGhostValues{A,C}(own_values,ghost_values,perm) where {A,C}
        T = eltype(A)
        new{A,C,T}(
          convert(A,own_values),
          convert(A,ghost_values),
          convert(C,perm))
    end
end
function OwnAndGhostValues{A}(own_values,ghost_values,perm) where A
    C = typeof(perm)
    OwnAndGhostValues{A,C}(own_values,ghost_values,perm)
end
function OwnAndGhostValues(own_values::A,ghost_values::A,perm) where A
    OwnAndGhostValues{A}(own_values,ghost_values,perm)
end
Base.IndexStyle(::Type{<:OwnAndGhostValues}) = IndexLinear()
Base.size(a::OwnAndGhostValues) = (length(a.own_values)+length(a.ghost_values),)
function Base.getindex(a::OwnAndGhostValues,local_id::Int)
    n_own = length(a.own_values)
    j = a.perm[local_id]
    if j > n_own
        a.ghost_values[j-n_own]
    else
        a.own_values[j]
    end
end
function Base.setindex!(a::OwnAndGhostValues,v,local_id::Int)
    n_own = length(a.own_values)
    j = a.perm[local_id]
    if j > n_own
        a.ghost_values[j-n_own] = v
    else
        a.own_values[j] = v
    end
    v
end

function get_own_values(values::OwnAndGhostValues,indices)
    values.own_values
end

function get_ghost_values(values::OwnAndGhostValues,indices)
    values.ghost_values
end

function allocate_local_values(values::OwnAndGhostValues,::Type{T},indices) where T
    n_own = get_n_own(indices)
    n_ghost = get_n_ghost(indices)
    own_values = similar(values.own_values,T,n_own)
    ghost_values = similar(values.ghost_values,T,n_ghost)
    perm = get_permutation(indices)
    OwnAndGhostValues(own_values,ghost_values,perm)
end

function allocate_local_values(::Type{<:OwnAndGhostValues{A}},indices) where {A}
    n_own = get_n_own(indices)
    n_ghost = get_n_ghost(indices)
    own_values = similar(A,n_own)
    ghost_values = similar(A,n_ghost)
    perm = get_permutation(indices)
    OwnAndGhostValues{A}(own_values,ghost_values,perm)
end

"""
    struct PVector{V,A,B,T}

`PVector` (partitioned vector) is a type representing a vector whose entries are
distributed (a.k.a. partitioned) over different parts for distributed-memory
parallel computations.

This type overloads numerous array-like operations with corresponding
parallel implementations.

# Properties

- `values::A`: A vector such that `values[i]` contains the vector of local values of the `i`-th part in the data distribution. The first type parameter `V` corresponds to `typeof(values[i])` i.e. the vector type used to store the local values.
- `rows::B`: An instance of `PRange` describing the distributed data layout.

# Supertype hierarchy

    PVector{V,A,B,T} <: AbstractVector{T}
"""
struct PVector{V,A,B,T} <: AbstractVector{T}
    values::A
    rows::B
    @doc """
        PVector(values,rows::PRange)

    Create an instance of [`PVector`](@ref) from the underlying fields
    `values` and `rows`.
    """
    function PVector(values,rows::PRange)
        V = eltype(values)
        T = eltype(V)
        A = typeof(values)
        B = typeof(rows)
        new{V,A,B,T}(values,rows)
    end
end

function Base.show(io::IO,k::MIME"text/plain",data::PVector)
    println(io,typeof(data)," on $(length(data.values)) parts")
end


"""
    get_local_values(a::PVector)

Get a vector of vectors containing the local values
in each part of `a`.
"""
function get_local_values(a::PVector)
    #map(get_local_values,a.values,a.rows.indices)
    a.values
end

"""
    get_own_values(a::PVector)

Get a vector of vectors containing the own values
in each part of `a`.
"""
function get_own_values(a::PVector)
    map(get_own_values,a.values,a.rows.indices)
end

"""
    get_ghost_values(a::PVector)

Get a vector of vectors containing the ghost values
in each part of `a`.
"""
function get_ghost_values(a::PVector)
    map(get_ghost_values,a.values,a.rows.indices)
end

Base.size(a::PVector) = (length(a.rows),)
Base.axes(a::PVector) = (a.rows,)
Base.IndexStyle(::Type{<:PVector}) = IndexLinear()
function Base.getindex(a::PVector,gid::Int)
    scalar_indexing_action(a)
end
function Base.setindex(a::PVector,v,gid::Int)
    scalar_indexing_action(a)
end

"""
    assemble!([op,] a::PVector) -> Task

Transfer the ghost values to its owner part
and insert them according with the insertion operation `op` (`+` by default).
It returns a task that produces `a` with updated values. After the transfer,
the source ghost values are set to zero.

# Examples

    julia> using PartitionedArrays
    
    julia> rank = LinearIndices((2,));
    
    julia> rows = PRange(ConstantBlockSize(),rank,2,6,true)
    1:1:6
    
    julia> a = pones(rows);
    
    julia> get_local_values(a)
    2-element Vector{Vector{Float64}}:
     [1.0, 1.0, 1.0, 1.0]
     [1.0, 1.0, 1.0, 1.0]
    
    julia> assemble!(a) |> wait
    
    julia> get_local_values(a)
    2-element Vector{Vector{Float64}}:
     [1.0, 1.0, 2.0, 0.0]
     [0.0, 2.0, 1.0, 1.0]
"""
function assemble!(a::PVector)
    assemble!(+,a)
end

function assemble!(o,a::PVector)
    t = assemble!(o,a.values,a.rows.assembler)
    @async begin
        wait(t)
        map(get_ghost_values(a)) do a
            fill!(a,zero(eltype(a)))
        end
        a
    end
end

"""
    consistent!(a::PVector) -> Task

Make the local values of `a` globally consistent. I.e., the
ghost values are updated with the corresponding own value in the
part that owns the associated global global id.

# Examples

    julia> using PartitionedArrays
    
    julia> rank = LinearIndices((2,));
    
    julia> rows = PRange(ConstantBlockSize(),rank,2,6,true)
    1:1:6
    
    julia> a = pvector(inds->fill(get_owner(inds),length(inds)),rows);
    
    julia> get_local_values(a)
    2-element Vector{Vector{Int32}}:
     [1, 1, 1, 1]
     [2, 2, 2, 2]
    
    julia> consistent!(a) |> wait
    
    julia> get_local_values(a)
    2-element Vector{Vector{Int32}}:
     [1, 1, 1, 2]
     [1, 2, 2, 2]
"""
function consistent!(a::PVector)
    insert(a,b) = b
    t = assemble!(insert,a.values,reverse(a.rows.assembler))
    @async begin
        wait(t)
        a
    end
end

function Base.similar(a::PVector,::Type{T},inds::Tuple{<:PRange}) where T
    rows = inds[1]
    values = map(a.values,rows.indices) do values, indices
        allocate_local_values(values,T,indices)
    end
    PVector(values,rows)
end

function Base.similar(::Type{<:PVector{V}},inds::Tuple{<:PRange}) where V
  rows = inds[1]
  values = map(rows.indices) do indices
      allocate_local_values(V,indices)
  end
  PVector(values,rows)
end

function Base.copy!(a::PVector,b::PVector)
    @assert length(a) == length(b)
    copyto!(a,b)
end

function Base.copyto!(a::PVector,b::PVector)
    if a.rows.indices === b.rows.indices
        map(copy!,a.values,b.values)
    elseif matching_own_indices(a,b)
        map(copy!,get_own_values(a),get_own_values(b))
    else
        error("Trying to copy a PVector into another one with a different data layout. This case is not implemented yet. It would require communications.")
    end
    a
end

function Base.fill!(a::PVector,v)
    map(a.values) do values
        fill!(values,v)
    end
    a
end

"""
    pvector(f,rows::PRange)

Create a [`PVector`](@ref) instance defined over the partitioned range `rows`,
by initializing the local values with function `f`. The signature of `f` is
`f(indices)`, where `indices` are the local indices in the corresponding part.

Equivalent to 

    values = map(f,rows.indices)
    PVector(values,rows)

# Examples

    julia> using PartitionedArrays
    
    julia> rank = LinearIndices((2,));
    
    julia> rows = PRange(ConstantBlockSize(),rank,2,6,true)
    1:1:6

    julia> a = pvector(i->rand(1:3,length(i)),rows);
    
    julia> get_local_values(a)
    2-element Vector{Vector{Int64}}:
     [2, 1, 1, 3]
     [3, 3, 2, 3]
"""
function pvector(f,rows)
    values = map(f,rows.indices)
    PVector(values,rows)
end

"""
    PVector{V}(::UndefInitializer,rows::PRange) where V
    PVector(::UndefInitializer,rows::PRange)

Allocate an uninitialized instance of [`Pvector`](@ref) with the partition in `rows`
whose local values are a vector of type `V`.
The default value for `V` is `Vector{Float64}`.

# Examples

    julia> using PartitionedArrays
    
    julia> rank = LinearIndices((2,));
    
    julia> rows = PRange(ConstantBlockSize(),rank,2,6,true)
    1:1:6

    julia> a = PVector(undef,rows);
    
    julia> get_local_values(a)
    2-element Vector{Vector{Float64}}:
     [6.90182496006166e-310, 4.925e-320, NaN, 0.0]
     [6.90182496007115e-310, 2.9614e-320, NaN, 0.0]
    
    julia> a = PVector{OwnAndGhostValues{Vector{Int32}}}(undef,rows);
    
    julia> get_local_values(a)
    2-element Vector{OwnAndGhostValues{Vector{Int32}, Vector{Int32}, Int32}}:
     [-1831434400, 32524, -1832234496, 66]
     [-1808351792, -1803592080, 32524, -1803592016]
"""
PVector{V}(::UndefInitializer,rows::PRange) where V = pvector(indices->allocate_local_values(V,indices),rows)
PVector(::UndefInitializer,rows::PRange) = PVector{Vector{Float64}}(undef,rows)


"""
    pfill(v,rows::PRange)

Create a [`Pvector`](@ref) object with the data partition in `rows`
with all entries equal to `v`.
"""
pfill(v,rows) = pvector(indices->fill(v,get_n_local(indices)),rows)

"""
    pzeros(rows::PRange)
    pzeros(::Type{T},rows::PRange) where T

Equivalent to

    pfill(zero(T),rows)
"""
pzeros(rows) = pzeros(Float64,rows)
pzeros(::Type{T},rows) where T = pvector(indices->zeros(T,get_n_local(indices)),rows)

"""
    pones(rows::PRange)
    pones(::Type{T},rows::PRange) where T

Equivalent to

    pfill(one(T),rows)
"""
pones(rows) = pones(Float64,rows)
pones(::Type{T},rows) where T = pvector(indices->ones(T,get_n_local(indices)),rows)

"""
    prand([rng,][s,]rows::PRange)

Create a [`Pvector`](@ref) object with uniform random values and the data partition in `rows`.
The optional arguments have the same meaning and default values as in [`rand`](@ref).
"""
prand(rows) = pvector(indices->rand(get_n_local(indices)),rows)
prand(s,rows) = pvector(indices->rand(S,get_n_local(indices)),rows)
prand(rng,s,rows) = pvector(indices->rand(rng,S,get_n_local(indices)),rows)

"""
    prandn([rng,][s,]rows::PRange)

Create a [`Pvector`](@ref) object with normally distributed random values and the data partition in `rows`.
The optional arguments have the same meaning and default values as in [`randn`](@ref).
"""
prandn(rows) = pvector(indices->randn(get_n_local(indices)),rows)
prandn(s,rows) = pvector(indices->randn(S,get_n_local(indices)),rows)
prandn(rng,s,rows) = pvector(indices->randn(rng,S,get_n_local(indices)),rows)

function Base.:(==)(a::PVector,b::PVector)
    length(a) == length(b) &&
    length(a.values) == length(b.values) &&
    reduce(&,map(==,get_own_values(a),get_own_values(b)),init=true)
end

function Base.any(f::Function,x::PVector)
    partials = map(get_own_values(x)) do o
        any(f,o)
    end
    reduce(|,partials,init=false)
end

function Base.all(f::Function,x::PVector)
    partials = map(get_own_values(x)) do o
        all(f,o)
    end
    reduce(&,partials,init=true)
end

Base.maximum(x::PVector) = maximum(identity,x)
function Base.maximum(f::Function,x::PVector)
    partials = map(get_own_values(x)) do o
        maximum(f,o,init=typemin(eltype(x)))
    end
    reduce(max,partials,init=typemin(eltype(x)))
end

Base.minimum(x::PVector) = minimum(identity,x)
function Base.minimum(f::Function,x::PVector)
    partials = map(get_own_values(x)) do o
        minimum(f,o,init=typemax(eltype(x)))
    end
    reduce(min,partials,init=typemax(eltype(x)))
end

function Base.:*(a::Number,b::PVector)
    values = map(b.values) do values
        a*values
    end
    PVector(values,b.rows)
end

function Base.:*(b::PVector,a::Number)
    a*b
end

function Base.:/(b::PVector,a::Number)
    (1/a)*b
end

for op in (:+,:-)
    @eval begin
        function Base.$op(a::PVector)
            values = map($op,a.values)
            PVector(values,a.rows)
        end
        function Base.$op(a::PVector,b::PVector)
            $op.(a,b)
        end
    end
end

function neutral_element end
neutral_element(::typeof(+),::Type{T}) where T = zero(T)
neutral_element(::typeof(&),::Type) = true
neutral_element(::typeof(|),::Type) = false
neutral_element(::typeof(min),::Type{T}) where T = typemax(T)
neutral_element(::typeof(max),::Type{T}) where T = typemin(T)

function Base.reduce(op,a::PVector;neutral=neutral_element(op,eltype(a)),kwargs...)
    b = map(get_own_values(a)) do a
        reduce(op,a,init=neutral)
    end
    reduce(op,b;kwargs...)
end

function Base.sum(a::PVector)
  reduce(+,a,init=zero(eltype(a)))
end

function LinearAlgebra.dot(a::PVector,b::PVector)
    c = map(dot,get_own_values(a),get_own_values(b))
    sum(c)
end

function LinearAlgebra.rmul!(a::PVector,v::Number)
    map(a.values) do l
        rmul!(l,v)
    end
    a
end

function LinearAlgebra.norm(a::PVector,p::Real=2)
    contibs = map(get_own_values(a)) do oid_to_value
        norm(oid_to_value,p)^p
    end
    reduce(+,contibs;init=zero(eltype(contibs)))^(1/p)
end

struct PBroadcasted{A,B,C}
    own_values::A
    ghost_values::B
    rows::C
end
get_own_values(a::PBroadcasted) = a.own_values
get_ghost_values(a::PBroadcasted) = a.ghost_values

function Base.broadcasted(f, args::Union{PVector,PBroadcasted}...)
    a1 = first(args)
    @boundscheck @assert all(ai->matching_own_indices(ai.rows,a1.rows),args)
    owned_values_in = map(get_own_values,args)
    own_values = map((largs...)->Base.broadcasted(f,largs...),owned_values_in...)
    if all(ai->ai.rows===a1.rows,args) && !any(ai->get_ghost_values(ai)===nothing,args)
        ghost_values_in = map(get_ghost_values,args)
        ghost_values = map((largs...)->Base.broadcasted(f,largs...),ghost_values_in...)
    else
        ghost_values = nothing
    end
    PBroadcasted(own_values,ghost_values,a1.rows)
end

function Base.broadcasted( f, a::Number, b::Union{PVector,PBroadcasted})
    own_values = map(b->Base.broadcasted(f,a,b),get_own_values(b))
    if b.ghost_values !== nothing
        ghost_values = map(b->Base.broadcasted(f,a,b),get_ghost_values(b))
    else
        ghost_values = nothing
    end
    PBroadcasted(own_values,ghost_values,b.rows)
end

function Base.broadcasted( f, a::Union{PVector,PBroadcasted}, b::Number)
    own_values = map(a->Base.broadcasted(f,a,b),get_own_values(a))
    if a.ghost_values !== nothing
        ghost_values = map(a->Base.broadcasted(f,a,b),get_ghost_values(a))
    else
        ghost_values = nothing
    end
    PBroadcasted(own_values,ghost_values,a.rows)
end

function Base.materialize(b::PBroadcasted)
    T = eltype(eltype(b.own_values))
    a = PVector{Vector{T}}(undef,b.rows)
    Base.materialize!(a,b)
    a
end

function Base.materialize!(a::PVector,b::PBroadcasted)
    map(Base.materialize!,get_own_values(a),get_own_values(b))
    if b.ghost_values !== nothing && a.rows === b.rows
        map(Base.materialize!,get_ghost_values(a),get_ghost_values(b))
    end
    a
end

for M in Distances.metrics
    @eval begin
        function (d::$M)(a::PVector,b::PVector)
            if Distances.parameters(d) !== nothing
                error("Only distances without parameters are implemented at this moment")
            end
            partials = map(get_own_values(a),get_own_values(b)) do a,b
                @boundscheck if length(a) != length(b)
                    throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
                end
                if length(a) == 0
                    return zero(Distances.result_type(d, a, b))
                end
                @inbounds begin
                    s = Distances.eval_start(d, a, b)
                    if (IndexStyle(a, b) === IndexLinear() && eachindex(a) == eachindex(b)) || axes(a) == axes(b)
                        @simd for I in eachindex(a, b)
                            ai = a[I]
                            bi = b[I]
                            s = Distances.eval_reduce(d, s, Distances.eval_op(d, ai, bi))
                        end
                    else
                        for (ai, bi) in zip(a, b)
                            s = Distances.eval_reduce(d, s, Distances.eval_op(d, ai, bi))
                        end
                    end
                    return s
                end
            end
            s = reduce((i,j)->Distances.eval_reduce(d,i,j),
                       partials,
                       init=Distances.eval_start(d, a, b))
            Distances.eval_end(d,s)
        end
    end
end

"""
    pvector_coo!([[op,] T,] I, V, rows; [owners,] [init,]) -> Task

Generate a [`PVector`](@ref) object with the data partition in `rows` from the coordinate
vectors `I` and `V`. At each part `part`, the vectors `I[part]` and `V[part]` contain 
global row ids and their corresponding values. The type of the generated local
values is `T` (`Vector{Float64}` by default) and it is initialized 
with value `init`, which defaults to `neutral_element(op,eltype(T))`. Values
in `V` are inserted using operation `op` (`+` by default). Repeated row ids
are allowed. In this case, they will be combined with operation `op`.
`owners` can be provided
to skip the discovery of the part owner for the global ids in `rows`. It defaults to
`find_owner(rows,I)`. The result is a task that produces the [`PVector`](@ref) object.
This function modifies `I` and `V`.

# Examples


    julia> using PartitionedArrays

    julia> rank = LinearIndices((2,));

    julia> rows = PRange(ConstantBlockSize(),rank,2,10)
    1:1:10

    julia> I = [[3,6,2,3],[1,7,9,7,10,1]]
    2-element Vector{Vector{Int64}}:
     [3, 6, 2, 3]
     [1, 7, 9, 7, 10, 1]
    
    julia> V = 10*I
    2-element Vector{Vector{Int64}}:
     [30, 60, 20, 30]
     [10, 70, 90, 70, 100, 10]
    
    julia> t = pvector_coo!(Vector{Float32},I,V,rows)
    Task (done) @0x00007f0c92f933d0
    
    julia> a = fetch(t);
    
    julia> get_local_values(a)
    2-element Vector{Vector{Float32}}:
     [0.0, 20.0, 30.0, 0.0, 0.0, 0.0]
     [0.0, 70.0, 0.0, 90.0, 100.0, 0.0, 0.0]

"""
function pvector_coo!(I,V,rows;kwargs...)
    pvector_coo!(+,Vector{Float64},I,V,rows;kwargs...)
end

function pvector_coo!(::Type{T},I,V,rows;kwargs...) where T
    pvector_coo!(+,T,I,V,rows;kwargs...)
end

function pvector_coo!(op,::Type{T},I,V,rows;owners=find_owner(rows,I),init=neutral_element(op,eltype(T))) where T
    rows = union_ghost(rows,I,owners)
    t = assemble_coo!(I,V,rows)
    a = PVector{T}(undef,rows)
    @async begin
        I,V = fetch(t)
        insert_coo!(op,a,I,V;init)
    end
end

function insert_coo!(op,values,I,V,indices;init)
    fill!(values,init)
    global_to_local = get_global_to_local(indices)
    for k in 1:length(I)
        gi = I[k]
        li = global_to_local[gi]
        values[li] = V[k]
    end
    values
end

function insert_coo!(op,a::PVector,I,V;init)
    map(a.values,I,V,a.rows.indices) do values,I,V,indices
        insert_coo!(op,values,I,V,indices;init)
    end
    a
end

"""
    assemble_coo!(I [,J], V, rows) -> Task

Assemble the coordinate vectors `I`, `J`, `V`. Entries corresponding to
ghost values are sent to the owner part and appended at the end 
of the given entries. Note: global ids `I` should be in the local
values of `rows`. You can achieve this, e.g., with `rows=union_ghost(rows,I)`.
The source ghost values are set to zero.

# Examples

    julia> using PartitionedArrays
    
    julia> rank = LinearIndices((2,));
    
    julia> rows = PRange(ConstantBlockSize(),rank,2,10)
    1:1:10
    
    julia> I = [[3,6,2,3],[1,7,9,7,10,1]]
    2-element Vector{Vector{Int64}}:
     [3, 6, 2, 3]
     [1, 7, 9, 7, 10, 1]
    
    julia> V = 10*I
    2-element Vector{Vector{Int64}}:
     [30, 60, 20, 30]
     [10, 70, 90, 70, 100, 10]
    
    julia> rows = union_ghost(rows,I)
    1:1:10
    
    julia> assemble_coo!(I,V,rows) |> wait
    
    julia> I
    2-element Vector{Vector{Int64}}:
     [3, 6, 2, 3, 1, 1]
     [1, 7, 9, 7, 10, 1, 6]
    
    julia> V
    2-element Vector{Vector{Int64}}:
     [30, 0, 20, 30, 10, 10]
     [0, 70, 90, 70, 100, 0, 60]

"""
function assemble_coo!(I,V,rows)
    t = assemble_coo!(I,V,V,rows)
    @async begin
        I,J,V = fetch(t)
        I,V
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
    function setup_rcv(part,row_lids,gi_rcv,gj_rcv,v_rcv,coo_values)
        k_gi, k_gj, k_v = coo_values
        ptrs = gi_rcv.ptrs
        current_n = length(k_gi)
        new_n = current_n + length(gi_rcv.data)
        resize!(k_gi,new_n)
        resize!(k_gj,new_n)
        resize!(k_v,new_n)
        for p in 1:length(gi_rcv.data)
            gi = gi_rcv.data[p]
            gj = gj_rcv.data[p]
            v = v_rcv.data[p]
            k_gi[current_n+p] = gi
            k_gj[current_n+p] = gj
            k_v[current_n+p] = v
        end
    end
    part = linear_indices(rows.indices)
    parts_snd = rows.assembler.parts_snd
    parts_rcv = rows.assembler.parts_rcv
    graph = ExchangeGraph(parts_snd,parts_rcv)
    coo_values = map(tuple,I,J,V)
    aux1 = map(setup_snd,part,parts_snd,rows.indices,coo_values)
    gi_snd, gj_snd, v_snd = aux1 |> unpack
    t1 = exchange(gi_snd,graph)
    t3 = exchange(v_snd,graph)
    if J !== V
        t2 = exchange(gj_snd,graph)
    else
        t2 = t3
    end
    @async begin
        gi_rcv = fetch(t1)
        gj_rcv = fetch(t2)
        v_rcv = fetch(t3) 
        map(setup_rcv,part,rows.indices,gi_rcv,gj_rcv,v_rcv,coo_values)
        I,J,V
    end
end


