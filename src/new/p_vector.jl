
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
    struct PVector{T,A,B}

`PVector` (partitioned vector) is a type representing a vector whose entries are
distributed (a.k.a. partitioned) over different parts for distributed-memory
parallel computations.

This type overloads numerous array-like operations with corresponding
parallel implementations.

# Properties

- `values::A`: A vector such that `values[i]` contains the vector of local values of the `i`-th part in the data distribution. The first type parameter `V` corresponds to `typeof(values[i])` i.e. the vector type used to store the local values.
- `rows::B`: An instance of `PRange` describing the distribution of the rows.

# Supertype hierarchy

    PVector{T,A,B} <: AbstractVector{T}
"""
struct PVector{V,A,B,T} <: AbstractVector{T}
    values::A
    rows::B
    @doc """
        PVector(values,rows)

    Create an instance of [`PVector`](@ref) from the underlying properties
    `values` and `rows`.
    """
    function PVector(values,rows)
        T = eltype(eltype(values))
        V = eltype(values)
        A = typeof(values)
        B = typeof(rows)
        new{V,A,B,T}(values,rows)
    end
end

function Base.show(io::IO,k::MIME"text/plain",data::PVector)
    println(io,"PVector with $(length(data)) items on $(length(data.values)) parts")
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

function PVector{V}(::UndefInitializer,rows::PRange) where V
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

function pvector(f,I,V,rows)
    values = map(f,I,V,rows.indices)
    rows = union_ghost(rows,I)
    PVector(values,rows)
end

function pvector(rows)
    pvector(default_local_values,rows)
end

function pvector(I,V,rows)
    rows = union_ghost(rows,I)
    pvector(default_local_values,I,V,rows)
end

function default_local_values(indices)
    Vector{Float64}(undef,get_n_local(indices))
end

function default_local_values(I,V,indices)
    values = Vector{Float64}(undef,get_n_local(indices))
    fill!(values,zero(eltype(values)))
    global_to_local = get_global_to_local(indices)
    for k in 1:length(I)
        gi = I[k]
        li = global_to_local[gi]
        values[li] = V[k]
    end
    values
end

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
prand(s,rows) = pvector(indices->rand(s,get_n_local(indices)),rows)
prand(rng,s,rows) = pvector(indices->rand(rng,s,get_n_local(indices)),rows)

"""
    prandn([rng,][s,]rows::PRange)

Create a [`Pvector`](@ref) object with normally distributed random values and the data partition in `rows`.
The optional arguments have the same meaning and default values as in [`randn`](@ref).
"""
prandn(rows) = pvector(indices->randn(get_n_local(indices)),rows)
prandn(s,rows) = pvector(indices->randn(s,get_n_local(indices)),rows)
prandn(rng,s,rows) = pvector(indices->randn(rng,s,get_n_local(indices)),rows)

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

function assembly_cache(values,assembly::SymbolicAssembly)
    V = eltype(values)
    assembly_cache_impl(V,values,assembly)
end

function assemble!(f,values,assembly::SymbolicAssembly,cache)
    V = eltype(values)
    assemble_impl!(f,V,values,assembly,cache)
end

function assembly_cache_impl(::Type{<:AbstractVector},values,assembly)
    T = eltype(eltype(values))
    assembly_buffers(T,assembly.local_indices)
end

function assemble_impl!(f,::Type{<:AbstractVector},values,assembly,buffers)
    neighbors = assembly.neighbors
    local_indices_snd = assembly.local_indices.snd
    local_indices_rcv = assembly.local_indices.rcv
    buffer_snd = buffers.snd
    buffer_rcv = buffers.rcv
    # Fill snd buffer
    map(values,local_indices_snd,buffer_snd) do values,local_indices_snd,buffer_snd
        for (p,lid) in enumerate(local_indices_snd.data)
            buffer_snd.data[p] = values[lid]
        end
    end
    t = exchange!(buffer_rcv,buffer_snd,neighbors)
    # Fill values from rcv buffer asynchronously
    @async begin
        wait(t)
        map(values,local_indices_rcv,buffer_rcv) do values,local_indices_rcv,buffer_rcv
            for (p,lid) in enumerate(local_indices_rcv.data)
                values[lid] = f(values[lid],buffer_rcv.data[p])
            end
        end
    end
end

function assembly_cache(a::PVector)
    assembly_cache(a.values,a.rows.assembly)
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
function assemble!(a::PVector,args...)
    assemble!(+,a,args...)
end

function assemble!(o,a::PVector,cache=assembly_cache(a))
    t = assemble!(o,a.values,a.rows.assembly,cache)
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
function consistent!(a::PVector,cache=assembly_cache(a))
    insert(a,b) = b
    t = assemble!(insert,a.values,reverse(a.rows.assembly),reverse(cache))
    @async begin
        wait(t)
        a
    end
end

