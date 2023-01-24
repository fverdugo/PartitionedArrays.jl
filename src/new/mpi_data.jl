
"""
    mpi_data(a,comm::MPI.Comm;duplicate_comm=true)

Create an `MPIData{T,N}` instance (`T=eltype(a)`, `N=ndims(a)`) by distributing
the items in the collection `a` over the ranks of the given MPI
communicator `comm`. Each rank receives
exactly one item, thus `length(a)`  and the communicator size need to match.
For arrays that can store more than one item per
rank see [`PVector`](@ref) or [`PSparseMatrix`](@ref).
If `duplicate_comm=false` the result will take ownership of the given communicator.
Otherwise, a copy will be done with `MPI.Comm_dup(comm)`.

!!! note

    Initialize MPI with `MPI.Init()` before using this function,
    or use [`with_mpi_data`](@ref).

# Examples

    julia> using PartitionedArrays
    
    julia> using MPI
    
    julia> MPI.Init()
    MPI.ThreadLevel(2)
    
    julia> mpi_data([10],MPI.COMM_WORLD)
    1-element MPIData{Int64, 1}:
    [1] = 10

    julia> with_mpi_data(MPI.COMM_WORLD) do distribute
             distribute([10])
           end
    1-element MPIData{Int64, 1}:
    [1] = 10
"""
function mpi_data(a,comm::MPI.Comm;duplicate_comm=true)
    msg = "Number of MPI ranks needs to be the same as items in the given array"
    @assert length(a) == MPI.Comm_size(comm) msg
    if duplicate_comm
        comm = MPI.Comm_dup(comm)
    end
    i = MPI.Comm_rank(comm)+1
    MPIData(Ref(a[i]),comm,size(a))
end

"""
    with_mpi_data(f,comm=MPI.COMM_WORLD;kwargs...)

Initialize MPI if need, call `f(a->mpi_data(a,comm;kwargs...))`
and abort MPI if there was an error.

This is the safest way of running the function `f` using MPI.
"""
function with_mpi_data(f,comm=MPI.COMM_WORLD;kwargs...)
    if !MPI.Initialized()
        MPI.Init()
    end
    distribute = a -> mpi_data(a,comm;kwargs...)
    if MPI.Comm_size(comm) == 1
        f(distribute)
    else
        try
            f(distribute)
        catch e
            @error "" exception=(e, catch_backtrace())
            if MPI.Initialized() && !MPI.Finalized()
                MPI.Abort(MPI.COMM_WORLD,1)
            end
        end
    end
    # We are NOT invoking MPI.Finalize() here because we rely on
    # MPI.jl, which registers MPI.Finalize() in atexit()
end

"""
    MPIData{T,N}

Represent an array of element type `T` and number of dimensions `N`, where
each item in the array is stored in a separate MPI process. I.e., each MPI
rank stores only one item. For arrays that can store more than one item per
rank see [`PVector`](@ref) or [`PSparseMatrix`](@ref). This struct implements
the Julia array interface.
However, using [`setindex!`](@ref) and [`getindex!`](@ref) is strongly discouraged
for performance reasons (communication cost).

# Properties

The fields of this struct (and the inner constructors) are private. To
generate an instance of `MPIData` use function [`mpi_data`](@ref).

# Supertype hierarchy

    MPIData{T,N} <: AbstractArray{T,N}
"""
struct MPIData{T,N} <: AbstractArray{T,N}
    item::Base.RefValue{T}
    comm::MPI.Comm
    size::NTuple{N,Int}
    function MPIData{T,N}(item::Base.RefValue{T},comm::MPI.Comm,size::Dims{N}) where {T,N}
        @assert MPI.Comm_size(comm) == prod(size)
        new{T,N}(item,comm,size)
    end
    function MPIData{T,N}(item::Base.RefValue,comm::MPI.Comm,size::Dims{N}) where {T,N}
        @assert MPI.Comm_size(comm) == prod(size)
        new{T,N}(Ref{T}(item[]),comm,size)
    end
    function MPIData(item::Base.RefValue,comm::MPI.Comm,size::Dims)
        T = eltype(item)
        N = length(size)
        @assert MPI.Comm_size(comm) == prod(size)
        new{T,N}(item,comm,size)
    end
end

Base.size(a::MPIData) = a.size
Base.IndexStyle(::Type{<:MPIData}) = IndexLinear()
function Base.getindex(a::MPIData,i::Int)
    scalar_indexing_action(a)
    if i == MPI.Comm_rank(a.comm)+1
        a.item[]
    else
        error("Indexing of MPIData at arbitrary indices not implemented yet.")
    end
end
function Base.setindex!(a::MPIData,v,i::Int)
    scalar_indexing_action(a)
    if i == MPI.Comm_rank(a.comm)+1
        a.item[]=v
    else
        error("Indexing of MPIData at arbitrary indices not implemented yet.")
    end
    v
end
linear_indices(a::MPIData) = mpi_data(LinearIndices(a),a.comm,duplicate_comm=false)
cartesian_indices(a::MPIData) = mpi_data(CartesianIndices(a),a.comm,duplicate_comm=false)
function Base.show(io::IO,k::MIME"text/plain",data::MPIData)
    header = ""
    if ndims(data) == 1
        header *= "$(length(data))-element"
    else
        for n in 1:ndims(data)
            if n!=1
                header *= "×"
            end
            header *= "$(size(data,n))"
        end
    end
    header *= " $(typeof(data)):"
    if MPI.Comm_rank(data.comm) == 0
        println(io,header)
    end
    MPI.Barrier(data.comm)
    linds = LinearIndices(data)
    for i in CartesianIndices(data)
        index = "["
        for (j,t) in enumerate(Tuple(i))
            if j != 1
                index *=","
            end
            index *= "$t"
        end
        index *= "]"
        if MPI.Comm_rank(data.comm) == linds[i]-1
            println(io,"$index = $(data.item[])")
        end
        MPI.Barrier(data.comm)
    end
end

getany(a::MPIData) = a.item[]
i_am_main(a::MPIData) = get_part_id(a.comm) == MAIN

function Base.similar(a::MPIData,::Type{T},dims::Dims) where T
    MPIData(Ref{T}(),a.comm,dims)
end

function Base.copyto!(b::MPIData,a::MPIData)
    b.item[] = a.item[]
    b
end

function Base.map(f,args::MPIData...)
    a = first(args)
    @assert all(i->size(a)==size(i),args)
    item = Ref(f(map(i->i.item[],args)...))
    MPIData(item,a.comm,a.size)
end

function Base.map!(f,r::MPIData,args::MPIData...)
    a = first(args)
    @assert all(i->size(a)==size(i),args)
    r.item[] = f(map(i->i.item[],args)...)
    r
end

function gather_impl!(
    rcv::MPIData, snd::MPIData,
    destination, ::Type{T}) where T
    @assert rcv.comm === snd.comm
    comm = snd.comm
    if isa(destination,Integer)
        root = destination-1
        if MPI.Comm_rank(comm) == root
            @assert length(rcv.item[]) == MPI.Comm_size(comm)
            rcv.item[][destination] = snd.item[]
            rcv_buffer = MPI.UBuffer(rcv.item[],1)
            MPI.Gather!(MPI.IN_PLACE,rcv_buffer,root,comm)
        else
            MPI.Gather!(snd.item,nothing,root,comm)
        end
    else
        @assert destination === :all
        @assert length(rcv.item[]) == MPI.Comm_size(comm)
        rcv_buffer = MPI.UBuffer(rcv.item[],1)
        MPI.Allgather!(snd.item,rcv_buffer,snd.comm)
    end
    rcv
end

function gather_impl!(
    rcv::MPIData, snd::MPIData,
    destination, ::Type{T}) where T <: AbstractVector
    @assert rcv.comm === snd.comm
    @assert isa(rcv.item[],JaggedArray)
    comm = snd.comm
    if isa(destination,Integer)
        root = destination-1
        if MPI.Comm_rank(comm) == root
            @assert length(rcv.item[]) == MPI.Comm_size(comm)
            rcv.item[][destination] = snd.item[]
            counts = ptrs_to_counts(rcv.item[].ptrs)
            rcv_buffer = MPI.VBuffer(rcv.item[].data,counts)
            MPI.Gatherv!(MPI.IN_PLACE,rcv_buffer,root,comm)
        else
            MPI.Gatherv!(snd.item[],nothing,root,comm)
        end
    else
        @assert destination === :all
        @assert length(rcv.item[]) == MPI.Comm_size(comm)
        counts = ptrs_to_counts(rcv.item[].ptrs)
        rcv_buffer = MPI.VBuffer(rcv.item[].data,counts)
        MPI.Allgatherv!(snd.item[],rcv_buffer,comm)
    end
    rcv
end

function scatter_impl!(
    rcv::MPIData,snd::MPIData,
    source,::Type{T}) where T
    @assert source !== :all "All to all not implemented"
    @assert rcv.comm === snd.comm
    @assert eltype(snd.item[]) == typeof(rcv.item[])
    comm = snd.comm
    root = source - 1
    if MPI.Comm_rank(comm) == root
        snd_buffer = MPI.UBuffer(snd.item[],1)
        rcv.item[] = snd.item[][source]
        MPI.Scatter!(snd_buffer,MPI.IN_PLACE,root,comm)
    else
        MPI.Scatter!(nothing,rcv.item,root,comm)
    end
    rcv
end

function scatter_impl!(
    rcv::MPIData,snd::MPIData,
    source,::Type{T}) where T<:AbstractVector
    @assert source !== :all "All to all not implemented"
    @assert rcv.comm === snd.comm
    @assert isa(snd.item[],JaggedArray)
    @assert eltype(eltype(snd.item[])) == eltype(rcv.item[])
    comm = snd.comm
    root = source - 1
    if MPI.Comm_rank(comm) == root
        counts = ptrs_to_counts(snd.item[].ptrs)
        snd_buffer = MPI.VBuffer(snd.item[].data,counts)
        rcv.item[] .= snd.item[][source]
        MPI.Scatterv!(snd_buffer,MPI.IN_PLACE,root,comm)
    else
        # This void Vbuffer is required to circumvent a deadlock
        # that we found with OpenMPI 4.1.X on Gadi. In particular, the
        # deadlock arises whenever buf is set to nothing
        S = eltype(eltype(snd.item[]))
        snd_buffer = MPI.VBuffer(S[],Int[])
        MPI.Scatterv!(snd_buffer,rcv.item[],root,comm)
    end
    rcv
end

function emit_impl!(
    rcv::MPIData,snd::MPIData,
    source,::Type{T}) where T
    @assert rcv.comm === snd.comm
    comm = snd.comm
    root = source - 1
    if MPI.Comm_rank(comm) == root
        rcv.item[] = snd.item[]
    end
    MPI.Bcast!(rcv.item,root,comm)
end

function emit_impl!(
    rcv::MPIData,snd::MPIData,
    source,::Type{T}) where T<:AbstractVector
    @assert rcv.comm === snd.comm
    comm = snd.comm
    root = source - 1
    if MPI.Comm_rank(comm) == root
        rcv.item[] = snd.item[]
    end
    MPI.Bcast!(rcv.item[],root,comm)
end

function scan!(op,b::MPIData,a::MPIData;init,type)
    @assert b.comm === a.comm
    @assert type in (:inclusive,:exclusive)
    T = eltype(a)
    @assert eltype(b) == T
    comm = a.comm
    opr = MPI.Op(op,T)
    if type === :inclusive
        if a.item !== b.item
            MPI.Scan!(a.item,b.item,opr,comm)
        else
            MPI.Scan!(b.item,opr,comm)
        end
        b.item[] = op(b.item[],init)
    else
        if a.item !== b.item
            MPI.Exscan!(a.item,b.item,opr,comm)
        else
            MPI.Exscan!(b.item,opr,comm)
        end
        if MPI.Comm_rank(comm) == 0
            b.item[] = init
        else
            b.item[] = op(b.item[],init)
        end
    end
    b
end

function reduction!(op,b::MPIData,a::MPIData;destination=1,init=nothing)
    @assert b.comm === a.comm
    T = eltype(a)
    @assert eltype(b) == T
    comm = a.comm
    opr = MPI.Op(op,T)
    if destination !== :all
        root = destination-1
        if a.item !== b.item
            MPI.Reduce!(a.item,b.item,opr,root,comm)
        else
            MPI.Reduce!(b.item,opr,root,comm)
        end
        if MPI.Comm_rank(comm) == root
            if init !== nothing
                b.item[] = op(b.item[],init)
            end
        end
    else
        if a.item !== b.item
            MPI.Allreduce!(a.item,b.item,opr,comm)
        else
            MPI.Allreduce!(b.item,opr,comm)
        end
        if init !== nothing
            b.item[] = op(b.item[],init)
        end
    end
    b
end

function Base.reduce(op,a::MPIData;kwargs...)
   r = reduction(op,a;destination=:all,kwargs...)
   r.item[]
end
Base.sum(a::MPIData) = reduce(+,a)
function Base.collect(a::MPIData)
    T = eltype(a)
    N = ndims(a)
    b = Array{T,N}(undef,size(a))
    c = MPIData(Ref(b),a.comm,size(a))
    gather!(c,a,destination=:all)
    c.item[]
end

function exchange_impl!(
    rcv::MPIData,
    snd::MPIData,
    graph::ExchangeGraph{<:MPIData},
    ::Type{T}) where T

    @assert size(rcv) == size(snd)
    @assert graph.rcv.comm === graph.rcv.comm
    @assert graph.rcv.comm === graph.snd.comm
    comm = graph.rcv.comm
    req_all = MPI.Request[]
    for (i,id_rcv) in enumerate(graph.rcv.item[])
        rank_rcv = id_rcv-1
        buff_rcv = view(rcv.item[],i:i)
        tag_rcv = rank_rcv
        reqr = MPI.Irecv!(buff_rcv,rank_rcv,tag_rcv,comm)
        push!(req_all,reqr)
    end
    for (i,id_snd) in enumerate(graph.snd.item[])
        rank_snd = id_snd-1
        buff_snd = view(snd.item[],i:i)
        tag_snd = MPI.Comm_rank(comm)
        reqs = MPI.Isend(buff_snd,rank_snd,tag_snd,comm)
        push!(req_all,reqs)
    end
    @async begin
        @static if isdefined(MPI,:Testall)
            while ! MPI.Testall(req_all)
                yield()
            end
        else
            while ! MPI.Testall!(req_all)[1]
                yield()
            end
        end
        rcv
    end
end

function exchange_impl!(
    rcv::MPIData,
    snd::MPIData,
    graph::ExchangeGraph{<:MPIData},
    ::Type{T}) where T <: AbstractVector

    @assert size(rcv) == size(snd)
    @assert graph.rcv.comm === graph.rcv.comm
    @assert graph.rcv.comm === graph.snd.comm
    comm = graph.rcv.comm
    req_all = MPI.Request[]
    data_snd = JaggedArray(snd.item[])
    data_rcv = rcv.item[]
    @assert isa(data_rcv,JaggedArray)
    for (i,id_rcv) in enumerate(graph.rcv.item[])
        rank_rcv = id_rcv-1
        ptrs_rcv = data_rcv.ptrs
        buff_rcv = view(data_rcv.data,ptrs_rcv[i]:(ptrs_rcv[i+1]-1))
        tag_rcv = rank_rcv
        reqr = MPI.Irecv!(buff_rcv,rank_rcv,tag_rcv,comm)
        push!(req_all,reqr)
    end
    for (i,id_snd) in enumerate(graph.snd.item[])
        rank_snd = id_snd-1
        ptrs_snd = data_snd.ptrs
        buff_snd = view(data_snd.data,ptrs_snd[i]:(ptrs_snd[i+1]-1))
        tag_snd = MPI.Comm_rank(comm)
        reqs = MPI.Isend(buff_snd,rank_snd,tag_snd,comm)
        push!(req_all,reqs)
    end
    @async begin
        @static if isdefined(MPI,:Testall)
            while ! MPI.Testall(req_all)
                yield()
            end
        else
            while ! MPI.Testall!(req_all)[1]
                yield()
            end
        end
        rcv
    end
end

