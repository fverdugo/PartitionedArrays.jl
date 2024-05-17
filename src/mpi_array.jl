
const EXCHANGE_IMPL_TAG  = 0 
const EXCHANGE_GRAPH_IMPL_TAG = Ref(Int64(1))

function new_exchange_graph_impl_tag()
    function new_tag()
      EXCHANGE_GRAPH_IMPL_TAG[] = 
           (EXCHANGE_GRAPH_IMPL_TAG[] + 1)%(MPI.tag_ub()+1)
      return EXCHANGE_GRAPH_IMPL_TAG[]     
    end 
   new_tag()
   if EXCHANGE_GRAPH_IMPL_TAG[] == EXCHANGE_IMPL_TAG
      return new_tag()
   else 
      return EXCHANGE_GRAPH_IMPL_TAG[]
   end
end 

function ptrs_to_counts(ptrs)
    counts = similar(ptrs,eltype(ptrs),length(ptrs)-1)
    @inbounds for i in 1:length(counts)
        counts[i] = ptrs[i+1]-ptrs[i]
    end
    counts
end

"""
    distribute_with_mpi(a;comm::MPI.Comm=MPI.COMM_WORLD,duplicate_comm=true)

Create an [`MPIArray`](@ref) instance by distributing
the items in the collection `a` over the ranks of the given MPI
communicator `comm`. Each rank receives
exactly one item, thus `length(a)`  and the communicator size need to match.
For arrays that can store more than one item per
rank see [`PVector`](@ref) or [`PSparseMatrix`](@ref).
If `duplicate_comm=false` the result will take ownership of the given communicator.
Otherwise, a copy will be done with `MPI.Comm_dup(comm)`.

!!! note
    This function calls `MPI.Init()` if MPI is not initialized yet.
"""
function distribute_with_mpi(a;comm::MPI.Comm=MPI.COMM_WORLD,duplicate_comm=true)
    if !MPI.Initialized()
        MPI.Init()
    end
    msg = "Number of MPI ranks needs to be the same as items in the given array"
    @assert length(a) == MPI.Comm_size(comm) msg
    if duplicate_comm
        comm = MPI.Comm_dup(comm)
    end
    i = MPI.Comm_rank(comm)+1
    MPIArray(a[i],comm,size(a))
end

"""
    with_mpi(f;comm=MPI.COMM_WORLD,duplicate_comm=true)

Call `f(a->distribute_with_mpi(a;comm,duplicate_comm))`
and abort MPI if there was an error.  This is the safest way of running the function `f` using MPI.

!!! note
    This function calls `MPI.Init()` if MPI is not initialized yet.
"""
function with_mpi(f;comm::MPI.Comm=MPI.COMM_WORLD,duplicate_comm=true)
    if !MPI.Initialized()
        MPI.Init()
    end
    distribute = a -> distribute_with_mpi(a;comm,duplicate_comm)
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
    MPIArray{T,N}

Represent an array of element type `T` and number of dimensions `N`, where
each item in the array is stored in a separate MPI process. I.e., each MPI
rank stores only one item. For arrays that can store more than one item per
rank see [`PVector`](@ref) or [`PSparseMatrix`](@ref). This struct implements
the Julia array interface.
However, using `setindex!` and `getindex!` is disabled
for performance reasons (communication cost).

# Properties

The fields of this struct (and the inner constructors) are private. To
generate an instance of `MPIArray` use function [`distribute_with_mpi`](@ref).

# Supertype hierarchy

    MPIArray{T,N} <: AbstractArray{T,N}
"""
struct MPIArray{T,N} <: AbstractArray{T,N}
    item_ref::Base.RefValue{T}
    comm::MPI.Comm
    size::NTuple{N,Int}
    function MPIArray{T,N}(item,comm::MPI.Comm,size::Dims{N}) where {T,N}
        @assert MPI.Comm_size(comm) == prod(size)
        new{T,N}(Ref{T}(item),comm,size)
    end
    function MPIArray{T,N}(::UndefInitializer,comm::MPI.Comm,size::Dims{N}) where {T,N}
        @assert MPI.Comm_size(comm) == prod(size)
        new{T,N}(Ref{T}(),comm,size)
    end
end

function MPIArray(item,comm::MPI.Comm,size::Dims)
    T = typeof(item)
    N = length(size)
    MPIArray{T,N}(item,comm,size)
end
function MPIArray(::UndefInitializer,comm::MPI.Comm,size::Dims)
    error("MPIArray(undef,comm,size) not allowed. Use MPIArray{T,N}(undef,comm,size) instead.")
end
function Base.getproperty(x::MPIArray,sym::Symbol)
    if sym === :item
        x.item_ref[]
    else
        getfield(x,sym)
    end
end
function Base.propertynames(x::MPIArray, private::Bool=false)
  (fieldnames(typeof(x))...,:item)
end
function Base.setproperty!(x::MPIArray, sym::Symbol, v, order::Symbol=:order)
    if sym === :item
       x.item_ref[] = v
    else
        setfield!(x,sym,v)
    end
end

Base.size(a::MPIArray) = a.size
Base.IndexStyle(::Type{<:MPIArray}) = IndexLinear()
function Base.getindex(a::MPIArray,i::Int)
    scalar_indexing_action(a)
    if i == MPI.Comm_rank(a.comm)+1
        a.item
    else
        error("Indexing of MPIArray at arbitrary indices not implemented yet.")
    end
end
function Base.setindex!(a::MPIArray,v,i::Int)
    scalar_indexing_action(a)
    if i == MPI.Comm_rank(a.comm)+1
        a.item = v
    else
        error("Indexing of MPIArray at arbitrary indices not implemented yet.")
    end
    v
end
linear_indices(a::MPIArray) = distribute_with_mpi(LinearIndices(a);comm=a.comm,duplicate_comm=false)
cartesian_indices(a::MPIArray) = distribute_with_mpi(CartesianIndices(a);comm=a.comm,duplicate_comm=false)
function Base.show(io::IO,k::MIME"text/plain",data::MPIArray)
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
            println(io,"$index = $(data.item)")
        end
        MPI.Barrier(data.comm)
    end
end
function Base.show(io::IO,data::MPIArray)
    if MPI.Comm_rank(data.comm) == 0
        print(io,"MPIArray(…)")
    end
end

getany(a::MPIArray) = a.item
i_am_main(a::MPIArray) = MPI.Comm_rank(a.comm)+1 == MAIN

function Base.similar(a::MPIArray,::Type{T},dims::Dims) where T
    N = length(dims)
    MPIArray{T,N}(undef,a.comm,dims)
end

function Base.copyto!(b::MPIArray,a::MPIArray)
    b.item = a.item
    b
end

function Base.map(f,args::Vararg{MPIArray,N}) where N
    a = first(args)
    @assert all(i->size(a)==size(i),args)
    item = f(map(i->i.item_ref[],args)...)
    MPIArray(item,a.comm,a.size)
end

function Base.map(f,a::MPIArray)
    item = f(a.item_ref[])
    MPIArray(item,a.comm,a.size)
end

function Base.map(f,args::Vararg{MPIArray,2})
    a = first(args)
    @assert all(i->size(a)==size(i),args)
    t1,t2 = map(i->i.item_ref[],args)
    item = f(t1,t2)
    MPIArray(item,a.comm,a.size)
end

function Base.map(f,args::Vararg{MPIArray,3})
    a = first(args)
    @assert all(i->size(a)==size(i),args)
    t1,t2,t3 = map(i->i.item_ref[],args)
    item = f(t1,t2,t3)
    MPIArray(item,a.comm,a.size)
end

function Base.map(f,args::Vararg{MPIArray,4})
    a = first(args)
    @assert all(i->size(a)==size(i),args)
    t1,t2,t3,t4 = map(i->i.item_ref[],args)
    item = f(t1,t2,t3,t4)
    MPIArray(item,a.comm,a.size)
end

function Base.map(f,args::Vararg{MPIArray,5})
    a = first(args)
    @assert all(i->size(a)==size(i),args)
    t1,t2,t3,t4,t5 = map(i->i.item_ref[],args)
    item = f(t1,t2,t3,t4,t5)
    MPIArray(item,a.comm,a.size)
end

function Base.map(f,args::Vararg{MPIArray,6})
    a = first(args)
    @assert all(i->size(a)==size(i),args)
    t1,t2,t3,t4,t5,t6 = map(i->i.item_ref[],args)
    item = f(t1,t2,t3,t4,t5,t6)
    MPIArray(item,a.comm,a.size)
end

function Base.map!(f,r::MPIArray,args::MPIArray...)
    a = first(args)
    @assert all(i->size(a)==size(i),args)
    r.item = f(map(i->i.item,args)...)
    r
end

function gather_impl!(
    rcv::MPIArray, snd::MPIArray,
    destination, ::Type{T}) where T
    @assert rcv.comm === snd.comm
    comm = snd.comm
    if isa(destination,Integer)
        root = destination-1
        if isbitstype(T)
            if MPI.Comm_rank(comm) == root
                @assert length(rcv.item) == MPI.Comm_size(comm)
                rcv.item[destination] = snd.item
                rcv_buffer = MPI.UBuffer(rcv.item,1)
                MPI.Gather!(MPI.IN_PLACE,rcv_buffer,root,comm)
            else
                MPI.Gather!(snd.item_ref,nothing,root,comm)
            end
        else
            if MPI.Comm_rank(comm) == root
                rcv.item[:] = MPI.gather(snd.item,comm;root)
            else
                MPI.gather(snd.item,comm;root)
            end
        end
    else
        @assert destination === :all
        @assert length(rcv.item) == MPI.Comm_size(comm)
        rcv_buffer = MPI.UBuffer(rcv.item,1)
        MPI.Allgather!(snd.item_ref,rcv_buffer,snd.comm)
    end
    rcv
end

function gather_impl!(
    rcv::MPIArray, snd::MPIArray,
    destination, ::Type{T}) where T <: AbstractVector
    Tv = eltype(snd.item)
    @assert rcv.comm === snd.comm
    @assert isa(rcv.item,JaggedArray)
    @assert eltype(eltype(rcv.item)) == Tv
    comm = snd.comm
    if isa(destination,Integer)
        root = destination-1
        if MPI.Comm_rank(comm) == root
            @assert length(rcv.item) == MPI.Comm_size(comm)
            rcv.item[destination] = snd.item
            counts = ptrs_to_counts(rcv.item.ptrs)
            rcv_buffer = MPI.VBuffer(rcv.item.data,counts)
            MPI.Gatherv!(MPI.IN_PLACE,rcv_buffer,root,comm)
        else
            MPI.Gatherv!(convert(Vector{Tv},snd.item),nothing,root,comm)
        end
    else
        @assert destination === :all
        @assert length(rcv.item) == MPI.Comm_size(comm)
        counts = ptrs_to_counts(rcv.item.ptrs)
        rcv_buffer = MPI.VBuffer(rcv.item.data,counts)
        MPI.Allgatherv!(convert(Vector{Tv},snd.item),rcv_buffer,comm)
    end
    rcv
end

function scatter_impl!(
    rcv::MPIArray,snd::MPIArray,
    source,::Type{T}) where T
    comm = snd.comm
    root = source - 1
    @assert source !== :all "All to all not implemented"
    @assert rcv.comm === snd.comm
    if isbitstype(T)
        @assert eltype(snd.item) == typeof(rcv.item)
        if MPI.Comm_rank(comm) == root
            snd_buffer = MPI.UBuffer(snd.item,1)
            rcv.item = snd.item[source]
            MPI.Scatter!(snd_buffer,MPI.IN_PLACE,root,comm)
        else
            MPI.Scatter!(nothing,rcv.item_ref,root,comm)
        end
    else
        if MPI.Comm_rank(comm) == root
            rcv.item_ref[] = MPI.scatter(snd.item,comm;root)
        else
            rcv.item_ref[] = MPI.scatter(nothing,comm;root)
        end
    end
    rcv
end

function scatter_impl!(
    rcv::MPIArray,snd::MPIArray,
    source,::Type{T}) where T<:AbstractVector
    @assert source !== :all "All to all not implemented"
    @assert rcv.comm === snd.comm
    @assert isa(snd.item,JaggedArray)
    @assert eltype(eltype(snd.item)) == eltype(rcv.item)
    comm = snd.comm
    root = source - 1
    if MPI.Comm_rank(comm) == root
        counts = ptrs_to_counts(snd.item.ptrs)
        snd_buffer = MPI.VBuffer(snd.item.data,counts)
        rcv.item .= snd.item[source]
        MPI.Scatterv!(snd_buffer,MPI.IN_PLACE,root,comm)
    else
        # This void Vbuffer is required to circumvent a deadlock
        # that we found with OpenMPI 4.1.X on Gadi. In particular, the
        # deadlock arises whenever buf is set to nothing
        S = eltype(eltype(snd.item))
        snd_buffer = MPI.VBuffer(S[],Int[])
        MPI.Scatterv!(snd_buffer,rcv.item,root,comm)
    end
    rcv
end

function multicast_impl!(
    rcv::MPIArray,snd::MPIArray,
    source,::Type{T}) where T
    @assert rcv.comm === snd.comm
    comm = snd.comm
    root = source - 1
    if isbitstype(T)
        if MPI.Comm_rank(comm) == root
            rcv.item = snd.item
        end
        MPI.Bcast!(rcv.item_ref,root,comm)
    else
        if MPI.Comm_rank(comm) == root
            rcv.item_ref[] = MPI.bcast(snd.item,comm;root)
        else
            rcv.item_ref[] = MPI.bcast(nothing,comm;root)
        end
    end
    rcv
end

function multicast_impl!(
    rcv::MPIArray,snd::MPIArray,
    source,::Type{T}) where T<:AbstractVector
    @assert rcv.comm === snd.comm
    comm = snd.comm
    root = source - 1
    if MPI.Comm_rank(comm) == root
        rcv.item = snd.item
    end
    MPI.Bcast!(rcv.item,root,comm)
    rcv
end

function scan!(op,b::MPIArray,a::MPIArray;init,type)
    @assert b.comm === a.comm
    @assert type in (:inclusive,:exclusive)
    T = eltype(a)
    @assert eltype(b) == T
    comm = a.comm
    opr = MPI.Op(op,T)
    if type === :inclusive
        if a.item_ref !== b.item_ref
            MPI.Scan!(a.item_ref,b.item_ref,opr,comm)
        else
            MPI.Scan!(b.item_ref,opr,comm)
        end
        b.item = op(b.item,init)
    else
        if a.item_ref !== b.item_ref
            MPI.Exscan!(a.item_ref,b.item_ref,opr,comm)
        else
            MPI.Exscan!(b.item_ref,opr,comm)
        end
        if MPI.Comm_rank(comm) == 0
            b.item = init
        else
            b.item = op(b.item,init)
        end
    end
    b
end

function reduction!(op,b::MPIArray,a::MPIArray;destination=1,init=nothing)
    @assert b.comm === a.comm
    T = eltype(a)
    @assert eltype(b) == T
    comm = a.comm
    opr = MPI.Op(op,T)
    if destination !== :all
        root = destination-1
        if a.item_ref !== b.item_ref
            MPI.Reduce!(a.item_ref,b.item_ref,opr,root,comm)
        else
            MPI.Reduce!(b.item_ref,opr,root,comm)
        end
        if MPI.Comm_rank(comm) == root
            if init !== nothing
                b.item = op(b.item,init)
            end
        end
    else
        if a.item_ref !== b.item_ref
            MPI.Allreduce!(a.item_ref,b.item_ref,opr,comm)
        else
            MPI.Allreduce!(b.item_ref,opr,comm)
        end
        if init !== nothing
            b.item = op(b.item,init)
        end
    end
    b
end

function Base.reduce(op,a::MPIArray;kwargs...)
   r = reduction(op,a;destination=:all,kwargs...)
   r.item
end
Base.sum(a::MPIArray) = reduce(+,a)
function Base.collect(a::MPIArray)
    T = eltype(a)
    N = ndims(a)
    b = Array{T,N}(undef,size(a))
    c = MPIArray(b,a.comm,size(a))
    gather!(c,a,destination=:all)
    c.item
end

function Base.all(a::MPIArray)
    reduce(&,a;init=true)
end
function Base.all(p::Function,a::MPIArray)
    b = map(p,a)
    all(b)
end

function exchange_impl!(
    rcv::MPIArray,
    snd::MPIArray,
    graph::ExchangeGraph{<:MPIArray},
    ::Type{T}) where T

    @assert size(rcv) == size(snd)
    @assert graph.rcv.comm === graph.rcv.comm
    @assert graph.rcv.comm === graph.snd.comm
    comm = graph.rcv.comm
    req_all = MPI.Request[]
    for (i,id_rcv) in enumerate(graph.rcv.item)
        rank_rcv = id_rcv-1
        buff_rcv = view(rcv.item,i:i)
        reqr = MPI.Irecv!(buff_rcv,rank_rcv,EXCHANGE_IMPL_TAG,comm)
        push!(req_all,reqr)
    end
    for (i,id_snd) in enumerate(graph.snd.item)
        rank_snd = id_snd-1
        buff_snd = view(snd.item,i:i)
        reqs = MPI.Isend(buff_snd,rank_snd,EXCHANGE_IMPL_TAG,comm)
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
    rcv::MPIArray,
    snd::MPIArray,
    graph::ExchangeGraph{<:MPIArray},
    ::Type{T}) where T <: AbstractVector

    @assert size(rcv) == size(snd)
    @assert graph.rcv.comm === graph.rcv.comm
    @assert graph.rcv.comm === graph.snd.comm
    comm = graph.rcv.comm
    req_all = MPI.Request[]
    data_snd = JaggedArray(snd.item)
    data_rcv = rcv.item
    @assert isa(data_rcv,JaggedArray)
    for (i,id_rcv) in enumerate(graph.rcv.item)
        rank_rcv = id_rcv-1
        ptrs_rcv = data_rcv.ptrs
        buff_rcv = view(data_rcv.data,ptrs_rcv[i]:(ptrs_rcv[i+1]-1))
        reqr = MPI.Irecv!(buff_rcv,rank_rcv,EXCHANGE_IMPL_TAG,comm)
        push!(req_all,reqr)
    end
    for (i,id_snd) in enumerate(graph.snd.item)
        rank_snd = id_snd-1
        ptrs_snd = data_snd.ptrs
        buff_snd = view(data_snd.data,ptrs_snd[i]:(ptrs_snd[i+1]-1))
        reqs = MPI.Isend(buff_snd,rank_snd,EXCHANGE_IMPL_TAG,comm)
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

# This should go eventually into MPI.jl! 
Issend(data, comm::MPI.Comm, req=MPI.Request(); dest::Integer, tag::Integer=0) =
    Issend(data, dest, tag, comm, req)

function Issend(buf::MPI.Buffer, dest::Integer, tag::Integer, comm::MPI.Comm, req=MPI.Request())
    @assert MPI.isnull(req)
    # int MPI_Issend(const void* buf, int count, MPI_Datatype datatype, int dest,
    #               int tag, MPI_Comm comm, MPI_Request *request)
    MPI.API.MPI_Issend(buf.data, buf.count, buf.datatype, dest, tag, comm, req)
    MPI.setbuffer!(req, buf)
    return req
end
Issend(data, dest::Integer, tag::Integer, comm::MPI.Comm, req=MPI.Request()) =
    Issend(MPI.Buffer_send(data), dest, tag, comm, req)


function default_find_rcv_ids(::MPIArray)
    find_rcv_ids_gather_scatter
end

"""
 Implements Alg. 2 in https://dl.acm.org/doi/10.1145/1837853.1693476
 The algorithm's complexity is claimed to be O(log(p))
"""
function find_rcv_ids_ibarrier(snd_ids::MPIArray{<:AbstractVector{T}}) where T
    comm = snd_ids.comm
    map(snd_ids) do snd_ids 
        requests=MPI.Request[]
        tag=new_exchange_graph_impl_tag()
        for snd_part in snd_ids
          snd_rank = snd_part-1
          push!(requests,Issend(T(0),snd_rank,tag,comm))
        end
        rcv_ids=T[]
        done=false
        barrier_multicastted=false
        all_sends_done=false
        barrier_req=nothing
        status = Ref(MPI.STATUS_ZERO)
        while (!done)
            # Check whether any message has arrived
            ismsg = MPI.Iprobe(comm, status; tag=tag)
            
            # If message has arrived ...
            if (ismsg)
                push!(rcv_ids, status[].source+1)
                tag_rcv = status[].tag
                dummy=T[0]
                MPI.Recv!(dummy, comm; source=rcv_ids[end]-1, tag=tag_rcv)
                @boundscheck @assert tag_rcv == tag "Inconsistent tag in ExchangeGraph_impl()!" 
            end     
    
            if (barrier_multicastted)
                done=MPI.Test(barrier_req)
            else
                all_sends_done = MPI.Testall(requests)
                if (all_sends_done)
                    barrier_req=MPI.Ibarrier(comm)
                    barrier_multicastted=true
                end
            end
        end
        sort(rcv_ids)
    end
end
