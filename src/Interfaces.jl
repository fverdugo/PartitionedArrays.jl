
abstract type Communicator end

function num_parts(::Communicator)
  @abstractmethod
end

function num_workers(::Communicator)
  @abstractmethod
end

function do_on_parts(task::Function,::Communicator,args...)
  @abstractmethod
end

function do_on_parts(task::Function,args...)
  comm = get_comm(get_distributed_data(first(args)))
  do_on_parts(task,comm,args...)
end

# Like do_on_parts put task does not take part as its first argument
function map_on_parts(task::Function,args...)
  do_on_parts(args...) do part, x...
    task(x...)
  end
end

function i_am_master(::Communicator)
  @abstractmethod
end

# We need to compare communicators to perform some checks
function Base.:(==)(a::Communicator,b::Communicator)
  @abstractmethod
end

# All communicators that are to be executed in the master to workers
# model inherit from these one
abstract type OrchestratedCommunicator <: Communicator end

function i_am_master(::OrchestratedCommunicator)
  true
end

# This is for the communicators to be executed in MPI mode
abstract type CollaborativeCommunicator <: Communicator end

# Data distributed in parts of type T in a communicator
# Formerly, ScatteredVector
abstract type DistributedData{T} end

function get_comm(a::DistributedData)
  @abstractmethod
end

function num_parts(a)
  num_parts(get_comm(a))
end

# Construct a DistributedData object in a communicator
function DistributedData{T}(initializer::Function,::Communicator,args...) where T
  @abstractmethod
end

function DistributedData(initializer::Function,::Communicator,args...)
  @abstractmethod
end

# The comm argument can be omitted if it can be determined from the first
# data argument.
function DistributedData{T}(initializer::Function,args...) where T
  comm = get_comm(get_distributed_data(first(args)))
  DistributedData{T}(initializer,comm,args...)
end

function DistributedData(initializer::Function,args...)
  comm = get_comm(get_distributed_data(first(args)))
  DistributedData(initializer,comm,args...)
end

get_part_type(::Type{<:DistributedData{T}}) where T = T

get_part_type(::DistributedData{T}) where T = T

Base.iterate(a::DistributedData)  = @abstractmethod

Base.iterate(a::DistributedData,state)  = @abstractmethod

function gather!(a::AbstractVector,b::DistributedData)
  @abstractmethod
end

function gather(b::DistributedData{T}) where T
  if i_am_master(get_comm(b))
    a = Vector{T}(undef,num_parts(b))
  else
    a = Vector{T}(undef,0)
  end
  gather!(a,b)
  a
end

function scatter(comm::Communicator,b::AbstractVector)
  @abstractmethod
end

function bcast(comm::Communicator,v)
  if i_am_master(comm)
    part_to_v = fill(v,num_parts(comm))
  else
    T = eltype(v)
    part_to_v = T[]
  end
  scatter(comm,part_to_v)
end

# return an object for which
# its restriction to the parts of a communicator is defined.
# The returned object is not necessarily an instance
# of DistributedData
# Do nothing by default.
function get_distributed_data(object)
  object
end

get_comm(a) = get_comm(get_distributed_data(a))

# Non-blocking in-place exchange
# In this version, sending a number per part is enough
# We have another version below to send a vector of numbers per part (compressed in a Table)
# Starts a non-blocking exchange. It returns a DistributedData of Julia Tasks. Calling wait on these
# tasks will wait until the exchange is done in the corresponding part
# (i.e., at this point it is save to read/write the buffers again).
function spawn_exchange!(
  data_rcv::DistributedData,
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  @abstractmethod
end

# Blocking in-place exchange
function exchange!(
  data_rcv::DistributedData,
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  t = spawn_exchange!(data_rcv,data_snd,parts_rcv,parts_snd)
  map_on_parts(wait,t)
  data_rcv
end

# Blocking allocating exchange
function exchange(
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  data_rcv = DistributedData(data_snd,parts_rcv) do partid, data_snd, parts_rcv
    similar(data_snd,eltype(data_snd),length(parts_rcv))
  end

  exchange!(data_rcv,data_snd,parts_rcv,parts_snd)
  data_rcv
end

# Non-blocking in-place exchange variable length (compressed in a Table)
function spawn_exchange!(
  data_rcv::DistributedData{<:Table},
  data_snd::DistributedData{<:Table},
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  @abstractmethod
end

# Blocking allocating exchange variable length (compressed in a Table)
function exchange(
  data_snd::DistributedData{<:Table},
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  # Count how many we snd to each part
  n_snd = DistributedData(data_snd) do part, data_snd
    n_snd = zeros(eltype(data_snd.ptrs),length(data_snd))
    for i in 1:length(n_snd)
      n_snd[i] = data_snd.ptrs[i+1] - data_snd.ptrs[i]
    end
    n_snd
  end

  # Count how many we rcv from each part
  n_rcv = exchange(n_snd,parts_rcv,parts_snd)

  # Allocate rcv tables
  data_rcv = DistributedData(n_rcv,data_snd) do part, n_rcv, data_snd
    ptrs = similar(data_snd.ptrs,eltype(data_snd.ptrs),length(n_rcv)+1)
    for i in 1:length(n_rcv)
      ptrs[i+1] = n_rcv[i]
    end
    length_to_ptrs!(ptrs)
    ndata = ptrs[end]-1
    data = similar(data_snd.data,eltype(data_snd.data),ndata)
    Table(data,ptrs)
  end

  # Do the exchange
  exchange!(data_rcv,data_snd,parts_rcv,parts_snd)
  data_rcv
end

# Discover snd parts from rcv assuming that srd is a subset of neighbors
# Assumes that neighbors is a symmetric communication graph
function discover_parts_snd(parts_rcv::DistributedData, neighbors::DistributedData)
  @assert get_comm(parts_rcv) == get_comm(neighbors)

  # Tell the neighbors whether I want to receive data from them
  data_snd = DistributedData(neighbors,parts_rcv) do part, neighbors, parts_rcv
    dict_snd = Dict(( n=>-1 for n in neighbors))
    for i in parts_rcv
      dict_snd[i] = part
    end
    [ dict_snd[n] for n in neighbors ]
  end
  data_rcv = exchange(data_snd,neighbors,neighbors)

  # build parts_snd
  parts_snd = DistributedData(data_rcv) do part, data_rcv
    k = findall(j->j>0,data_rcv)
    data_rcv[k]
  end

  parts_snd
end

# If neighbors not provided, all procs are considered neighbors (to be improved)
function discover_parts_snd(parts_rcv::DistributedData)
  comm = get_comm(parts_rcv)
  nparts = num_parts(comm)
  neighbors = DistributedData(parts_rcv) do part, parts_rcv
    T = eltype(parts_rcv)
    [T(i) for i in 1:nparts if i!=part]
  end
  discover_parts_snd(parts_rcv,neighbors)
end

function discover_parts_snd(parts_rcv::DistributedData,::Nothing)
  discover_parts_snd(parts_rcv)
end

struct IndexPartition{A,B,C}
  part::Int
  oid_to_gid::A
  gid_to_oid::B
  gid_to_part::C
end

num_gids(a::IndexPartition) = length(a.gid_to_part)
num_oids(a::IndexPartition) = length(a.oid_to_gid)

function UniformIndexPartition(ngids,np,p)
  gids = _oid_to_gid(ngids,np,p)
  oids = Int32(1):Int32(length(gids))
  oid_to_gid = OidToGid(gids)
  gid_to_oid = GidToOid(gids,oids)
  gid_to_part = GidToPart(ngids,np)
  IndexPartition(
    p,
    oid_to_gid,
    gid_to_oid,
    gid_to_part)
end

struct OidToGid <: AbstractVector{Int}
  gids::UnitRange{Int}
end
Base.size(a::OidToGid) = (length(a.gids),)
Base.IndexStyle(::Type{<:OidToGid}) = IndexLinear()
Base.getindex(a::OidToGid,oid::Integer) = a.gids[oid]

struct GidToOid <: AbstractDict{Int,Int32}
  gids::UnitRange{Int}
  oids::UnitRange{Int32}
end
Base.length(a::GidToOid) = length(a.gids)
Base.keys(a::GidToOid) = a.gids
Base.values(a::GidToOid) = a.oids
function Base.getindex(a::GidToOid,gid::Int)
  @boundscheck begin
    if ! (gid in a.gids)
      throw(KeyError(gid))
    end
  end
  oid = Int32(gid - a.gids.start + 1)
  oid
end
function Base.iterate(a::GidToOid)
  if length(a) == 0
    return nothing
  end
  state = 1
  a.gids[state]=>a.oids[state], state
end
function Base.iterate(a::GidToOid,state)
  if length(a) <= state
    return nothing
  end
  s = state + 1
  a.gids[s]=>a.oids[s], s
end

struct GidToPart <: AbstractVector{Int}
  ngids::Int
  np::Int
end
Base.size(a::GidToPart) = (a.ngids,)
Base.IndexStyle(::Type{<:GidToPart}) = IndexLinear()
function Base.getindex(a::GidToPart,gid::Integer)
  @boundscheck begin
    if !( 1<=gid && gid<=length(a) )
      throw(BoundsError(a,gid))
    end
  end
  _gid_to_part(a.ngids,a.np,gid)
end

function _oid_to_gid(ngids,np,p)
  _olength = ngids ÷ np
  _offset = _olength * (p-1)
  _rem = ngids % np
  if _rem < (np-p+1)
    olength = _olength
    offset = _offset
  else
    olength = _olength + 1
    offset = _offset + p - (np-_rem) - 1
  end
  Int(1+offset):Int(olength+offset)
end

function _gid_to_part(ngids,np,gid)
  @check 1<=gid && gid<=ngids "gid=$gid is not in [1,$ngids]"
  # TODO this can be heavily optimized
  for p in 1:np
    if _is_gid_in_part(ngids,np,p,gid)
      return p
    end
  end
end

function _is_gid_in_part(ngids,np,p,gid)
  gids = _oid_to_gid(ngids,np,p)
  gid >= gids.start && gid <= gids.stop
end

struct DistributedIndexPartition{A}
  ngids::Int
  oids::A
  function DistributedIndexPartition(ngids::Integer, oids::DistributedData{<:IndexPartition})
    A = typeof(oids)
    new{A}(ngids,oids)
  end
end

get_distributed_data(a::DistributedIndexPartition) = a.oids
num_gids(a::DistributedIndexPartition) = a.ngids

function UniformDistributedIndexPartition(comm::Communicator,ngids::Integer)
  np = num_parts(comm)
  oids = DistributedData(comm) do p
    UniformIndexPartition(ngids,np,p)
  end
  DistributedIndexPartition(ngids,oids)
end

# A, B, C should be the type of some indexable collection, e.g. ranges or vectors or dicts
struct IndexSet{A,B,C}
  part::Int
  ngids::Int
  lid_to_gid::A
  lid_to_part::B
  gid_to_lid::C
end

function IndexSet(part,ngids,lid_to_gid,lid_to_part)
  gid_to_lid = Dict{Int,Int32}()
  for (lid,gid) in enumerate(lid_to_gid)
    gid_to_lid[gid] = lid
  end
  IndexSet(part,ngids,lid_to_gid,lid_to_part,gid_to_lid)
end

num_gids(a::IndexSet) = a.ngids
num_lids(a::IndexSet) = length(a.lid_to_part)

struct Exchanger{B,C}
  parts_rcv::B
  parts_snd::B
  lids_rcv::C
  lids_snd::C
  function Exchanger(
    parts_rcv::DistributedData{<:AbstractVector{<:Integer}},
    parts_snd::DistributedData{<:AbstractVector{<:Integer}},
    lids_rcv::DistributedData{<:Table{<:Integer}},
    lids_snd::DistributedData{<:Table{<:Integer}})

    B = typeof(parts_rcv)
    C = typeof(lids_rcv)
    new{B,C}(parts_rcv,parts_snd,lids_rcv,lids_snd)
  end
end

function Exchanger(ids::DistributedData{<:IndexSet},neighbors=nothing)

  parts_rcv = DistributedData(ids) do part, ids
    parts_rcv = Dict((owner=>true for owner in ids.lid_to_part if owner!=part))
    sort(collect(keys(parts_rcv)))
  end

  lids_rcv, gids_rcv = DistributedData(ids,parts_rcv) do part, ids, parts_rcv

    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_rcv) ))

    ptrs = zeros(Int32,length(parts_rcv)+1)
    for owner in ids.lid_to_part
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)

    data_lids = zeros(Int32,ptrs[end]-1)
    data_gids = zeros(Int,ptrs[end]-1)

    for (lid,owner) in enumerate(ids.lid_to_part)
      if owner != part
        p = ptrs[owner_to_i[owner]]
        data_lids[p]=lid
        data_gids[p]=ids.lid_to_gid[lid]
        ptrs[owner_to_i[owner]] += 1
      end
    end
    rewind_ptrs!(ptrs)

    lids_rcv = Table(data_lids,ptrs)
    gids_rcv = Table(data_gids,ptrs)

    lids_rcv, gids_rcv
  end

  parts_snd = discover_parts_snd(parts_rcv,neighbors)

  gids_snd = exchange(gids_rcv,parts_snd,parts_rcv)

  lids_snd = DistributedData(ids, gids_snd) do part, ids, gids_snd
    ptrs = gids_snd.ptrs
    data_lids = zeros(Int32,ptrs[end]-1)
    for (k,gid) in enumerate(gids_snd.data)
      data_lids[k] = ids.gid_to_lid[gid]
    end
    lids_snd = Table(data_lids,ptrs)
    lids_snd
  end

  Exchanger(parts_rcv,parts_snd,lids_rcv,lids_snd)
end

function allocate_rcv_buffer(::Type{T},a::Exchanger) where T
  data_rcv = DistributedData(a.lids_rcv) do part, lids_rcv
    ptrs = lids_rcv.ptrs
    data = zeros(T,ptrs[end]-1)
    Table(data,ptrs)
  end
  data_rcv
end

function allocate_snd_buffer(::Type{T},a::Exchanger) where T
  data_snd = DistributedData(a.lids_snd) do part, lids_snd
    ptrs = lids_snd.ptrs
    data = zeros(T,ptrs[end]-1)
    Table(data,ptrs)
  end
  data_snd
end

struct DistributedIndexSet{A,B}
  ngids::Int
  lids::A
  exchanger::B
  function DistributedIndexSet(
    ngids::Integer,
    lids::DistributedData{<:IndexSet},
    exchanger::Exchanger=Exchanger(lids))

    A = typeof(lids)
    B = typeof(exchanger)
    new{A,B}(ngids,lids,exchanger)
  end
end

get_distributed_data(a::DistributedIndexSet) = a.lids
num_gids(a::DistributedIndexSet) = a.ngids

function non_overlaping(ids::DistributedIndexSet)
  lids = DistributedData(ids) do part, ids
    lid_to_gid = similar(ids.lid_to_gid,eltype(ids.lid_to_gid),0)
    for (lid,owner) in enumerate(ids.lid_to_part)
      if owner == part
        gid = ids.lid_to_gid[lid]
        push!(lid_to_gid,gid)
      end
    end
    lid_to_part = similar(ids.lid_to_part,eltype(ids.lid_to_part),length(lid_to_gid))
    fill!(lid_to_part,part)
    IndexSet(ids.part,ids.ngids,lid_to_gid,lid_to_part)
  end
  neighbors = DistributedData(ids.exchanger.parts_rcv) do part,parts_rcv
    similar(parts_rcv,eltype(parts_rcv),0)
  end
  exchanger = Exchanger(lids,neighbors)
  DistributedIndexSet(ids.ngids,lids,exchanger)
end

struct DistributedVector{T,A,B} <: AbstractVector{T}
  values::A
  ids::B
  function DistributedVector(
    values::DistributedData{<:AbstractVector{T}},
    ids::DistributedIndexSet) where T

    A = typeof(values)
    B = typeof(ids)
    new{T,A,B}(values,ids)
  end
end

function DistributedVector{T}(::UndefInitializer,ids::DistributedIndexSet) where T
  values = DistributedData(ids) do part, ids
    nlids = length(ids.lid_to_part)
    Vector{T}(undef,nlids)
  end
  DistributedVector(values,ids)
end

function Base.fill!(a::DistributedVector,v)
  do_on_parts(a.values) do part, values
    fill!(values,v)
  end
  a
end

function Base.getindex(a::DistributedVector,ids::DistributedIndexSet)
  exchange!(a)
  if ids === a.ids
    a
  else
    ids_out = ids
    ids_in = a.ids
    values_in = a.values
    values_out = DistributedData(values_in,ids_in,ids_out) do part, values_in, ids_in, ids_out
      values_out = similar(values_in,eltype(values_in),num_lids(ids_out))
      for lid_out in 1:num_lids(ids_out)
        if ids_out.lid_to_part[lid_out] == part
          gid = ids_out.lid_to_gid[lid_out]
          @notimplementedif ! haskey(ids_in.gid_to_lid,gid)
          lid_in = ids_in.gid_to_lid[gid]
          values_out[lid_out] = values_in[lid_in]
        end
      end
      values_out
    end
    v = DistributedVector(values_out,ids_out)
    exchange!(v)
    v
  end
end

#get_distributed_data(a::DistributedVector) = a.values

Base.length(a::DistributedVector) = num_gids(a.ids)

function exchange!(a::DistributedVector{T}) where T

  # Allocate buffers
  data_rcv = allocate_rcv_buffer(T,a.ids.exchanger)
  data_snd = allocate_snd_buffer(T,a.ids.exchanger)

  # Fill snd buffer
  do_on_parts(a.values,data_snd,a.ids.exchanger.lids_snd) do part,values,data_snd,lids_snd 
    for p in 1:length(lids_snd.data)
      lid = lids_snd.data[p]
      data_snd.data[p] = values[lid]
    end
  end

  # communicate
  exchange!(
    data_rcv,
    data_snd,
    a.ids.exchanger.parts_rcv,
    a.ids.exchanger.parts_snd)

  # Fill non-owned values from rcv buffer
  do_on_parts( a.values,data_rcv,a.ids.exchanger.lids_rcv) do part,values,data_rcv,lids_rcv 
    for p in 1:length(lids_rcv.data)
      lid = lids_rcv.data[p]
      values[lid] = data_rcv.data[p]  
    end
  end

  a
end

struct DistributedSparseMatrix{T,A,B} <: AbstractMatrix{T}
  values::A
  row_ids::B
  col_ids::B
  function DistributedSparseMatrix(
    values::DistributedData{<:AbstractSparseMatrix{T}},
    row_ids::DistributedIndexSet,
    col_ids::DistributedIndexSet) where T

    A = typeof(values)
    B = typeof(col_ids)
    new{T,A,B}(values,row_ids,col_ids)
  end
end

Base.size(a::DistributedSparseMatrix) = (num_gids(a.row_ids),num_gids(a.col_ids))

function LinearAlgebra.mul!(
  c::DistributedVector,
  a::DistributedSparseMatrix,
  b::DistributedVector,
  α::Number,
  β::Number)

  @assert c.ids === a.row_ids
  @assert b.ids === a.col_ids
  exchange!(b)
  do_on_parts(c.values,a.values,b.values) do part,c,a,b
    mul!(c,a,b,α,β)
  end
  c
end

function Base.getindex(
  a::DistributedSparseMatrix,
  row_ids::DistributedIndexSet,
  col_ids::DistributedIndexSet)

  @notimplementedif a.row_ids !== row_ids
  if a.col_ids === col_ids
    a
  else
    ids_out = col_ids
    ids_in = a.col_ids
    values_in = a.values
    values_out = DistributedData(values_in,ids_in,ids_out) do part, values_in, ids_in, ids_out
      i_to_lid_in = Int32[]
      i_to_lid_out = Int32[]
      for lid_in in 1:num_lids(ids_in)
         gid = ids_in.lid_to_gid[lid_in]
         if haskey(ids_out.gid_to_lid,gid)
           lid_out = ids_out.gid_to_lid[gid]
           push!(i_to_lid_in,lid_in)
           push!(i_to_lid_out,lid_out)
         end
      end
      I,J_in,V = findnz(values_in[:,i_to_lid_in])
      J_out = similar(J_in)
      J_out .= i_to_lid_out[J_in]
      sparse(I,J_out,V,size(values_in,1),num_lids(ids_out))
    end
    DistributedSparseMatrix(values_out,row_ids,col_ids)
  end
end

struct AdditiveSchwarz{A,B,C}
  problems::A
  solvers::B
  row_ids::C
  col_ids::C
end

function AdditiveSchwarz(a::DistributedSparseMatrix)
  problems = a[a.row_ids,a.row_ids].values
  solvers = DistributedData(problems) do part, problem
    return \
  end
  AdditiveSchwarz(problems,solvers,a.row_ids,a.col_ids)
end

function LinearAlgebra.mul!(c::DistributedVector,a::AdditiveSchwarz,b::DistributedVector)
  @assert c.ids === a.col_ids
  @assert b.ids === a.row_ids
  do_on_parts(c.values,a.problems,a.solvers,b.values,a.col_ids,a.row_ids) do part,c_col,p,s,b,col_ids,row_ids
    c_row = s(p,b)
    for lid_row in 1:num_lids(row_ids)
      gid = row_ids.lid_to_gid[lid_row]
      lid_col = col_ids.gid_to_lid[gid]
      c_col[lid_col] = c_row[lid_row]
    end
  end
  c
end




