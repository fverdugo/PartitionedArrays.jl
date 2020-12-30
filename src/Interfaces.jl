
abstract type Backend end

# Should return a DistributedData{Int}
function get_part_ids(b::Backend,nparts::Integer)
  @abstractmethod
end

function get_part_ids(b::Backend,nparts::Tuple)
  get_part_ids(b,prod(nparts))
end

# This can be overwritten to add a finally clause
function distributed_run(driver::Function,b::Backend,nparts)
  part = get_part_ids(b,nparts)
  driver(part)
end

# Data distributed in parts of type T
abstract type DistributedData{T} end

num_parts(a::DistributedData) = @abstractmethod

get_backend(a::DistributedData) = @abstractmethod

Base.iterate(a::DistributedData)  = @abstractmethod

Base.iterate(a::DistributedData,state)  = @abstractmethod

get_part_ids(a::DistributedData) = get_part_ids(get_backend(a),num_parts(a))

map_parts(task::Function,a::DistributedData...) = @abstractmethod

#function map_parts(task::Function,a...)
#  map_parts(task,map(DistributedData,a)...)
#end
#
#DistributedData(a::DistributedData) = a

# Non-blocking in-place exchange
# In this version, sending a number per part is enough
# We have another version below to send a vector of numbers per part (compressed in a Table)
# Starts a non-blocking exchange. It returns a DistributedData of Julia Tasks. Calling schedule and wait on these
# tasks will wait until the exchange is done in the corresponding part
# (i.e., at this point it is save to read/write the buffers again).
function async_exchange!(
  data_rcv::DistributedData,
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData,
  t_in::DistributedData)

  @abstractmethod
end

function async_exchange!(
  data_rcv::DistributedData,
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData)

  t_in = _empty_tasks(parts_rcv)
  async_exchange!(data_rcv,data_snd,parts_rcv,parts_snd,t_in)
end

function _empty_tasks(a)
  map_parts(a) do a
    @task nothing
  end
end

# Non-blocking allocating exchange
# the returned data_rcv cannot be consumed in a part until the corresponding task in t is done.
function async_exchange(
  data_snd::DistributedData,
  parts_rcv::DistributedData,
  parts_snd::DistributedData,
  t_in::DistributedData=_empty_tasks(parts_rcv))

  data_rcv = map_parts(data_snd,parts_rcv) do data_snd, parts_rcv
    similar(data_snd,eltype(data_snd),length(parts_rcv))
  end

  t_out = async_exchange!(data_rcv,data_snd,parts_rcv,parts_snd,t_in)

  data_rcv, t_out
end

# Non-blocking in-place exchange variable length (compressed in a Table)
function async_exchange!(
  data_rcv::DistributedData{<:Table},
  data_snd::DistributedData{<:Table},
  parts_rcv::DistributedData,
  parts_snd::DistributedData,
  t_in::DistributedData)

  @abstractmethod
end

# Non-blocking allocating exchange variable length (compressed in a Table)
function async_exchange(
  data_snd::DistributedData{<:Table},
  parts_rcv::DistributedData,
  parts_snd::DistributedData,
  t_in::DistributedData)

  # Allocate empty data
  data_rcv = map_parts(empty_table,data_snd)
  n_snd = map_parts(parts_snd) do parts_snd
    Int[]
  end

  # wait data_snd to be in a correct state and
  # Count how many we snd to each part
  t1 = map_parts(n_snd,data_snd,t_in) do n_snd,data_snd,t_in
    @task begin
      wait(schedule(t_in))
      resize!(n_snd,length(data_snd))
      for i in 1:length(n_snd)
        n_snd[i] = data_snd.ptrs[i+1] - data_snd.ptrs[i]
      end
    end
  end

  # Count how many we rcv from each part
  n_rcv, t2 = async_exchange(n_snd,parts_rcv,parts_snd,t1)

  # Wait n_rcv to be in a correct state and
  # resize data_rcv to the correct size
  t3 = map_parts(n_rcv,t2,data_rcv) do n_rcv,t2,data_rcv
    @task begin
      wait(schedule(t2))
      resize!(data_rcv.ptrs,length(n_rcv)+1)
      for i in 1:length(n_rcv)
        data_rcv.ptrs[i+1] = n_rcv[i]
      end
      length_to_ptrs!(data_rcv.ptrs)
      ndata = data_rcv.ptrs[end]-1
      resize!(data_rcv.data,ndata)
    end
  end

  # Do the actual exchange
  t4 = async_exchange!(data_rcv,data_snd,parts_rcv,parts_snd,t3)

  data_rcv, t4
end

# Blocking in-place exchange
function exchange!(args...;kwargs...)
  t = async_exchange!(args...;kwargs...)
  map_parts(schedule,t)
  map_parts(wait,t)
  first(args)
end

# Blocking allocating exchange
function exchange(args...;kwargs...)
  data_rcv, t = async_exchange(args...;kwargs...)
  map_parts(schedule,t)
  map_parts(wait,t)
  data_rcv
end

# Discover snd parts from rcv assuming that srd is a subset of neighbors
# Assumes that neighbors is a symmetric communication graph
function discover_parts_snd(parts_rcv::DistributedData, neighbors::DistributedData)
  @assert num_parts(parts_rcv) == num_parts(neighbors)

  parts = get_part_ids(parts_rcv)

  # Tell the neighbors whether I want to receive data from them
  data_snd = map_parts(parts,neighbors,parts_rcv) do part, neighbors, parts_rcv
    dict_snd = Dict(( n=>-1 for n in neighbors))
    for i in parts_rcv
      dict_snd[i] = part
    end
    [ dict_snd[n] for n in neighbors ]
  end
  data_rcv = exchange(data_snd,neighbors,neighbors)

  # build parts_snd
  parts_snd = map_parts(data_rcv) do data_rcv
    k = findall(j->j>0,data_rcv)
    data_rcv[k]
  end

  parts_snd
end

# If neighbors not provided, all procs are considered neighbors (to be improved)
function discover_parts_snd(parts_rcv::DistributedData)
  parts = get_part_ids(parts_rcv)
  nparts = num_parts(parts)
  neighbors = map_parts(parts,parts_rcv) do part, parts_rcv
    T = eltype(parts_rcv)
    [T(i) for i in 1:nparts if i!=part]
  end
  discover_parts_snd(parts_rcv,neighbors)
end

function discover_parts_snd(parts_rcv::DistributedData,::Nothing)
  discover_parts_snd(parts_rcv)
end

# Arbitrary set of global indices stored in a part
# gid_to_part can be omitted with nothing since only for some particular parallel
# data layouts (e.g. uniform partitions) it is efficient to recover this information. 
# oid: owned id
# hig: ghost (aka halo) id
# gid: global id
# lid: local id (ie union of owned + ghost)
struct IndexSet{A,B,C,D,E,F,G}
  part::Int
  ngids::Int
  lid_to_gid::A
  lid_to_part::B
  gid_to_part::C
  oid_to_lid::D
  hid_to_lid::E
  lid_to_ohid::F
  gid_to_lid::G
  function IndexSet(
    part::Integer,
    ngids::Integer,
    lid_to_gid::AbstractVector,
    lid_to_part::AbstractVector,
    gid_to_part::Union{AbstractVector,Nothing},
    oid_to_lid::Union{AbstractVector,AbstractRange},
    hid_to_lid::Union{AbstractVector,AbstractRange},
    lid_to_ohid::AbstractVector,
    gid_to_lid::AbstractDict)
    A = typeof(lid_to_gid)
    B = typeof(lid_to_part)
    C = typeof(gid_to_part)
    D = typeof(oid_to_lid)
    E = typeof(hid_to_lid)
    F = typeof(lid_to_ohid)
    G = typeof(gid_to_lid)
    new{A,B,C,D,E,F,G}(
      part,
      ngids,
      lid_to_gid,
      lid_to_part,
      gid_to_part,
      oid_to_lid,
      hid_to_lid,
      lid_to_ohid,
      gid_to_lid)
  end
end

num_gids(a::IndexSet) = a.ngids
num_lids(a::IndexSet) = length(a.lid_to_part)
num_oids(a::IndexSet) = length(a.oid_to_lid)
num_hids(a::IndexSet) = length(a.hid_to_lid)

function IndexSet(
  part::Integer,
  ngids::Integer,
  lid_to_gid::AbstractVector,
  lid_to_part::AbstractVector,
  gid_to_part::Union{AbstractVector,Nothing},
  oid_to_lid::Union{AbstractVector,AbstractRange},
  hid_to_lid::Union{AbstractVector,AbstractRange},
  lid_to_ohid::AbstractVector)

  gid_to_lid = Dict{Int,Int32}()
  for (lid,gid) in enumerate(lid_to_gid)
    gid_to_lid[gid] = lid
  end
  IndexSet(
    part,
    ngids,
    lid_to_gid,
    lid_to_part,
    gid_to_part,
    oid_to_lid,
    hid_to_lid,
    lid_to_ohid,
    gid_to_lid)
end

function IndexSet(
  part::Integer,
  ngids::Integer,
  lid_to_gid::AbstractVector,
  lid_to_part::AbstractVector,
  gid_to_part::Union{AbstractVector,Nothing},
  oid_to_lid::Union{AbstractVector,AbstractRange},
  hid_to_lid::Union{AbstractVector,AbstractRange})

  lid_to_ohid = zeros(Int32,length(lid_to_gid))
  lid_to_ohid[oid_to_lid] = 1:length(oid_to_lid)
  lid_to_ohid[hid_to_lid] = -(1:length(hid_to_lid))

  IndexSet(
    part,
    ngids,
    lid_to_gid,
    lid_to_part,
    gid_to_part,
    oid_to_lid,
    hid_to_lid,
    lid_to_ohid)
end

function IndexSet(
  part::Integer,
  ngids::Integer,
  lid_to_gid::AbstractVector,
  lid_to_part::AbstractVector,
  gid_to_part::Union{AbstractVector,Nothing}=nothing)

  oid_to_lid = findall(owner->owner==part,lid_to_part)
  hid_to_lid = findall(owner->owner!=part,lid_to_part)
  IndexSet(
    part,
    ngids,
    lid_to_gid,
    lid_to_part,
    gid_to_part,
    oid_to_lid,
    hid_to_lid)
end

function IndexSet(a::IndexSet,xid_to_lid::AbstractVector)
  xid_to_gid = a.lid_to_gid[xid_to_lid]
  xid_to_part = a.lid_to_part[xid_to_lid]
  IndexSet(a.part,a.ngids,xid_to_gid,xid_to_part,a.gid_to_part)
end

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

  parts = get_part_ids(ids)

  parts_rcv = map_parts(parts,ids) do part, ids
    parts_rcv = Dict((owner=>true for owner in ids.lid_to_part if owner!=part))
    sort(collect(keys(parts_rcv)))
  end

  lids_rcv, gids_rcv = map_parts(parts,ids,parts_rcv) do part,ids,parts_rcv

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

  lids_snd = map_parts(ids,gids_snd) do ids,gids_snd
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

function Base.reverse(a::Exchanger)
  Exchanger(a.parts_snd,a.parts_rcv,a.lids_snd,a.lids_rcv)
end

function allocate_rcv_buffer(::Type{T},a::Exchanger) where T
  data_rcv = map_parts(a.lids_rcv) do lids_rcv
    ptrs = lids_rcv.ptrs
    data = zeros(T,ptrs[end]-1)
    Table(data,ptrs)
  end
  data_rcv
end

function allocate_snd_buffer(::Type{T},a::Exchanger) where T
  data_snd = map_parts(a.lids_snd) do lids_snd
    ptrs = lids_snd.ptrs
    data = zeros(T,ptrs[end]-1)
    Table(data,ptrs)
  end
  data_snd
end

function async_exchange!(
  values::DistributedData{<:AbstractVector{T}},
  exchanger::Exchanger,
  t0::DistributedData=_empty_tasks(exchanger.parts_rcv);
  reduce_op=_replace) where T

  # Allocate buffers
  data_rcv = allocate_rcv_buffer(T,exchanger)
  data_snd = allocate_snd_buffer(T,exchanger)

  # Fill snd buffer
  t1 = map_parts(t0,values,data_snd,exchanger.lids_snd) do t0,values,data_snd,lids_snd 
    @task begin
      wait(schedule(t0))
      for p in 1:length(lids_snd.data)
        lid = lids_snd.data[p]
        data_snd.data[p] = values[lid]
      end
    end
  end

  # communicate
  t2 = async_exchange!(
    data_rcv,
    data_snd,
    exchanger.parts_rcv,
    exchanger.parts_snd,
    t1)

  # Fill values from rcv buffer
  # asynchronously
  t3 = map_parts(t2,values,data_rcv,exchanger.lids_rcv) do t2,values,data_rcv,lids_rcv 
    @task begin
      wait(schedule(t2))
      for p in 1:length(lids_rcv.data)
        lid = lids_rcv.data[p]
        values[lid] = reduce_op(values[lid],data_rcv.data[p])
      end
    end
  end

  t3
end

_replace(x,y) = y

struct DistributedRange{A,B} <: AbstractUnitRange{Int}
  ngids::Int
  lids::A
  exchanger::B
  function DistributedRange(
    ngids::Integer,
    lids::DistributedData{<:IndexSet},
    exchanger::Exchanger=Exchanger(lids))
  
    A = typeof(lids)
    B = typeof(exchanger)
    new{A,B}(
      ngids,
      lids,
      exchanger)
  end
end

Base.first(a::DistributedRange) = 1
Base.last(a::DistributedRange) = a.ngids

num_gids(a::DistributedRange) = a.ngids
num_parts(a::DistributedRange) = num_parts(a.lids)

struct DistributedVector{T,A,B} <: AbstractVector{T}
  values::A
  ids::B
  function DistributedVector(
    values::DistributedData{<:AbstractVector{T}},
    ids::DistributedRange) where T

    A = typeof(values)
    B = typeof(ids)
    new{T,A,B}(values,ids)
  end
end

function DistributedVector{T}(
  ::UndefInitializer,
  ids::DistributedRange) where T

  values = map_parts(ids.lids) do lids
    nlids = num_lids(lids)
    Vector{T}(undef,nlids)
  end
  DistributedVector(values,ids)
end

function Base.fill!(a::DistributedVector,v)
  map_parts(a.values) do lid_to_value
    fill!(lid_to_value,v)
  end
  a
end

Base.length(a::DistributedVector) = length(a.ids)

function async_exchange!(
  a::DistributedVector,
  t0::DistributedData=_empty_tasks(a.ids.exchanger.parts_rcv))
  async_exchange!(a.values,a.ids.exchanger,t0)
end

# Non-blocking assembly
function async_assemble!(
  a::DistributedVector,
  t0::DistributedData=_empty_tasks(a.ids.exchanger.parts_rcv);
  reduce_op=+)

  exchanger_rcv = a.ids.exchanger # receives data at ghost ids from remote parts
  exchanger_snd = reverse(exchanger_rcv) # sends data at ghost ids to remote parts
  t1 = async_exchange!(a.values,exchanger_snd,t0;reduce_op=reduce_op)
  async_exchange!(a.values,exchanger_rcv,t1)
end

# Blocking assembly
function assemble!(args...;kwargs...)
  t = async_assemble!(args...;kwargs...)
  map_parts(schedule,t)
  map_parts(wait,t)
  first(args)
end

struct DistributedSparseMatrix{T,A,B,C,D} <: AbstractMatrix{T}
  values::A
  row_ids::B
  col_ids::C
  exchanger::D
  function DistributedSparseMatrix(
    values::DistributedData{<:AbstractSparseMatrix{T}},
    row_ids::DistributedRange,
    col_ids::DistributedRange,
    exchanger=_matrix_exchanger(values,row_ids.exchanger,row_ids.lids,col_ids.lids)) where T

    A = typeof(values)
    B = typeof(row_ids)
    C = typeof(col_ids)
    D = typeof(exchanger)
    new{T,A,B,C,D}(values,row_ids,col_ids,exchanger)
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
  t = async_exchange!(b)
  map_parts(t,c.values,a.values,b.values) do t,c,a,b
    # TODO start multiplying the diagonal block
    # before waiting for the ghost values of b
    wait(schedule(t))
    # If a is in an assembled state the c ghost values of c will be correct
    # otherwise one would need to call exchange!(c)
    # Note that it is cheaper to call exchange!(c) than assemble!(a)
    mul!(c,a,b,α,β)
  end
  c
end

function _matrix_exchanger(values,row_exchanger,row_lids,col_lids)

  part = get_part_ids(row_lids)
  parts_rcv = row_exchanger.parts_rcv
  parts_snd = row_exchanger.parts_snd
  findnz_values = map_parts(findnz,values)

  function setup_rcv(part,parts_rcv,row_lids,col_lids,findnz_values)
    k_li, k_lj, = findnz_values
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_rcv) ))
    ptrs = zeros(Int32,length(parts_rcv)+1)
    for k in 1:length(k_li)
      li = k_li[k]
      owner = row_lids.lid_to_part[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)
    k_rcv_data = zeros(Int,ptrs[end]-1)
    gi_rcv_data = zeros(Int,ptrs[end]-1)
    gj_rcv_data = zeros(Int,ptrs[end]-1)
    for k in 1:length(k_li)
      li = k_li[k]
      lj = k_lj[k]
      owner = row_lids.lid_to_part[li]
      if owner != part
        p = ptrs[owner_to_i[owner]]
        k_rcv_data[p] = k
        gi_rcv_data[p] = row_lids.lid_to_gid[li]
        gj_rcv_data[p] = col_lids.lid_to_gid[lj]
        ptrs[owner_to_i[owner]] += 1
      end
    end
    rewind_ptrs!(ptrs)
    k_rcv = Table(k_rcv_data,ptrs)
    gi_rcv = Table(gi_rcv_data,ptrs)
    gj_rcv = Table(gj_rcv_data,ptrs)
    k_rcv, gi_rcv, gj_rcv
  end
  
  k_rcv, gi_rcv, gj_rcv = map_parts(setup_rcv,part,parts_rcv,row_lids,col_lids,findnz_values)

  gi_snd = exchange(gi_rcv,parts_snd,parts_rcv)
  gj_snd = exchange(gj_rcv,parts_snd,parts_rcv)

  function setup_snd(part,row_lids,col_lids,gi_snd,gj_snd,findnz_values)
    ptrs = gi_snd.ptrs
    k_snd_data = zeros(Int,ptrs[end]-1)
    k_li, k_lj, = findnz_values
    k_v = collect(1:length(k_li))
    # TODO this can be optimized:
    li_lj_to_k = sparse(k_li,k_lj,k_v,num_lids(row_lids),num_lids(col_lids))
    for p in 1:length(gi_snd.data)
      gi = gi_snd.data[p]
      gj = gj_snd.data[p]
      li = row_lids.gid_to_lid[gi]
      lj = col_lids.gid_to_lid[gj]
      k = li_lj_to_k[li,lj]
      @assert k > 0 "The sparsity patern of the ghost layer is inconsistent"
      k_snd_data[p] = k
    end
    k_snd = Table(k_snd_data,ptrs)
    k_snd
  end

  k_snd = map_parts(setup_snd,part,row_lids,col_lids,gi_snd,gj_snd,findnz_values)

  Exchanger(parts_rcv,parts_snd,k_rcv,k_snd)
end

function async_exchange!(
  a::DistributedSparseMatrix,
  t0::DistributedData=_empty_tasks(a.exchanger.parts_rcv))
  nzval = map_parts(nonzeros,a.values)
  async_exchange!(nzval,a.exchanger,t0)
end

# Non-blocking assembly
function async_assemble!(
  a::DistributedSparseMatrix,
  t0::DistributedData=_empty_tasks(a.exchanger.parts_rcv);
  reduce_op=+)

  exchanger_rcv = a.exchanger # receives data at ghost ids from remote parts
  exchanger_snd = reverse(exchanger_rcv) # sends data at ghost ids to remote parts
  nzval = map_parts(nonzeros,a.values)
  t1 = async_exchange!(nzval,exchanger_snd,t0;reduce_op=reduce_op)
  async_exchange!(nzval,exchanger_rcv,t1)
end


