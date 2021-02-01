module IndexSetsTests

using PartitionedArrays
using Test

part = 2
noids = 20
firstgid = 11
ids = IndexRange(part,noids,firstgid)
@test ids.part == part
@test ids.lid_to_gid == 11:30
push!(ids.lid_to_gid,48)
push!(ids.lid_to_gid,49)
@test ids.lid_to_gid == vcat(collect(11:30),[48,49])
@test ids.lid_to_part == fill(part,20)
push!(ids.lid_to_part,3)
push!(ids.lid_to_part,3)
@test ids.lid_to_part == vcat(fill(part,20),[3,3])
@test ids.oid_to_lid == 1:20
@test ids.hid_to_lid == Int32[]
push!(ids.hid_to_lid,21)
push!(ids.hid_to_lid,22)
@test ids.hid_to_lid == Int32[21,22]
@test ids.lid_to_ohid == 1:20
push!(ids.lid_to_ohid,-1)
push!(ids.lid_to_ohid,-2)
@test ids.lid_to_ohid == vcat(collect(1:20),[-1,-2])
@test collect(keys(ids.gid_to_lid)) == 11:30
@test collect(values(ids.gid_to_lid)) == 1:20
ids.gid_to_lid[48] = 21
ids.gid_to_lid[49] = 22
@test haskey(ids.gid_to_lid,49)
@test haskey(ids.gid_to_lid,48)
@test haskey(ids.gid_to_lid,12)
@test haskey(ids.gid_to_lid,30)
@test haskey(ids.gid_to_lid,11)
@test !haskey(ids.gid_to_lid,10)
@test !haskey(ids.gid_to_lid,31)
@test length(collect(ids.gid_to_lid)) == 22

hid_to_gid = [48,49]
hid_to_part = Int32[3,3]
ids = IndexRange(part,noids,firstgid,hid_to_gid,hid_to_part)
@test ids.part == part
@test ids.lid_to_gid == vcat(collect(11:30),[48,49])
@test ids.lid_to_part == vcat(fill(part,20),[3,3])
@test ids.hid_to_lid == Int32[21,22]
@test ids.lid_to_ohid == vcat(collect(1:20),[-1,-2])
@test haskey(ids.gid_to_lid,49)
@test haskey(ids.gid_to_lid,48)
@test haskey(ids.gid_to_lid,12)
@test haskey(ids.gid_to_lid,30)
@test haskey(ids.gid_to_lid,11)
@test !haskey(ids.gid_to_lid,10)
@test !haskey(ids.gid_to_lid,31)
@test length(collect(ids.gid_to_lid)) == 22

lid_to_gid = collect(ids.lid_to_gid)
lid_to_part = collect(ids.lid_to_part)
ids = ExtendedIndexRange(part,lid_to_gid,lid_to_part,firstgid)
@test ids.part == part
@test ids.lid_to_gid == vcat(collect(11:30),[48,49])
@test ids.lid_to_part == vcat(fill(part,20),[3,3])
@test ids.hid_to_lid == Int32[21,22]
@test ids.lid_to_ohid == vcat(collect(1:20),[-1,-2])
@test haskey(ids.gid_to_lid,49)
@test haskey(ids.gid_to_lid,48)
@test haskey(ids.gid_to_lid,12)
@test haskey(ids.gid_to_lid,30)
@test haskey(ids.gid_to_lid,11)
@test !haskey(ids.gid_to_lid,10)
@test !haskey(ids.gid_to_lid,31)
@test length(collect(ids.gid_to_lid)) == 22
push!(ids.lid_to_gid,62)
push!(ids.lid_to_gid,63)
push!(ids.lid_to_part,4)
push!(ids.lid_to_part,4)
push!(ids.hid_to_lid,23)
push!(ids.hid_to_lid,24)
push!(ids.lid_to_ohid,-3)
push!(ids.lid_to_ohid,-4)
ids.gid_to_lid[62] = 23
ids.gid_to_lid[63] = 24
@test ids.lid_to_gid == vcat(collect(11:30),[48,49,62,63])
@test ids.lid_to_part == vcat(fill(part,20),[3,3,4,4])
@test ids.hid_to_lid == Int32[21,22,23,24]
@test ids.lid_to_ohid == vcat(collect(1:20),[-1,-2,-3,-4])
@test haskey(ids.gid_to_lid,62)
@test haskey(ids.gid_to_lid,63)
@test !haskey(ids.gid_to_lid,64)
@test !haskey(ids.gid_to_lid,61)
@test length(collect(ids.gid_to_lid)) == 24

end # module
