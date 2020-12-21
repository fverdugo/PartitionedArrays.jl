module InterfacesTests

using DistributedDataDraft
using Test

ngids = 10
np = 3
p = 3
oids = UniformIndexPartition(ngids,np,p)
@test oids.part == p
@test oids.oid_to_gid == [7, 8, 9, 10]
@test oids.gid_to_part == [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
@test collect(oids.gid_to_oid) == [7 => 1, 8 => 2, 9 => 3, 10 => 4]
@test keys(oids.gid_to_oid) == 7:10
@test values(oids.gid_to_oid) == 1:4
@test oids.gid_to_oid[7] == 1
@test oids.gid_to_oid[8] == 2
@test oids.gid_to_oid[9] == 3
@test oids.gid_to_oid[10] == 4


end # module
