# Advanced

## Custom partitions

```@docs
LocalIndices
LocalIndices(a,b,c,d)
OwnAndGhostIndices
OwnAndGhostIndices(own::OwnIndices,ghost::GhostIndices)
OwnIndices
OwnIndices(a,b,c)
GhostIndices
GhostIndices(a,b,c)
```
## Transform partitions

```@docs
replace_ghost
union_ghost
permute_indices
find_owner
```

## Transform indices

```@docs
to_global!
to_local!
```

## Local vector storage

```@docs
OwnAndGhostVectors
OwnAndGhostVectors(a,b,c)
```

## Assembly

```@docs
assembly_graph
assembly_neighbors
assembly_local_indices
```

