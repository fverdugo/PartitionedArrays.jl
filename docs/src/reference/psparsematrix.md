# OldPSparseMatrix

## Type signature

```@docs
OldPSparseMatrix
```

## Accessors

```@docs
local_values(::OldPSparseMatrix)
own_values(::OldPSparseMatrix)
ghost_values(::OldPSparseMatrix)
own_ghost_values(::OldPSparseMatrix)
ghost_own_values(::OldPSparseMatrix)
```
## Constructors

```@docs
OldPSparseMatrix(a,b,c)
psparse
old_psparse!
```
## Assembly

```@docs
assemble!(o,::OldPSparseMatrix)
```
