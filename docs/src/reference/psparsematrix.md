# PSparseMatrix

## Type signature

```@docs
PSparseMatrix
```

## Accessors

```@docs
local_values(::PSparseMatrix)
own_values(::PSparseMatrix)
ghost_values(::PSparseMatrix)
own_ghost_values(::PSparseMatrix)
ghost_own_values(::PSparseMatrix)
```
## Constructors

```@docs
PSparseMatrix(a,b,c)
psparse
psparse!
```
## Assembly

```@docs
assemble!(o,::PSparseMatrix)
```
