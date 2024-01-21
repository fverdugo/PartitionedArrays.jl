# PSparseMatrix

## Type signature

```@docs
PSparseMatrix
```

## Accessors

```@docs
local_values(::PSparseMatrix)
own_own_values(::PSparseMatrix)
own_ghost_values(::PSparseMatrix)
ghost_own_values(::PSparseMatrix)
ghost_ghost_values(::PSparseMatrix)
```
## Constructors

```@docs
PSparseMatrix(a,b,c,d)
psparse(f,b,c)
psparse(f,a,b,c,d,e)
psparse!
psystem
psystem!
```
## Assembly

```@docs
assemble(::PSparseMatrix,rows)
assemble!(::PSparseMatrix,::PSparseMatrix,cache)
consistent(::PSparseMatrix,rows)
consistent!(::PSparseMatrix,::PSparseMatrix,cache)
```

## Re-partition

```@docs
repartition(::PSparseMatrix,rows,cols)
repartition!(::PSparseMatrix,::PSparseMatrix,cache)
repartition(::PSparseMatrix,::PVector,rows,cols)
repartition!(::PSparseMatrix,::PVector,::PSparseMatrix,::PVector,cache)
```
