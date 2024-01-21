# PVector

## Type signature

```@docs
PVector
```
## Accessors

```@docs
local_values(::PVector)
own_values(::PVector)
ghost_values(::PVector)
```
## Constructors

```@docs
PVector(a,b)
PVector{V}(::UndefInitializer,b) where V
pvector(f,a)
pvector(f,a,b,c)
pvector!
pfill
pzeros
pones
prand
prandn
```
## Assembly

```@docs
assemble!(::PVector)
assemble(::PVector,rows)
assemble!(::PVector,::PVector,cache)
consistent!(::PVector)
consistent(::PVector,cols)
consistent!(::PVector,::PVector,cache)
```

## Re-partition

```@docs
repartition(::PVector,rows)
repartition!(::PVector,::PVector,cache)
```
