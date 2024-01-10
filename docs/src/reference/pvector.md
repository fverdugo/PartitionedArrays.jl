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
pvector
old_pvector!
pfill
pzeros
pones
prand
prandn
```
## Assembly

```@docs
assemble!(::PVector)
consistent!(::PVector)
```
