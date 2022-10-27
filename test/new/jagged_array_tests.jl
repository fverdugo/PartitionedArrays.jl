module GenericJaggedArrayTests

using GalerkinToolkit
using Test

a = [[1,2],[3,4,5],Int[],[3,4]]
b = JaggedArray(a)
@test a == b
@test b === JaggedArray(b)
T = typeof(b)
c = T(b.data,b.ptrs)
@test c == b

d = collect(c)

a = [[1,2],[3,4,5],Int[],[3,4]]
b = JaggedArray(a)
c = GenericJaggedArray(b.data,b.ptrs)
@test a == c
T = typeof(c)
d = T(c.data,c.ptrs)
@test c == b

end

