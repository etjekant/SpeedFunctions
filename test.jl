using BenchmarkTools
using Random
include("abs.jl")
include("any.jl")
include("general.jl")
include("max.jl")
include("sum.jl")

y = rand(Float64, 100000);
@benchmark sum(y)
@benchmark my_sum(y)

y = rand(Float16, 100000);
@benchmark sum(y)
@benchmark my_sum(y)

y = rand(Float64, 1000, 100);
@benchmark sum(y)
@benchmark my_sum(y)

y = rand(Float16, 1000, 100);
@benchmark sum(y)
@benchmark my_sum(y)

y = rand(Int64, 100000);
@benchmark sum(y)
@benchmark my_sum(y)

y = rand(Int16, 100000);
@benchmark sum(y)
@benchmark my_sum(y)

y = rand(Int64, 1000, 100);
@benchmark sum(y)
@benchmark my_sum(y)

y = rand(Int16, 1000, 100);
@benchmark sum(y)
@benchmark my_sum(y)


y = rand(Float64, 1000, 100);
@benchmark sum(abs2, y, dims=2)
@benchmark my_sum(abs2, y, dims=2)
@benchmark sum(abs2.(y), dims=2)

y = -1*rand(Float64, 100000);
@benchmark abs.(y)
@benchmark my_abs(y)

y = -1*rand(Float16, 100000);
@benchmark abs.(y)
@benchmark my_abs(y)

y = rand(Float64, 1000, 100);
@benchmark abs.(y)
@benchmark my_abs(y)

y = rand(Int64, 100000);
@benchmark abs.(y)
@benchmark my_abs(y)

y = rand(Int64, 1000, 1000);
@benchmark abs.(y)
@benchmark my_abs(y)

y = rand(Int64, 1000, 1000);
@benchmark abs.(y)
@benchmark my_abs!(y, y)

y = rand(Int64, 1000, 1000);
@benchmark maximum(y)
@benchmark my_max(y)

y = rand(Int64, 1000000);
@benchmark maximum(y)
@benchmark my_max(y)

y = rand(Float64, 1000, 1000);
@benchmark maximum(abs2, y)
@benchmark my_max(abs2, y)

y = rand(Int8, 1000, 1000);
@benchmark maximum(abs2, y)
@benchmark my_max(abs2, y)
my_max(abs, y)

y = rand(Int64, 100000);
@benchmark argmax(abs, y)
@benchmark my_argmax(abs, y)

argmax(abs2, y)
my_argmax(abs2, y)
