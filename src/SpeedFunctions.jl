__precompile__()
module SpeedFunctions
using LoopVectorization


include("general.jl")
include("abs.jl")
include("max.jl")
include("min.jl")
include("sum.jl")


export my_abs, my_abs!, my_maximum, my_findmax, my_argmax, my_minimum, my_findmin, my_argmin, my_sum

end

