using LoopVectorization
using BenchmarkTools

function my_max(f::Function, itr::VecOrMat; dims::Int = 0)
    if dims == 0
        return Base.mapreduce(f, max, itr)
    else
        return Base.mapreduce(f, max, itr; dims)
    end
end

function my_max(itr::VecOrMat; dims::Int = 0)
    max_value = itr[1]
    if dims == 0
        @tturbo for i in 1:length(itr)
            max_value = ifelse(max_value > itr[i], max_value, itr[i])
        end
        return max_value
    else
        return Base.mapreduce(identity, max, itr; dims)
    end
end

function my_max(itr; dims)
    if dims == 0
        return Base.mapreduce(identity, max, itr)
    else
        return Base.mapreduce(identity, max, itr; dims)
    end
end

function my_max(f, itr; dims)
    if dims == 0
        return Base.mapreduce(f, max, itr)
    else
        return Base.mapreduce(f, max, itr; dims)
    end
end

function my_argmax(itr::Vector; dims::Int64 = 0)
    max_value = itr[1]
    pos = 1
    if dims == 0
        @turbo for i in 1:length(itr)
            value = max_value > itr[i]
            max_value = ifelse(value, max_value, itr[i])
            pos = ifelse(value, pos, i)        
        end
        return pos
    else
        return argmax(itr, dims)
    end
end

function my_argmax(f::Function, itr::Vector{T}; dims = 0) where T <: Real
    max_value = f(Int128(itr[1]))
    pos = 1
    if dims == 0
        @avxt warn_check_args=false for i in 1:length(itr)
            value = max_value >= f(Int128(itr[i]))
            max_value = ifelse(value, max_value, f(Int128(itr[i])))
            pos = ifelse(value, pos, i)        
        end
        return pos
    end
end

function my_argmax(f::Function, itr::Vector{T}; dims = 0) where T <: Integer
    max_value = f(Int128(itr[1]))
    pos = 1
    if dims == 0
        @avxt warn_check_args=false for i in 1:length(itr)
            value = max_value >= f(Int128(itr[i]))
            max_value = ifelse(value, max_value, f(Int128(itr[i])))
            pos = ifelse(value, pos, i)        
        end
        return pos
    end
end
