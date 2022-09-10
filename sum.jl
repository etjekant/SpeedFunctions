using LoopVectorization

function my_sum(f::Function, itr::Matrix{T}; dims::Int64 = 0, init::T = 0.0) where T <: Real
    if dims == 0
        @tturbo for i in 1:length(itr)
            init += f(itr[i])
        end
        return init
    else
        return Base._mapreduce_dim(f, Base.add_sum, init, itr, dims)
    end
end   

function my_sum(itr::Matrix{T}; dims::Int64 = 0, init::Float64 = 0.0) where T <: Real
    if dims == 0
        @tturbo for i in 1:length(itr)
            init += itr[i]
        end
        return init
    else
        return Base._mapreduce_dim(identity, Base.add_sum, init, itr, dims)
    end
end    

function my_sum(itr::Vector{T}; init::Float64 = 0.0) where T <: Real
    @tturbo for i in 1:length(itr)
        init += itr[i]
    end
    return init
end

function my_sum(f::Function, itr::Matrix{T}; dims::Int64 = 0, init::Int64 = 0) where T <: Integer
    if dims == 0
        @tturbo for i in 1:length(itr)
            init += f(itr[i])
        end
        return init
    else
        return Base._mapreduce_dim(f, Base.add_sum, init, itr, dims)
    end
end 

function my_sum(itr::Matrix{T}; dims::Int64 = 0, init::Int64 = 0) where T <: Integer
    if dims == 0
        @tturbo for i in 1:length(itr)
            init += itr[i]
        end
        return init
    else
        return Base._mapreduce_dim(identity, Base.add_sum, init, itr, dims)
    end
end  

function my_sum(itr::Vector{T}; init::Int64 = 0) where T <: Integer
    @tturbo for i in 1:length(itr)
        init += itr[i]
    end
    return init
end

function my_sum(f::Function, itr; kw...)
    return sum(f, itr; kw...)
end

function my_sum(itr; kw...)
    return sum(itr; kw...)
end
