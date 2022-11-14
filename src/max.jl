
function my_maximum(itr::Vector{T}) where T
    max_value::T = itr[1]
    @tturbo for i in 1:length(itr)
        max_value = ifelse(max_value > itr[i], max_value, itr[i])
    end
    return max_value
end

function my_maximum(f::Function, itr::Vector{T}) where T
    max_value::T = itr[1]
    @tturbo for i in 1:length(itr)
        max_value = ifelse(max_value > f(itr[i]), max_value, f(itr[i]))
    end
    return max_value
end

function my_maximum(itr::Matrix{T}; dims::Int = 0) where T
    if dims == 0
        max_value = itr[1]
        @tturbo for i in 2:length(itr)
            max_value = ifelse(max_value > itr[i], max_value, itr[i])
        end
        return max_value
    elseif dims == 1
        itr_size = size(itr)
        max_values = zeros(T, 1, itr_size[2])

        @inbounds for j in 1:itr_size[2]
            max_value = itr[1, j]
            @turbo for i in 1:itr_size[1]
                max_value = ifelse(max_value > itr[i, j], max_value, itr[i, j])
            end
            max_values[j] = max_value
        end
        return max_values
    else
        return Base.mapreduce(identity, max, itr; dims)
    end
end

function my_maximum(f::Function, itr::Matrix{T}; dims::Int=0) where T
    max_value::T = itr[1]
    if dims == 0
        @tturbo for i in 1:length(itr)
            max_value = ifelse(max_value > f(itr[i]), max_value, f(itr[i]))
        end
        return max_value
    elseif dims == 1
        itr_size = size(itr)
        max_values = zeros(T, 1, itr_size[2])

        @inbounds for j in 1:itr_size[2]
            max_value = f(itr[1, j])
            @avx for i in 1:itr_size[1]
                max_value = ifelse(max_value > f(itr[i, j]), max_value, f(itr[i, j]))
            end
            max_values[j] = max_value
        end
        return max_values
    else
        return Base.mapreduce(f, max, itr; dims)
    end
end

function my_maximum(itr; dims::Int=0)
    if dims == 0
        return Base.mapreduce(identity, max, itr)
    else
        return Base.mapreduce(identity, max, itr; dims)
    end
end

function my_maximum(f,itr; dims::Int=0)
    if dims == 0
        return Base.mapreduce(f, max, itr)
    else
        return Base.mapreduce(f, max, itr; dims)
    end
end



function my_findmax(itr::Vector{T}) where T
    max_value::T = itr[1]
    pos = 1

    @turbo for i in 1:length(itr)
        value = max_value > itr[i]
        max_value = ifelse(value, max_value, itr[i])
        pos = ifelse(value, pos, i)        
    end
    return max_value, pos

end

function my_findmax(f::Function, itr::Vector{T}) where T
    max_value::T = f(itr[1])
    pos = 1

    @avx warn_check_args=false for i in 1:length(itr)
        value = max_value > f(itr[i])
        max_value = ifelse(value, max_value, f(itr[i]))
        pos = ifelse(value, pos, i)        
    end
    return max_value, pos

end

function my_findmax(itr::Matrix{T}; dims::Int64 = 0) where T
    max_value::T = itr[1]
    pos::Int64 = 1
    if dims == 0

        @avx for i in 1:length(itr)
            value = max_value > itr[i]
            max_value = ifelse(value, max_value, itr[i])
            pos = ifelse(value, pos, i)        
        end

        return max_value, col_row_calc(pos, size(itr, 1))
    elseif dims == 1

        itr_size = size(itr)
        max_values = zeros(T, 1, itr_size[2])
        pos_value = zeros(CartesianIndex{2}, 1, itr_size[2])

        @inbounds for j in 1:itr_size[2]
            max_value = itr[1, j]
            @avx for i in 1:itr_size[1]
                value = max_value > itr[i, j]
                max_value = ifelse(value, max_value, itr[i, j])
                pos = ifelse(value, pos, i)   
            end

            max_values[j] = max_value
            pos_value[j] = CartesianIndex(pos, j)
        end
        return max_values, pos_value

    else 
        return findmax(itr, dims=dims)
    end
end

function my_findmax(f::Function, itr::Matrix{T}; dims = 0) where T
    max_value::T = f(itr[1])
    pos::Int64 = 1

    if dims == 0

        @avxt warn_check_args=false for i in 1:length(itr)
            value = max_value > f(itr[i])

            max_value = ifelse(value, max_value, f(itr[i]))
            pos = ifelse(value, pos, i)   
        end

        return max_value, col_row_calc(pos, size(itr, 1))

    elseif dims == 1
        itr_size = size(itr)

        max_values = zeros(T, 1, itr_size[2])
        pos_value = zeros(CartesianIndex{2}, 1, itr_size[2])

        @inbounds for j in 1:itr_size[2]
            max_value = itr[1, j]
            @turbo for i in 1:itr_size[1]
                value = max_value > f(itr[i, j])
                max_value = ifelse(value, max_value, f(itr[i, j]))
                pos = ifelse(value, pos, i)   
            end
            max_values[j] = max_value
            pos_value[j] = CartesianIndex(pos, j)
        end

        return max_values, pos_value 

    else
        itr_size = size(itr)

        max_values = zeros(T, itr_size[1], 1)
        pos_value = zeros(CartesianIndex{2}, itr_size[1], 1)

        @inbounds for i in 1:itr_size[1]
            max_value = itr[1, i]
            @turbo for j in 1:itr_size[2]
                value = max_value > f(itr[i, j])
                max_value = ifelse(value, max_value, f(itr[i, j]))
                pos = ifelse(value, pos, i)   
            end
            max_values[i] = max_value
            pos_value[i] = CartesianIndex(pos, i)
        end

        return max_values, pos_value 
    end
end

function my_findmax(itr, kw...)
    return findmax(itr, kw...)
end




function my_argmax(itr::Vector{T}) where T
    max_value::T = itr[1]
    pos = 1

    @turbo for i in 1:length(itr)
        value = max_value > itr[i]
        max_value = ifelse(value, max_value, itr[i])
        pos = ifelse(value, pos, i)        
    end
    return pos

end

function my_argmax(itr::Matrix{T}; dims::Int64 = 0) where T
    max_value::T = itr[1]
    pos::Int64 = 1
    if dims == 0
        @turbo for i in 1:length(itr)
            value = max_value > itr[i]
            max_value = ifelse(value, max_value, itr[i])
            pos = ifelse(value, pos, i)        
        end

        return col_row_calc(pos, size(itr, 1))
    else 
        return argmax(itr, dims=dims)
    end
end

function my_argmax(itr, kw...)
    return argmax(itr, kw...)
end
function my_argmax(f, itr, kw...)
    return argmax(f, itr, kw...)
end