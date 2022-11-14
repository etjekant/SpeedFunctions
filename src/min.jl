
function my_minimum(itr::Vector{T}) where T
    min_value::T = itr[1]
    @avx for i in 1:length(itr)
        min_value = ifelse(min_value < itr[i], min_value, itr[i])
    end
    return min_value
end

function my_minimum(f::Function, itr::Vector{T}) where T
    min_value::T = itr[1]
    @avx for i in 1:length(itr)
        min_value = ifelse(min_value < f(itr[i]), min_value, f(itr[i]))
    end
    return min_value
end

function my_minimum(itr::Matrix{T}; dims::Int = 0) where T
    if dims == 0
        min_value = itr[1]
        @avx for i in 2:length(itr)
            min_value = ifelse(min_value < itr[i], min_value, itr[i])
        end
        return min_value
    elseif dims == 1
        itr_size = size(itr)
        min_values = zeros(T, 1, itr_size[2])

        @inbounds for j in 1:itr_size[2]
            min_value = itr[1, j]
            @avx for i in 1:itr_size[1]
                min_value = ifelse(min_value < itr[i, j], min_value, itr[i, j])
            end
            min_values[j] = min_value
        end
        return min_values
    else
        return Base.mapreduce(identity, min, itr; dims)
    end
end

function my_minimum(f::Function, itr::Matrix{T}; dims::Int=0) where T
    min_value::T = f(itr[1])
    if dims == 0
        @avx for i in 1:length(itr)
            min_value = ifelse(min_value < f(itr[i]), min_value, f(itr[i]))
        end
        return min_value
    elseif dims == 1
        itr_size = size(itr)
        min_values = zeros(T, 1, itr_size[2])

        @inbounds for j in 1:itr_size[2]
            min_value = f(itr[1, j])
            @avx for i in 1:itr_size[1]
                min_value = ifelse(min_value < f(itr[i, j]), min_value, f(itr[i, j]))
            end
            min_values[j] = min_value
        end
        return min_values
    else
        return Base.mapreduce(f, min, itr; dims)
    end
end

function my_minimum(itr; dims::Int=0)
    if dims == 0
        return Base.mapreduce(identity, min, itr)
    else
        return Base.mapreduce(identity, min, itr; dims)
    end
end

function my_minimum(f,itr; dims::Int=0)
    if dims == 0
        return Base.mapreduce(f, min, itr)
    else
        return Base.mapreduce(f, min, itr; dims)
    end
end




function my_findmin(itr::Vector{T}) where T
    min_value::T = itr[1]
    pos = 1

    @avx for i in 1:length(itr)
        value = min_value < itr[i]
        min_value = ifelse(value, min_value, itr[i])
        pos = ifelse(value, pos, i)        
    end
    return min_value, pos

end

function my_findmin(f::Function, itr::Vector{T}) where T
    min_value::T = f(itr[1])
    pos = 1

    @avx warn_check_args=false for i in 1:length(itr)
        value = min_value < f(itr[i])
        min_value = ifelse(value, min_value, f(itr[i]))
        pos = ifelse(value, pos, i)        
    end
    return min_value, pos

end

function my_findmin(itr::Matrix{T}; dims::Int64 = 0) where T
    min_value::T = itr[1]
    pos::Int64 = 1
    if dims == 0

        @avx for i in 1:length(itr)
            value = min_value < itr[i]
            min_value = ifelse(value, min_value, itr[i])
            pos = ifelse(value, pos, i)        
        end

        return min_value, col_row_calc(pos, size(itr, 1))
    elseif dims == 1

        itr_size = size(itr)
        min_values = zeros(T, 1, itr_size[2])
        pos_value = zeros(CartesianIndex{2}, 1, itr_size[2])

        @inbounds for j in 1:itr_size[2]
            min_value = itr[1, j]
            @avx for i in 1:itr_size[1]
                value = min_value < itr[i, j]
                min_value = ifelse(value, min_value, itr[i, j])
                pos = ifelse(value, pos, i)   
            end

            min_values[j] = min_value
            pos_value[j] = CartesianIndex(pos, j)
        end
        return min_values, pos_value

    else 
        return findmin(itr, dims=dims)
    end
end

function my_findmin(f::Function, itr::Matrix{T}; dims = 0) where T
    min_value::T = f(itr[1])
    pos::Int64 = 1

    if dims == 0

        @avxt warn_check_args=false for i in 1:length(itr)
            value = min_value < f(itr[i])

            min_value = ifelse(value, min_value, f(itr[i]))
            pos = ifelse(value, pos, i)   
        end

        return min_value, col_row_calc(pos, size(itr, 1))

    elseif dims == 1
        itr_size = size(itr)

        min_values = zeros(T, 1, itr_size[2])
        pos_value = zeros(CartesianIndex{2}, 1, itr_size[2])

        @inbounds for j in 1:itr_size[2]
            min_value = itr[1, j]
            @avx for i in 1:itr_size[1]
                value = min_value < f(itr[i, j])
                min_value = ifelse(value, min_value, f(itr[i, j]))
                pos = ifelse(value, pos, i)   
            end
            min_values[j] = min_value
            pos_value[j] = CartesianIndex(pos, j)
        end

        return min_values, pos_value 

    else
        itr_size = size(itr)

        min_values = zeros(T, itr_size[1], 1)
        pos_value = zeros(CartesianIndex{2}, itr_size[1], 1)

        @inbounds for i in 1:itr_size[1]
            min_value = itr[1, i]
            @avx for j in 1:itr_size[2]
                value = min_value < f(itr[i, j])
                min_value = ifelse(value, min_value, f(itr[i, j]))
                pos = ifelse(value, pos, i)   
            end
            min_values[i] = min_value
            pos_value[i] = CartesianIndex(pos, i)
        end

        return min_values, pos_value 
    end
end

function my_findmin(itr, kw...)
    return findmin(itr, kw...)
end




function my_argmin(itr::Vector{T}) where T
    min_value::T = itr[1]
    pos = 1

    @avx for i in 1:length(itr)
        value = min_value < itr[i]
        min_value = ifelse(value, min_value, itr[i])
        pos = ifelse(value, pos, i)        
    end
    return pos

end

function my_argmin(itr::Matrix{T}; dims::Int64 = 0) where T
    min_value::T = itr[1]
    pos::Int64 = 1
    if dims == 0

        @avx for i in 1:length(itr)
            value = min_value < itr[i]
            min_value = ifelse(value, min_value, itr[i])
            pos = ifelse(value, pos, i)        
        end

        return col_row_calc(pos, size(itr, 1))
    else 
        return argmin(itr, dims=dims)
    end
end

function my_argmin(itr, kw...)
    return argmin(itr, kw...)
end

function my_argmin(f, itr, kw...)
    return argmin(f, itr, kw...)
end