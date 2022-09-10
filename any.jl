function my_any(y::Vector{Bool})
    @inbounds for i in 1:length(y)
        y[i] && return true
    end
    return false
end

function my_any(f::Function, y::VecOrMat{T}) where T <: Real
    @inbounds for i in 1:length(y)
        f(y[i]) && return true
    end
    return false
end

function my_any(f::Function, y::VecOrMat{T}) where T <: Integer
    @inbounds for i in 1:length(y)
        f(y[i]) && return true
    end
    return false
end

function my_any(y)
    return any(y)
end