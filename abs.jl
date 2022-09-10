using LoopVectorization
include("./general.jl")

function my_abs!(a2::Vector{T}, a::Vector{T}) where T <: Real
    @avxt for i in 1:length(a)
        a2[i] = signbit(a[i]) < 0 ? -a[i] : a[i]
    end
    return a2
end

function my_abs!(a2::Matrix{T}, a::Matrix{T}) where T <: Real
    @avxt for i in 1:length(a)
        a2[i] = signbit(a[i]) < 0 ? -a[i] : a[i]
    end
    return a2
end

function my_abs!(a2::Vector{T}, a::Vector{T}) where T <: Integer
    @avxt for i in 1:length(a)
        a2[i] = signbit(a[i]) < 0 ? -a[i] : a[i]
    end
    return a2
end

function my_abs!(a2::Matrix{T}, a::Matrix{T}) where T <: Integer
    @avxt for i in 1:length(a)
        a2[i] = signbit(a[i]) < 0 ? -a[i] : a[i]
    end
    return a
end

function my_abs(a::Vector{T}) where T <: Real
    a2 = zeros_via_malloc(T, size(a)[1])
    @avxt for i in 1:length(a)
        a2[i] = a[i] < 0 ? -a[i] : a[i]
    end
    return a2
end

function my_abs(a::Matrix{T}) where T <: Real
    a2 = zeros_via_malloc(T, size(a)[1], size(a)[2])
    @avxt for i in 1:length(a)
        a2[i] = a[i] < 0 ? -a[i] : a[i]
    end
    return a2
end

function my_abs(a::Vector{T}) where T <: Integer
    a2 = zeros_via_malloc(T, size(a)[1])
    @avxt for i in 1:length(a)
        a2[i] = signbit(a[i]) ? -a[i] : a[i]
    end
    return a2
end

function my_abs(a::Matrix{T}) where T <: Integer
    a2 = zeros_via_malloc(T, size(a)[1], size(a)[2])
    @avxt for i in 1:length(a)
        a2[i] = signbit(a[i]) ? -a[i] : a[i]
    end
    return a
end

function my_abs(a::Real)
    return signbit(a) ? -a : a
end

function my_abs(a::Number)
    return signbit(a) ? -a : a
end

