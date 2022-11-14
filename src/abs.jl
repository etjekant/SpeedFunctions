


function my_abs!(a2::Vector, a::Vector)
    @avxt for i in 1:length(a)
        a2[i] = signbit(a[i]) ? -a[i] : a[i]
    end
    return a2
end

function my_abs!(a2::Matrix, a::Matrix)
    @avxt for i in 1:length(a)
        a2[i] = signbit(a[i]) ? -a[i] : a[i]
    end
    return a2
end

function my_abs(a::Vector{T}) where T <: Real
    a2 = zeros_via_malloc(T, size(a)[1])
    @avxt for i in 1:length(a)
        a2[i] = signbit(a[i]) ? -a[i] : a[i]
    end
    return a2
end

function my_abs(a::Matrix{T}) where T <: Real
    a2 = zeros_via_malloc(T, size(a)[1], size(a)[2])
    @avxt for i in 1:length(a)
        a2[i] = signbit(a[i]) ? -a[i] : a[i]
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
    return a2
end

function my_abs(a)
    return signbit(a) ? -a : a
end
