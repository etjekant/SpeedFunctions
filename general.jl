function zeros_via_malloc(::Type{T}, dims::Integer...) where T
    ptr = Ptr{T}(Libc.malloc(prod(dims) * sizeof(T)))
    return unsafe_wrap(Array{T}, ptr, dims; own=true)
 end