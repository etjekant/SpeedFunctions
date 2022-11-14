function zeros_via_malloc(::Type{T}, dims::Integer...) where T
    ptr = Ptr{T}(ccall(:malloc, Ptr{Cvoid}, (Csize_t,), prod(dims) * sizeof(T)))
    return unsafe_wrap(Array{T}, ptr, dims; own=true)
 end

 function zeros_via_calloc(::Type{T}, dims::Integer...) where T
    ptr = Ptr{T}(Libc.calloc(prod(dims), sizeof(T)))
    return unsafe_wrap(Array{T}, ptr, dims; own=true)
 end

 function col_row_calc(pos, data_size)
    position1 = pos / data_size
    col = Int(ceil(position1))
    row = position1 - floor(position1)
    row = Int(round(row * data_size))
    value = row == 0
    row = value * data_size + !value * row 
    return CartesianIndex(row, col)
end