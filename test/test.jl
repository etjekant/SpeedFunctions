using SpeedFunctions
using Test
using Pkg
using LoopVectorization

@testset "my_abs checks" begin
    y = -1*rand(1000)
    @test my_abs(y) == abs.(y)

    y = -1*rand(1000, 1000)
    @test my_abs(y) == abs.(y)

    y = rand(Int64, 100000)
    @test my_abs(y) == abs.(y)

    y = rand(Int64, 1000, 1000)
    @test my_abs(y) == abs.(y)
    
    y2 = zeros(Int64, size(y))
    @test abs.(y) == my_abs!(y2, y)

    @test my_abs(-1) == abs(-1)
end


@testset "my_sum checks" begin
    y = rand(Float64, 100000)
    value1 = sum(y)
    value2 = my_sum(y)
    @test my_abs(value1 - value2) < my_abs(value1) / 1_000_000_000_000_000

    y = rand(Float64, 100, 1000)
    value1 = sum(y)
    value2 = my_sum(y)
    @test my_abs(value1 - value2) < my_abs(value1) / 1_000_000_000_000_000

    y = rand(Int64, 100000)
    value1 = sum(y)
    value2 = my_sum(y)
    @test my_abs(value1 - value2) < my_abs(value1) / 1_000_000_000_000_000

    y = rand(Int64, 100, 1000)
    value1 = sum(y)
    value2 = my_sum(y)
    @test my_abs(value1 - value2) < my_abs(value1) / 1_000_000_000_000_000

    y = rand(Int64, 1000, 100);
    value1 = sum(abs2, y)
    value2 = my_sum(abs2, y)
    @test my_abs(value1 - value2) < my_abs(value1) / 1_000_000_000_000_000

    y = rand(Float64, 1000, 100);
    value1 = sum(abs2, y)
    value2 = my_sum(abs2, y)
    @test my_abs(value1 - value2) < my_abs(value1) / 1_000_000_000_000_000
end

@testset "my_maximum test" begin
    y = rand(Int64, 1000, 1000);
    @test maximum(y) == my_maximum(y)

    y = rand(Int64, 100000);
    @test maximum(y) == my_maximum(y)
    
    y = rand(Int64, 1000, 1000);
    @test maximum(abs2, y) == my_maximum(abs2, y)
    
    y = rand(Float64, 1000, 1000);
    @test maximum(y) == my_maximum(y)

    y = rand(Float64, 100000);
    @test maximum(y) == my_maximum(y)
    
    y = -1* rand(Float64, 1000, 1000);
    @test maximum(abs2, y) == my_maximum(abs2, y)
end


@testset "my_findmax test" begin
    y = rand(Int64, 1000, 1000);
    @test findmax(y) == my_findmax(y)

    y = rand(Int64, 100000);
    @test findmax(y) == my_findmax(y)
    
    y = rand(Int64, 100000);
    @test findmax(abs, y) == my_findmax(abs, y)

    y = rand(Float64, 1000, 1000);
    @test findmax(y) == my_findmax(y)

    y = rand(Float64, 100000);
    @test findmax(y) == my_findmax(y)
end

@testset "my_argmax test" begin
    y = rand(Int64, 1000, 1000);
    @test argmax(y) == my_argmax(y)

    y = rand(Int64, 100000);
    @test argmax(y) == my_argmax(y)
    
    y = -1*rand(Float64, 1000, 1000);
    @test argmax(y, dims=1) == my_argmax(y, dims=1)

    y = rand(Float64, 100000);
    @test argmax(y) == my_argmax(y)
end

@testset "my_minimum test" begin
    y = rand(Int64, 1000, 1000);
    @test minimum(y) == my_minimum(y)

    y = rand(Int64, 100000);
    @test minimum(y) == my_minimum(y)
    
    y = rand(Int64, 1000, 1000);
    @test minimum(abs2, y) == my_minimum(abs2, y)
    
    y = rand(Float64, 1000, 1000);
    @test minimum(y) == my_minimum(y)

    y = rand(Float64, 100000);
    @test minimum(y) == my_minimum(y)
    
    y = -1* rand(Float64, 1000, 1000);
    @test minimum(abs2, y) == my_minimum(abs2, y)
end

@testset "my_findmin test" begin
    y = rand(Int64, 1000, 1000);
    @test findmin(y) == my_findmin(y)

    y = rand(Int64, 100000);
    @test findmin(y) == my_findmin(y)
    
    y = rand(Float64, 1000, 1000);
    @test findmin(y) == my_findmin(y)

    y = rand(Float64, 100000);
    @test findmin(y) == my_findmin(y)

    y = rand(Float64, 1000, 1000)
    @test findmin(abs, y) == my_findmin(abs, y)
end

@testset "my_argmmin test" begin
    y = rand(Int64, 1000, 1000);
    @test argmin(y) == my_argmin(y)

    y = rand(Int64, 100000);
    @test argmin(y) == my_argmin(y)
    
    y = -1*rand(Float64, 1000, 1000);
    @test argmin(y, dims=1) == my_argmin(y, dims=1)

    y = rand(Float64, 100000);
    @test argmin(y) == my_argmin(y)
end
