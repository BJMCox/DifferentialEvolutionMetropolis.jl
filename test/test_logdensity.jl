@testset "Untransformed LogDensityProblems target" begin
    target = IsotropicNormalModel([-5.0, 5.0])
    wrapped = AbstractMCMC.LogDensityModel(target)
    sampler = setup_de_update()
    options = (;
        n_chains = 4, num_warmup = 10, silent = true, progress = false,
        chain_type = DifferentialEvolutionOutput,
    )
    reference = sample(backwards_compat_rng(123), wrapped, sampler, 10; options...)
    result = sample(backwards_compat_rng(123), target, sampler, 10; options...)
    @test result.samples == reference.samples
    @test result.ld == reference.ld
    @test all(isfinite, result.ld)
end

mutable struct CountingLogDensityModel
    count::Int
    nan_at::Int
    value::Float64
end
LogDensityProblems.dimension(::CountingLogDensityModel) = 1
LogDensityProblems.capabilities(::CountingLogDensityModel) = LogDensityProblems.LogDensityOrder{0}()
function LogDensityProblems.logdensity(model::CountingLogDensityModel, x)
    model.count += 1
    return model.count == model.nan_at ? NaN : model.value
end

struct IntervalLogDensityModel
    outside::Float64
end
LogDensityProblems.dimension(::IntervalLogDensityModel) = 1
LogDensityProblems.capabilities(::IntervalLogDensityModel) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.logdensity(model::IntervalLogDensityModel, x) = 0 <= x[1] <= 1 ? 0.0 : model.outside

@testset "Non-finite log densities" begin
    options = (; n_chains = 4, memory = false, silent = true, progress = false, chain_type = DifferentialEvolutionOutput)
    sampler = setup_de_update()

    @testset "$stage" for (stage, nan_at, n_tries) in (
            ("initial", 1, 1), ("proposal", 5, 1),
            ("forward trial", 5, 2), ("reference trial", 13, 2),
        )
        model = AbstractMCMC.LogDensityModel(CountingLogDensityModel(0, nan_at, 0.0))
        @test_throws DomainError sample(
            backwards_compat_rng(42), model, sampler, 2; n_tries = n_tries, options...
        )
    end

    @testset "undefined ratio at $value" for value in (-Inf, Inf)
        model = AbstractMCMC.LogDensityModel(CountingLogDensityModel(0, 0, value))
        output = sample(backwards_compat_rng(42), model, sampler, 2; options...)
        @test output.samples[2, :, :] == output.samples[1, :, :]
        @test all(==(value), output.ld)
    end

    @testset "zero-density support" begin
        model = AbstractMCMC.LogDensityModel(IntervalLogDensityModel(-Inf))
        output = sample(
            backwards_compat_rng(42), model, sampler, 100;
            initial_position = [[0.1], [0.3], [0.7], [0.9]], options...
        )
        @test all(x -> 0 <= x <= 1, output.samples)
        @test all(iszero, output.ld)
        recovered = sample(
            backwards_compat_rng(42), model, sampler, 100;
            initial_position = [[-1.0], [0.3], [0.7], [0.9]], options...
        )
        @test all(iszero, recovered.ld[end, :])
    end

    @testset "infinite-density region" begin
        model = AbstractMCMC.LogDensityModel(IntervalLogDensityModel(Inf))
        output = sample(
            backwards_compat_rng(42), model, sampler, 100;
            initial_position = [[0.1], [0.3], [0.7], [0.9]], options...
        )
        @test all(iszero, output.ld[1, :])
        @test all(==(Inf), output.ld[end, :])
        for densities in eachcol(output.ld)
            first_infinite = findfirst(isinf, densities)
            @test all(==(Inf), densities[first_infinite:end])
        end
    end
end
