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

@testset "Starting position dimensions" begin
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel(zeros(2)))
    options = (; n_chains = 4, silent = true, progress = false, chain_type = DifferentialEvolutionOutput)
    @test_throws ErrorException sample(
        backwards_compat_rng(42), model, setup_de_update(), 1;
        initial_position = [[1.0, 2.0], [1.0], [2.0, 3.0], [4.0, 5.0]],
        memory = false, options...
    )
    @test_throws ErrorException sample(
        backwards_compat_rng(42), model, setup_de_update(), 1;
        initial_position = [zeros(2), ones(2), -ones(2), [1.0, -1.0], [1.0]],
        memory = true, options...
    )
    expanded = sample(
        backwards_compat_rng(42), model, setup_de_update(), 1;
        initial_position = [zeros(2)], memory = false, options...
    )
    @test size(expanded.samples) == (1, 4, 2)
    @test all(isfinite, expanded.ld)
end
