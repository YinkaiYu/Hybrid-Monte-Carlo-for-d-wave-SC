using Test
using Random
using LinearAlgebra
using DwaveHMC

@testset "Core model/cache/HMC smoke" begin
    Random.seed!(1234)

    p = ModelParameters(4, 4, 1.0, -0.35, -0.5, 0.0, 0.0, 5.0, 1.0, 1.0;
                        η=0.25, Δω=0.25, ω_max=2.0)
    state = initialize_state(p)
    cache = initialize_cache(p)

    @test p.N == 16
    @test size(state.Δ) == (p.N, 2)
    @test size(state.π) == (p.N, 2)
    @test size(cache.H_base) == (2p.N, 2p.N)
    @test size(cache.U) == (2p.N, 2p.N)
    @test length(cache.E_n) == 2p.N
    @test size(cache.ak_map) == (p.Lx, p.Ly)

    init_static_H!(cache, p, state)
    H_static = Matrix(Hermitian(cache.H_base, :U))
    @test isapprox(H_static, H_static'; atol=1e-12, rtol=1e-12)

    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)
    @test issorted(cache.E_n)
    @test all(isfinite, cache.E_n)
    @test isfinite(compute_total_energy(cache, p, state))

    compute_forces!(cache, p, state)
    @test all(z -> isfinite(real(z)) && isfinite(imag(z)), cache.forces)

    Random.seed!(5678)
    acc, dH = hmc_sweep!(cache, p, state; Nt=1, dt=1.0e-3)
    @test acc isa Bool
    @test isfinite(dH)
    @test all(z -> isfinite(real(z)) && isfinite(imag(z)), state.Δ)
end
