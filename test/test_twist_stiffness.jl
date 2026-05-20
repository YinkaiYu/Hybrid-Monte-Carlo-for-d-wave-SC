using Test
using LinearAlgebra
using Random
using DwaveHMC

function manual_kubo_qy_skip_diagonal(cache::ComputeCache, p::ModelParameters)
    N = p.N
    dim = 2 * N
    β = p.β
    U = cache.U
    E = cache.E_n
    f = cache.fermi_factors

    @inbounds @simd for n in 1:dim
        f[n] = 1.0 / (exp(β * E[n]) + 1.0)
    end

    val_dia = 0.0
    @inbounds for n in 1:dim
        En = E[n]
        if En > 0
            w_n = 0.0
            @simd for i in 1:N
                j_x = p.nn_table[i, 1]
                j_xpy = p.nnn_table[i, 1]
                j_xmy = p.nnn_table[i, 4]
                w_n += p.t * 2.0 * real(U[i+N, n] * conj(U[j_x+N, n]) -
                                         conj(U[i, n]) * U[j_x, n])
                w_n += p.tp * 2.0 * real(U[i+N, n] * conj(U[j_xpy+N, n]) -
                                          conj(U[i, n]) * U[j_xpy, n])
                w_n += p.tp * 2.0 * real(U[i+N, n] * conj(U[j_xmy+N, n]) -
                                          conj(U[i, n]) * U[j_xmy, n])
            end
            val_dia += w_n * tanh(0.5 * β * En) / N
        end
    end

    DwaveHMC.build_current_operator!(cache, p; qx=0.0, qy=2π / p.Ly, store=:qy)
    mul!(cache.temp_JU, cache.Jx_sparse_qy, U)
    mul!(cache.J_mn, U', cache.temp_JU)

    Lambda_xx = 0.0
    @inbounds for n in 1:dim
        for m in 1:dim
            m == n && continue
            diff_E = E[m] - E[n]
            ratio = if abs(diff_E) < 1.0e-8
                β * f[n] * (1.0 - f[n])
            else
                (f[n] - f[m]) / diff_E
            end
            Lambda_xx += ratio * abs2(cache.J_mn[n, m])
        end
    end

    return val_dia - Lambda_xx / N
end

@testset "Finite magnetic field rejects direct twist stiffness diagnostics" begin
    p = ModelParameters(4, 4, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                        n_flux_sc=2,
                        boundary_condition=:magnetic_pbc)
    state = initialize_state(p)
    cache = initialize_cache(p)
    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)

    @test_throws ErrorException DwaveHMC.measure_twist_stiffness(cache, p, state; Ax=1.0e-3)
    @test_throws ErrorException DwaveHMC.measure_twist_stiffness_qy(cache, p, state;
                                                                    Ax=1.0e-3,
                                                                    qy=2π / p.Ly)

    H_twist = zeros(ComplexF64, 2 * p.N, 2 * p.N)
    @test_throws ErrorException DwaveHMC.build_twisted_H_BdG!(H_twist, cache, p, state, 1.0e-3)
    @test_throws ErrorException DwaveHMC.build_twisted_H_BdG_qy!(H_twist, cache, p, state,
                                                                 1.0e-3, 2π / p.Ly, 0.0)
end

@testset "Twist stiffness estimator" begin
    @test !(:H_twist_base in fieldnames(ComputeCache))
    @test !(:H_twist_work in fieldnames(ComputeCache))
    @test !(:E_twist in fieldnames(ComputeCache))

    Lx, Ly = 4, 4
    p = ModelParameters(Lx, Ly, 1.0, -0.35, -0.5, 0.0, 0.0, 20.0, 1.0, 1.0)
    state = initialize_state(p)
    cache = initialize_cache(p)

    fill!(state.Δ, 0.0 + 0.0im)
    for i in 1:p.N
        state.Δ[i, 1] = 0.2
        state.Δ[i, 2] = -0.2
    end

    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)

    H_twist = zeros(ComplexF64, 2 * p.N, 2 * p.N)
    DwaveHMC.build_twisted_H_BdG!(H_twist, cache, p, state, 0.0)
    H0 = Matrix(Hermitian(cache.H_base, :U))
    Ht0 = Matrix(Hermitian(H_twist, :U))
    @test isapprox(H0, Ht0; atol=1e-12, rtol=1e-12)

    DwaveHMC.build_twisted_H_BdG!(H_twist, cache, p, state, 1.0e-3)
    Ht = Matrix(Hermitian(H_twist, :U))
    @test isapprox(Ht, Ht'; atol=1e-12, rtol=1e-12)

    H_before = copy(cache.H_base)
    E_before = copy(cache.E_n)
    U_before = copy(cache.U)
    res = DwaveHMC.measure_twist_stiffness(cache, p, state; Ax=1.0e-3)

    @test res.Ax == 1.0e-3
    @test isfinite(res.S0)
    @test isfinite(res.Splus)
    @test isfinite(res.Sminus)
    @test isfinite(res.s1)
    @test isfinite(res.s2)
    @test isfinite(res.rho_curvature_config)
    @test isapprox(res.rho_curvature_config, res.s2 / (p.β * p.N); atol=1e-12, rtol=1e-12)
    @test cache.H_base == H_before
    @test cache.E_n == E_before
    @test cache.U == U_before

    spec = measure_transport_and_spectra(cache, p; reuse_buffers=true)
    @test isfinite(spec.superfluid_stiffness)
    @test isapprox(spec.superfluid_stiffness,
                   manual_kubo_qy_skip_diagonal(cache, p); atol=1e-10, rtol=1e-10)

    for i in 1:p.N
        state.Δ[i, 1] = 0.10 * cis(0.37 * i)
        state.Δ[i, 2] = -0.07 * cis(0.23 * i + 0.41)
    end
    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)

    complex_twist = DwaveHMC.measure_twist_stiffness(cache, p, state; Ax=1.0e-3)
    complex_spec = measure_transport_and_spectra(cache, p; reuse_buffers=true)
    @test isfinite(complex_spec.superfluid_stiffness)
    @test isfinite(complex_twist.rho_curvature_config)
    @test isapprox(complex_spec.superfluid_stiffness,
                   manual_kubo_qy_skip_diagonal(cache, p); atol=1e-10, rtol=1e-10)

    Random.seed!(1)
    random_state = initialize_state(p)
    init_static_H!(cache, p, random_state)
    update_H_BdG!(cache, p, random_state)
    diagonalize_H_BdG!(cache, p)

    random_twist = DwaveHMC.measure_twist_stiffness(cache, p, random_state; Ax=1.0e-3)
    random_spec = measure_transport_and_spectra(cache, p; reuse_buffers=true)
    @test isfinite(random_twist.rho_curvature_config)
    @test isapprox(random_spec.superfluid_stiffness,
                   manual_kubo_qy_skip_diagonal(cache, p); atol=1e-10, rtol=1e-10)

    p_random = ModelParameters(6, 6, 1.0, -0.35, -0.8, 0.0, 0.0, 8.0, 1.0, 1.0;
                               η=8.0 / 36.0, Δω=4.0 / 36.0, ω_max=3.0)
    Random.seed!(1)
    random_state_6 = initialize_state(p_random)
    random_cache_6 = initialize_cache(p_random)
    init_static_H!(random_cache_6, p_random, random_state_6)
    update_H_BdG!(random_cache_6, p_random, random_state_6)
    diagonalize_H_BdG!(random_cache_6, p_random)

    random_twist_6 = DwaveHMC.measure_twist_stiffness(random_cache_6, p_random,
                                                      random_state_6; Ax=1.0e-3)
    random_spec_6 = measure_transport_and_spectra(random_cache_6, p_random; reuse_buffers=true)
    @test isfinite(random_twist_6.rho_curvature_config)
    @test isapprox(random_spec_6.superfluid_stiffness,
                   manual_kubo_qy_skip_diagonal(random_cache_6, p_random);
                   atol=1e-10, rtol=1e-10)

    random_twist_qy_6 = DwaveHMC.measure_twist_stiffness_qy(random_cache_6, p_random,
                                                            random_state_6;
                                                            Ax=1.0e-3,
                                                            qy=2π / p_random.Ly)
    @test isfinite(random_twist_qy_6.rho_curvature_avg)
    @test isfinite(random_twist_qy_6.diag_correction)
    @test isfinite(random_twist_qy_6.rho_offdiag_corrected)
    @test random_twist_qy_6.diag_correction > 0
    @test isapprox(random_twist_qy_6.rho_offdiag_corrected,
                   random_twist_qy_6.rho_curvature_avg + random_twist_qy_6.diag_correction;
                   atol=1e-12, rtol=1e-12)
    @test abs(random_spec_6.superfluid_stiffness -
              random_twist_qy_6.rho_curvature_avg) > 5e-6
    @test isapprox(random_spec_6.superfluid_stiffness,
                   random_twist_qy_6.rho_offdiag_corrected; atol=2e-6, rtol=1e-3)
end
