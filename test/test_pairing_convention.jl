using Test
using Random
using LinearAlgebra
using DwaveHMC

function local_pair_amplitude(cache, p, i::Int, dir::Int)
    N = p.N
    j = p.nn_table[i, dir]
    rho_1 = zero(ComplexF64)
    rho_2 = zero(ComplexF64)

    @inbounds for n in 1:(2N)
        f = 1 / (1 + exp(p.β * cache.E_n[n]))
        rho_1 += cache.U[i, n] * f * conj(cache.U[j+N, n])
        rho_2 += cache.U[j, n] * f * conj(cache.U[i+N, n])
    end

    return -rho_1 - rho_2
end

@testset "t-V pairing convention" begin
    @testset "boson action uses physical V with 2/V coefficient" begin
        p = ModelParameters(2, 2, 1.0, -0.35, -0.5, 0.0, 0.0, 3.0, 2.0, 1.0)
        state = initialize_state(p)
        cache = initialize_cache(p)
        fill!(state.π, 0)
        fill!(cache.E_n, 0)

        state.Δ .= ComplexF64[
            1.0+0.0im  0.0+0.5im
            0.2-0.3im -0.4+0.1im
            0.0+0.0im  0.7+0.2im
            -0.2+0.6im 0.3-0.8im
        ]

        expected_boson = (2p.β / p.V) * sum(abs2, state.Δ)
        @test compute_total_energy(cache, p, state) ≈ expected_boson
    end

    @testset "force and observables use g_pair = V/2" begin
        Random.seed!(20260519)
        p = ModelParameters(4, 4, 1.0, -0.35, -0.5, 0.0, 0.0, 6.0, 1.4, 1.0)
        state = initialize_state(p)
        cache = initialize_cache(p)

        init_static_H!(cache, p, state)
        update_H_BdG!(cache, p, state)
        diagonalize_H_BdG!(cache, p)
        compute_forces!(cache, p, state)
        obs = measure_observables(cache, p, state)

        g_pair = p.V / 2
        beta_over_g = p.β / g_pair
        max_force_err = 0.0
        sum_diff = 0.0
        sum_pair_global = 0.0 + 0.0im
        sum_pair_local = 0.0

        for i in 1:p.N
            P_x = local_pair_amplitude(cache, p, i, 1)
            P_y = local_pair_amplitude(cache, p, i, 2)

            expected_fx = -beta_over_g * (state.Δ[i, 1] - g_pair * P_x)
            expected_fy = -beta_over_g * (state.Δ[i, 2] - g_pair * P_y)
            max_force_err = max(max_force_err, abs(cache.forces[i, 1] - expected_fx))
            max_force_err = max(max_force_err, abs(cache.forces[i, 2] - expected_fy))

            sum_diff += 0.5 * (abs(state.Δ[i, 1] - g_pair * P_x) +
                               abs(state.Δ[i, 2] - g_pair * P_y))
            term = 0.5 * g_pair * (P_x - P_y)
            sum_pair_global += term
            sum_pair_local += abs(term)
        end

        @test max_force_err < 1e-10
        @test obs.Δ_diff ≈ sum_diff / p.N atol=1e-12 rtol=1e-12
        @test obs.Δ_pair ≈ abs(sum_pair_global / p.N) atol=1e-12 rtol=1e-12
        @test obs.Δ_localpair ≈ sum_pair_local / p.N atol=1e-12 rtol=1e-12
    end

    @testset "recommended HMC step uses g_pair = V/2" begin
        β = 8.0
        V = 1.6
        mass = 1.25
        Nt = 10

        expected_dt = 2π * sqrt(mass * (V / 2) / β) / (2Nt)
        @test calc_optimal_dt(β, V, mass, Nt) ≈ expected_dt
    end
end
