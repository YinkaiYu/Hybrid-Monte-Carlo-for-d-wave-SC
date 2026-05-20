using Test
using LinearAlgebra
using Random
using JLD2
using DwaveHMC

function magnetic_test_parameters(; Lx=4, Ly=4, n_flux_sc=0,
                                  n_vortices=nothing,
                                  boundary_condition=:periodic)
    return ModelParameters(Lx, Ly, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                           η=0.25, Δω=0.25, ω_max=2.0,
                           n_flux_sc=n_flux_sc,
                           n_vortices=n_vortices,
                           boundary_condition=boundary_condition)
end

function site_index(x::Int, y::Int, Lx::Int)
    return (y - 1) * Lx + x
end

@testset "Magnetic field validation" begin
    p0 = magnetic_test_parameters()
    mag0 = DwaveHMC.build_magnetic_cache(p0)
    @test mag0 isa DwaveHMC.NoFieldCache
    @test DwaveHMC.link_phase(mag0, 1, 1, 0) == 1.0 + 0.0im
    @test DwaveHMC.link_phase(mag0, 1, 0, 1) == 1.0 + 0.0im

    p0_magnetic_pbc = magnetic_test_parameters(boundary_condition=:magnetic_pbc)
    mag0_magnetic_pbc = DwaveHMC.build_magnetic_cache(p0_magnetic_pbc)
    @test mag0_magnetic_pbc isa DwaveHMC.NoFieldCache

    p_plus = magnetic_test_parameters(n_flux_sc=2, boundary_condition=:magnetic_pbc)
    mag_plus = DwaveHMC.build_magnetic_cache(p_plus)
    @test mag_plus isa DwaveHMC.LandauGaugeCache
    @test mag_plus.n_flux_sc == 2
    @test mag_plus.flux_density_sc == 2 / 16
    @test mag_plus.plaquette_phase ≈ π * 2 / 16

    p_minus = magnetic_test_parameters(n_flux_sc=-2, boundary_condition=:magnetic_pbc)
    mag_minus = DwaveHMC.build_magnetic_cache(p_minus)
    for i in 1:p_plus.N, (dx, dy) in ((1, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (0, -1))
        @test DwaveHMC.link_phase(mag_minus, i, dx, dy) ≈ conj(DwaveHMC.link_phase(mag_plus, i, dx, dy))
    end

    p_alias = magnetic_test_parameters(n_vortices=-2, boundary_condition=:magnetic_pbc)
    @test p_alias.n_vortices == p_alias.n_flux_sc == -2

    @test_throws ErrorException magnetic_test_parameters(n_flux_sc=1, boundary_condition=:magnetic_pbc)
    @test_throws ErrorException magnetic_test_parameters(n_flux_sc=2, boundary_condition=:periodic)
    @test_throws ErrorException magnetic_test_parameters(n_flux_sc=2, n_vortices=-2,
                                                         boundary_condition=:magnetic_pbc)
    @test_throws ErrorException magnetic_test_parameters(Lx=1, Ly=4)
    @test_throws ErrorException magnetic_test_parameters(Lx=4, Ly=1)
end

@testset "Landau gauge link phases" begin
    p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2, boundary_condition=:magnetic_pbc)
    mag = DwaveHMC.build_magnetic_cache(p)
    α = mag.plaquette_phase
    @test DwaveHMC.plaquette_phase(mag, 1, 1) ≈ α

    for y0 in 0:(p.Ly - 1), x0 in 0:(p.Lx - 1)
        i = site_index(x0 + 1, y0 + 1, p.Lx)
        ux = DwaveHMC.link_phase(mag, i, 1, 0)
        iy = site_index(mod(x0 + 1, p.Lx) + 1, y0 + 1, p.Lx)
        uy_right = DwaveHMC.link_phase(mag, iy, 0, 1)
        ix_up = site_index(x0 + 1, mod(y0 + 1, p.Ly) + 1, p.Lx)
        ux_up_reverse = DwaveHMC.link_phase(mag, ix_up, 1, 0)
        uy = DwaveHMC.link_phase(mag, i, 0, 1)
        wilson = ux * uy_right * conj(ux_up_reverse) * conj(uy)
        @test wilson ≈ cis(α)
    end

    i_boundary = site_index(p.Lx, 2, p.Lx) # zero-based (Lx-1, 1)
    @test DwaveHMC.link_phase(mag, i_boundary, 1, 0) ≈ cis(-α * p.Lx * 1)
    @test DwaveHMC.link_phase(mag, i_boundary, 1, 1) ≈
          cis(α * (p.Lx - 0.5)) * cis(-α * p.Lx * 2)
    @test DwaveHMC.link_phase(mag, i_boundary, 1, -1) ≈
          cis(-α * (p.Lx - 0.5)) * cis(-α * p.Lx * 0)
end

@testset "Landau gauge reverse links" begin
    p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2, boundary_condition=:magnetic_pbc)
    mag = DwaveHMC.build_magnetic_cache(p)
    for y0 in 0:(p.Ly - 1), x0 in 0:(p.Lx - 1), (dx, dy) in ((1, 0), (0, 1), (1, 1), (1, -1))
        i = site_index(x0 + 1, y0 + 1, p.Lx)
        xj = mod(x0 + dx, p.Lx)
        yj = mod(y0 + dy, p.Ly)
        j = site_index(xj + 1, yj + 1, p.Lx)
        @test DwaveHMC.link_phase(mag, i, dx, dy) ≈ conj(DwaveHMC.link_phase(mag, j, -dx, -dy))
    end
end

@testset "Magnetic Hamiltonian hopping phases" begin
    p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2, boundary_condition=:magnetic_pbc)
    state = initialize_state(p)
    cache = initialize_cache(p)
    fill!(state.Δ, 0.0 + 0.0im)

    init_static_H!(cache, p, state)
    H = cache.H_base
    Hfull = Matrix(Hermitian(H, :U))
    N = p.N
    i = site_index(p.Lx, 2, p.Lx)
    jx = p.nn_table[i, 1]
    jxpy = p.nnn_table[i, 1]
    jxmy = p.nnn_table[i, 4]

    ph_x = DwaveHMC.link_phase(cache.magnetic, i, 1, 0)
    ph_xpy = DwaveHMC.link_phase(cache.magnetic, i, 1, 1)
    ph_xmy = DwaveHMC.link_phase(cache.magnetic, i, 1, -1)
    @test Hfull[i, jx] ≈ -p.t * ph_x
    @test Hfull[i, jxpy] ≈ -p.tp * ph_xpy
    @test Hfull[i, jxmy] ≈ -p.tp * ph_xmy
    @test Hfull[i + N, jx + N] ≈ p.t * conj(ph_x)
    @test Hfull ≈ Hfull'
end

function random_finite_field_state(p)
    Random.seed!(20260521)
    state = initialize_state(p)
    state.disorder_pot .= randn(p.N) .* 0.05
    state.Δ .= (randn(p.N, 2) .+ im .* randn(p.N, 2)) .* 0.03
    return state
end

@testset "Kubo operators match Hamiltonian derivatives" begin
    for (Lx, Ly) in ((4, 4), (2, 2))
        p = magnetic_test_parameters(Lx=Lx, Ly=Ly, n_flux_sc=2, boundary_condition=:magnetic_pbc)
        state = random_finite_field_state(p)
        cache = initialize_cache(p)
        init_static_H!(cache, p, state)
        update_H_BdG!(cache, p, state)

        dim = 2 * p.N
        Hplus = zeros(ComplexF64, dim, dim)
        Hminus = zeros(ComplexF64, dim, dim)
        H0 = zeros(ComplexF64, dim, dim)
        eps = 1.0e-6

        for qy in (0.0, 2π / p.Ly)
            DwaveHMC.build_probe_H_BdG!(Hplus, cache, p, state; λ=eps, qx=0.0, qy=qy)
            DwaveHMC.build_probe_H_BdG!(Hminus, cache, p, state; λ=-eps, qx=0.0, qy=qy)
            DwaveHMC.build_probe_H_BdG!(H0, cache, p, state; λ=0.0, qx=0.0, qy=qy)

            J_fd = (Matrix(Hermitian(Hplus, :U)) - Matrix(Hermitian(Hminus, :U))) ./ (2eps)
            K_fd = (Matrix(Hermitian(Hplus, :U)) + Matrix(Hermitian(Hminus, :U)) -
                    2 .* Matrix(Hermitian(H0, :U))) ./ (eps^2)

            J_an = Matrix(DwaveHMC.probe_current_operator_matrix(cache, p; qx=0.0, qy=qy))
            K_an = DwaveHMC.diamagnetic_operator_matrix(cache, p; qx=0.0, qy=qy)

            @test norm(J_an - J_fd) / max(norm(J_fd), 1.0) < 1.0e-6
            @test norm(K_an - K_fd) / max(norm(K_fd), 1.0) < 5.0e-4
        end
    end
end

@testset "Finite-field transport is finite and diagnostic curvature is available" begin
    p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2, boundary_condition=:magnetic_pbc)
    state = random_finite_field_state(p)
    cache = initialize_cache(p)
    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)

    res = DwaveHMC.measure_transport_only(cache, p; eta_values=[p.η], reuse_buffers=false)
    @test isfinite(res.superfluid_stiffness)
    @test isfinite(res.dc_conductivity)
    @test length(res.optical_conductivity) == length(cache.omega_grid)

    K0 = DwaveHMC.diamagnetic_operator_matrix(cache, p; qx=0.0, qy=0.0)
    dia_from_matrix = 0.0
    for n in 1:(2 * p.N)
        if cache.E_n[n] > 0
            ψn = @view cache.U[:, n]
            dia_from_matrix += -real(dot(ψn, K0 * ψn)) * tanh(0.5 * p.β * cache.E_n[n]) / p.N
        end
    end
    @test DwaveHMC.diamagnetic_expectation_x(cache, p) ≈ dia_from_matrix atol=1.0e-10

    diag = DwaveHMC.measure_full_curvature_diagnostic(cache, p, state; Ax=1.0e-4, qy=2π / p.Ly)
    @test isfinite(diag.rho_full_curvature)
    @test isfinite(diag.lambda_diag)
end

@testset "Gauge-covariant pair bond output" begin
    mktempdir() do out_dir
        p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2, boundary_condition=:magnetic_pbc)
        Random.seed!(20260521)
        run_simulation(p, out_dir;
                       n_therm=0,
                       n_measure=2,
                       Nt_measure=1,
                       measure_transport_freq=2,
                       bin_size=1,
                       write_gauge_pair_bonds_freq=1,
                       verbose=false)
        jldopen(joinpath(out_dir, "pairing_scatter.jld2"), "r") do file
            @test haskey(file, "sweep_1/delta_bond_landau_gauge_covariant")
            @test haskey(file, "sweep_1/pair_bond_landau_gauge_covariant")
            @test size(file["sweep_1/delta_bond_landau_gauge_covariant"]) == (p.N, 2)
            @test size(file["sweep_1/pair_bond_landau_gauge_covariant"]) == (p.N, 2)
        end
    end
end
