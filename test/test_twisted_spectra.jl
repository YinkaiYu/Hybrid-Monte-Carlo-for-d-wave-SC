using Test
using Random
using LinearAlgebra
using FFTW
using DwaveHMC

@inline test_site_index_xy(x::Int, y::Int, Lx::Int, Ly::Int) =
    (mod1(y, Ly) - 1) * Lx + mod1(x, Lx)

function test_set_hermitian_pair!(H::Matrix{ComplexF64},
                                  row::Int,
                                  col::Int,
                                  val::ComplexF64)
    if row <= col
        H[row, col] = val
    else
        H[col, row] = conj(val)
    end
    return nothing
end

function test_add_hop!(H::Matrix{ComplexF64},
                       N::Int,
                       i::Int,
                       j::Int,
                       h::ComplexF64)
    test_set_hermitian_pair!(H, i, j, h)
    test_set_hermitian_pair!(H, i + N, j + N, -conj(h))
    return nothing
end

function build_repeated_supercell_H!(H::Matrix{ComplexF64},
                                     p::ModelParameters,
                                     state::SimulationState,
                                     Ltw::Int)
    Lx_eff = p.Lx * Ltw
    Ly_eff = p.Ly * Ltw
    N_eff = Lx_eff * Ly_eff
    fill!(H, 0.0 + 0.0im)

    @inbounds for y in 1:Ly_eff, x in 1:Lx_eff
        i_eff = test_site_index_xy(x, y, Lx_eff, Ly_eff)
        i_small = test_site_index_xy(x, y, p.Lx, p.Ly)

        onsite = state.disorder_pot[i_small] - state.μ_eff
        H[i_eff, i_eff] = onsite
        H[i_eff + N_eff, i_eff + N_eff] = -onsite

        for (dx, dy, tt) in ((1, 0, p.t), (0, 1, p.t),
                             (1, 1, p.tp), (1, -1, p.tp))
            j_eff = test_site_index_xy(x + dx, y + dy, Lx_eff, Ly_eff)
            test_add_hop!(H, N_eff, i_eff, j_eff, -tt + 0.0im)
        end

        jx_eff = test_site_index_xy(x + 1, y, Lx_eff, Ly_eff)
        jy_eff = test_site_index_xy(x, y + 1, Lx_eff, Ly_eff)
        H[i_eff, jx_eff + N_eff] = state.Δ[i_small, 1]
        H[jx_eff, i_eff + N_eff] = state.Δ[i_small, 1]
        H[i_eff, jy_eff + N_eff] = state.Δ[i_small, 2]
        H[jy_eff, i_eff + N_eff] = state.Δ[i_small, 2]
    end

    return nothing
end

function repeated_supercell_spectra_reference(p::ModelParameters,
                                              state::SimulationState,
                                              cache::ComputeCache,
                                              Ltw::Int)
    Lx_eff = p.Lx * Ltw
    Ly_eff = p.Ly * Ltw
    N_eff = Lx_eff * Ly_eff
    dim_eff = 2 * N_eff
    nω = length(cache.dos_omega_grid)

    H_super = zeros(ComplexF64, dim_eff, dim_eff)
    build_repeated_supercell_H!(H_super, p, state, Ltw)
    vals, vecs = eigen(Hermitian(H_super, :U))

    dos = zeros(Float64, nω)
    A_k0 = zeros(Float64, Lx_eff, Ly_eff)
    A_MX_path = zeros(Float64, fld(Ly_eff, 2) + 1, nω)
    A_XG_path = zeros(Float64, fld(min(Lx_eff, Ly_eff), 2) + 1, nω)
    A_XG_node_patch = zeros(Float64, fld(min(Lx_eff, Ly_eff), 2) + 1, nω)
    lor_cache = zeros(Float64, nω)
    u_r = zeros(ComplexF64, Lx_eff, Ly_eff)
    u_k = similar(u_r)
    fft_plan = plan_fft(u_r)
    Ix_pi = div(Lx_eff, 2)

    @inbounds for n in 1:dim_eff
        En = vals[n]
        electron_weight = 0.0
        @simd for i in 1:N_eff
            electron_weight += abs2(vecs[i, n])
        end

        for iw in eachindex(cache.dos_omega_grid)
            lor_cache[iw] = DwaveHMC.lorentzian_spectra(cache.dos_omega_grid[iw] - En, p.η)
            dos[iw] += electron_weight * lor_cache[iw]
        end

        for y in 1:Ly_eff, x in 1:Lx_eff
            i_eff = test_site_index_xy(x, y, Lx_eff, Ly_eff)
            u_r[x, y] = vecs[i_eff, n]
        end
        mul!(u_k, fft_plan, u_r)

        weight_at_zero = DwaveHMC.lorentzian_spectra(-En, p.η)
        if weight_at_zero > 1e-6
            for y in 1:Ly_eff, x in 1:Lx_eff
                A_k0[x, y] += abs2(u_k[x, y]) * weight_at_zero
            end
        end

        for Iy in 0:fld(Ly_eff, 2)
            wk = abs2(u_k[Ix_pi + 1, Iy + 1]) / N_eff
            for iw in eachindex(cache.dos_omega_grid)
                A_MX_path[Iy + 1, iw] += wk * lor_cache[iw]
            end
        end

        for I in 0:fld(min(Lx_eff, Ly_eff), 2)
            wk = abs2(u_k[I + 1, I + 1]) / N_eff
            for iw in eachindex(cache.dos_omega_grid)
                A_XG_path[I + 1, iw] += wk * lor_cache[iw]
            end

            neighbor_indices = Set{Tuple{Int, Int}}()
            for dx in -1:1, dy in -1:1
                Ix = mod(I + dx, Lx_eff)
                Iy = mod(I + dy, Ly_eff)
                push!(neighbor_indices, (Ix, Iy))
            end
            patch_weight = 1.0 / length(neighbor_indices)
            for (Ix, Iy) in neighbor_indices
                wk_patch = patch_weight * abs2(u_k[Ix + 1, Iy + 1]) / N_eff
                for iw in eachindex(cache.dos_omega_grid)
                    A_XG_node_patch[I + 1, iw] += wk_patch * lor_cache[iw]
                end
            end
        end
    end

    dos ./= N_eff
    A_k0 ./= N_eff

    return (dos=dos,
            A_k_ω0=A_k0,
            A_MX_path=A_MX_path,
            A_XG_path=A_XG_path,
            A_XG_node_patch=A_XG_node_patch)
end

function setup_tbc_fixture(; Lx::Int=4, Ly::Int=4)
    Random.seed!(20260510)
    p = ModelParameters(Lx, Ly, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                        η=0.25, Δω=0.25, ω_max=2.0)
    state = initialize_state(p)
    cache = initialize_cache(p)

    @inbounds for i in 1:p.N
        state.disorder_pot[i] = 0.03 * sin(0.37 * i)
        state.Δ[i, 1] = 0.11 * cis(0.19 * i)
        state.Δ[i, 2] = -0.08 * cis(0.23 * i + 0.4)
    end

    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)
    return p, state, cache
end

@testset "TBC Hamiltonian builder" begin
    p, state, cache = setup_tbc_fixture()
    dim = 2 * p.N
    H = zeros(ComplexF64, dim, dim)

    for nx in 0:1, ny in 0:1
        qx = 2π * nx / 2
        qy = 2π * ny / 2
        DwaveHMC.build_tbc_H_BdG!(H, p, state, qx, qy)
        Hfull = Matrix(Hermitian(H, :U))
        @test isapprox(Hfull, Hfull'; atol=1e-12, rtol=1e-12)
        @test all(isfinite, real.(Hfull))
        @test all(isfinite, imag.(Hfull))
    end

    DwaveHMC.build_tbc_H_BdG!(H, p, state, 0.0, 0.0)
    @test isapprox(Matrix(Hermitian(H, :U)),
                   Matrix(Hermitian(cache.H_base, :U));
                   atol=1e-12, rtol=1e-12)
end

@testset "TBC boundary phase convention" begin
    p, state, _ = setup_tbc_fixture(Lx=3, Ly=3)
    dim = 2 * p.N
    H = zeros(ComplexF64, dim, dim)
    qx = 2π / 3
    qy = 2π / 3

    DwaveHMC.build_tbc_H_BdG!(H, p, state, qx, qy)
    Hfull = Matrix(Hermitian(H, :U))

    i_x = test_site_index_xy(p.Lx, 1, p.Lx, p.Ly)
    j_x = test_site_index_xy(1, 1, p.Lx, p.Ly)
    ph_x = cis(-qx)
    @test Hfull[i_x, j_x] ≈ -p.t * ph_x
    @test Hfull[i_x + p.N, j_x + p.N] ≈ p.t * ph_x

    i_diag = test_site_index_xy(p.Lx, p.Ly, p.Lx, p.Ly)
    j_diag = test_site_index_xy(1, 1, p.Lx, p.Ly)
    ph_diag = cis(-(qx + qy))
    @test Hfull[i_diag, j_diag] ≈ -p.tp * ph_diag
    @test Hfull[i_diag + p.N, j_diag + p.N] ≈ p.tp * ph_diag

    i_diag_xmy = test_site_index_xy(p.Lx, 2, p.Lx, p.Ly)
    j_diag_xmy = test_site_index_xy(1, 1, p.Lx, p.Ly)
    ph_diag_xmy = cis(-qx)
    @test Hfull[i_diag_xmy, j_diag_xmy] ≈ -p.tp * ph_diag_xmy
    @test Hfull[i_diag_xmy + p.N, j_diag_xmy + p.N] ≈ p.tp * ph_diag_xmy

    @test Hfull[i_x, j_x + p.N] ≈ state.Δ[i_x, 1] * ph_x
    @test Hfull[j_x, i_x + p.N] ≈ state.Δ[i_x, 1] * conj(ph_x)

    i_y = test_site_index_xy(1, p.Ly, p.Lx, p.Ly)
    j_y = test_site_index_xy(1, 1, p.Lx, p.Ly)
    ph_y = cis(-qy)
    @test Hfull[i_y, j_y + p.N] ≈ state.Δ[i_y, 2] * ph_y
    @test Hfull[j_y, i_y + p.N] ≈ state.Δ[i_y, 2] * conj(ph_y)
end

@testset "TBC sectors reproduce repeated supercell spectrum" begin
    p, state, _ = setup_tbc_fixture(Lx=3, Ly=3)
    dim_small = 2 * p.N

    for Ltw in (2, 3)
        dim_super = 2 * p.N * Ltw^2

        sector_vals = Float64[]
        H_sector = zeros(ComplexF64, dim_small, dim_small)
        for nx in 0:Ltw-1, ny in 0:Ltw-1
            qx = 2π * nx / Ltw
            qy = 2π * ny / Ltw
            DwaveHMC.build_tbc_H_BdG!(H_sector, p, state, qx, qy)
            append!(sector_vals, eigvals(Hermitian(H_sector, :U)))
        end

        H_super = zeros(ComplexF64, dim_super, dim_super)
        build_repeated_supercell_H!(H_super, p, state, Ltw)
        super_vals = eigvals(Hermitian(H_super, :U))

        sort!(sector_vals)
        sort!(super_vals)
        @test length(sector_vals) == length(super_vals)
        @test isapprox(sector_vals, super_vals; atol=1e-9, rtol=1e-9)
    end
end

@testset "Transport and spectra helper recomposition" begin
    p, _, cache = setup_tbc_fixture()

    combined = measure_transport_and_spectra(cache, p; reuse_buffers=false)
    transport = DwaveHMC.measure_transport_only(cache, p; reuse_buffers=false)
    spectra = DwaveHMC.measure_untwisted_spectra(cache, p; reuse_buffers=false)

    @test combined.superfluid_stiffness ≈ transport.superfluid_stiffness
    @test combined.dc_conductivity ≈ transport.dc_conductivity
    @test combined.ω_grid == transport.ω_grid
    @test combined.optical_conductivity ≈ transport.optical_conductivity

    @test combined.dos_ω_grid == spectra.dos_ω_grid
    @test combined.dos ≈ spectra.dos
    @test combined.dos_M ≈ spectra.dos_M
    @test combined.A_k_ω0 ≈ spectra.A_k_ω0
    @test combined.A_MX_path ≈ spectra.A_MX_path
    @test combined.A_XG_path ≈ spectra.A_XG_path
end

@testset "Twisted spectra measurement" begin
    p, state, cache = setup_tbc_fixture()

    base = measure_transport_and_spectra(cache, p; reuse_buffers=false)
    H_before = copy(cache.H_base)
    E_before = copy(cache.E_n)
    U_before = copy(cache.U)

    tw1 = measure_twisted_spectra(cache, p, state;
                                  Ltw=1,
                                  m_point_patch_half_width=0.0,
                                  reuse_buffers=false)

    @test cache.H_base == H_before
    @test cache.E_n == E_before
    @test cache.U == U_before

    @test size(tw1.A_k_ω0) == size(base.A_k_ω0)
    @test size(tw1.A_MX_path) == size(base.A_MX_path)
    @test size(tw1.A_XG_path) == size(base.A_XG_path)
    @test isapprox(tw1.dos, base.dos; atol=1e-8, rtol=1e-8)
    @test isapprox(tw1.dos_M, base.dos_M; atol=1e-8, rtol=1e-8)
    @test isapprox(tw1.A_k_ω0, base.A_k_ω0; atol=1e-8, rtol=1e-8)
    @test isapprox(tw1.A_MX_path, base.A_MX_path; atol=1e-8, rtol=1e-8)
    @test isapprox(tw1.A_XG_path, base.A_XG_path; atol=1e-8, rtol=1e-8)
    @test isapprox(tw1.dos_M_patch, tw1.dos_M; atol=1e-8, rtol=1e-8)

    tw2 = measure_twisted_spectra(cache, p, state; Ltw=2, reuse_buffers=false)
    @test size(tw2.A_k_ω0) == (p.Lx * 2, p.Ly * 2)
    @test size(tw2.A_MX_path, 1) == fld(p.Ly * 2, 2) + 1
    @test size(tw2.A_MX_path, 2) == length(cache.dos_omega_grid)
    @test size(tw2.A_XG_path, 1) == fld(min(p.Lx * 2, p.Ly * 2), 2) + 1
    @test size(tw2.A_XG_path, 2) == length(cache.dos_omega_grid)
    @test size(tw2.A_XG_node_patch) == size(tw2.A_XG_path)
    @test length(tw2.kx_grid) == p.Lx * 2
    @test length(tw2.ky_grid) == p.Ly * 2
    @test all(isfinite, tw2.dos)
    @test all(isfinite, tw2.dos_M)
    @test all(isfinite, tw2.dos_M_patch)
    @test all(isfinite, tw2.A_k_ω0)
    @test all(isfinite, tw2.A_MX_path)
    @test all(isfinite, tw2.A_XG_path)
    @test all(isfinite, tw2.A_XG_node_patch)
end

@testset "Twisted spectra repeated-supercell regression" begin
    p, state, cache = setup_tbc_fixture(Lx=3, Ly=3)
    Ltw = 2

    tw = measure_twisted_spectra(cache, p, state; Ltw=Ltw, reuse_buffers=false)
    ref = repeated_supercell_spectra_reference(p, state, cache, Ltw)

    @test isapprox(tw.dos, ref.dos; atol=1e-8, rtol=1e-8)
    @test isapprox(tw.A_k_ω0, ref.A_k_ω0; atol=1e-8, rtol=1e-8)
    @test isapprox(tw.A_MX_path, ref.A_MX_path; atol=1e-8, rtol=1e-8)
    @test isapprox(tw.A_XG_path, ref.A_XG_path; atol=1e-8, rtol=1e-8)
    @test isapprox(tw.A_XG_node_patch, ref.A_XG_node_patch; atol=1e-8, rtol=1e-8)
end

@testset "Twisted spectra odd effective dimensions" begin
    p, state, cache = setup_tbc_fixture(Lx=3, Ly=3)

    err = try
        measure_twisted_spectra(cache, p, state; Ltw=1, reuse_buffers=false)
        nothing
    catch e
        e
    end

    @test err isa ErrorException
    @test occursin("TBC spectra require even effective dimensions",
                   sprint(showerror, err))
end
