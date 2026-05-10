using Test
using Random
using LinearAlgebra
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
