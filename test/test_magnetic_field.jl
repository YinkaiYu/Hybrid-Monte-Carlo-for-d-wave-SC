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
