using LinearAlgebra
using FFTW

struct TwistedSpectraResult
    dos_ω_grid::Vector{Float64}
    dos::Vector{Float64}
    dos_AN::Vector{Float64}
    dos_AN_patch::Vector{Float64}
    A_k_ω0::Matrix{Float64}
    A_kpath::Matrix{Float64}
    kx_grid::Vector{Float64}
    ky_grid::Vector{Float64}
    kpath_kx::Float64
    kpath_ky::Vector{Float64}
    Ltw::Int
    antinode_patch_half_width::Float64
end

@inline lorentzian_spectra(x::Float64, η::Float64) =
    (1.0 / π) * (η / (x * x + η * η))

@inline site_index_xy(x::Int, y::Int, Lx::Int, Ly::Int) =
    (mod1(y, Ly) - 1) * Lx + mod1(x, Lx)

@inline function boundary_winding(x::Int, y::Int, dx::Int, dy::Int,
                                  Lx::Int, Ly::Int)
    wx = x + dx > Lx ? 1 : (x + dx < 1 ? -1 : 0)
    wy = y + dy > Ly ? 1 : (y + dy < 1 ? -1 : 0)
    return wx, wy
end

@inline function tbc_phase(x::Int, y::Int, dx::Int, dy::Int,
                           Lx::Int, Ly::Int,
                           qx::Float64, qy::Float64)
    wx, wy = boundary_winding(x, y, dx, dy, Lx, Ly)
    return cis(-(qx * wx + qy * wy))
end

@inline function set_tbc_hermitian_pair!(H::Matrix{ComplexF64},
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

@inline function add_tbc_hop!(H::Matrix{ComplexF64},
                              N::Int,
                              i::Int,
                              j::Int,
                              h::ComplexF64)
    set_tbc_hermitian_pair!(H, i, j, h)
    set_tbc_hermitian_pair!(H, i + N, j + N, -h)
    return nothing
end

@inline function effective_index_to_twist_fft(I::Int, L::Int, Ltw::Int)
    n = mod(-I, Ltw)
    m = div(I + n, Ltw)
    return mod(n, Ltw), mod(m, L)
end

@inline function twist_fft_to_effective_index(m::Int, n::Int, L::Int, Ltw::Int)
    return mod(m * Ltw - n, L * Ltw)
end

function effective_k_grid(L::Int, Ltw::Int)
    Leff = L * Ltw
    vals = Vector{Float64}(undef, Leff)
    @inbounds for I in 0:Leff-1
        k = 2π * I / Leff
        vals[I + 1] = k > π ? k - 2π : k
    end
    return vals
end

@inline function periodic_k_distance(a::Real, b::Real)
    d = abs(a - b)
    return min(d, 2π - d)
end

function antinode_patch_mask(kx_grid::Vector{Float64},
                             ky_grid::Vector{Float64},
                             half_width::Float64)
    mask = falses(length(kx_grid), length(ky_grid))
    count = 0
    @inbounds for ix in eachindex(kx_grid), iy in eachindex(ky_grid)
        kx = kx_grid[ix]
        ky = ky_grid[iy]
        near_pi_0 = periodic_k_distance(kx, π) <= half_width &&
                    periodic_k_distance(ky, 0.0) <= half_width
        near_0_pi = periodic_k_distance(kx, 0.0) <= half_width &&
                    periodic_k_distance(ky, π) <= half_width
        if near_pi_0 || near_0_pi
            mask[ix, iy] = true
            count += 1
        end
    end
    return mask, count
end

"""
    build_tbc_H_BdG!(H, p, state, qx, qy)

Build the spectra twisted-boundary-condition BdG matrix in caller-provided
storage `H`. The matrix is filled in the upper-triangle representation consumed
by `Hermitian(H, :U)`.
"""
function build_tbc_H_BdG!(H::Matrix{ComplexF64},
                          p::ModelParameters,
                          state::SimulationState,
                          qx::Float64,
                          qy::Float64)
    N = p.N
    Lx = p.Lx
    Ly = p.Ly
    fill!(H, 0.0 + 0.0im)

    @inbounds for y in 1:Ly, x in 1:Lx
        i = site_index_xy(x, y, Lx, Ly)

        onsite = state.disorder_pot[i] - state.μ_eff
        H[i, i] = onsite
        H[i + N, i + N] = -onsite

        for (dx, dy, tt) in ((1, 0, p.t), (0, 1, p.t),
                             (1, 1, p.tp), (1, -1, p.tp))
            j = site_index_xy(x + dx, y + dy, Lx, Ly)
            ph = tbc_phase(x, y, dx, dy, Lx, Ly, qx, qy)
            add_tbc_hop!(H, N, i, j, -tt * ph)
        end

        jx = site_index_xy(x + 1, y, Lx, Ly)
        phx = tbc_phase(x, y, 1, 0, Lx, Ly, qx, qy)
        valx = state.Δ[i, 1] * phx
        H[i, jx + N] = valx
        H[jx, i + N] = state.Δ[i, 1] * conj(phx)

        jy = site_index_xy(x, y + 1, Lx, Ly)
        phy = tbc_phase(x, y, 0, 1, Lx, Ly, qx, qy)
        valy = state.Δ[i, 2] * phy
        H[i, jy + N] = valy
        H[jy, i + N] = state.Δ[i, 2] * conj(phy)
    end

    return nothing
end

function exact_effective_point(L_eff::Int, fraction::Float64, name::String)
    I = round(Int, fraction * L_eff)
    if !isapprox(I, fraction * L_eff; atol=1e-12, rtol=0.0)
        error("Effective grid cannot represent exact $name point")
    end
    return mod(I, L_eff)
end

function tbc_kpath_metadata(Lx::Int, Ly::Int, Ltw::Int)
    Lx_eff = Lx * Ltw
    Ly_eff = Ly * Ltw
    Ix_pi = exact_effective_point(Lx_eff, 0.5, "kx=π")
    ky_count = fld(Ly_eff, 2) + 1
    kx = 2π * Ix_pi / Lx_eff
    kx = kx > π ? kx - 2π : kx
    ky = Vector{Float64}(undef, ky_count)
    @inbounds for idx in 1:ky_count
        I = idx - 1
        ky[idx] = 2π * I / Ly_eff
    end
    return Ix_pi, kx, ky
end

"""
    measure_twisted_spectra(cache, p, state; Ltw=2, antinode_patch_half_width=π / max(p.Lx, p.Ly), reuse_buffers=false)

Measure DOS and spectral weights on the effective `Ltw`-refined momentum grid
by summing spectra twisted-boundary-condition sectors.
"""
function measure_twisted_spectra(cache::ComputeCache,
                                 p::ModelParameters,
                                 state::SimulationState;
                                 Ltw::Int=2,
                                 antinode_patch_half_width::Float64=π / max(p.Lx, p.Ly),
                                 reuse_buffers::Bool=false)
    Ltw <= 0 && error("Ltw must be positive")
    antinode_patch_half_width < 0 && error("antinode_patch_half_width must be nonnegative")

    N = p.N
    dim = 2 * N
    Lx = p.Lx
    Ly = p.Ly
    Lx_eff = Lx * Ltw
    Ly_eff = Ly * Ltw
    if isodd(Lx_eff) || isodd(Ly_eff)
        error("TBC spectra require even effective dimensions to represent exact antinodes and kx=π path")
    end

    dos_ω_grid = reuse_buffers ? cache.dos_omega_grid : copy(cache.dos_omega_grid)
    nω = length(dos_ω_grid)
    dos_vals = zeros(Float64, nω)
    dos_AN_vals = zeros(Float64, nω)
    dos_AN_patch_vals = zeros(Float64, nω)
    A_k0 = zeros(Float64, Lx_eff, Ly_eff)
    _, kpath_kx, kpath_ky = tbc_kpath_metadata(Lx, Ly, Ltw)
    A_kpath = zeros(Float64, length(kpath_ky), nω)
    lor_cache = zeros(Float64, nω)

    kx_grid = effective_k_grid(Lx, Ltw)
    ky_grid = effective_k_grid(Ly, Ltw)
    patch_mask, patch_count = antinode_patch_mask(kx_grid, ky_grid,
                                                  antinode_patch_half_width)
    patch_count == 0 && error("Antinodal patch contains no effective momentum points")

    Ix_pi = exact_effective_point(Lx_eff, 0.5, "kx=π")
    Iy_pi = exact_effective_point(Ly_eff, 0.5, "ky=π")
    Ix_zero = 0
    Iy_zero = 0

    nx_pi, mx_pi = effective_index_to_twist_fft(Ix_pi, Lx, Ltw)
    ny_zero, my_zero = effective_index_to_twist_fft(Iy_zero, Ly, Ltw)
    nx_zero, mx_zero = effective_index_to_twist_fft(Ix_zero, Lx, Ltw)
    ny_pi, my_pi = effective_index_to_twist_fft(Iy_pi, Ly, Ltw)

    Htw = zeros(ComplexF64, dim, dim)
    Uwork = similar(Htw)
    Etw = zeros(Float64, dim)

    @inbounds for nx in 0:Ltw-1, ny in 0:Ltw-1
        qx = 2π * nx / Ltw
        qy = 2π * ny / Ltw
        has_pi_0_sector = nx == nx_pi && ny == ny_zero
        has_0_pi_sector = nx == nx_zero && ny == ny_pi

        build_tbc_H_BdG!(Htw, p, state, qx, qy)
        copyto!(Uwork, Htw)
        vals, vecs = eigen!(Hermitian(Uwork, :U))
        copyto!(Etw, vals)

        for n in 1:dim
            En = Etw[n]
            w_n = 0.0
            @simd for i in 1:N
                w_n += abs2(vecs[i, n])
            end

            for iw in eachindex(dos_ω_grid)
                lor_cache[iw] = lorentzian_spectra(dos_ω_grid[iw] - En, p.η)
                dos_vals[iw] += w_n * lor_cache[iw]
            end

            for y in 1:Ly, x in 1:Lx
                i = (y - 1) * Lx + x
                ph = cis(qx * (x - 1) / Lx + qy * (y - 1) / Ly)
                cache.u_r_cache[x, y] = vecs[i, n] * ph
            end
            mul!(cache.u_k_cache, cache.fft_plan, cache.u_r_cache)

            exact_weight = 0.0
            if has_pi_0_sector
                exact_weight += 0.5 * abs2(cache.u_k_cache[mx_pi + 1, my_zero + 1]) / N
            end
            if has_0_pi_sector
                exact_weight += 0.5 * abs2(cache.u_k_cache[mx_zero + 1, my_pi + 1]) / N
            end
            if has_pi_0_sector || has_0_pi_sector
                for iw in eachindex(dos_ω_grid)
                    dos_AN_vals[iw] += exact_weight * lor_cache[iw]
                end
            end

            patch_weight = 0.0
            weight_at_zero = lorentzian_spectra(-En, p.η)
            for my in 0:Ly-1, mx in 0:Lx-1
                Ix = twist_fft_to_effective_index(mx, nx, Lx, Ltw)
                Iy = twist_fft_to_effective_index(my, ny, Ly, Ltw)
                wk = abs2(cache.u_k_cache[mx + 1, my + 1]) / N

                if patch_mask[Ix + 1, Iy + 1]
                    patch_weight += wk
                end

                if weight_at_zero > 1e-6
                    A_k0[Ix + 1, Iy + 1] += abs2(cache.u_k_cache[mx + 1, my + 1]) * weight_at_zero
                end
            end
            patch_weight /= patch_count
            for iw in eachindex(dos_ω_grid)
                dos_AN_patch_vals[iw] += patch_weight * lor_cache[iw]
            end

            if nx == nx_pi
                for my in 0:Ly-1
                    Iy = twist_fft_to_effective_index(my, ny, Ly, Ltw)
                    if Iy <= fld(Ly_eff, 2)
                        wk = abs2(cache.u_k_cache[mx_pi + 1, my + 1]) / N
                        path_idx = Iy + 1
                        for iw in eachindex(dos_ω_grid)
                            A_kpath[path_idx, iw] += wk * lor_cache[iw]
                        end
                    end
                end
            end
        end
    end

    dos_vals ./= (N * Ltw^2)
    A_k0 ./= N

    return TwistedSpectraResult(
        dos_ω_grid,
        reuse_buffers ? dos_vals : copy(dos_vals),
        reuse_buffers ? dos_AN_vals : copy(dos_AN_vals),
        reuse_buffers ? dos_AN_patch_vals : copy(dos_AN_patch_vals),
        reuse_buffers ? A_k0 : copy(A_k0),
        reuse_buffers ? A_kpath : copy(A_kpath),
        kx_grid,
        ky_grid,
        kpath_kx,
        kpath_ky,
        Ltw,
        antinode_patch_half_width,
    )
end
