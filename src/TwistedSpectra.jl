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
    set_tbc_hermitian_pair!(H, i + N, j + N, -conj(h))
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

@inline function periodic_k_distance(a::Float64, b::Float64)
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
        H[jx, i + N] = valx

        jy = site_index_xy(x, y + 1, Lx, Ly)
        phy = tbc_phase(x, y, 0, 1, Lx, Ly, qx, qy)
        valy = state.Δ[i, 2] * phy
        H[i, jy + N] = valy
        H[jy, i + N] = valy
    end

    return nothing
end
