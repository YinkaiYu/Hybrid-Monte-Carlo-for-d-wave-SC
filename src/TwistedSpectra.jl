using LinearAlgebra
using FFTW

struct TwistedSpectraResult
    dos_ω_grid::Vector{Float64}
    dos::Vector{Float64}
    dos_M::Vector{Float64}
    dos_M_patch::Vector{Float64}
    ldos_ω0::Vector{Float64}
    A_k_ω0::Matrix{Float64}
    A_MX_path::Matrix{Float64}
    A_XG_path::Matrix{Float64}
    A_XG_node_patch::Matrix{Float64}
    dos_eta::Matrix{Float64}
    dos_M_eta::Matrix{Float64}
    dos_M_patch_eta::Matrix{Float64}
    ldos_ω0_eta::Matrix{Float64}
    A_k_ω0_eta::Array{Float64, 3}
    A_MX_path_eta::Array{Float64, 3}
    A_XG_path_eta::Array{Float64, 3}
    A_XG_node_patch_eta::Array{Float64, 3}
    ldos_ω::Union{Nothing,Matrix{Float64}}
    ldos_ω_eta::Union{Nothing,Array{Float64, 3}}
    kx_grid::Vector{Float64}
    ky_grid::Vector{Float64}
    mx_path_kx::Float64
    mx_path_ky::Vector{Float64}
    xg_path_kx::Vector{Float64}
    xg_path_ky::Vector{Float64}
    Ltw::Int
    m_point_patch_half_width::Float64
    spectra_eta::Float64
    spectra_delta_omega::Float64
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

function m_point_patch_mask(kx_grid::Vector{Float64},
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

function tbc_mx_path_metadata(Lx::Int, Ly::Int, Ltw::Int)
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

function tbc_xg_path_metadata(Lx::Int, Ly::Int, Ltw::Int)
    Lx_eff = Lx * Ltw
    Ly_eff = Ly * Ltw
    if isodd(Lx_eff) || isodd(Ly_eff)
        error("TBC spectra require even effective dimensions to represent exact X-Gamma path")
    end

    path_len = fld(min(Lx_eff, Ly_eff), 2) + 1
    kx = Vector{Float64}(undef, path_len)
    ky = Vector{Float64}(undef, path_len)
    @inbounds for idx in 1:path_len
        I = idx - 1
        kx[idx] = 2π * I / Lx_eff
        ky[idx] = 2π * I / Ly_eff
    end
    return kx, ky
end

function spectra_dos_grid(p::ModelParameters, spectra_delta_omega::Float64)
    spectra_delta_omega > 0 || error("spectra_delta_omega must be positive")
    return collect(-p.ω_max:spectra_delta_omega:p.ω_max)
end

"""
    measure_twisted_spectra(cache, p, state; Ltw=2, m_point_patch_half_width=π / max(p.Lx, p.Ly), reuse_buffers=false)

Measure DOS and spectral weights on the effective `Ltw`-refined momentum grid
by summing spectra twisted-boundary-condition sectors.
"""
function measure_twisted_spectra(cache::ComputeCache,
                                 p::ModelParameters,
                                 state::SimulationState;
                                 Ltw::Int=2,
                                 m_point_patch_half_width::Float64=π / max(p.Lx, p.Ly),
                                 spectra_eta::Float64=p.η,
                                 spectra_delta_omega::Float64=p.Δω,
                                 eta_values::AbstractVector{<:Real}=Float64[spectra_eta],
                                 reuse_buffers::Bool=false,
                                 write_ldos_spectrum::Bool=false)
    p.n_flux_sc == 0 ||
        error("measure_twisted_spectra is not supported for finite magnetic field (n_flux_sc=$(p.n_flux_sc))")
    Ltw <= 0 && error("Ltw must be positive")
    m_point_patch_half_width < 0 && error("m_point_patch_half_width must be nonnegative")
    spectra_eta > 0 || error("spectra_eta must be positive")
    eta_vals = Float64.(eta_values)
    nη = length(eta_vals)
    nη > 0 || error("eta_values must be non-empty")
    all(isfinite, eta_vals) || error("eta_values must be finite")
    all(>(0.0), eta_vals) || error("eta_values must be positive")
    isapprox(eta_vals[1], spectra_eta; atol=ETA_FACTOR_ATOL, rtol=0.0) ||
        error("eta_values[1] must match spectra_eta")

    N = p.N
    dim = 2 * N
    Lx = p.Lx
    Ly = p.Ly
    Lx_eff = Lx * Ltw
    Ly_eff = Ly * Ltw
    if isodd(Lx_eff) || isodd(Ly_eff)
        error("TBC spectra require even effective dimensions to represent exact M points and kx=π path")
    end

    dos_ω_grid = spectra_dos_grid(p, spectra_delta_omega)
    nω = length(dos_ω_grid)
    dos_vals = zeros(Float64, nω)
    dos_M_vals = zeros(Float64, nω)
    dos_M_patch_vals = zeros(Float64, nω)
    ldos_ω0 = zeros(Float64, N)
    A_k0 = zeros(Float64, Lx_eff, Ly_eff)
    _, mx_path_kx, mx_path_ky = tbc_mx_path_metadata(Lx, Ly, Ltw)
    xg_path_kx, xg_path_ky = tbc_xg_path_metadata(Lx, Ly, Ltw)
    A_MX_path = zeros(Float64, length(mx_path_ky), nω)
    A_XG_path = zeros(Float64, length(xg_path_kx), nω)
    A_XG_node_patch = zeros(Float64, length(xg_path_kx), nω)
    dos_eta = zeros(Float64, nη, nω)
    dos_M_eta = zeros(Float64, nη, nω)
    dos_M_patch_eta = zeros(Float64, nη, nω)
    ldos_eta = zeros(Float64, nη, N)
    ldos_ω_eta = write_ldos_spectrum ? zeros(Float64, nη, N, nω) : nothing
    ldos_ω = write_ldos_spectrum ? zeros(Float64, N, nω) : nothing
    A_k0_eta = zeros(Float64, nη, Lx_eff, Ly_eff)
    A_MX_path_eta = zeros(Float64, nη, length(mx_path_ky), nω)
    A_XG_path_eta = zeros(Float64, nη, length(xg_path_kx), nω)
    A_XG_node_patch_eta = zeros(Float64, nη, length(xg_path_kx), nω)
    lor_eta = zeros(Float64, nη, nω)
    zero_lor_eta = zeros(Float64, nη)

    kx_grid = effective_k_grid(Lx, Ltw)
    ky_grid = effective_k_grid(Ly, Ltw)
    patch_mask, patch_count = m_point_patch_mask(kx_grid, ky_grid,
                                                 m_point_patch_half_width)
    patch_count == 0 && error("M-point patch contains no effective momentum points")

    Ix_pi = exact_effective_point(Lx_eff, 0.5, "kx=π")
    Iy_pi = exact_effective_point(Ly_eff, 0.5, "ky=π")
    Ix_zero = 0
    Iy_zero = 0

    nx_pi, mx_pi = effective_index_to_twist_fft(Ix_pi, Lx, Ltw)
    ny_zero, my_zero = effective_index_to_twist_fft(Iy_zero, Ly, Ltw)
    nx_zero, mx_zero = effective_index_to_twist_fft(Ix_zero, Lx, Ltw)
    ny_pi, my_pi = effective_index_to_twist_fft(Iy_pi, Ly, Ltw)

    xg_nx = Vector{Int}(undef, length(xg_path_kx))
    xg_mx = Vector{Int}(undef, length(xg_path_kx))
    xg_ny = Vector{Int}(undef, length(xg_path_kx))
    xg_my = Vector{Int}(undef, length(xg_path_kx))
    @inbounds for path_idx in eachindex(xg_path_kx)
        I = path_idx - 1
        xg_nx[path_idx], xg_mx[path_idx] = effective_index_to_twist_fft(I, Lx, Ltw)
        xg_ny[path_idx], xg_my[path_idx] = effective_index_to_twist_fft(I, Ly, Ltw)
    end

    xg_patch_terms_by_sector = [Tuple{Int, Int, Int, Float64}[] for _ in 1:(Ltw^2)]
    for path_idx in eachindex(xg_path_kx)
        I = path_idx - 1
        neighbors = Tuple{Int, Int, Int, Int}[]
        neighbor_indices = Set{Tuple{Int, Int}}()
        for dx in -1:1, dy in -1:1
            Ix = mod(I + dx, Lx_eff)
            Iy = mod(I + dy, Ly_eff)
            if (Ix, Iy) in neighbor_indices
                continue
            end
            push!(neighbor_indices, (Ix, Iy))
            nx_term, mx_term = effective_index_to_twist_fft(Ix, Lx, Ltw)
            ny_term, my_term = effective_index_to_twist_fft(Iy, Ly, Ltw)
            push!(neighbors, (nx_term, ny_term, mx_term, my_term))
        end

        patch_weight = 1.0 / length(neighbors)
        for (nx_term, ny_term, mx_term, my_term) in neighbors
            sector_idx = nx_term * Ltw + ny_term + 1
            push!(xg_patch_terms_by_sector[sector_idx],
                  (path_idx, mx_term, my_term, patch_weight))
        end
    end

    Htw = zeros(ComplexF64, dim, dim)
    Uwork = similar(Htw)
    Etw = zeros(Float64, dim)

    @inbounds for nx in 0:Ltw-1, ny in 0:Ltw-1
        qx = 2π * nx / Ltw
        qy = 2π * ny / Ltw
        sector_idx = nx * Ltw + ny + 1
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

            for iw in 1:nω
                x = dos_ω_grid[iw] - En
                @simd for iη in 1:nη
                    lor_eta[iη, iw] = lorentzian_spectra(x, eta_vals[iη])
                    dos_eta[iη, iw] += w_n * lor_eta[iη, iw]
                end
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
                for iw in 1:nω
                    @simd for iη in 1:nη
                        dos_M_eta[iη, iw] += exact_weight * lor_eta[iη, iw]
                    end
                end
            end

            patch_weight = 0.0
            @inbounds @simd for iη in 1:nη
                zero_lor_eta[iη] = lorentzian_spectra(-En, eta_vals[iη])
            end
            @inbounds for i in 1:N
                site_weight = abs2(vecs[i, n])
                @simd for iη in 1:nη
                    ldos_eta[iη, i] += site_weight * zero_lor_eta[iη]
                end
                if write_ldos_spectrum
                    for iw in 1:nω
                        @simd for iη in 1:nη
                            ldos_ω_eta[iη, i, iw] += site_weight * lor_eta[iη, iw]
                        end
                    end
                end
            end
            for my in 0:Ly-1, mx in 0:Lx-1
                Ix = twist_fft_to_effective_index(mx, nx, Lx, Ltw)
                Iy = twist_fft_to_effective_index(my, ny, Ly, Ltw)
                wk = abs2(cache.u_k_cache[mx + 1, my + 1]) / N

                if patch_mask[Ix + 1, Iy + 1]
                    patch_weight += wk
                end

                uk2 = abs2(cache.u_k_cache[mx + 1, my + 1])
                for iη in 1:nη
                    if zero_lor_eta[iη] > 1e-6
                        A_k0_eta[iη, Ix + 1, Iy + 1] += uk2 * zero_lor_eta[iη]
                    end
                end
            end
            patch_weight /= patch_count
            for iw in 1:nω
                @simd for iη in 1:nη
                    dos_M_patch_eta[iη, iw] += patch_weight * lor_eta[iη, iw]
                end
            end

            if nx == nx_pi
                for my in 0:Ly-1
                    Iy = twist_fft_to_effective_index(my, ny, Ly, Ltw)
                    if Iy <= fld(Ly_eff, 2)
                        wk = abs2(cache.u_k_cache[mx_pi + 1, my + 1]) / N
                        path_idx = Iy + 1
                        for iw in 1:nω
                            @simd for iη in 1:nη
                                A_MX_path_eta[iη, path_idx, iw] += wk * lor_eta[iη, iw]
                            end
                        end
                    end
                end
            end

            for path_idx in eachindex(xg_path_kx)
                if nx == xg_nx[path_idx] && ny == xg_ny[path_idx]
                    wk = abs2(cache.u_k_cache[xg_mx[path_idx] + 1, xg_my[path_idx] + 1]) / N
                    for iw in 1:nω
                        @simd for iη in 1:nη
                            A_XG_path_eta[iη, path_idx, iw] += wk * lor_eta[iη, iw]
                        end
                    end
                end
            end

            for (path_idx, mx_term, my_term, weight_factor) in xg_patch_terms_by_sector[sector_idx]
                wk = weight_factor * abs2(cache.u_k_cache[mx_term + 1, my_term + 1]) / N
                for iw in 1:nω
                    @simd for iη in 1:nη
                        A_XG_node_patch_eta[iη, path_idx, iw] += wk * lor_eta[iη, iw]
                    end
                end
            end
        end
    end

    dos_eta ./= (N * Ltw^2)
    ldos_eta ./= Ltw^2
    if write_ldos_spectrum
        ldos_ω_eta ./= Ltw^2
    end
    A_k0_eta ./= N

    copyto!(dos_vals, @view dos_eta[1, :])
    copyto!(dos_M_vals, @view dos_M_eta[1, :])
    copyto!(dos_M_patch_vals, @view dos_M_patch_eta[1, :])
    copyto!(ldos_ω0, @view ldos_eta[1, :])
    if write_ldos_spectrum
        copyto!(ldos_ω, @view ldos_ω_eta[1, :, :])
    end
    copyto!(A_k0, @view A_k0_eta[1, :, :])
    copyto!(A_MX_path, @view A_MX_path_eta[1, :, :])
    copyto!(A_XG_path, @view A_XG_path_eta[1, :, :])
    copyto!(A_XG_node_patch, @view A_XG_node_patch_eta[1, :, :])

    return TwistedSpectraResult(
        dos_ω_grid,
        reuse_buffers ? dos_vals : copy(dos_vals),
        reuse_buffers ? dos_M_vals : copy(dos_M_vals),
        reuse_buffers ? dos_M_patch_vals : copy(dos_M_patch_vals),
        reuse_buffers ? ldos_ω0 : copy(ldos_ω0),
        reuse_buffers ? A_k0 : copy(A_k0),
        reuse_buffers ? A_MX_path : copy(A_MX_path),
        reuse_buffers ? A_XG_path : copy(A_XG_path),
        reuse_buffers ? A_XG_node_patch : copy(A_XG_node_patch),
        reuse_buffers ? dos_eta : copy(dos_eta),
        reuse_buffers ? dos_M_eta : copy(dos_M_eta),
        reuse_buffers ? dos_M_patch_eta : copy(dos_M_patch_eta),
        reuse_buffers ? ldos_eta : copy(ldos_eta),
        reuse_buffers ? A_k0_eta : copy(A_k0_eta),
        reuse_buffers ? A_MX_path_eta : copy(A_MX_path_eta),
        reuse_buffers ? A_XG_path_eta : copy(A_XG_path_eta),
        reuse_buffers ? A_XG_node_patch_eta : copy(A_XG_node_patch_eta),
        reuse_buffers ? ldos_ω : _copy_optional(ldos_ω),
        reuse_buffers ? ldos_ω_eta : _copy_optional(ldos_ω_eta),
        kx_grid,
        ky_grid,
        mx_path_kx,
        mx_path_ky,
        xg_path_kx,
        xg_path_ky,
        Ltw,
        m_point_patch_half_width,
        spectra_eta,
        spectra_delta_omega,
    )
end
