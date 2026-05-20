@inline function _site_xy0(i::Int, Lx::Int)
    return mod1(i, Lx) - 1, cld(i, Lx) - 1
end

function validate_magnetic_field(p::ModelParameters)
    if p.n_flux_sc == 0
        p.boundary_condition in (:periodic, :magnetic_pbc) ||
            error("boundary_condition must be :periodic or :magnetic_pbc")
    else
        p.boundary_condition === :magnetic_pbc ||
            error("Finite n_flux_sc requires boundary_condition=:magnetic_pbc")
        iseven(p.n_flux_sc) ||
            error("magnetic PBC requires even n_flux_sc")
    end
    return nothing
end

@inline function _landau_link_phase(Lx::Int, Ly::Int, α::Float64,
                                    x0::Int, y0::Int, dx::Int, dy::Int)
    wx = fld(x0 + dx, Lx)
    y_end = y0 + dy
    raw = cis(α * dy * (x0 + 0.5 * dx))
    patch = cis(-α * wx * Lx * y_end)
    return raw * patch
end

function build_magnetic_cache(p::ModelParameters)
    validate_magnetic_field(p)
    flux_density_sc = p.n_flux_sc / p.N
    α = π * flux_density_sc
    if p.n_flux_sc == 0
        return NoFieldCache(p.Lx, p.Ly, 0, 0.0, 0.0)
    end

    U_x = Vector{ComplexF64}(undef, p.N)
    U_y = Vector{ComplexF64}(undef, p.N)
    U_xpy = Vector{ComplexF64}(undef, p.N)
    U_xmy = Vector{ComplexF64}(undef, p.N)
    @inbounds for i in 1:p.N
        x0, y0 = _site_xy0(i, p.Lx)
        U_x[i] = _landau_link_phase(p.Lx, p.Ly, α, x0, y0, 1, 0)
        U_y[i] = _landau_link_phase(p.Lx, p.Ly, α, x0, y0, 0, 1)
        U_xpy[i] = _landau_link_phase(p.Lx, p.Ly, α, x0, y0, 1, 1)
        U_xmy[i] = _landau_link_phase(p.Lx, p.Ly, α, x0, y0, 1, -1)
    end
    return LandauGaugeCache(p.Lx, p.Ly, p.n_flux_sc, flux_density_sc, α,
                            U_x, U_y, U_xpy, U_xmy)
end

@inline link_phase(::NoFieldCache, i::Int, dx::Int, dy::Int) = 1.0 + 0.0im

@inline function link_phase(mag::LandauGaugeCache, i::Int, dx::Int, dy::Int)
    if dx == 1 && dy == 0
        return mag.U_x[i]
    elseif dx == 0 && dy == 1
        return mag.U_y[i]
    elseif dx == 1 && dy == 1
        return mag.U_xpy[i]
    elseif dx == 1 && dy == -1
        return mag.U_xmy[i]
    else
        x0, y0 = _site_xy0(i, mag.Lx)
        return _landau_link_phase(mag.Lx, mag.Ly, mag.plaquette_phase, x0, y0, dx, dy)
    end
end

@inline plaquette_phase(mag::AbstractMagneticCache, x::Int, y::Int) =
    getfield(mag, :plaquette_phase)

function magnetic_metadata(mag::AbstractMagneticCache)
    return (n_flux_sc=mag.n_flux_sc,
            flux_density_sc=mag.flux_density_sc,
            plaquette_phase=mag.plaquette_phase,
            magnetic_gauge=mag.n_flux_sc == 0 ? "none" : "Landau gauge",
            magnetic_pbc=mag.n_flux_sc != 0)
end
