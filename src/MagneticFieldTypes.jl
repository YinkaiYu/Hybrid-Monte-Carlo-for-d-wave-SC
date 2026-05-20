abstract type AbstractMagneticCache end

function normalize_magnetic_field_inputs(n_flux_sc::Int,
                                         n_vortices::Union{Nothing,Int},
                                         boundary_condition::Symbol)
    actual_n_flux_sc = n_vortices === nothing ? n_flux_sc : Int(n_vortices)
    if n_vortices !== nothing && n_flux_sc != 0 && n_flux_sc != actual_n_flux_sc
        error("n_flux_sc and n_vortices must match when both are provided")
    end
    if actual_n_flux_sc == 0
        boundary_condition in (:periodic, :magnetic_pbc) ||
            error("boundary_condition must be :periodic or :magnetic_pbc")
    else
        boundary_condition === :magnetic_pbc ||
            error("Finite n_flux_sc requires boundary_condition=:magnetic_pbc")
        iseven(actual_n_flux_sc) || error("magnetic PBC requires even n_flux_sc")
    end
    return actual_n_flux_sc, boundary_condition
end

struct NoFieldCache <: AbstractMagneticCache
    Lx::Int
    Ly::Int
    n_flux_sc::Int
    flux_density_sc::Float64
    plaquette_phase::Float64
end

struct LandauGaugeCache <: AbstractMagneticCache
    Lx::Int
    Ly::Int
    n_flux_sc::Int
    flux_density_sc::Float64
    plaquette_phase::Float64
    U_x::Vector{ComplexF64}
    U_y::Vector{ComplexF64}
    U_xpy::Vector{ComplexF64}
    U_xmy::Vector{ComplexF64}
end
