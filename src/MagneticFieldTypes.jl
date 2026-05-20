abstract type AbstractMagneticCache end

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
