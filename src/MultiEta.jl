const DEFAULT_SPECTRA_ETA_FACTORS = Float64[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
const ETA_FACTOR_ATOL = 1.0e-12

function validate_spectra_eta_factors(factors)::Vector{Float64}
    vals = Float64.(collect(factors))
    isempty(vals) && error("spectra_eta_factors must be non-empty")
    all(isfinite, vals) || error("spectra_eta_factors must be finite")
    all(vals .> 0.0) || error("spectra_eta_factors must be positive")
    isapprox(vals[1], 1.0; atol=ETA_FACTOR_ATOL, rtol=0.0) ||
        error("spectra_eta_factors must start with 1")

    @inbounds for i in eachindex(vals)
        for j in firstindex(vals):(i - 1)
            if isapprox(vals[i], vals[j]; atol=ETA_FACTOR_ATOL, rtol=0.0)
                error("spectra_eta_factors must not contain duplicate factors")
            end
        end
    end

    return vals
end

function eta_factor_index(factors, eta_factor)::Int
    factor = Float64(eta_factor)
    @inbounds for i in eachindex(factors)
        if isapprox(Float64(factors[i]), factor; atol=ETA_FACTOR_ATOL, rtol=0.0)
            return i
        end
    end
    error("eta_factor=$factor not found. Available factors: $(collect(factors))")
end

eta_values_from_base(base_eta::Real, factors::AbstractVector{<:Real}) =
    Float64(base_eta) .* Float64.(factors)
