using Statistics
using Printf

const DEFAULT_PATH_WINDOW_RADIUS = 1

function calc_stats(data_list)
    n_samples = length(data_list)
    if n_samples == 0
        return nothing, nothing
    end

    raw_shape = size(data_list[1])
    sum_val = zeros(Float64, raw_shape)
    sum_sq = zeros(Float64, raw_shape)

    for d in data_list
        sum_val .+= d
        sum_sq .+= d .^ 2
    end

    mean_val = sum_val ./ n_samples
    var_val = (sum_sq ./ n_samples) .- (mean_val .^ 2)
    var_val = max.(var_val, 0.0)
    sem_val = sqrt.(var_val ./ n_samples)

    return mean_val, sem_val
end

function selected_eta_index(file, eta_factor)
    factor = Float64(eta_factor)
    if haskey(file, "spectra_eta_factors")
        return DwaveHMC.eta_factor_index(collect(file["spectra_eta_factors"]), factor)
    end
    isapprox(factor, 1.0; atol=DwaveHMC.ETA_FACTOR_ATOL, rtol=0.0) ||
        error("Old spectra file has no multi-eta data; only eta_factor=1 is available")
    return 1
end

function selected_vector(group, multi_key::AbstractString, old_key::AbstractString, eta_idx::Int)
    if haskey(group, multi_key)
        arr = group[multi_key]
        ndims(arr) == 2 ||
            error("$multi_key expected a 2D eta-first array with eta dimension >= $eta_idx; actual size $(size(arr))")
        size(arr, 1) >= eta_idx ||
            error("$multi_key expected eta dimension >= $eta_idx; actual size $(size(arr))")
        if haskey(group, old_key)
            expected_len = length(group[old_key])
            size(arr, 2) == expected_len ||
                error("$multi_key expected shape (eta, $expected_len) compatible with $old_key; actual size $(size(arr))")
        end
        return vec(arr[eta_idx, :])
    end
    eta_idx == 1 || error("Missing $multi_key for selected eta factor")
    return group[old_key]
end

function selected_scalar(group, multi_key::AbstractString, old_key::AbstractString, eta_idx::Int)
    if haskey(group, multi_key)
        arr = group[multi_key]
        ndims(arr) == 1 ||
            error("$multi_key expected a 1D eta array with eta dimension >= $eta_idx; actual size $(size(arr))")
        length(arr) >= eta_idx ||
            error("$multi_key expected eta dimension >= $eta_idx; actual size $(size(arr))")
        return Float64(arr[eta_idx])
    end
    eta_idx == 1 || error("Missing $multi_key for selected eta factor")
    return Float64(group[old_key])
end

function selected_matrix(group, multi_key::AbstractString, old_key::AbstractString, eta_idx::Int)
    if haskey(group, multi_key)
        arr = group[multi_key]
        ndims(arr) == 3 ||
            error("$multi_key expected a 3D eta-first array with eta dimension >= $eta_idx; actual size $(size(arr))")
        size(arr, 1) >= eta_idx ||
            error("$multi_key expected eta dimension >= $eta_idx; actual size $(size(arr))")
        if haskey(group, old_key)
            old = group[old_key]
            size(arr, 2) == size(old, 1) && size(arr, 3) == size(old, 2) ||
                error("$multi_key expected shape (eta, $(size(old, 1)), $(size(old, 2))) compatible with $old_key; actual size $(size(arr))")
        end
        return arr[eta_idx, :, :]
    end
    eta_idx == 1 || error("Missing $multi_key for selected eta factor")
    return group[old_key]
end

function has_any_key(group, names)
    return any(name -> haskey(group, name), names)
end

function selected_vector_any(group, key_pairs, eta_idx::Int)
    for (multi_key, old_key) in key_pairs
        if haskey(group, multi_key) || haskey(group, old_key)
            return selected_vector(group, multi_key, old_key, eta_idx)
        end
    end
    return nothing
end

function selected_matrix_any(group, key_pairs, eta_idx::Int)
    for (multi_key, old_key) in key_pairs
        if haskey(group, multi_key) || haskey(group, old_key)
            return selected_matrix(group, multi_key, old_key, eta_idx)
        end
    end
    return nothing
end

function momentum_source_for_group(group)
    ordinary_keys = ("dos_M", "dos_M_eta", "A_k0", "A_k0_eta",
                     "A_MX_path", "A_MX_path_eta", "A_XG_path", "A_XG_path_eta")
    diagnostic_keys = ("dos_M_landau_gauge_diagnostic",
                       "dos_M_eta_landau_gauge_diagnostic",
                       "A_k_omega0_landau_gauge_diagnostic",
                       "A_k_omega0_eta_landau_gauge_diagnostic",
                       "A_MX_path_landau_gauge_diagnostic",
                       "A_MX_path_eta_landau_gauge_diagnostic",
                       "A_XG_path_landau_gauge_diagnostic",
                       "A_XG_path_eta_landau_gauge_diagnostic")
    has_ordinary = has_any_key(group, ordinary_keys)
    has_diagnostic = has_any_key(group, diagnostic_keys)
    has_ordinary && has_diagnostic &&
        error("Incompatible spectra config: ordinary and Landau-gauge diagnostic momentum spectra are both present")
    return has_diagnostic ? :landau_gauge_diagnostic : (has_ordinary ? :ordinary : :none)
end

function calc_scalar_stats(values::AbstractVector{<:Real})
    n_samples = length(values)
    n_samples > 0 || error("Cannot calculate scalar stats for empty data")
    mean_val = sum(values) / n_samples
    var_val = 0.0
    for v in values
        var_val += abs2(v - mean_val)
    end
    var_val /= n_samples
    return mean_val, sqrt(max(var_val, 0.0) / n_samples)
end

function write_selected_dc_csv(path, eta_factor, mean_dc::Real, err_dc::Real)
    open(path, "w") do io
        println(io, "eta_factor,DC_Conductivity,Error")
        @printf(io, "%.6g,%.6e,%.6e\n", Float64(eta_factor), mean_dc, err_dc)
    end
end

function write_series_csv(path, header, grid, mean_values, err_values)
    open(path, "w") do io
        println(io, header)
        for i in eachindex(mean_values)
            @printf(io, "%.6f,%.6e,%.6e\n", grid[i], mean_values[i], err_values[i])
        end
    end
end

function write_ak_csv(path, mean_ak, err_ak)
    Lx, Ly = size(mean_ak)
    open(path, "w") do io
        println(io, "kx_idx,ky_idx,kx,ky,A_val,Error")
        for x in 1:Lx, y in 1:Ly
            kx = 2π * (x - 1) / Lx
            ky = 2π * (y - 1) / Ly
            if kx > π kx -= 2π end
            if ky > π ky -= 2π end
            @printf(io, "%d,%d,%.6f,%.6f,%.6e,%.6e\n",
                    x, y, kx, ky, mean_ak[x, y], err_ak[x, y])
        end
    end
end

function write_ldos_csv(path, mean_ldos, err_ldos, Lx::Int, Ly::Int)
    length(mean_ldos) == Lx * Ly || error("LDOS length does not match lattice size")
    length(err_ldos) == length(mean_ldos) || error("LDOS error length mismatch")
    open(path, "w") do io
        println(io, "x,y,site,LDOS_0,Error")
        for y in 1:Ly, x in 1:Lx
            site = (y - 1) * Lx + x
            @printf(io, "%d,%d,%d,%.6e,%.6e\n",
                    x, y, site, mean_ldos[site], err_ldos[site])
        end
    end
end

function path_peak_window(path::AbstractMatrix,
                          omega_grid::AbstractVector;
                          radius::Int=DEFAULT_PATH_WINDOW_RADIUS)
    radius >= 0 || error("path window radius must be nonnegative")
    idx0 = argmin(abs.(omega_grid))
    peak_idx = argmax(@view path[:, idx0])
    lo = max(1, peak_idx - radius)
    hi = min(size(path, 1), peak_idx + radius)
    return peak_idx, lo, hi, idx0
end

function average_path_window(path::AbstractMatrix, lo::Int, hi::Int)
    return vec(mean(view(path, lo:hi, :); dims=1))
end

function average_path_window_error(err_path::AbstractMatrix, lo::Int, hi::Int)
    n = hi - lo + 1
    return vec(sqrt.(sum(abs2, view(err_path, lo:hi, :); dims=1)) ./ n)
end

function path_observable(path::AbstractMatrix,
                         omega_grid::AbstractVector,
                         kx_vals::AbstractVector,
                         ky_vals::AbstractVector;
                         err_path=nothing,
                         radius::Int=DEFAULT_PATH_WINDOW_RADIUS)
    size(path, 1) == length(kx_vals) || error("path rows and kx metadata size mismatch")
    size(path, 1) == length(ky_vals) || error("path rows and ky metadata size mismatch")
    size(path, 2) == length(omega_grid) || error("path columns and omega grid size mismatch")

    peak_idx, lo, hi, idx0 = path_peak_window(path, omega_grid; radius=radius)
    vals = average_path_window(path, lo, hi)
    errs = err_path === nothing ? nothing : average_path_window_error(err_path, lo, hi)

    peak = (k_idx=peak_idx,
            kx=Float64(kx_vals[peak_idx]),
            ky=Float64(ky_vals[peak_idx]),
            omega0=Float64(omega_grid[idx0]),
            A_omega0=Float64(path[peak_idx, idx0]),
            window_start=lo,
            window_end=hi)
    return vals, errs, peak
end

function write_path_csv(path, path_data, err_data, omega_grid,
                        kx_vals, ky_vals, kx_indices, ky_indices)
    open(path, "w") do io
        println(io, "k_idx,kx_idx,ky_idx,kx,ky,omega,A_val,Error")
        for k in 1:size(path_data, 1), iw in eachindex(omega_grid)
            @printf(io, "%d,%d,%d,%.6f,%.6f,%.6f,%.6e,%.6e\n",
                    k, kx_indices[k], ky_indices[k], kx_vals[k], ky_vals[k],
                    omega_grid[iw], path_data[k, iw], err_data[k, iw])
        end
    end
end

function write_peak_summary(path, rows)
    open(path, "w") do io
        println(io, "source,kind,k_idx,kx,ky,omega0,A_omega0,window_start,window_end")
        for row in rows
            @printf(io, "%s,%s,%d,%.6f,%.6f,%.6f,%.6e,%d,%d\n",
                    row.source, row.kind, row.k_idx, row.kx, row.ky,
                    row.omega0, row.A_omega0, row.window_start, row.window_end)
        end
    end
end
