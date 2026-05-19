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
