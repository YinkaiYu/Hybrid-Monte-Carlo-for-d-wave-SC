using JLD2
using Statistics
using DelimitedFiles
using Printf
using DwaveHMC # 加载你的包以识别 ModelParameters 类型

# ==========================================
# 设置
# ==========================================
# 这里填写你刚才运行的数据目录
target_dir = "data/test_spectra_L24_J0.8_W1.0_imp0.0_T0.001_mu-1.4"

# ==========================================
# 辅助函数：计算 Mean 和 SEM
# ==========================================
function calc_stats(data_list)
    # data_list 是一个 Vector{Vector} 或 Vector{Matrix}
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

function read_spectra_metadata(file, params)
    dos_omega_grid = haskey(file, "dos_omega_grid") ? file["dos_omega_grid"] :
                     collect(-params.ω_max:params.Δω:params.ω_max)
    spectra_Ltw = haskey(file, "spectra_Ltw") ? Int(file["spectra_Ltw"]) : 1
    spectra_Lx_eff = haskey(file, "spectra_Lx_eff") ? Int(file["spectra_Lx_eff"]) : params.Lx
    spectra_Ly_eff = haskey(file, "spectra_Ly_eff") ? Int(file["spectra_Ly_eff"]) : params.Ly

    return collect(dos_omega_grid), spectra_Ltw, spectra_Lx_eff, spectra_Ly_eff
end

function read_omega_grid(file, params)
    return haskey(file, "omega_grid") ? collect(file["omega_grid"]) :
           collect(params.ω_min:params.Δω:params.ω_max)
end

function compatible_grid(grid, values, label; source="")
    if length(grid) == length(values)
        return grid
    end

    @warn "$label grid size mismatch; using index grid." source=source grid_length=length(grid) data_length=length(values)
    return collect(1:length(values))
end

function resolve_ak_dims(mean_ak, Lx_eff, Ly_eff; source="")
    actual_dims = size(mean_ak)
    if actual_dims != (Lx_eff, Ly_eff)
        @warn "A_k0 size differs from spectra metadata; using actual array size." source=source metadata=(Lx_eff, Ly_eff) actual=actual_dims
        return actual_dims
    end
    return Lx_eff, Ly_eff
end

function write_series_csv(path, header, grid, mean_values, err_values)
    open(path, "w") do io
        println(io, header)
        for i in eachindex(mean_values)
            @printf(io, "%.6f,%.6f,%.6f\n", grid[i], mean_values[i], err_values[i])
        end
    end
end

function write_ak_csv(path, mean_ak, err_ak, Lx_eff, Ly_eff; source="")
    Lx, Ly = resolve_ak_dims(mean_ak, Lx_eff, Ly_eff; source=source)
    open(path, "w") do io
        println(io, "kx_idx,ky_idx,kx,ky,A_val,Error")
        for x in 1:Lx
            for y in 1:Ly
                kx = 2π * (x - 1) / Lx
                ky = 2π * (y - 1) / Ly

                if kx > π
                    kx -= 2π
                end
                if ky > π
                    ky -= 2π
                end

                @printf(io, "%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                        x, y, kx, ky, mean_ak[x, y], err_ak[x, y])
            end
        end
    end
end

function collect_sweep_data(file; source="")
    list_opt = Vector{Vector{Float64}}()
    list_dos = Vector{Vector{Float64}}()
    list_dos_AN = Vector{Vector{Float64}}()
    list_dos_AN_patch = Vector{Vector{Float64}}()
    list_ak = Vector{Matrix{Float64}}()

    count = 0
    patch_count = 0

    for key in keys(file)
        if startswith(key, "sweep_")
            g = file[key]
            push!(list_opt, g["opt_cond"])
            push!(list_dos, g["dos"])
            push!(list_dos_AN, g["dos_AN"])
            push!(list_ak, g["A_k0"])
            count += 1

            if haskey(g, "dos_AN_patch")
                push!(list_dos_AN_patch, g["dos_AN_patch"])
                patch_count += 1
            end
        end
    end

    if 0 < patch_count < count
        @warn "Mixed dos_AN_patch presence across sweep groups; skipping patch output." source=source patch_sweeps=patch_count sweep_count=count
        empty!(list_dos_AN_patch)
    elseif patch_count == 0
        empty!(list_dos_AN_patch)
    end

    return (opt=list_opt,
            dos=list_dos,
            dos_AN=list_dos_AN,
            dos_AN_patch=list_dos_AN_patch,
            ak=list_ak,
            count=count)
end

function process_spectra_directory(target_dir::AbstractString=target_dir)
    jld_file = joinpath(target_dir, "spectra_bins.jld2")

    output_opt = joinpath(target_dir, "processed_opt_cond.csv")
    output_dos = joinpath(target_dir, "processed_dos.csv")
    output_dos_AN = joinpath(target_dir, "processed_dos_AN.csv")
    output_dos_AN_patch = joinpath(target_dir, "processed_dos_AN_patch.csv")
    output_ak = joinpath(target_dir, "processed_ak0.csv")

    println("Opening file: $jld_file")

    jldopen(jld_file, "r") do file
        params = file["params"] # ModelParameters
        omega_grid = read_omega_grid(file, params)
        dos_omega_grid, spectra_Ltw, Lx_eff, Ly_eff = read_spectra_metadata(file, params)

        println("Params: L=$(params.Lx)x$(params.Ly), Beta=$(params.β), spectra_Ltw=$spectra_Ltw, effective=$(Lx_eff)x$(Ly_eff)")

        data = collect_sweep_data(file; source=jld_file)
        if data.count == 0
            @warn "No 'sweep_' data found in $jld_file."
            return
        end

        println("Found $(data.count) bins. Calculating statistics...")

        mean_opt, err_opt = calc_stats(data.opt)
        mean_dos, err_dos = calc_stats(data.dos)
        mean_dos_AN, err_dos_AN = calc_stats(data.dos_AN)
        mean_ak, err_ak = calc_stats(data.ak)

        opt_grid = compatible_grid(omega_grid, mean_opt, "Optical conductivity"; source=jld_file)
        write_series_csv(output_opt, "omega,Re_Sigma,Error", opt_grid, mean_opt, err_opt)
        println("Saved: $output_opt")

        dos_grid = compatible_grid(dos_omega_grid, mean_dos, "DOS"; source=jld_file)
        write_series_csv(output_dos, "omega,DOS,Error", dos_grid, mean_dos, err_dos)
        println("Saved: $output_dos")

        dos_AN_grid = compatible_grid(dos_omega_grid, mean_dos_AN, "DOS_AN"; source=jld_file)
        write_series_csv(output_dos_AN, "omega,DOS_AN,Error", dos_AN_grid, mean_dos_AN, err_dos_AN)
        println("Saved: $output_dos_AN")

        if !isempty(data.dos_AN_patch)
            mean_dos_AN_patch, err_dos_AN_patch = calc_stats(data.dos_AN_patch)
            patch_grid = compatible_grid(dos_omega_grid, mean_dos_AN_patch, "DOS_AN_patch"; source=jld_file)
            write_series_csv(output_dos_AN_patch, "omega,DOS_AN_patch,Error",
                             patch_grid, mean_dos_AN_patch, err_dos_AN_patch)
            println("Saved: $output_dos_AN_patch")
        else
            rm(output_dos_AN_patch; force=true)
        end

        write_ak_csv(output_ak, mean_ak, err_ak, Lx_eff, Ly_eff; source=jld_file)
        println("Saved: $output_ak")
    end
end

function main()
    process_spectra_directory(target_dir)
    println("Processing Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
