using JLD2
using Statistics
using DelimitedFiles
using Printf
using Glob
using DwaveHMC

target_dir = @__DIR__

# --- 1. 鲁棒的单构型处理 ---
function process_single_config(jld_path)
    # 文件存在且有内容
    if !isfile(jld_path) || filesize(jld_path) == 0
        return nothing 
    end
    
    try
        jldopen(jld_path, "r") do file
            keys_in_file = keys(file)
            # 必须包含 sweep_ 数据
            sweep_keys = filter(k -> startswith(k, "sweep_"), keys_in_file)
            if isempty(sweep_keys) return nothing end
            
            # 必须包含 params
            if !haskey(file, "params") return nothing end
            
            # 读取第一个 bin 初始化
            g1 = file[sweep_keys[1]]
            sum_opt = copy(g1["opt_cond"])
            sum_dos = copy(g1["dos"])
            sum_dos_AN = copy(g1["dos_AN"])
            sum_ak = copy(g1["A_k0"])
            sum_akpath = haskey(g1, "A_kpath") ? copy(g1["A_kpath"]) : nothing
            sum_dos_AN_patch = haskey(g1, "dos_AN_patch") ? copy(g1["dos_AN_patch"]) : nothing
            patch_count = sum_dos_AN_patch === nothing ? 0 : 1
            mixed_patch = false
            count = 1
            
            # 累加后续 bin
            for i in 2:length(sweep_keys)
                g = file[sweep_keys[i]]
                sum_opt .+= g["opt_cond"]
                sum_dos .+= g["dos"]
                sum_dos_AN .+= g["dos_AN"]
                sum_ak .+= g["A_k0"]
                has_patch = haskey(g, "dos_AN_patch")
                if has_patch
                    patch_count += 1
                end
                if sum_dos_AN_patch !== nothing
                    if has_patch
                        sum_dos_AN_patch .+= g["dos_AN_patch"]
                    else
                        mixed_patch = true
                    end
                elseif has_patch
                    mixed_patch = true
                end
                if sum_akpath !== nothing
                    if !haskey(g, "A_kpath")
                        return nothing
                    end
                    sum_akpath .+= g["A_kpath"]
                end
                count += 1
            end
            
            p = file["params"]
            meta = Dict{String, Any}()
            if haskey(file, "omega_grid") meta["omega_grid"] = file["omega_grid"] end
            if haskey(file, "dos_omega_grid") meta["dos_omega_grid"] = file["dos_omega_grid"] end
            if haskey(file, "kpath_kx") meta["kpath_kx"] = file["kpath_kx"] end
            if haskey(file, "kpath_ky") meta["kpath_ky"] = file["kpath_ky"] end
            if haskey(file, "kpath_kx_idx") meta["kpath_kx_idx"] = file["kpath_kx_idx"] end
            if haskey(file, "kpath_ky_idx") meta["kpath_ky_idx"] = file["kpath_ky_idx"] end
            meta["spectra_Ltw"] = haskey(file, "spectra_Ltw") ? Int(file["spectra_Ltw"]) : 1
            meta["spectra_Lx_eff"] = haskey(file, "spectra_Lx_eff") ? Int(file["spectra_Lx_eff"]) : p.Lx
            meta["spectra_Ly_eff"] = haskey(file, "spectra_Ly_eff") ? Int(file["spectra_Ly_eff"]) : p.Ly

            dos_AN_patch = nothing
            if mixed_patch
                @warn "Mixed dos_AN_patch presence inside config; skipping patch for this config." file=jld_path patch_sweeps=patch_count sweep_count=count
            elseif sum_dos_AN_patch !== nothing
                dos_AN_patch = sum_dos_AN_patch ./ count
            end
            
            # 计算平均
            res = (opt=sum_opt ./ count, dos=sum_dos ./ count, dos_AN=sum_dos_AN ./ count,
                   ak0=sum_ak ./ count,
                   akpath=sum_akpath === nothing ? nothing : (sum_akpath ./ count),
                   dos_AN_patch=dos_AN_patch,
                   params=p, meta=meta)
            
            # 检查 NaN
            if any(isnan, res.opt) || any(isnan, res.dos) ||
               (res.dos_AN_patch !== nothing && any(isnan, res.dos_AN_patch))
                return nothing
            end
            
            return res
        end
    catch
        return nothing
    end
end

function calc_final_stats(list_of_means)
    n = length(list_of_means)
    if n == 0 return nothing, nothing end
    
    total_sum = sum(list_of_means)
    final_mean = total_sum ./ n
    
    if n > 1
        sq_diff_sum = sum((x .- final_mean).^2 for x in list_of_means)
        var = sq_diff_sum ./ (n - 1)
        sem = sqrt.(var ./ n)
    else
        sem = zeros(eltype(final_mean), size(final_mean))
    end
    return final_mean, sem
end

function compatible_grid(grid, values, label; source="")
    if length(grid) == length(values)
        return grid
    end
    @warn "$label grid size mismatch; using fallback range." source=source grid_length=length(grid) data_length=length(values)
    return range(first(grid), stop=last(grid), length=length(values))
end

function resolve_ak_dims(final_ak, Lx_eff, Ly_eff; source="")
    actual_dims = size(final_ak)
    if actual_dims != (Lx_eff, Ly_eff)
        @warn "A_k0 size differs from spectra metadata; using actual array size." source=source metadata=(Lx_eff, Ly_eff) actual=actual_dims
        return actual_dims
    end
    return Lx_eff, Ly_eff
end

const SPECTRA_OUTPUT_FILES = [
    "spectra_opt_cond.csv",
    "spectra_dos.csv",
    "spectra_dos_AN_patch.csv",
    "spectra_ak0.csv",
    "spectra_akpath.csv",
]

function remove_spectra_outputs!(dir_path)
    for filename in SPECTRA_OUTPUT_FILES
        rm(joinpath(dir_path, filename); force=true)
    end
end

function resolved_output_grids(res)
    p = res.params
    meta = res.meta
    omega_grid = haskey(meta, "omega_grid") ? collect(meta["omega_grid"]) :
                 collect(p.ω_min : p.Δω : p.ω_max)
    dos_omega_grid = haskey(meta, "dos_omega_grid") ? collect(meta["dos_omega_grid"]) :
                     collect(-p.ω_max : p.Δω : p.ω_max)

    if length(omega_grid) != length(res.opt)
        omega_grid = collect(range(p.ω_min, stop=p.ω_max, length=length(res.opt)))
    end
    if length(dos_omega_grid) != length(res.dos)
        dos_omega_grid = collect(range(-p.ω_max, stop=p.ω_max, length=length(res.dos)))
    end

    return omega_grid, dos_omega_grid
end

function resolved_kpath_signature(res)
    if res.akpath === nothing
        return (present=false, shape=nothing, kx=nothing, ky=nothing, ky_idx=nothing)
    end

    meta = res.meta
    kx_val = haskey(meta, "kpath_kx") ? meta["kpath_kx"] : nothing
    ky_vals = haskey(meta, "kpath_ky") ? meta["kpath_ky"] : nothing
    ky_indices = haskey(meta, "kpath_ky_idx") ? meta["kpath_ky_idx"] : nothing

    if kx_val === nothing || ky_vals === nothing
        _, ky_indices_fallback, kx_val_fallback, ky_vals_fallback = DwaveHMC.antinode_kpath(res.params)
        kx_val = kx_val === nothing ? kx_val_fallback : kx_val
        ky_vals = ky_vals === nothing ? ky_vals_fallback : ky_vals
        ky_indices = ky_indices === nothing ? ky_indices_fallback : ky_indices
    end
    if ky_indices === nothing
        ky_indices = collect(1:length(ky_vals))
    end

    return (present=true,
            shape=size(res.akpath),
            kx=kx_val,
            ky=collect(ky_vals),
            ky_idx=collect(ky_indices))
end

function compatibility_signature(res)
    omega_grid, dos_omega_grid = resolved_output_grids(res)
    meta = res.meta
    return (
        omega_grid=omega_grid,
        dos_omega_grid=dos_omega_grid,
        spectra_Lx_eff=Int(get(meta, "spectra_Lx_eff", res.params.Lx)),
        spectra_Ly_eff=Int(get(meta, "spectra_Ly_eff", res.params.Ly)),
        opt_size=size(res.opt),
        dos_size=size(res.dos),
        dos_AN_size=size(res.dos_AN),
        ak0_size=size(res.ak0),
        dos_AN_patch_size=res.dos_AN_patch === nothing ? nothing : size(res.dos_AN_patch),
        kpath=resolved_kpath_signature(res),
    )
end

function same_metadata_value(a, b)
    if a === nothing || b === nothing
        return a === nothing && b === nothing
    elseif a isa AbstractArray || b isa AbstractArray
        return a isa AbstractArray && b isa AbstractArray && size(a) == size(b) && a == b
    else
        return isequal(a, b)
    end
end

function compatibility_mismatches(reference, candidate)
    mismatches = String[]

    for field in (:omega_grid, :dos_omega_grid, :spectra_Lx_eff, :spectra_Ly_eff,
                  :opt_size, :dos_size, :dos_AN_size, :ak0_size)
        if !same_metadata_value(getfield(reference, field), getfield(candidate, field))
            push!(mismatches, String(field))
        end
    end

    if reference.dos_AN_patch_size !== nothing && candidate.dos_AN_patch_size !== nothing &&
       !same_metadata_value(reference.dos_AN_patch_size, candidate.dos_AN_patch_size)
        push!(mismatches, "dos_AN_patch_size")
    end

    if reference.kpath.present != candidate.kpath.present
        push!(mismatches, "A_kpath_presence")
    elseif reference.kpath.present
        for field in (:shape, :kx, :ky, :ky_idx)
            if !same_metadata_value(getfield(reference.kpath, field), getfield(candidate.kpath, field))
                push!(mismatches, "A_kpath_$(field)")
            end
        end
    end

    return mismatches
end

# --- 2. 处理单个 T 目录 ---
function process_T_directory(dir_path)
    conf_dirs = glob("conf_*", dir_path)
    if isempty(conf_dirs)
        remove_spectra_outputs!(dir_path)
        return
    end
    sort!(conf_dirs)
    
    println("Processing $(basename(dir_path))...")
    
    samples_opt = []
    samples_dos = []
    samples_dos_AN = []
    samples_dos_AN_patch = []
    samples_ak = []
    samples_akpath = []
    reference_params = nothing
    reference_meta = Dict{String, Any}()
    reference_signature = nothing
    
    # 收集有效样本
    for c_dir in conf_dirs
        jld_path = joinpath(c_dir, "spectra_bins.jld2")
        res = process_single_config(jld_path)
        if res !== nothing
            sig = compatibility_signature(res)
            if reference_signature === nothing
                reference_signature = sig
                reference_params = res.params
                reference_meta = res.meta
            else
                mismatches = compatibility_mismatches(reference_signature, sig)
                if !isempty(mismatches)
                    @warn "Skipping incompatible spectra config." config=c_dir mismatches=join(mismatches, ", ")
                    continue
                end
            end

            push!(samples_opt, res.opt)
            push!(samples_dos, res.dos)
            push!(samples_dos_AN, res.dos_AN)
            push!(samples_ak, res.ak0)
            if res.dos_AN_patch !== nothing
                push!(samples_dos_AN_patch, res.dos_AN_patch)
            end
            if res.akpath !== nothing
                push!(samples_akpath, res.akpath)
            end
        end
    end
    
    real_n = length(samples_opt)
    if real_n == 0
        remove_spectra_outputs!(dir_path)
        println("  -> Skipped: No valid JLD2 data.")
        return
    end
    println("  -> Valid Samples: $real_n / $(length(conf_dirs))")
    
    # 计算统计
    final_opt, err_opt = calc_final_stats(samples_opt)
    final_dos, err_dos = calc_final_stats(samples_dos)
    final_dos_AN, err_dos_AN = calc_final_stats(samples_dos_AN)
    final_dos_AN_patch = nothing
    err_dos_AN_patch = nothing
    if length(samples_dos_AN_patch) == real_n
        final_dos_AN_patch, err_dos_AN_patch = calc_final_stats(samples_dos_AN_patch)
    elseif !isempty(samples_dos_AN_patch)
        @warn "Mixed dos_AN_patch presence across configs; skipping patch output." dir=dir_path patch_configs=length(samples_dos_AN_patch) valid_configs=real_n
    end
    final_ak, err_ak = calc_final_stats(samples_ak)
    final_akpath = nothing
    err_akpath = nothing
    if !isempty(samples_akpath)
        final_akpath, err_akpath = calc_final_stats(samples_akpath)
    end
    
    # 重建网格
    p = reference_params
    omega_grid = reference_signature.omega_grid
    dos_omega_grid = reference_signature.dos_omega_grid
    
    # 写入 CSV 到 T 目录内
    open(joinpath(dir_path, "spectra_opt_cond.csv"), "w") do io
        println(io, "omega,Re_Sigma,Error")
        for i in 1:length(final_opt)
            @printf(io, "%.6f,%.6e,%.6e\n", omega_grid[i], final_opt[i], err_opt[i])
        end
    end
    
    open(joinpath(dir_path, "spectra_dos.csv"), "w") do io
        println(io, "omega,DOS,DOS_Error,DOS_AN,DOS_AN_Error")
        for i in 1:length(final_dos)
            @printf(io, "%.6f,%.6e,%.6e,%.6e,%.6e\n", 
                    dos_omega_grid[i], final_dos[i], err_dos[i], 
                    final_dos_AN[i], err_dos_AN[i])
        end
    end

    if final_dos_AN_patch !== nothing && err_dos_AN_patch !== nothing
        patch_grid = compatible_grid(dos_omega_grid, final_dos_AN_patch, "DOS_AN_patch"; source=dir_path)
        open(joinpath(dir_path, "spectra_dos_AN_patch.csv"), "w") do io
            println(io, "omega,DOS_AN_patch,Error")
            for i in 1:length(final_dos_AN_patch)
                @printf(io, "%.6f,%.6e,%.6e\n",
                        patch_grid[i], final_dos_AN_patch[i], err_dos_AN_patch[i])
            end
        end
    else
        rm(joinpath(dir_path, "spectra_dos_AN_patch.csv"); force=true)
    end
    
    open(joinpath(dir_path, "spectra_ak0.csv"), "w") do io
        println(io, "kx_idx,ky_idx,kx,ky,A_val,Error")
        Lx_eff = Int(get(reference_meta, "spectra_Lx_eff", p.Lx))
        Ly_eff = Int(get(reference_meta, "spectra_Ly_eff", p.Ly))
        Lx, Ly = resolve_ak_dims(final_ak, Lx_eff, Ly_eff; source=dir_path)
        for x in 1:Lx
            for y in 1:Ly
                kx = 2π * (x - 1) / Lx
                ky = 2π * (y - 1) / Ly
                if kx > π kx -= 2π end
                if ky > π ky -= 2π end
                @printf(io, "%d,%d,%.6f,%.6f,%.6e,%.6e\n", 
                        x, y, kx, ky, final_ak[x, y], err_ak[x, y])
            end
        end
    end

    if final_akpath !== nothing && err_akpath !== nothing
        kpath = reference_signature.kpath
        kx_val = kpath.kx
        ky_vals = kpath.ky
        ky_indices = kpath.ky_idx

        open(joinpath(dir_path, "spectra_akpath.csv"), "w") do io
            println(io, "k_idx,ky_idx,kx,ky,omega,A_val,Error")
            for (k_idx, ky_idx) in enumerate(ky_indices)
                ky = ky_vals[k_idx]
                for i in 1:length(dos_omega_grid)
                    @printf(io, "%d,%d,%.6f,%.6f,%.6f,%.6e,%.6e\n",
                            k_idx, ky_idx, kx_val, ky, dos_omega_grid[i],
                            final_akpath[k_idx, i], err_akpath[k_idx, i])
                end
            end
        end
    else
        rm(joinpath(dir_path, "spectra_akpath.csv"); force=true)
    end
end

function main()
    println("Starting Robust T-scan Spectra Processing...")

    T_dirs = glob("T_*", target_dir)
    # 数字排序
    sort!(T_dirs, by = d -> try parse(Float64, split(basename(d), "_")[2]) catch; 0.0 end)

    for t_dir in T_dirs
        process_T_directory(t_dir)
    end

    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
