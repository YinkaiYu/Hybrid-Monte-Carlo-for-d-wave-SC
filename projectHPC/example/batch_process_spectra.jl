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
            count = 1
            
            # 累加后续 bin
            for i in 2:length(sweep_keys)
                g = file[sweep_keys[i]]
                sum_opt .+= g["opt_cond"]
                sum_dos .+= g["dos"]
                sum_dos_AN .+= g["dos_AN"]
                sum_ak .+= g["A_k0"]
                count += 1
            end
            
            p = file["params"]
            
            # 计算平均
            res = (sum_opt ./ count, sum_dos ./ count, sum_dos_AN ./ count, sum_ak ./ count, p)
            
            # 检查 NaN
            if any(isnan, res[1]) || any(isnan, res[2])
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

# --- 2. 处理单个 T 目录 ---
function process_T_directory(dir_path)
    conf_dirs = glob("conf_*", dir_path)
    if isempty(conf_dirs) return end
    
    println("Processing $(basename(dir_path))...")
    
    samples_opt = []
    samples_dos = []
    samples_dos_AN = []
    samples_ak = []
    last_params = nothing
    
    # 收集有效样本
    for c_dir in conf_dirs
        jld_path = joinpath(c_dir, "spectra_bins.jld2")
        res = process_single_config(jld_path)
        if res !== nothing
            push!(samples_opt, res[1])
            push!(samples_dos, res[2])
            push!(samples_dos_AN, res[3])
            push!(samples_ak, res[4])
            last_params = res[5]
        end
    end
    
    real_n = length(samples_opt)
    if real_n == 0
        println("  -> Skipped: No valid JLD2 data.")
        return
    end
    println("  -> Valid Samples: $real_n / $(length(conf_dirs))")
    
    # 计算统计
    final_opt, err_opt = calc_final_stats(samples_opt)
    final_dos, err_dos = calc_final_stats(samples_dos)
    final_dos_AN, err_dos_AN = calc_final_stats(samples_dos_AN)
    final_ak, err_ak = calc_final_stats(samples_ak)
    
    # 重建网格
    p = last_params
    omega_grid = collect(p.ω_min : p.Δω : p.ω_max)
    dos_omega_grid = collect(-p.ω_max : p.Δω : p.ω_max)
    
    # 防御性修正网格长度
    if length(omega_grid) != length(final_opt)
        omega_grid = range(p.ω_min, stop=p.ω_max, length=length(final_opt))
    end
    if length(dos_omega_grid) != length(final_dos)
        dos_omega_grid = range(-p.ω_max, stop=p.ω_max, length=length(final_dos))
    end
    
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
    
    open(joinpath(dir_path, "spectra_ak0.csv"), "w") do io
        println(io, "kx_idx,ky_idx,kx,ky,A_val,Error")
        Lx, Ly = p.Lx, p.Ly
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
end

# --- 3. 主程序 ---
println("Starting Robust T-scan Spectra Processing...")

T_dirs = glob("T_*", target_dir)
# 数字排序
sort!(T_dirs, by = d -> try parse(Float64, split(basename(d), "_")[2]) catch; 0.0 end)

for t_dir in T_dirs
    process_T_directory(t_dir)
end

println("Done.")