using JLD2
using Statistics
using DelimitedFiles
using Printf
using DwaveHMC # 加载你的包以识别 ModelParameters 类型

# ==========================================
# 设置：根目录
# ==========================================
# 指定包含多个 T_xxx 子目录的父目录
root_dir = "data/T_scan_L24_J0.8_W1.0_imp0.05_mu_-1.08" 

# ==========================================
# 辅助函数：计算 Mean 和 SEM
# ==========================================
function calc_stats(data_list)
    n_samples = length(data_list)
    if n_samples == 0
        return nothing, nothing
    end
    
    # 假设所有元素形状相同
    raw_shape = size(data_list[1])
    
    # 累加
    sum_val = zeros(Float64, raw_shape)
    sum_sq = zeros(Float64, raw_shape)
    
    for d in data_list
        sum_val .+= d
        sum_sq .+= d.^2
    end
    
    mean_val = sum_val ./ n_samples
    
    # 方差 Var
    var_val = (sum_sq ./ n_samples) .- (mean_val.^2)
    var_val = max.(var_val, 0.0) # 防止数值误差导致的负值
    
    # 标准误差 SEM
    sem_val = sqrt.(var_val ./ n_samples)
    
    return mean_val, sem_val
end

# ==========================================
# 核心处理函数：处理单个目录
# ==========================================
function process_single_directory(target_dir)
    jld_file = joinpath(target_dir, "spectra_bins.jld2")
    
    # 检查文件是否存在
    if !isfile(jld_file)
        @warn "File not found: $jld_file. Skipping this directory."
        return
    end

    println("========================================")
    println("Processing: $target_dir")
    println("Opening file: $jld_file")

    output_opt = joinpath(target_dir, "processed_opt_cond.csv")
    output_dos = joinpath(target_dir, "processed_dos.csv")
    output_dos_AN = joinpath(target_dir, "processed_dos_AN.csv")
    output_ak = joinpath(target_dir, "processed_ak0.csv")

    jldopen(jld_file, "r") do file
        # 1. 读取参数和网格
        if !haskey(file, "params")
            @warn "No 'params' found in $jld_file. Skipping."
            return
        end
        
        params = file["params"] # ModelParameters
        omega_grid = file["omega_grid"] 
        
        # 重建 DOS 网格 (对称)
        dos_omega_grid = collect(-params.ω_max : params.Δω : params.ω_max)
        
        println("  Params: L=$(params.Lx)x$(params.Ly), Beta=$(params.β)")
        
        # 2. 遍历所有 bins 收集数据
        list_opt = Vector{Vector{Float64}}()
        list_dos = Vector{Vector{Float64}}()
        list_dos_AN = Vector{Vector{Float64}}()
        list_ak = Vector{Matrix{Float64}}()
        
        group_names = keys(file)
        count = 0
        
        for key in group_names
            if startswith(key, "sweep_")
                g = file[key]
                push!(list_opt, g["opt_cond"])
                push!(list_dos, g["dos"])
                push!(list_dos_AN, g["dos_AN"])
                push!(list_ak, g["A_k0"])
                count += 1
            end
        end
        
        if count == 0
            @warn "  No 'sweep_' data found in $jld_file."
            return
        end

        println("  Found $count bins. Calculating statistics...")
        
        # 3. 计算统计量
        mean_opt, err_opt = calc_stats(list_opt)
        mean_dos, err_dos = calc_stats(list_dos)
        mean_dos_AN, err_dos_AN = calc_stats(list_dos_AN)
        mean_ak, err_ak = calc_stats(list_ak)
        
        # 4. 写入 CSV
        
        # --- Optical Conductivity ---
        open(output_opt, "w") do io
            println(io, "omega,Re_Sigma,Error")
            for i in 1:length(omega_grid)
                @printf(io, "%.6f,%.6f,%.6f\n", omega_grid[i], mean_opt[i], err_opt[i])
            end
        end
        # println("  Saved: opt_cond")
        
        # --- DOS ---
        if length(dos_omega_grid) != length(mean_dos)
            @warn "  DOS grid size mismatch! Using index grid."
            dos_omega_grid = 1:length(mean_dos)
        end
        
        open(output_dos, "w") do io
            println(io, "omega,DOS,Error")
            for i in 1:length(mean_dos)
                @printf(io, "%.6f,%.6f,%.6f\n", dos_omega_grid[i], mean_dos[i], err_dos[i])
            end
        end
        # println("  Saved: DOS")
        
        open(output_dos_AN, "w") do io
            println(io, "omega,DOS_AN,Error")
            for i in 1:length(mean_dos_AN)
                @printf(io, "%.6f,%.6f,%.6f\n", dos_omega_grid[i], mean_dos_AN[i], err_dos_AN[i])
            end
        end
        # println("  Saved: DOS_AN")
        
        # --- A(k, 0) Fermi Surface ---
        open(output_ak, "w") do io
            println(io, "kx_idx,ky_idx,kx,ky,A_val,Error")
            Lx, Ly = params.Lx, params.Ly
            for x in 1:Lx
                for y in 1:Ly
                    kx = 2π * (x - 1) / Lx
                    ky = 2π * (y - 1) / Ly
                    
                    if kx > π kx -= 2π end
                    if ky > π ky -= 2π end
                    
                    @printf(io, "%d,%d,%.6f,%.6f,%.6f,%.6f\n", 
                            x, y, kx, ky, mean_ak[x, y], err_ak[x, y])
                end
            end
        end
        # println("  Saved: A_k0")
        
        println("  Done processing $target_dir")
    end
end

# ==========================================
# 主逻辑：扫描并循环
# ==========================================

if !isdir(root_dir)
    error("Root directory does not exist: $root_dir")
end

println("Scanning directory: $root_dir")

# 获取所有文件/文件夹
all_entries = readdir(root_dir)

# 过滤：必须是文件夹，且名字以 "T_" 开头
subdirs = filter(entry -> startswith(entry, "T_") && isdir(joinpath(root_dir, entry)), all_entries)

# (可选) 按照 T 的数值大小排序，这样处理日志比较好看
# 尝试解析 "T_xxx" 中的 xxx 为数字进行排序，如果解析失败则按字母序
sort!(subdirs, by = x -> try parse(Float64, replace(x, "T_" => "")) catch; x end)

if isempty(subdirs)
    println("No directories starting with 'T_' found in $root_dir")
else
    println("Found $(length(subdirs)) directories to process.")
    
    for subdir in subdirs
        full_path = joinpath(root_dir, subdir)
        try
            process_single_directory(full_path)
        catch e
            @error "Error processing $subdir: $e"
            # 继续处理下一个，不中断整个脚本
        end
    end
end

println("\nAll tasks completed.")