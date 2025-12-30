using JLD2
using Statistics
using DelimitedFiles
using Printf
using DwaveHMC # 加载你的包以识别 ModelParameters 类型

# ==========================================
# 设置
# ==========================================
# 这里填写你刚才运行的数据目录
target_dir = "data/T_scan_L24_J0.8_W1.0_imp0.0_mu_-1.4/T_0.0001" 
jld_file = joinpath(target_dir, "spectra_bins.jld2")

output_opt = joinpath(target_dir, "processed_opt_cond.csv")
output_dos = joinpath(target_dir, "processed_dos.csv")
output_dos_AN = joinpath(target_dir, "processed_dos_AN.csv")
output_ak = joinpath(target_dir, "processed_ak0.csv")

# ==========================================
# 辅助函数：计算 Mean 和 SEM
# ==========================================
function calc_stats(data_list)
    # data_list 是一个 Vector{Vector} 或 Vector{Matrix}
    # 我们将其堆叠成高维数组
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
    
    # 方差 Var = E[x^2] - (E[x])^2
    # 考虑到数值稳定性，如果样本很少，用 var函数更好，这里手动算
    # 为了得到无偏估计，乘 n/(n-1)
    var_val = (sum_sq ./ n_samples) .- (mean_val.^2)
    # 防止数值误差导致的负值
    var_val = max.(var_val, 0.0)
    
    # 标准误差 SEM = sqrt(Var / N)
    sem_val = sqrt.(var_val ./ n_samples)
    
    return mean_val, sem_val
end

# ==========================================
# 主逻辑
# ==========================================

println("Opening file: $jld_file")

jldopen(jld_file, "r") do file
    # 1. 读取参数和网格
    # 注意：我们需要重建 DOS 的网格，因为 Simulation.jl 里没存 dos_grid
    params = file["params"] # ModelParameters
    omega_grid = file["omega_grid"] # Re sigma 的网格 (ω > 0)
    
    # 重建 DOS 网格 (对称)
    dos_omega_grid = collect(-params.ω_max : params.dω : params.ω_max)
    
    println("Params: L=$(params.Lx)x$(params.Ly), Beta=$(params.β)")
    
    # 2. 遍历所有 bins 收集数据
    list_opt = Vector{Vector{Float64}}()
    list_dos = Vector{Vector{Float64}}()
    list_dos_AN = Vector{Vector{Float64}}()
    list_ak = Vector{Matrix{Float64}}()
    
    # JLD2 的 keys 是 group names
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
    
    println("Found $count bins. Calculating statistics...")
    
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
    println("Saved: $output_opt")
    
    # --- DOS ---
    # 注意：DOS网格和数据长度必须匹配。
    # 之前Observables.jl里 dos_vals 是 zeros(length(dos_ω_grid))
    if length(dos_omega_grid) != length(mean_dos)
        @warn "DOS grid size mismatch! Re-checking params."
        # 简单的回退：只存索引
        dos_omega_grid = 1:length(mean_dos)
    end
    
    open(output_dos, "w") do io
        println(io, "omega,DOS,Error")
        for i in 1:length(mean_dos)
            @printf(io, "%.6f,%.6f,%.6f\n", dos_omega_grid[i], mean_dos[i], err_dos[i])
        end
    end
    println("Saved: $output_dos")
    
    open(output_dos_AN, "w") do io
        println(io, "omega,DOS_AN,Error")
        for i in 1:length(mean_dos_AN)
            @printf(io, "%.6f,%.6f,%.6f\n", dos_omega_grid[i], mean_dos_AN[i], err_dos_AN[i])
        end
    end
    println("Saved: $output_dos_AN")
    
    # --- A(k, 0) Fermi Surface ---
    # 保存为: kx_idx, ky_idx, kx, ky, A_val, Error
    # 我们把 index 映射到动量空间 [-pi, pi]
    open(output_ak, "w") do io
        println(io, "kx_idx,ky_idx,kx,ky,A_val,Error")
        Lx, Ly = params.Lx, params.Ly
        for x in 1:Lx
            for y in 1:Ly
                # 简单的动量映射: k = 2pi * (idx-1) / L
                # 或者是移位到中心: k = 2pi * (idx - 1 - L/2) / L
                kx = 2π * (x - 1) / Lx
                ky = 2π * (y - 1) / Ly
                
                # 为了美观，通常将 k 移到 [-π, π] 区间
                if kx > π kx -= 2π end
                if ky > π ky -= 2π end
                
                @printf(io, "%d,%d,%.6f,%.6f,%.6f,%.6f\n", 
                        x, y, kx, ky, mean_ak[x, y], err_ak[x, y])
            end
        end
    end
    println("Saved: $output_ak")
    
end

println("Processing Done.")