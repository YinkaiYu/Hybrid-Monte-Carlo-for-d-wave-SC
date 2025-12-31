using DwaveHMC
using LinearAlgebra
using Statistics
using Printf
using Plots
using DelimitedFiles

# ==========================================
# 辅助函数
# ==========================================

function calc_optimal_dt(β, J, mass, Nt)
    T = 2 * π * sqrt(mass * J / β) 
    return T / (2 * Nt) 
end

function calc_BCS_RHS(Δ_in, Lx, Ly, t, tp, μ, β, J)
    if abs(Δ_in) < 1e-9; return 0.0; end
    sum_term = 0.0
    for ny in 0:Ly-1, nx in 0:Lx-1
        kx = 2π * nx / Lx; ky = 2π * ny / Ly
        ε_k = -2*t*(cos(kx) + cos(ky)) - 4*tp*cos(kx)*cos(ky) - μ
        g_k = cos(kx) - cos(ky) 
        E_k = sqrt(ε_k^2 + abs2(Δ_in * g_k))
        term = (g_k^2) / (2 * E_k) * tanh(0.5 * β * E_k)
        sum_term += term
    end
    return (J / (Lx*Ly)) * sum_term * Δ_in
end

# ==========================================
# 主程序
# ==========================================

function run_beta_scan()
    # 1. 扫描配置
    beta_start = 1.0
    beta_end = 5000.0
    n_points = 12 
    betas = 10 .^ range(log10(beta_start), stop=log10(beta_end), length=n_points)

    println("=== Benchmark: Beta Scan ($(beta_start) -> $(beta_end)) ===")
    
    # 2. 物理与算法参数
    Lx, Ly = 12, 12
    t, tp = 1.0, -0.35
    μ = -1.08
    W, n_imp = 3.0, 0.0
    J = 1.6
    mass = 1.0
    dt_dummy = 0.05
    n_therm = 60    
    n_measure = 120 
    Nt_therm = 20
    Nt_measure = 5
    
    # 3. 存储结果
    avg_Δ_global = zeros(n_points); err_Δ_global = zeros(n_points)
    avg_Δ_pair   = zeros(n_points); err_Δ_pair   = zeros(n_points)
    avg_Δ_rhs    = zeros(n_points)
    avg_Δ_diff   = zeros(n_points); err_Δ_diff   = zeros(n_points)
    acc_rates    = zeros(n_points) # 存储接受率
    
    # 4. 初始化
    p_init = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, betas[1], J, mass)
    state = initialize_state(p_init)
    cache = initialize_cache(p_init)
    
    # 初始种子
    state.Δ[:, 1] .= 0.01
    state.Δ[:, 2] .= -0.01
    init_static_H!(cache, p_init, state)
    
    # 5. 循环扫描
    println("Starting scan over $(n_points) beta points...")
    # 修改 Header，增加 AccRate 列
    @printf("%-10s | %-8s | %-10s | %-10s | %-10s | %-10s\n", "Beta", "AccRate", "Global", "Pair", "RHS", "Diff")
    println("-"^75)
    
    for (idx, β) in enumerate(betas)
        p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, mass)
        
        # 热化
        dt_therm = calc_optimal_dt(p.β, p.J, p.mass, Nt_therm)
        for _ in 1:n_therm
            hmc_sweep!(cache, p, state; Nt=Nt_therm, dt=dt_therm)
        end
        
        # 测量
        samples_global = zeros(n_measure)
        samples_pair   = zeros(n_measure)
        samples_diff   = zeros(n_measure)
        
        dt_meas = calc_optimal_dt(p.β, p.J, p.mass, Nt_measure)
        
        n_accepted = 0 # 计数器
        
        for m in 1:n_measure
            accepted, _ = hmc_sweep!(cache, p, state; Nt=Nt_measure, dt=dt_meas)
            
            if accepted
                n_accepted += 1
            end
            
            obs = measure_observables(cache, p, state)
            samples_global[m] = obs.Δ_global
            samples_pair[m]   = obs.Δ_pair
            samples_diff[m]   = obs.Δ_diff
        end
        
        # 计算统计量
        current_acc_rate = n_accepted / n_measure
        acc_rates[idx] = current_acc_rate
        
        avg_Δ_global[idx] = mean(samples_global)
        avg_Δ_pair[idx]   = mean(samples_pair)
        avg_Δ_diff[idx]   = mean(samples_diff)
        
        err_Δ_global[idx] = std(samples_global) / sqrt(n_measure)
        err_Δ_pair[idx]   = std(samples_pair) / sqrt(n_measure)
        err_Δ_diff[idx]   = std(samples_diff) / sqrt(n_measure)
        
        avg_Δ_rhs[idx] = calc_BCS_RHS(avg_Δ_pair[idx], Lx, Ly, t, tp, μ, β, J)
        
        # 打印包含接受率的信息
        @printf("%-10.2f | %-8.2f | %-10.5f | %-10.5f | %-10.5f | %-10.5f\n", 
                β, current_acc_rate, avg_Δ_global[idx], avg_Δ_pair[idx], avg_Δ_rhs[idx], avg_Δ_diff[idx])
    end
    
    # # ==========================================
    # # 6. 画图 (修改 X 轴刻度)
    # # ==========================================
    # println("\nPlotting results...")
    
    # # 定义自定义刻度：1, 10, 100, 1000
    # my_xticks = [1, 10, 100, 1000]
    
    # default(grid=true, frame=:box, lw=1.5, markersize=4, xscale=:log10, minorgrid=true)

    # # 图 1
    # p1 = plot(betas, avg_Δ_global, yerror=err_Δ_global, label="HMC Global", marker=:circle)
    # plot!(p1, betas, avg_Δ_pair, yerror=err_Δ_pair, label="HMC Pair", marker=:rect)
    # plot!(p1, betas, avg_Δ_rhs, label="BCS RHS", ls=:dash, color=:black)
    
    # plot!(p1, xlabel="Inverse Temperature (β)", ylabel="|Δ|",
    #       title="Order Parameter Benchmark",
    #       legend=:bottomright,
    #       xticks=my_xticks) # <--- 显式指定刻度
        
    # # 图 2
    # diff_GP = avg_Δ_global .- avg_Δ_pair
    # diff_GR = avg_Δ_global .- avg_Δ_rhs
    # err_GP = sqrt.(err_Δ_global.^2 .+ err_Δ_pair.^2)
    
    # p2 = plot(betas, diff_GP, yerror=err_GP, label="Global - Pair", marker=:diamond)
    # plot!(p2, betas, diff_GR, label="Global - RHS", marker=:utriangle)
    # plot!(p2, betas, avg_Δ_diff, yerror=err_Δ_diff, label="HMC Δ_diff", marker=:hline)
    
    # plot!(p2, xlabel="Inverse Temperature (β)", ylabel="Difference",
    #       title="Consistency Check",
    #       legend=:topright,
    #       xticks=my_xticks) # <--- 显式指定刻度
    
    # savefig(p1, "benchmark_beta_scan_values.png")
    # savefig(p2, "benchmark_beta_scan_errors.png")
    
    # 保存 CSV
    header = "Beta,AccRate,Global,Err_Global,Pair,Err_Pair,RHS,Diff,Err_Diff"
    data = [betas acc_rates avg_Δ_global err_Δ_global avg_Δ_pair err_Δ_pair avg_Δ_rhs avg_Δ_diff err_Δ_diff]
    open("benchmark_beta_scan.csv", "w") do io
        println(io, header)
        writedlm(io, data, ',')
    end
    
    println("Done! AccRates saved to CSV and printed to console.")
end

run_beta_scan()