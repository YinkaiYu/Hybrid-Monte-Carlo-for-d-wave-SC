using DwaveHMC
using DelimitedFiles
using Statistics
using Printf
using Plots

# ==========================================
# 1. 参数设置
# ==========================================
Lx, Ly = 24, 24
t, tp = 1.0, -0.35
μ = -1.08
W, n_imp = 3.0, 0.04
J = 0.8
mass = 1.0
dt_dummy = 0.05 # 占位符，会被自动计算覆盖

# 扫描 Beta
beta_start = 1.0
beta_end = 5000.0
n_points = 24
betas = 10 .^ range(log10(beta_start), stop=log10(beta_end), length=n_points)

# 模拟控制参数
n_therm = 20
n_sweep = 100
Nt_therm = 20
Nt_measure = 6

# 关键：由于 sweep 次数很少，我们需要每步都测超流刚度才能算出误差
measure_freq = 1 
bin_size = 10 # JLD2 分箱大小 (这里主要关注 CSV，这个参数不太影响)

# 输出总目录
base_dir = "data/beta_test_L$(Lx)_J$(J)_imp$(n_imp)"
if !isdir(base_dir)
    mkpath(base_dir)
end

println("==================================================")
println("Start Beta Sweep Test")
println("L=$Lx, J=$J, n_imp=$(n_imp)")
println("Betas: $betas")
println("==================================================")

# 用于存储最终汇总数据
results_beta = Float64[]
results_rho_mean = Float64[]
results_rho_err = Float64[]
results_Delta_mean = Float64[]
results_Delta_err = Float64[]
results_sigma_mean = Float64[]
results_sigma_err = Float64[]

# ==========================================
# 2. 循环运行模拟
# ==========================================
for (i, β) in enumerate(betas)
    @printf("\n--- Processing Point %d/%d: Beta=%.2f ---\n", i, n_points, β)
    
    # 构造参数
    # 注意：eta_scale 和 domega 即使不测光谱也需要给默认值
    p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, dt_dummy, mass, eta_scale=2.0, domega=0.01)
    
    # 子目录
    work_dir = joinpath(base_dir, "beta_$(round(β, digits=3))")
    
    # 运行模拟
    # 这里的日志会输出到 work_dir/simulation.log，屏幕上也会有简略输出
    run_simulation(p, work_dir; 
                   n_therm=n_therm, 
                   n_sweep=n_sweep, 
                   Nt_therm_init=Nt_therm, 
                   Nt_measure=Nt_measure,
                   measure_transport_freq=measure_freq,
                   bin_size=bin_size)
    
    # ==========================================
    # 3. 读取并分析数据
    # ==========================================
    # 读取 transport.csv
    # 格式: Sweep, Superfluid_Stiffness, DC_Conductivity
    csv_path = joinpath(work_dir, "transport.csv")
    try
        # readdlm 返回 (data, header) 元组
        raw_data, header = readdlm(csv_path, ',', header=true)
        
        # 第2列是 Superfluid Stiffness
        rhos = raw_data[:, 2]
        sigmas = raw_data[:, 3]
        
        # 简单的统计分析
        rho_avg = mean(rhos)
        rho_err = std(rhos) / sqrt(length(rhos)-1)
        sigma_avg = mean(sigmas)
        sigma_err = std(sigmas) / sqrt(length(sigmas)-1)
        
        push!(results_beta, β)
        push!(results_rho_mean, rho_avg)
        push!(results_rho_err, rho_err)
        push!(results_sigma_mean, sigma_avg)
        push!(results_sigma_err, sigma_err)
        
        @printf("Result: rho_s = %.4f +/- %.4f\n", rho_avg, rho_err)
        @printf("        sigma_DC = %.4f +/- %.4f\n", sigma_avg, sigma_err)
    catch e
        println("Error reading data for beta=$β: $e")
    end

    # 读取 observables.csv 
    # 格式: Sweep,Accepted,dH,Energy,Delta_Amp,Delta_Loc,Delta_Glob,S_Delta,Hole_p,Delta_Diff,Delta_Pair
    obs_csv_path = joinpath(work_dir, "observables.csv")
    try
        raw_obs, header = readdlm(obs_csv_path, ',', header=true)
        Delta_Pairs = raw_obs[:, 11]
        Delta_avg = mean(Delta_Pairs)
        Delta_err = std(Delta_Pairs) / sqrt(length(Delta_Pairs)-1)
        push!(results_Delta_mean, Delta_avg)
        push!(results_Delta_err, Delta_err)
        @printf("Result: <Delta_Pair> = %.4f +/- %.4f\n", Delta_avg, Delta_err)
    catch e
        println("Error reading observables for beta=$β: $e")
    end
end

# ==========================================
# 4. 保存汇总数据
# ==========================================
summary_path = joinpath(base_dir, "summary_stiffness.csv")
open(summary_path, "w") do io
    println(io, "Beta,Rho_s_mean,Rho_s_err,Delta_mean,Delta_err,sigma_mean,sigma_err")
    for k in 1:length(results_beta)
        @printf(io, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", 
                results_beta[k], 
                results_rho_mean[k], results_rho_err[k],
                results_Delta_mean[k], results_Delta_err[k],
                results_sigma_mean[k], results_sigma_err[k])
    end
end
println("\nSummary saved to $summary_path")

# # ==========================================
# # 5. 画图
# # ==========================================
# println("Generating Plot...")

# # 设置绘图风格
# default(fmt = :png, dpi = 300, grid = true, framestyle = :box)

# p1 = scatter(results_T, results_rho_mean, 
#     yerror = results_rho_err,
#     label = "Simulation Data",
#     xlabel = "Temperature (T)",
#     ylabel = "Superfluid Stiffness ρ_s",
#     title = "Superfluid Stiffness vs T (L=$Lx, J=$J)",
#     marker = (:circle, 8),
#     color = :blue,
#     legend = :topright
# )

# # 连线
# plot!(p1, results_T, results_rho_mean, label="", line=(:blue, 1))

# # 保存图片
# plot_path = joinpath(base_dir, "stiffness_vs_T.png")
# savefig(p1, plot_path)
# println("Plot saved to $plot_path")

# println("Test Completed.")