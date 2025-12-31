using DwaveHMC
using DelimitedFiles
using Statistics
using Printf
using Plots

# ==========================================
# 1. 参数设置
# ==========================================
Lx, Ly = 12, 12
t, tp = 1.0, -0.35
μ = -1.08
W, n_imp = 1.0, 0.0
J = 0.8
mass = 1.0

η = 8.0 / (Lx*Ly) * 1.0
Δω = 0.2 * η
ω_max = 4.0

# 扫描 Beta
beta_start = 0.01
beta_end = 100000.0
n_points = 24
betas = 10 .^ range(log10(beta_start), stop=log10(beta_end), length=n_points)

# 模拟控制参数
n_therm = 20
n_measure = 100
Nt_therm = 20
Nt_measure = 6

# 关键：由于 sweep 次数很少，我们需要每步都测超流刚度才能算出误差
measure_freq = 1 
bin_size = 10 # JLD2 分箱大小 (这里主要关注 CSV，这个参数不太影响)

# 输出总目录
base_dir = "data/beta_test_L$(Lx)_J$(J)_W$(W)_imp$(n_imp)"
if !isdir(base_dir)
    mkpath(base_dir)
end

println("==================================================")
println("Start Beta Sweep Test")
println("L=$Lx, J=$J, n_imp=$(n_imp)")
println("Betas: $betas")
println("==================================================")

# ==========================================
# 2. 循环运行模拟
# ==========================================
for (i, β) in enumerate(betas)
    @printf("\n--- Processing Point %d/%d: Beta=%.2f ---\n", i, n_points, β)
    
    # 构造参数
    # 注意：eta_scale 和 domega 即使不测光谱也需要给默认值
    p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, mass, η=η, Δω=Δω, ω_max=ω_max)
    
    # 子目录
    work_dir = joinpath(base_dir, "beta_$(round(β, digits=3))")
    
    # 运行模拟
    # 这里的日志会输出到 work_dir/simulation.log，屏幕上也会有简略输出
    run_simulation(p, work_dir; 
                   n_therm=n_therm, 
                   n_measure=n_measure, 
                   Nt_therm_init=Nt_therm, 
                   Nt_measure=Nt_measure,
                   measure_transport_freq=measure_freq,
                   bin_size=bin_size)
end