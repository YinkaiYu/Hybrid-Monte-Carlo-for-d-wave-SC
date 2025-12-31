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
W, n_imp = 1.0, 0.05
J = 0.8
mass = 1.0

η = 8.0 / (Lx*Ly) * 1.0
Δω = 0.2 * η
ω_max = 4.0

T_start = 0.0001  
T_end = 1000.0   
n_points = 24
Ts = 10 .^ range(log10(T_start), stop=log10(T_end), length=n_points)

# 如果想用线性刻度，注释掉上面一行，使用下面这行：
# Ts = range(T_start, stop=T_end, length=n_points)

# 模拟控制参数
n_therm = 20
n_measure = 100
Nt_therm = 20
Nt_measure = 6

measure_freq = 1 
bin_size = 10 

# 输出目录名称为 T_scan
base_dir = "data/T_scan_L$(Lx)_J$(J)_W$(W)_imp$(n_imp)_mu_$(μ)"
if !isdir(base_dir)
    mkpath(base_dir)
end

println("==================================================")
println("Start Temperature Sweep Test")
println("L=$Lx, J=$J, n_imp=$(n_imp)")
println("Temperatures: $Ts")
println("==================================================")

# ==========================================
# 2. 循环运行模拟
# ==========================================
# 修改：遍历 T 数组
for (i, T) in enumerate(Ts)
    β = 1.0 / T
    
    @printf("\n--- Processing Point %d/%d: T=%.5f (Beta=%.2f) ---\n", i, n_points, T, β)
    
    # 构造参数
    # 将计算出的 β 传入 ModelParameters
    p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, mass, η=η, Δω=Δω, ω_max=ω_max)
    
    # 子目录以 T 命名
    work_dir = joinpath(base_dir, "T_$(round(T, sigdigits=3))")
    
    # 运行模拟
    run_simulation(p, work_dir; 
                   n_therm=n_therm, 
                   n_measure=n_measure, 
                   Nt_therm_init=Nt_therm, 
                   Nt_measure=Nt_measure,
                   measure_transport_freq=measure_freq,
                   bin_size=bin_size)
end