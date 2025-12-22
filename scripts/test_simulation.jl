using DwaveHMC

# 参数设置
Lx, Ly = 10, 10 # 稍微大一点的系统
t, tp, μ = 1.0, -0.35, -1.08
W, n_imp = 3.0, 0.08
β, J = 40.0, 0.8
mass = 1.0
dt_dummy = 0.1 # 这里的值会被 run_simulation 里的自动计算覆盖，给个初始值即可

p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, dt_dummy, mass)

# 输出目录
out_dir = "data/test_output_L$(Lx)_W$(W)_n_imp$(n_imp)"

# 运行模拟
# 本地测试步数少一点
println("Calling run_simulation...")
run_simulation(p, out_dir; 
               n_therm=50, 
               n_sweep=100, 
               Nt_therm=20, 
               Nt_measure=5)

println("Simulation finished. Check directory: $out_dir")