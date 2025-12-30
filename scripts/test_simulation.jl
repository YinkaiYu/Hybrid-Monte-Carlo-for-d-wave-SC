using DwaveHMC

# 参数设置
Lx, Ly = 24, 24 # 稍微大一点的系统
t, tp, μ = 1.0, -0.35, -1.4
W, n_imp = 1.0, 0.19
T, J = 0.0001, 0.8
mass = 1.0
dt_dummy = 0.1 # 这里的值会被 run_simulation 里的自动计算覆盖，给个初始值即可

p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, 1/T, J, dt_dummy, mass, eta_scale=1.0, domega=0.001)
# 输出目录
out_dir = "data/test_output_L$(Lx)_J$(J)_W$(W)_imp$(n_imp)_T$(T)_mu$(μ)"

# 运行模拟
# 本地测试步数少一点
println("Calling run_simulation...")
run_simulation(p, out_dir; 
               n_therm=20, 
               n_sweep=100, 
               Nt_therm_init=20, 
               Nt_measure=6,
               measure_transport_freq=1, # 每1步测一次谱
               bin_size=2) # 每2次测量存一次盘 (即每10步存一次JLD2)

println("Simulation finished. Check directory: $out_dir") 