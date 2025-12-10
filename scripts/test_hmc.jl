using DwaveHMC
using Printf

# 1. 设置小规模系统进行调试
Lx, Ly = 4, 4
t, tp, μ = 1.0, -0.35, -0.5
W, n_imp = 0.0, 0.0 # 无杂质
β, J = 5.0, 1.0
dt, mass = 0.02, 1.0
p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, dt, mass) 
# 注意：beta=5.0, dt=0.02 (较小), mass=1.0

println("Initialize System...")
state = initialize_state(p)
cache = initialize_cache(p)

# 初始化 H_BdG
init_static_H!(cache, p, state)
update_H_BdG!(cache, p, state)
diagonalize_H_BdG!(cache, p)

println("Initial Energy: $(compute_total_energy(cache, p, state))")

# 2. 运行几个 HMC 步
println("\nStarting HMC warmup (10 steps)...")
for i in 1:10
    acc, dH = hmc_sweep!(cache, p, state; Nt=10)
    @printf("Step %d: dH = %+.6f, Accepted = %s\n", i, dH, acc)
end

println("\nTest Finished.")