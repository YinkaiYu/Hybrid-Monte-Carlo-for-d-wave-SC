using DwaveHMC

# 模型参数 (平均场, 大参数)
Lx, Ly = 24, 24
t, tp, μ = 1.0, -0.35, -1.0294222
T, J = 0.08, 0.8
β = 1.0 / T
mass = 1.0
η = 8.0 / (Lx*Ly) * 1.0
Δω = 0.2 * η
ω_max = 4.0

# 平均场参数
p = ModelParameters(Lx, Ly, t, tp, μ, β, J, mass;
                    W=0.0, n_imp=0.0,
                    η=η, Δω=Δω, ω_max=ω_max,
                    Δ_MF_0=0.2, α=0.5, Δ_MF_tol=1e-6, Δ_MF_max_iter=2000)

out_dir = "data/test_admf_L$(Lx)_J$(J)_T$(T)_mu$(μ)"

# 运行平均场迭代
run_simulation(p, out_dir; max_iter=p.Δ_MF_max_iter)
