using DwaveHMC

# 模型参数
Lx, Ly = 24, 24
t, tp, μ = 1.0, -0.35, -1.4
W, n_imp = 1.0, 0.0
T, J = 0.001, 0.8
β = 1.0 / T
mass = 1.0
η = 8.0 / (Lx*Ly) * 1.0
Δω = 0.2 * η
ω_max = 4.0
p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, mass, η=η, Δω=Δω, ω_max=ω_max)

# HMC 参数
n_therm = 20
n_measure = 100
Nt_therm_init = 20
Nt_measure = 6
measure_transport_freq = 1
bin_size = 5
out_dir = "data/test_spectra_L$(Lx)_J$(J)_W$(W)_imp$(n_imp)_T$(T)_mu$(μ)"

# 运行模拟
run_simulation(p, out_dir; 
               n_therm=n_therm, 
               n_measure=n_measure, 
               Nt_therm_init=Nt_therm_init, 
               Nt_measure=Nt_measure,
               measure_transport_freq=measure_transport_freq,
               bin_size=bin_size) 