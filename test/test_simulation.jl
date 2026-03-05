using DwaveHMC
using DelimitedFiles

# 模型参数
Lx, Ly = 12, 12
t, tp, μ = 1.0, -0.35, -1.4
W, n_imp = 1.0, 0.0
T, J = 0.001, 0.8
β = 1.0 / T
mass = 1.0
η = 8.0 / (Lx*Ly) * 1.0
Δω = 0.2 * η
ω_max = 4.0
p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, mass, η=η, Δω=Δω, ω_max=ω_max)

# HMC 参数 (默认本地快速 smoke test，可通过环境变量覆盖)
n_therm = parse(Int, get(ENV, "HMC_TEST_N_THERM", "8"))
n_measure = parse(Int, get(ENV, "HMC_TEST_N_MEASURE", "12"))
Nt_therm_init = parse(Int, get(ENV, "HMC_TEST_NT_THERM", "12"))
Nt_measure = parse(Int, get(ENV, "HMC_TEST_NT_MEASURE", "6"))
measure_transport_freq = parse(Int, get(ENV, "HMC_TEST_TRANS_FREQ", "2"))
bin_size = parse(Int, get(ENV, "HMC_TEST_BIN_SIZE", "4"))
out_dir = "data/test_for_L$(Lx)_J$(J)_W$(W)_imp$(n_imp)_T$(T)_mu$(μ)"

# 运行模拟
run_simulation(p, out_dir; 
               n_therm=n_therm, 
               n_measure=n_measure, 
               Nt_therm_init=Nt_therm_init, 
               Nt_measure=Nt_measure,
               measure_transport_freq=measure_transport_freq,
               bin_size=bin_size)

# 检查 observables.csv 是否包含新增相位关联列
obs_csv = joinpath(out_dir, "observables.csv")
data, header = readdlm(obs_csv, ',', header=true)
header_names = String.(vec(header))
required_cols = ["S_L2_L2", "S_L2_0", "S_1_0", "F_0_0"]

for c in required_cols
    @assert c in header_names "Missing required observable column: $c"
end
@assert size(data, 1) > 0 "observables.csv has no data rows"

println("Local simulation smoke test passed.")
println("Verified phase-correlation columns: $(join(required_cols, ", "))")
