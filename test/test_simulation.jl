using DwaveHMC
using DelimitedFiles

# 模型参数
Lx, Ly = 12, 12
t, tp, μ = 1.0, -0.35, -1.4
W, n_imp = 0.0, 0.0
T, V = 0.001, 0.8
β = 1.0 / T
mass = 1.0
η = 8.0 / (Lx*Ly) * 1.0
Δω = 0.2 * η
ω_max = 4.0

# HMC 参数 (默认本地快速 smoke test，可通过环境变量覆盖)
n_therm = parse(Int, get(ENV, "HMC_TEST_N_THERM", "8"))
n_measure = parse(Int, get(ENV, "HMC_TEST_N_MEASURE", "12"))
Nt_therm_init = parse(Int, get(ENV, "HMC_TEST_NT_THERM", "12"))
Nt_measure = parse(Int, get(ENV, "HMC_TEST_NT_MEASURE", "6"))
measure_transport_freq = parse(Int, get(ENV, "HMC_TEST_TRANS_FREQ", "2"))
bin_size = parse(Int, get(ENV, "HMC_TEST_BIN_SIZE", "4"))

required_cols = ["S_L2_L2", "S_L2_0", "S_1_0", "F_0_0", "Hole_p"]
base_transport_cols = ["Sweep", "Superfluid_Stiffness", "DC_Conductivity"]
twist_transport_cols = ["Twist_Qy",
                        "Twist_Qy_Rho_Curv_Cos",
                        "Twist_Qy_Rho_Curv_Sin",
                        "Twist_Qy_Rho_Curv_Avg",
                        "Twist_Qy_Lambda_Diag",
                        "Twist_Qy_Rho_OffdiagCorrected"]

function check_output(out_dir::String; expect_twist::Bool=false)
    obs_csv = joinpath(out_dir, "observables.csv")
    data, header = readdlm(obs_csv, ',', header=true)
    header_names = String.(vec(header))
    for c in required_cols
        @assert c in header_names "Missing required observable column: $c"
    end
    @assert size(data, 1) > 0 "observables.csv has no data rows"

    trans_csv = joinpath(out_dir, "transport.csv")
    trans_data, trans_header = readdlm(trans_csv, ',', header=true)
    trans_header_names = String.(vec(trans_header))
    for c in base_transport_cols
        @assert c in trans_header_names "Missing required transport column: $c"
    end
    if expect_twist
        for c in twist_transport_cols
            @assert c in trans_header_names "Missing required twist column: $c"
        end
    else
        for c in twist_transport_cols
            @assert !(c in trans_header_names) "Twist column $c should be opt-in"
        end
    end
    @assert size(trans_data, 1) > 0 "transport.csv has no data rows"
    return data, header_names
end

# Case 1: 固定 μ
p_fixed = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, V, mass, η=η, Δω=Δω, ω_max=ω_max)
out_dir_fixed = "data/test_fixed_mu_L$(Lx)_V$(V)_T$(T)_mu$(μ)"
run_simulation(p_fixed, out_dir_fixed;
               n_therm=n_therm,
               n_measure=n_measure,
               Nt_therm_init=Nt_therm_init,
               Nt_measure=Nt_measure,
               measure_transport_freq=measure_transport_freq,
               bin_size=bin_size)
check_output(out_dir_fixed)

# Case 1b: twist benchmark is explicit opt-in
p_twist = ModelParameters(4, 4, t, tp, μ, W, n_imp, 10.0, V, mass;
                          η=8.0 / 16.0, Δω=4.0 / 16.0, ω_max=3.0)
out_dir_twist = "data/test_twist_enabled_L4"
run_simulation(p_twist, out_dir_twist;
               n_therm=1,
               n_measure=1,
               Nt_therm_init=2,
               Nt_measure=1,
               measure_transport_freq=1,
               bin_size=1,
               measure_twist=true,
               verbose=false)
check_output(out_dir_twist; expect_twist=true)

# Case 2: 目标 n (热化阶段调 μ)
target_n = parse(Float64, get(ENV, "HMC_TEST_TARGET_N", "0.85"))
μ_init = parse(Float64, get(ENV, "HMC_TEST_MU_INIT", "-1.4"))
μ_gain = parse(Float64, get(ENV, "HMC_TEST_MU_GAIN", "0.50"))
μ_interval = parse(Int, get(ENV, "HMC_TEST_MU_INTERVAL", "1"))
n_tol = parse(Float64, get(ENV, "HMC_TEST_TARGET_N_TOL", "0.10"))
p_target = ModelParameters(Lx, Ly, t, tp, W, n_imp, β, V, mass;
                           target_n=target_n, μ_init=μ_init,
                           μ_tune_gain=μ_gain, μ_tune_interval=μ_interval,
                           η=η, Δω=Δω, ω_max=ω_max)
out_dir_target = "data/test_target_n_L$(Lx)_V$(V)_T$(T)_n$(target_n)"
run_simulation(p_target, out_dir_target;
               n_therm=n_therm,
               n_measure=n_measure,
               Nt_therm_init=Nt_therm_init,
               Nt_measure=Nt_measure,
               measure_transport_freq=measure_transport_freq,
               bin_size=bin_size)
data_target, header_target = check_output(out_dir_target)
hole_idx = findfirst(==("Hole_p"), header_target)
@assert hole_idx !== nothing "Hole_p column not found in target-n run"
last_hole = Float64(data_target[end, hole_idx])
last_n = 1.0 - last_hole
@assert abs(last_n - target_n) <= n_tol "Final n=$last_n not close to target_n=$target_n (tol=$n_tol)"

println("Local simulation smoke tests passed.")
println("Verified columns: $(join(required_cols, ", "))")
println("Verified default transport columns: $(join(base_transport_cols, ", "))")
println("Target-n case: final n=$(last_n), target=$(target_n), tol=$(n_tol)")
