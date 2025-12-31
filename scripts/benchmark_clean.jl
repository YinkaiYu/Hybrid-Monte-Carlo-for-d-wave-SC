using DwaveHMC
using Printf
using LinearAlgebra
using Statistics
using DelimitedFiles

function calc_optimal_dt(β, J, mass, Nt)
    T = 2 * π * sqrt(mass * J / β) # 周期
    return T / (2 * Nt) # 半周期分 Nt 步
end

# ==========================================
# 1. 定义解析计算函数 (RHS of Gap Equation)
# ==========================================
function calc_BCS_RHS(Δ_in, Lx, Ly, t, tp, μ, β, J)
    N = Lx * Ly
    sum_term = 0.0
    
    # 遍历 k 空间 (离散化)
    # k_x = 2π * nx / Lx
    for ny in 0:Ly-1, nx in 0:Lx-1
        kx = 2π * nx / Lx
        ky = 2π * ny / Ly
        
        # 色散关系
        ε_k = -2*t*(cos(kx) + cos(ky)) - 4*tp*cos(kx)*cos(ky) - μ
        
        # d-wave 结构因子 g_k = cos(kx) - cos(ky)
        g_k = cos(kx) - cos(ky)
        Δ_k = Δ_in * g_k
        
        E_k = sqrt(ε_k^2 + abs2(Δ_k))
        
        # 自洽方程右边项: (J/N) * sum_k g_k * (Δ_k / 2E_k) * tanh(βE_k/2)
        # 注意 Δ_k 包含 Δ_in，这里我们提取 Δ_in，
        # term = (g_k * g_k) * (1 / 2E_k) * tanh(...)
        
        val = (g_k^2) / (2 * E_k) * tanh(0.5 * β * E_k)
        sum_term += val
    end
    
    return (J / N) * sum_term * Δ_in
end

# ==========================================
# 2. HMC 模拟与比较
# ==========================================
function run_benchmark_clean()
    println("=== Benchmark: Clean Limit (Momentum Space Check) ===")
    
    # 参数设置
    Lx, Ly = 10, 10 
    t, tp = 1.0, -0.35
    μ = -1.08
    W, n_imp = 0.0, 0.0 # Clean limit
    β = 180.0  
    J = 1.6   
    
    
    # 自动步长
    dt_dummy = 0.05
    mass = 1.0

    # 采样参数
    n_therm = 50 # 热化样本数
    n_measure = 100 # 测量样本数
    Nt_therm = 20 # 热化时的 leapfrog 步数
    Nt_measure = 5 # 测量时的 leapfrog 步数
    
    p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, mass)
    
    # 初始化
    state = initialize_state(p)
    cache = initialize_cache(p)
    
    # 为了加快收敛，给一个稍微合理的初态 (uniform d-wave)
    # Δ_x = 0.2, Δ_y = -0.2
    fill!(state.Δ, 0.0 + 0.0im)
    state.Δ[:, 1] .= 0.2
    state.Δ[:, 2] .= -0.2
    
    init_static_H!(cache, p, state)
    
    # 热化
    println("Thermalizing...")
    dt_therm = calc_optimal_dt(p.β, p.J, p.mass, Nt_therm)
    for i in 1:n_therm
        hmc_sweep!(cache, p, state; Nt=Nt_therm, dt=dt_therm)
    end
    
    # 测量
    println("Measuring...")
    dt_meas = calc_optimal_dt(p.β, p.J, p.mass, Nt_measure)
    Δ_history = Float64[]
    
    for i in 1:n_measure
        hmc_sweep!(cache, p, state; Nt=Nt_measure, dt=dt_meas)
        obs = measure_observables(cache, p, state)
        println(@sprintf("Sweep %d: |Δ_diff| = %.6f, |Δ_pair| = %.6f, |Δ_global| = %.6f", i, obs.Δ_diff, obs.Δ_pair, obs.Δ_global))
        push!(Δ_history, obs.Δ_global)
    end
    
    # HMC 结果
    Δ_hmc_mean = mean(Δ_history)
    Δ_hmc_std = std(Δ_history)
    
    println("\n--- Results ---")
    @printf("HMC <|Δ_global|>: %.6f +/- %.6f\n", Δ_hmc_mean, Δ_hmc_std)
    
    # 代入自洽方程检验
    # 我们把 HMC 算出来的 Δ 扔进方程右边，看看能不能算回自己
    Δ_rhs = calc_BCS_RHS(Δ_hmc_mean, Lx, Ly, t, tp, μ, β, J)
    
    @printf("BCS RHS(Δ_hmc)  : %.6f\n", Δ_rhs)
    
    diff = abs(Δ_hmc_mean - Δ_rhs)
    @printf("Difference      : %.6f (%.2f%%)\n", diff, diff/Δ_hmc_mean*100)
    
    if diff < 0.02 # 允许一定的误差 (有限尺寸 + 统计涨落)
        println(">>> Benchmark 1 Passed! HMC result matches Mean Field theory.")
    else
        println(">>> Benchmark 1 Warning! Deviation too large. Check params or equilibration.")
    end
end

run_benchmark_clean()