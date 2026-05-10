using Dates
using Printf
using DelimitedFiles
using JLD2

"""
    calc_optimal_dt(β, V, mass, Nt)

根据谐振子近似计算推荐的时间步长 δt。
"""
function calc_optimal_dt(β, V, mass, Nt)
    T = 2 * π * sqrt(mass * V / β)
    return T / (2 * Nt) 
end

mutable struct MuRootTracker
    has_prev::Bool
    μ_prev::Float64
    err_prev::Float64
    has_lo::Bool
    μ_lo::Float64
    err_lo::Float64
    has_hi::Bool
    μ_hi::Float64
    err_hi::Float64
end

function MuRootTracker()
    return MuRootTracker(false, 0.0, 0.0, false, 0.0, 0.0, false, 0.0, 0.0)
end

function update_bracket!(tracker::MuRootTracker, μ::Float64, err::Float64)
    if err > 0
        # n < target，根在更大的 μ 方向，正误差侧取最大的 μ 以收紧区间
        if !tracker.has_lo || μ > tracker.μ_lo
            tracker.has_lo = true
            tracker.μ_lo = μ
            tracker.err_lo = err
        end
    elseif err < 0
        # n > target，根在更小的 μ 方向，负误差侧取最小的 μ 以收紧区间
        if !tracker.has_hi || μ < tracker.μ_hi
            tracker.has_hi = true
            tracker.μ_hi = μ
            tracker.err_hi = err
        end
    end
    return nothing
end

function propose_next_mu(tracker::MuRootTracker, p::ModelParameters, μ::Float64, err::Float64)
    if abs(err) <= p.μ_tune_tol
        return μ, :converged
    end
    α = clamp(p.μ_tune_gain, 0.0, 1.0)

    if tracker.has_lo && tracker.has_hi && tracker.μ_lo < tracker.μ_hi
        μ_lo = tracker.μ_lo
        μ_hi = tracker.μ_hi
        err_lo = tracker.err_lo
        err_hi = tracker.err_hi
        μ_mid = 0.5 * (μ_lo + μ_hi)
        denom = err_hi - err_lo
        μ_sec = μ_mid
        if isfinite(denom) && abs(denom) > eps(Float64)
            μ_tmp = μ_hi - err_hi * (μ_hi - μ_lo) / denom
            if μ_lo < μ_tmp < μ_hi
                μ_sec = μ_tmp
            end
        end
        μ_next = μ_mid + α * (μ_sec - μ_mid)
        return clamp(μ_next, p.μ_min, p.μ_max), :bracketed_secant
    end

    μ_prop = NaN
    mode = :step
    if tracker.has_prev
        denom = err - tracker.err_prev
        if isfinite(denom) && abs(denom) > 1e-12
            μ_try = μ - err * (μ - tracker.μ_prev) / denom
            # 未成区间时，要求更新方向与误差符号一致，避免噪声导致反向跳步
            if sign(μ_try - μ) == sign(err) || abs(μ_try - μ) < 1e-12
                μ_prop = μ_try
                mode = :secant
            end
        end
    end
    if !isfinite(μ_prop)
        step_scale = max(α, 0.25)
        μ_prop = μ + sign(err) * step_scale * p.μ_tune_step_max
        mode = :step
    end
    δμ = μ_prop - μ
    if mode == :secant
        δμ *= α
    end
    δμ = clamp(δμ, -p.μ_tune_step_max, p.μ_tune_step_max)
    μ_next = clamp(μ + δμ, p.μ_min, p.μ_max)
    return μ_next, mode
end

function tune_chemical_potential!(cache::ComputeCache, p::ModelParameters, state::SimulationState, tracker::MuRootTracker)
    old_μ = state.μ_eff
    n_meas = electron_density_from_cache(cache, p)
    err = p.target_n - n_meas
    update_bracket!(tracker, old_μ, err)
    new_μ, mode = propose_next_mu(tracker, p, old_μ, err)
    state.μ_eff = new_μ

    if new_μ != old_μ
        init_static_H!(cache, p, state)
        update_H_BdG!(cache, p, state)
        diagonalize_H_BdG!(cache, p)
    end

    tracker.has_prev = true
    tracker.μ_prev = old_μ
    tracker.err_prev = err
    δμ = new_μ - old_μ
    bracketed = tracker.has_lo && tracker.has_hi && tracker.μ_lo < tracker.μ_hi
    return old_μ, new_μ, n_meas, err, δμ, mode, bracketed
end

"""
    run_simulation(p::ModelParameters, out_dir::String; 
                   n_therm::Int=100, 
                   n_measure::Int=500, 
                   Nt_therm_init::Int=10,
                   Nt_measure::Int=5,
                   measure_transport_freq::Int=1,
                   bin_size::Int=5,
                   measure_twist::Bool=false,
                   twist_Ax::Float64=1e-3,
                   twist_qy::Float64=2π/p.Ly)

运行完整的 HMC 模拟。

# 参数
- `n_therm`: 热化步数
- `n_measure`: 测量步数
- `Nt_therm_init`: 热化初始 Leapfrog 步数
- `measure_transport_freq`: 每隔多少个 MC 步进行一次重量级测量（输运/谱）
- `bin_size`: 谱学数据分箱大小。即累积 `bin_size` 次测量后，求平均并存入 JLD2 一次。
- `measure_twist`: 是否额外计算 twist benchmark；默认关闭，避免额外对角化
- `twist_Ax`: twist 有限差分步长
- `twist_qy`: 横向调制 twist 的动量，默认 `2π/Ly`
"""
function run_simulation(p::ModelParameters, out_dir::String; 
                        n_therm::Int=100, 
                        n_measure::Int=500, 
                        Nt_therm_init::Int=10, 
                        Nt_measure::Int=5,
                        measure_transport_freq::Int=1,
                        bin_size::Int=5,
                        measure_twist::Bool=false,
                        twist_Ax::Float64=1.0e-3,
                        twist_qy::Float64=2π / p.Ly,
                        verbose::Bool=true)
    
    # --- 1. 环境准备 ---
    if !isdir(out_dir)
        mkpath(out_dir)
    end
    
    # 文件句柄
    log_path = joinpath(out_dir, "simulation.log")
    obs_csv_path = joinpath(out_dir, "observables.csv")
    trans_csv_path = joinpath(out_dir, "transport.csv") # 存标量输运结果
    spectra_jld_path = joinpath(out_dir, "spectra_bins.jld2") # 存谱学数组
    pair_scatter_jld_path = joinpath(out_dir, "pairing_scatter.jld2") # 局域配对散点
    
    f_log = open(log_path, "a")
    f_obs = open(obs_csv_path, "w")
    f_trans = open(trans_csv_path, "w")
    
    # 辅助打印 (同时打印到屏幕和日志)
    function tee_println(msg)
        ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        full_msg = "[$ts] $msg"
        println(f_log, full_msg)
        flush(f_log)
        if verbose
            println(full_msg)
        end
    end
    
    # 写入 CSV 表头
    # 基础物理量
    println(f_obs, "Sweep,Accepted,dH,Energy,Delta_Amp,Delta_Loc,Delta_Glob,S_Delta,Hole_p,Delta_Diff,Delta_Pair,Delta_LocalPair,D2,D4,Avg_d2,Avg_d4,S_L2_L2,S_L2_0,S_1_0,F_0_0")
    # 输运标量
    if measure_twist
        println(f_trans, "Sweep,Superfluid_Stiffness,DC_Conductivity,Twist_Qy,Twist_Qy_Rho_Curv_Cos,Twist_Qy_Rho_Curv_Sin,Twist_Qy_Rho_Curv_Avg,Twist_Qy_Lambda_Diag,Twist_Qy_Rho_OffdiagCorrected")
    else
        println(f_trans, "Sweep,Superfluid_Stiffness,DC_Conductivity")
    end
    
    tee_println("Starting Simulation...")
    tee_println("System: $(p.Lx)x$(p.Ly), β=$(p.β), V=$(p.V), W=$(p.W), n_imp=$(p.n_imp)")
    tee_println("Config: Therm=$n_therm, Sweep=$n_measure, TransFreq=$measure_transport_freq, BinSize=$bin_size")

    # --- 2. 初始化 ---
    tee_println("Initializing State...")
    state = initialize_state(p)
    cache = initialize_cache(p)
    if p.has_target_n
        tee_println("Mode: target_n=$(p.target_n), μ_init=$(state.μ_eff), μ_range=[$(p.μ_min), $(p.μ_max)], gain=$(p.μ_tune_gain)")
    else
        tee_println("Mode: fixed_μ=$(state.μ_eff)")
    end
    
    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)
    
    # 初始化 JLD2 文件 (写入参数信息)
    omega_grid = cache.omega_grid
    dos_omega_grid = cache.dos_omega_grid
    kx_idx, ky_indices, kx_val, ky_vals = antinode_kpath(p)
    jldsave(spectra_jld_path; params=p,
            omega_grid=omega_grid,
            dos_omega_grid=dos_omega_grid,
            kpath_kx=kx_val,
            kpath_ky=ky_vals,
            kpath_kx_idx=kx_idx,
            kpath_ky_idx=ky_indices)
    jldsave(pair_scatter_jld_path; params=p)

    # --- 3. 热化阶段 (Adaptive Thermalization) ---
    Nt_current = Nt_therm_init
    dt_current = calc_optimal_dt(p.β, p.V, p.mass, Nt_current)
    
    tee_println("--- Thermalization Start ---")
    tee_println("Init: Nt=$Nt_current, dt=$(round(dt_current, digits=5))")
    
    # 用于计算接受率窗口
    acc_window = 5
    recent_acc = 0
    μ_tracker = MuRootTracker()
    
    start_time = time()
    
    for i in 1:n_therm
        acc, dH = hmc_sweep!(cache, p, state; Nt=Nt_current, dt=dt_current)
        if acc recent_acc += 1 end
        
        # 自适应调整逻辑
        if i % acc_window == 0
            rate = recent_acc / acc_window
            recent_acc = 0 # 重置计数器
            
            old_Nt = Nt_current
            
            # 目标接受率区间: [0.60, 0.85]
            if rate < 0.60
                Nt_current += 2 # 步子太大了，多切几份
            elseif rate > 0.95 && Nt_current > 4
                Nt_current -= 2 # 步子太小了，浪费算力
            end
            
            if Nt_current != old_Nt
                dt_current = calc_optimal_dt(p.β, p.V, p.mass, Nt_current)
                tee_println(@sprintf("Therm %d/%d. Rate=%.2f. Adjust Nt: %d -> %d, dt: %.4f", 
                                     i, n_therm, rate, old_Nt, Nt_current, dt_current))
            elseif i % 20 == 0
                tee_println(@sprintf("Therm %d/%d. Rate=%.2f. Nt=%d (Stable)", i, n_therm, rate, Nt_current))
            end
        end

        if p.has_target_n && (i % p.μ_tune_interval == 0)
            old_μ, new_μ, n_meas, err, δμ, mode, bracketed = tune_chemical_potential!(cache, p, state, μ_tracker)
            tee_println(@sprintf("Therm %d/%d. Tune μ[%s%s]: %.5f -> %.5f, n=%.6f (target=%.6f, err=%+.3e, dμ=%+.3e)",
                                 i, n_therm, String(mode), bracketed ? ",bracket" : "",
                                 old_μ, new_μ, n_meas, p.target_n, err, δμ))
        end
    end
    
    if p.has_target_n
        n_final = electron_density_from_cache(cache, p)
        tee_println(@sprintf("Therm target summary: μ=%.5f, n=%.6f, target=%.6f, err=%+.3e",
                             state.μ_eff, n_final, p.target_n, p.target_n - n_final))
    end
    tee_println("Thermalization Done. Time: $(round(time() - start_time, digits=2))s")
    
    # --- 4. 测量阶段 ---
    dt_meas = calc_optimal_dt(p.β, p.V, p.mass, Nt_measure)
    tee_println("--- Measurement Start ---")
    tee_println("Settings: Nt=$Nt_measure, dt=$(round(dt_meas, digits=5))")
    
    meas_start_time = time()
    acc_total = 0
    
    # 谱学分箱缓存初始化
    bin_count = 0
    # 我们需要缓存累加值，维度需与 Observables.jl 中的 SpectrumResult 数组一致
    # 这里采用 lazy initialization (第一次测量时分配内存)
    accum_opt_cond = Vector{Float64}()
    accum_dos = Vector{Float64}()
    accum_dos_AN = Vector{Float64}()
    accum_Ak0 = Matrix{Float64}(undef, 0, 0)
    accum_Akpath = Matrix{Float64}(undef, 0, 0)
    
    for i in 1:n_measure
        # 1. HMC 演化
        acc, dH = hmc_sweep!(cache, p, state; Nt=Nt_measure, dt=dt_meas)
        if acc acc_total += 1 end
        
        # 2. 轻量级测量 (Every Step)
        obs = measure_observables(cache, p, state)
        
        # 写入 Observables CSV
        # Sweep, Accepted, dH, ...
        line = @sprintf("%d,%d,%.5e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n", 
                i, acc, dH, obs.total_energy, 
                obs.Δ_amp, obs.Δ_local, obs.Δ_global, obs.S_Δ, obs.hole_conc,
                obs.Δ_diff, obs.Δ_pair, obs.Δ_localpair,
                obs.D2, obs.D4, obs.d2_avg, obs.d4_avg,
                obs.S_L2_L2, obs.S_L2_0, obs.S_1_0, obs.F_0_0)
        write(f_obs, line)
        flush(f_obs) # 实时落盘

        jldopen(pair_scatter_jld_path, "a+") do file
            group_name = "sweep_$i"
            g = JLD2.Group(file, group_name)
            g["d_local"] = obs.d_local
        end
        
        # 3. 重量级测量 (Every Freq Step)
        if i % measure_transport_freq == 0
            # 计算输运和谱
            spec_res = measure_transport_and_spectra(cache, p; reuse_buffers=true)
            
            # A. 写入 Transport CSV (Scalars)
            if measure_twist
                twist_qy_res = measure_twist_stiffness_qy(cache, p, state;
                                                          Ax=twist_Ax, qy=twist_qy)
                line_trans = @sprintf("%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                                      i, spec_res.superfluid_stiffness,
                                      spec_res.dc_conductivity,
                                      twist_qy_res.qy,
                                      twist_qy_res.rho_curvature_cos,
                                      twist_qy_res.rho_curvature_sin,
                                      twist_qy_res.rho_curvature_avg,
                                      twist_qy_res.diag_correction,
                                      twist_qy_res.rho_offdiag_corrected)
            else
                line_trans = @sprintf("%d,%.6e,%.6e\n",
                                      i, spec_res.superfluid_stiffness,
                                      spec_res.dc_conductivity)
            end
            write(f_trans, line_trans)
            flush(f_trans)
            
            # B. 谱学数据分箱 (Binning)
            # 初始化累加器
            if bin_count == 0
                accum_opt_cond = copy(spec_res.optical_conductivity)
                accum_dos = copy(spec_res.dos)
                accum_dos_AN = copy(spec_res.dos_AN)
                accum_Ak0 = copy(spec_res.A_k_ω0)
                accum_Akpath = copy(spec_res.A_kpath)
                bin_count = 1
            else
                accum_opt_cond .+= spec_res.optical_conductivity
                accum_dos .+= spec_res.dos
                accum_dos_AN .+= spec_res.dos_AN
                accum_Ak0 .+= spec_res.A_k_ω0
                accum_Akpath .+= spec_res.A_kpath
                bin_count += 1
            end
            
            # 达到 Bin Size，写入 JLD2 并清空缓存
            if bin_count >= bin_size
                # 求平均
                accum_opt_cond ./= bin_count
                accum_dos ./= bin_count
                accum_dos_AN ./= bin_count
                accum_Ak0 ./= bin_count
                accum_Akpath ./= bin_count
                
                # JLD2 追加写入
                # 使用 string key 来区分不同的 bin，例如 "bin_100", "bin_200" 表示到第几步的 bin
                # 注意：频繁打开关闭文件有开销，但对于 bin_size * measure_freq 步才一次的操作，这是安全的
                jldopen(spectra_jld_path, "a+") do file
                    group_name = "sweep_$i"
                    g = JLD2.Group(file, group_name)
                    g["opt_cond"] = accum_opt_cond
                    g["dos"] = accum_dos
                    g["dos_AN"] = accum_dos_AN
                    g["A_k0"] = accum_Ak0
                    g["A_kpath"] = accum_Akpath
                    g["count"] = bin_count # 记录这个 bin 包含了多少个样本
                end
                
                # 重置计数器
                bin_count = 0
                # accum_... 会在下一次循环开头被覆盖，无需手动清零，但为了安全可以置空
                # 这里依赖 if bin_count == 0 分支来重新 copy
            end
        end
        
        # 进度打印
        if i % 10 == 0
             rate = acc_total / i
             if p.has_target_n
                 n_curr = 1.0 - obs.hole_conc
                 tee_println(@sprintf("Meas %d/%d. Acc=%.2f. E=%.4f. μ=%.5f, n=%.6f",
                                      i, n_measure, rate, obs.total_energy, state.μ_eff, n_curr))
             else
                 tee_println(@sprintf("Meas %d/%d. Acc=%.2f. E=%.4f", i, n_measure, rate, obs.total_energy))
             end
        end
    end
    
    tee_println("Measurement Done. Total Time: $(round(time() - meas_start_time, digits=2))s")
    
    close(f_log)
    close(f_obs)
    close(f_trans)
    # JLD2 已经在循环中关闭了
end
