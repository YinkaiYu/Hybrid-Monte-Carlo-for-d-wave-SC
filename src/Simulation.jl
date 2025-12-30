using Dates
using Printf
using DelimitedFiles
using JLD2

"""
    calc_optimal_dt(β, J, mass, Nt)

根据谐振子近似计算推荐的时间步长 δt。
"""
function calc_optimal_dt(β, J, mass, Nt)
    T = 2 * π * sqrt(mass * J / β) 
    return T / (2 * Nt) 
end

"""
    run_simulation(p::ModelParameters, out_dir::String; 
                   n_therm::Int=100, 
                   n_sweep::Int=500, 
                   Nt_therm_init::Int=10, 
                   Nt_measure::Int=5,
                   measure_transport_freq::Int=10,
                   bin_size::Int=5)

运行完整的 HMC 模拟。

# 参数
- `n_therm`: 热化步数
- `n_sweep`: 测量步数
- `Nt_therm_init`: 热化初始 Leapfrog 步数
- `measure_transport_freq`: 每隔多少个 MC 步进行一次重量级测量（输运/谱）
- `bin_size`: 谱学数据分箱大小。即累积 `bin_size` 次测量后，求平均并存入 JLD2 一次。
"""
function run_simulation(p::ModelParameters, out_dir::String; 
                        n_therm::Int=100, 
                        n_sweep::Int=500, 
                        Nt_therm_init::Int=10, 
                        Nt_measure::Int=5,
                        measure_transport_freq::Int=1,
                        bin_size::Int=5)
    
    # --- 1. 环境准备 ---
    if !isdir(out_dir)
        mkpath(out_dir)
    end
    
    # 文件句柄
    log_path = joinpath(out_dir, "simulation.log")
    obs_csv_path = joinpath(out_dir, "observables.csv")
    trans_csv_path = joinpath(out_dir, "transport.csv") # 存标量输运结果
    spectra_jld_path = joinpath(out_dir, "spectra_bins.jld2") # 存谱学数组
    
    f_log = open(log_path, "a")
    f_obs = open(obs_csv_path, "w")
    f_trans = open(trans_csv_path, "w")
    
    # 辅助打印 (同时打印到屏幕和日志)
    function tee_println(msg)
        ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        full_msg = "[$ts] $msg"
        println(full_msg)
        println(f_log, full_msg)
        flush(f_log)
    end
    
    # 写入 CSV 表头
    # 基础物理量
    println(f_obs, "Sweep,Accepted,dH,Energy,Delta_Amp,Delta_Loc,Delta_Glob,S_Delta,Hole_p,Delta_Diff,Delta_Pair,Delta_LocalPair")
    # 输运标量
    println(f_trans, "Sweep,Superfluid_Stiffness,DC_Conductivity")
    
    tee_println("Starting Simulation...")
    tee_println("System: $(p.Lx)x$(p.Ly), β=$(p.β), n_imp=$(p.n_imp), J=$(p.J)")
    tee_println("Config: Therm=$n_therm, Sweep=$n_sweep, TransFreq=$measure_transport_freq, BinSize=$bin_size")

    # --- 2. 初始化 ---
    tee_println("Initializing State...")
    state = initialize_state(p)
    cache = initialize_cache(p)
    
    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)
    
    # 初始化 JLD2 文件 (写入参数信息)
    jldsave(spectra_jld_path; params=p, omega_grid=collect(p.ω_min:p.dω:p.ω_max))

    # --- 3. 热化阶段 (Adaptive Thermalization) ---
    Nt_current = Nt_therm_init
    dt_current = calc_optimal_dt(p.β, p.J, p.mass, Nt_current)
    
    tee_println("--- Thermalization Start ---")
    tee_println("Init: Nt=$Nt_current, dt=$(round(dt_current, digits=5))")
    
    # 用于计算接受率窗口
    therm_window = 5 
    recent_acc = 0
    
    start_time = time()
    
    for i in 1:n_therm
        acc, dH = hmc_sweep!(cache, p, state; Nt=Nt_current, dt=dt_current)
        if acc recent_acc += 1 end
        
        # 自适应调整逻辑
        if i % therm_window == 0
            rate = recent_acc / therm_window
            recent_acc = 0 # 重置计数器
            
            old_Nt = Nt_current
            
            # 目标接受率区间: [0.60, 0.85]
            if rate < 0.60
                Nt_current += 2 # 步子太大了，多切几份
            elseif rate > 0.95 && Nt_current > 4
                Nt_current -= 1 # 步子太小了，浪费算力
            end
            
            if Nt_current != old_Nt
                dt_current = calc_optimal_dt(p.β, p.J, p.mass, Nt_current)
                tee_println(@sprintf("Therm %d/%d. Rate=%.2f. Adjust Nt: %d -> %d, dt: %.4f", 
                                     i, n_therm, rate, old_Nt, Nt_current, dt_current))
            elseif i % 20 == 0
                tee_println(@sprintf("Therm %d/%d. Rate=%.2f. Nt=%d (Stable)", i, n_therm, rate, Nt_current))
            end
        end
    end
    
    tee_println("Thermalization Done. Time: $(round(time() - start_time, digits=2))s")
    
    # --- 4. 测量阶段 ---
    dt_meas = calc_optimal_dt(p.β, p.J, p.mass, Nt_measure)
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
    
    for i in 1:n_sweep
        # 1. HMC 演化
        acc, dH = hmc_sweep!(cache, p, state; Nt=Nt_measure, dt=dt_meas)
        if acc acc_total += 1 end
        
        # 2. 轻量级测量 (Every Step)
        obs = measure_observables(cache, p, state)
        
        # 写入 Observables CSV
        # Sweep, Accepted, dH, ...
        line = @sprintf("%d,%d,%.5e,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", 
                i, acc, dH, obs.total_energy, 
                obs.Δ_amp, obs.Δ_local, obs.Δ_global, obs.S_Δ, obs.hole_conc,
                obs.Δ_diff, obs.Δ_pair, obs.Δ_localpair)
        write(f_obs, line)
        flush(f_obs) # 实时落盘
        
        # 3. 重量级测量 (Every Freq Step)
        if i % measure_transport_freq == 0
            # 计算输运和谱
            spec_res = measure_transport_and_spectra(cache, p)
            
            # A. 写入 Transport CSV (Scalars)
            line_trans = @sprintf("%d,%.6f,%.6f\n", 
                                  i, spec_res.superfluid_stiffness, spec_res.dc_conductivity)
            write(f_trans, line_trans)
            flush(f_trans)
            
            # B. 谱学数据分箱 (Binning)
            # 初始化累加器
            if bin_count == 0
                accum_opt_cond = copy(spec_res.optical_conductivity)
                accum_dos = copy(spec_res.dos)
                accum_dos_AN = copy(spec_res.dos_AN)
                accum_Ak0 = copy(spec_res.A_k_ω0)
                bin_count = 1
            else
                accum_opt_cond .+= spec_res.optical_conductivity
                accum_dos .+= spec_res.dos
                accum_dos_AN .+= spec_res.dos_AN
                accum_Ak0 .+= spec_res.A_k_ω0
                bin_count += 1
            end
            
            # 达到 Bin Size，写入 JLD2 并清空缓存
            if bin_count >= bin_size
                # 求平均
                accum_opt_cond ./= bin_count
                accum_dos ./= bin_count
                accum_dos_AN ./= bin_count
                accum_Ak0 ./= bin_count
                
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
             tee_println(@sprintf("Meas %d/%d. Acc=%.2f. E=%.4f", i, n_sweep, rate, obs.total_energy))
        end
    end
    
    tee_println("Measurement Done. Total Time: $(round(time() - meas_start_time, digits=2))s")
    
    close(f_log)
    close(f_obs)
    close(f_trans)
    # JLD2 已经在循环中关闭了
end