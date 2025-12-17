using Dates
using Printf
using DelimitedFiles

"""
    calc_optimal_dt(β, J, mass, Nt)

根据谐振子近似计算推荐的时间步长 δt。
δt = (2π / Nt) * sqrt(mJ / β)
"""
function calc_optimal_dt(β, J, mass, Nt)
    T = 2 * π * sqrt(mass * J / β) # 周期
    return T / (2 * Nt) # 半周期分 Nt 步
end

"""
    run_simulation(p::ModelParameters, out_dir::String; 
                   n_therm::Int=1000, 
                   n_sweep::Int=5000, 
                   Nt_therm::Int=10, 
                   Nt_measure::Int=5)

运行完整的 HMC 模拟。
"""
function run_simulation(p::ModelParameters, out_dir::String; 
                        n_therm::Int=100, 
                        n_sweep::Int=500, 
                        Nt_therm::Int=10, 
                        Nt_measure::Int=5)
    
    # 1. 准备输出目录和文件
    if !isdir(out_dir)
        mkpath(out_dir)
    end
    
    # 日志文件
    log_path = joinpath(out_dir, "simulation.log")
    f_log = open(log_path, "a") # append mode
    
    # 数据文件
    data_path = joinpath(out_dir, "observables.csv")
    f_data = open(data_path, "w")
    
    # 辅助打印函数 (同时打印到屏幕和日志)
    function tee_println(msg)
        ts = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
        full_msg = "[$ts] $msg"
        println(full_msg)
        println(f_log, full_msg)
        flush(f_log)
    end
    
    tee_println("Starting Simulation...")
    tee_println("Params: Lx=$(p.Lx), Ly=$(p.Ly), β=$(p.β), n_imp=$(p.n_imp)")
    
    # 2. 初始化
    tee_println("Initializing State and Cache...")
    state = initialize_state(p)
    cache = initialize_cache(p)
    
    # 必须先初始化 H_base
    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p) # 初始对角化
    
    # 3. 热化阶段 (Thermalization)
    dt_therm = calc_optimal_dt(p.β, p.J, p.mass, Nt_therm)
    tee_println("--- Thermalization Start ---")
    tee_println("Target Sweeps: $n_therm, Nt=$Nt_therm, dt=$(round(dt_therm, digits=4))")
    
    acc_count = 0
    start_time = time()
    
    for i in 1:n_therm
        # 运行一步 HMC，强制使用热化参数
        accepted, dH = hmc_sweep!(cache, p, state; Nt=Nt_therm, dt=dt_therm)
        
        if accepted
            acc_count += 1
        end
        
        # 每 10% 进度打印一次
        if i % max(1, n_therm ÷ 10) == 0
            rate = acc_count / i
            tee_println(@sprintf("Therm Step %d/%d. AccRate=%.2f. dH=%.4f", i, n_therm, rate, dH))

            # # 自适应调整 dt_therm 以维持接受率在合理范围
            # old_Nt = Nt_therm
            # if rate < 0.5
            #     Nt_therm += 2
            # elseif rate > 0.9 && Nt_therm > 2
            #     Nt_therm -= 1
            # end
            # if Nt_therm != old_Nt
            #     dt_therm = calc_optimal_dt(p.β, p.J, p.mass, Nt_therm)
            #     tee_println(@sprintf("Therm Step %d/%d. Adjusting Nt to %d, new dt=%.4f to target AccRate ~0.65", Nt_therm, dt_therm))
            # end
        end
    end
    
    therm_time = time() - start_time
    tee_println("Thermalization Done. Time: $(round(therm_time, digits=2))s")
    
    # 4. 测量阶段 (Measurement)
    dt_meas = calc_optimal_dt(p.β, p.J, p.mass, Nt_measure)
    tee_println("--- Measurement Start ---")
    tee_println("Target Sweeps: $n_sweep, Nt=$Nt_measure, dt=$(round(dt_meas, digits=4))")
    
    # 写入 CSV 表头
    # 格式: Sweep, Acc, Energy, D_amp, D_loc, D_glob, S_D, hole_p
    println(f_data, "Sweep,Accepted,Energy,Delta_Amp,Delta_Loc,Delta_Glob,S_Delta,Hole_p")
    
    acc_count = 0
    meas_start_time = time()
    
    for i in 1:n_sweep
        # HMC Step
        accepted, dH = hmc_sweep!(cache, p, state; Nt=Nt_measure, dt=dt_meas)
        
        if accepted
            acc_count += 1
        end
        
        # 测量物理量
        # 注意：这里我们只测量，不重新对角化，因为 hmc_sweep! 结束时 cache 里的数据就是最新的
        obs = measure_observables(cache, p, state)
        
        # 写入文件 (CSV格式)
        # 使用 @printf 无法直接输出到文件流，用 write + sprintf 组合
        line = @sprintf("%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", 
                        i, accepted, obs.total_energy, 
                        obs.Δ_amp, obs.Δ_local, obs.Δ_global, obs.S_Δ, obs.hole_conc)
        write(f_data, line)
        flush(f_data) # <--- 关键！确保断电/Kill不丢数据
        
        # 可以在某些步数输出到日志
        if i % max(1, n_sweep ÷ 10) == 0
             rate = acc_count / i
             tee_println(@sprintf("Meas Step %d/%d. AccRate=%.2f. E=%.4f", i, n_sweep, rate, obs.total_energy))
        end
    end
    
    meas_time = time() - meas_start_time
    tee_println("Measurement Done. Time: $(round(meas_time, digits=2))s")
    
    close(f_log)
    close(f_data)
end