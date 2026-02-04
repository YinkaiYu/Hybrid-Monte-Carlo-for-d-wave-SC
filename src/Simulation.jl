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
                   max_iter::Int=p.Δ_MF_max_iter,
                   verbose::Bool=true, ...)

运行均匀 d-wave 平均场迭代。
注意：为兼容旧脚本，保留 HMC 相关关键字，但在平均场模式下会被忽略。
"""
function run_simulation(p::ModelParameters, out_dir::String;
                        max_iter::Int=p.Δ_MF_max_iter,
                        n_therm::Int=100,
                        n_measure::Int=500,
                        Nt_therm_init::Int=10,
                        Nt_measure::Int=5,
                        measure_transport_freq::Int=1,
                        bin_size::Int=5,
                        verbose::Bool=true)

    # --- 1. 环境准备 ---
    if !isdir(out_dir)
        mkpath(out_dir)
    end

    # 文件句柄
    log_path = joinpath(out_dir, "simulation.log")
    obs_csv_path = joinpath(out_dir, "observables.csv")
    hist_csv_path = joinpath(out_dir, "mf_history.csv")
    trans_csv_path = joinpath(out_dir, "transport.csv")
    spectra_jld_path = joinpath(out_dir, "spectra_bins.jld2")
    pair_scatter_jld_path = joinpath(out_dir, "pairing_scatter.jld2")

    f_log = open(log_path, "a")
    f_obs = open(obs_csv_path, "w")
    f_hist = open(hist_csv_path, "w")
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
    println(f_obs, "Iter,Delta_MF,Energy,Delta_Amp,Delta_Loc,Delta_Glob,S_Delta,Hole_p,Delta_Diff,Delta_Pair,Delta_LocalPair,D2,D4,Avg_d2,Avg_d4")
    println(f_hist, "Iter,Delta_Old,Delta_New,Delta_Mix,Residual,Diff,Alpha")
    println(f_trans, "Iter,Superfluid_Stiffness,DC_Conductivity")

    tee_println("Starting Mean-Field Iteration...")
    tee_println("System: $(p.Lx)x$(p.Ly), β=$(p.β), n_imp=$(p.n_imp), J=$(p.J)")
    tee_println("Config: max_iter=$max_iter, α=$(p.α), tol=$(p.Δ_MF_tol)")

    # --- 2. 初始化 ---
    tee_println("Initializing State...")
    state = initialize_state(p)
    cache = initialize_cache(p)

    init_static_H!(cache, p, state)

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

    # --- 3. 平均场迭代 ---
    tee_println("--- Mean-Field Iteration Start ---")
    start_time = time()

    converged = false
    Δ_old = state.Δ_MF
    Δ_new = Δ_old
    α_current = p.α
    prev_res_abs = Inf
    α_min = 0.05
    α_max = 0.9
    α_up = 1.1
    α_down = 0.5

    for iter in 1:max_iter
        # 用旧的 Δ 更新哈密顿量并对角化
        update_H_BdG!(cache, p, state)
        diagonalize_H_BdG!(cache, p)

        # 计算新的 Δ_MF 并线性混合
        Δ_new = compute_Δ_MF(cache, p)
        res = Δ_new - Δ_old
        res_abs = abs(res)

        # 自适应线性混合系数
        if iter > 1
            if res_abs > prev_res_abs
                α_current = max(α_min, α_current * α_down)
            else
                α_current = min(α_max, α_current * α_up)
            end
        end

        Δ_mix = (1.0 - α_current) * Δ_old + α_current * Δ_new
        diff = abs(Δ_mix - Δ_old)

        # 更新状态
        state.Δ_MF = Δ_mix
        state.Δ[:, 1] .= Δ_mix
        state.Δ[:, 2] .= -Δ_mix

        # 记录迭代历史
        line_hist = @sprintf("%d,%.6e,%.6e,%.6e,%.3e,%.3e,%.3e\n",
                             iter, Δ_old, Δ_new, Δ_mix, res_abs, diff, α_current)
        write(f_hist, line_hist)
        flush(f_hist)

        if iter == 1 || iter % 10 == 0
            tee_println(@sprintf("Iter %d/%d: Δ_old=%.6e, Δ_new=%.6e, Δ=%.6e, res=%.3e, α=%.3f",
                                 iter, max_iter, Δ_old, Δ_new, Δ_mix, res_abs, α_current))
        end

        if res_abs < p.Δ_MF_tol
            converged = true
            tee_println(@sprintf("Converged at iter %d with Δ=%.6e (res=%.3e)", iter, Δ_mix, res_abs))
            break
        end

        Δ_old = Δ_mix
        prev_res_abs = res_abs
    end

    if !converged
        tee_println(@sprintf("Warning: not converged after %d iterations. Final Δ=%.6e", max_iter, state.Δ_MF))
    end

    # 用收敛后的 Δ 重新计算谱
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)

    # --- 4. 输出可观测量 ---
    obs = measure_observables(cache, p, state)
    line = @sprintf("%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                    1, state.Δ_MF, obs.total_energy,
                    obs.Δ_amp, obs.Δ_local, obs.Δ_global, obs.S_Δ, obs.hole_conc,
                    obs.Δ_diff, obs.Δ_pair, obs.Δ_localpair,
                    obs.D2, obs.D4, obs.d2_avg, obs.d4_avg)
    write(f_obs, line)
    flush(f_obs)

    jldopen(pair_scatter_jld_path, "a+") do file
        g = JLD2.Group(file, "final")
        g["d_local"] = obs.d_local
    end

    spec_res = measure_transport_and_spectra(cache, p; reuse_buffers=true)
    line_trans = @sprintf("%d,%.6e,%.6e\n",
                          1, spec_res.superfluid_stiffness, spec_res.dc_conductivity)
    write(f_trans, line_trans)
    flush(f_trans)

    jldopen(spectra_jld_path, "a+") do file
        g = JLD2.Group(file, "final")
        g["opt_cond"] = spec_res.optical_conductivity
        g["dos"] = spec_res.dos
        g["dos_AN"] = spec_res.dos_AN
        g["A_k0"] = spec_res.A_k_ω0
        g["A_kpath"] = spec_res.A_kpath
        g["count"] = 1
    end

    tee_println("Mean-Field Done. Total Time: $(round(time() - start_time, digits=2))s")

    close(f_log)
    close(f_obs)
    close(f_hist)
    close(f_trans)
end
