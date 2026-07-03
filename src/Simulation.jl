using Dates
using Printf
using DelimitedFiles
using JLD2

"""
    calc_optimal_dt(β, V, mass, Nt)

根据谐振子近似计算推荐的时间步长 δt。这里的输入 V 是 t-V/PRB
物理耦合，辅助场有效配对耦合为 g_pair = V/2。
"""
function calc_optimal_dt(β, V, mass, Nt)
    g_pair = V / 2
    T = 2 * π * sqrt(mass * g_pair / β)
    return T / (2 * Nt) 
end

function round_even_nt(x::Real)
    return 2 * round(Int, x / 2)
end

function adjust_thermalization_nt(Nt_current::Int, rate::Real; Nt_min::Int, Nt_max::Int)
    Nt_min > 0 || error("Nt_min must be positive")
    Nt_max >= Nt_min || error("Nt_max must be >= Nt_min")

    Nt_next = if rate <= 0.0
        Nt_current + 6
    elseif rate < 0.30
        Nt_current + 4
    elseif rate < 0.55
        Nt_current + 2
    elseif rate <= 0.85
        Nt_current
    elseif rate <= 0.95
        Nt_current - 4
    else
        round_even_nt(0.65 * Nt_current)
    end

    return clamp(Nt_next, Nt_min, Nt_max)
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
                   spectra_Ltw::Int=1,
                   use_twisted_spectra::Bool=spectra_Ltw > 1,
                   m_point_patch_half_width::Float64=π / max(p.Lx, p.Ly),
                   spectra_eta=nothing,
                   spectra_eta_factors=DEFAULT_SPECTRA_ETA_FACTORS,
                   spectra_delta_omega=nothing,
                   measure_twist::Bool=false,
                   twist_Ax::Float64=1e-3,
                   twist_qy::Float64=2π/p.Ly,
                   allow_gauge_dependent_spectra::Bool=false,
                   write_gauge_pair_bonds_freq::Int=0,
                   write_ldos_spectrum::Bool=false)

运行完整的 HMC 模拟。

# 参数
- `n_therm`: 热化步数
- `n_measure`: 测量步数
- `Nt_therm_init`: 热化初始 Leapfrog 步数
- `measure_transport_freq`: 每隔多少个 MC 步进行一次重量级测量（输运/谱）
- `bin_size`: 谱学数据分箱大小。即累积 `bin_size` 次测量后，求平均并存入 JLD2 一次。
- `spectra_Ltw`: 谱函数动量网格的 twist 细分倍数；默认 `1`
- `use_twisted_spectra`: 是否用 TBC 谱函数替代默认谱函数；默认在 `spectra_Ltw > 1` 时开启
- `m_point_patch_half_width`: TBC M 点 patch 半宽；仅在 `use_twisted_spectra=true` 时写入和使用
- `spectra_eta`: TBC 谱函数展宽；默认 `p.η`
- `spectra_eta_factors`: 谱学多展宽因子；必须以 `1` 开头，默认 `DEFAULT_SPECTRA_ETA_FACTORS`
- `spectra_delta_omega`: TBC 谱函数频率步长；默认 `p.Δω`
- `measure_twist`: 是否额外计算 twist benchmark；默认关闭，避免额外对角化
- `twist_Ax`: twist 有限差分步长
- `twist_qy`: 横向调制 twist 的动量，默认 `2π/Ly`
- `allow_gauge_dependent_spectra`: 有限轨道磁场下是否显式输出 Landau gauge 诊断动量谱；默认关闭
- `write_gauge_pair_bonds_freq`: 每隔多少个测量步向 pairing_scatter.jld2 写出规范协变 bond 配对数组；`0` 表示关闭
- `write_ldos_spectrum`: 是否输出完整 LDOS(ω) 谱；默认关闭以避免 JLD2 文件过大
"""
function run_simulation(p::ModelParameters, out_dir::String; 
                        n_therm::Int=100, 
                        n_measure::Int=500, 
                        Nt_therm_init::Int=10, 
                        Nt_measure::Int=5,
                        measure_transport_freq::Int=1,
                        bin_size::Int=5,
                        spectra_Ltw::Int=1,
                        use_twisted_spectra::Bool=spectra_Ltw > 1,
                        m_point_patch_half_width::Float64=π / max(p.Lx, p.Ly),
                        spectra_eta::Union{Nothing,Real}=nothing,
                        spectra_eta_factors=DEFAULT_SPECTRA_ETA_FACTORS,
                        spectra_delta_omega::Union{Nothing,Real}=nothing,
                        measure_twist::Bool=false,
                        twist_Ax::Float64=1.0e-3,
                        twist_qy::Float64=2π / p.Ly,
                        allow_gauge_dependent_spectra::Bool=false,
                        write_gauge_pair_bonds_freq::Int=0,
                        write_ldos_spectrum::Bool=false,
                        verbose::Bool=true)
    
    # --- 1. 环境准备 ---
    spectra_Ltw > 0 || error("spectra_Ltw must be positive")
    m_point_patch_half_width >= 0 || error("m_point_patch_half_width must be nonnegative")
    write_gauge_pair_bonds_freq >= 0 || error("write_gauge_pair_bonds_freq must be nonnegative")
    finite_field = p.n_flux_sc != 0
    if finite_field && use_twisted_spectra
        error("use_twisted_spectra is not supported for finite magnetic field (n_flux_sc=$(p.n_flux_sc))")
    end
    if finite_field && measure_twist
        error("measure_twist is not supported for finite magnetic field (n_flux_sc=$(p.n_flux_sc))")
    end
    include_momentum_spectra = !finite_field || allow_gauge_dependent_spectra
    gauge_dependent_spectra = finite_field && allow_gauge_dependent_spectra
    spectra_gauge = finite_field ? "Landau gauge" : "none"
    spectra_interpretation = if gauge_dependent_spectra
        "diagnostic only; not a gauge-invariant momentum-resolved spectral function"
    elseif finite_field
        "ordinary momentum-resolved spectra disabled for finite magnetic field"
    else
        "ordinary momentum-resolved spectral function"
    end

    actual_spectra_Ltw = use_twisted_spectra ? spectra_Ltw : 1
    spectra_Lx_eff = p.Lx * actual_spectra_Ltw
    spectra_Ly_eff = p.Ly * actual_spectra_Ltw
    if use_twisted_spectra && (isodd(spectra_Lx_eff) || isodd(spectra_Ly_eff))
        error("TBC spectra require even effective dimensions")
    end
    actual_spectra_eta = use_twisted_spectra ?
                         Float64(spectra_eta === nothing ? p.η : spectra_eta) :
                         p.η
    actual_spectra_delta_omega = use_twisted_spectra ?
                                 Float64(spectra_delta_omega === nothing ? p.Δω : spectra_delta_omega) :
                                 p.Δω
    actual_spectra_eta > 0 || error("spectra_eta must be positive")
    actual_spectra_delta_omega > 0 || error("spectra_delta_omega must be positive")
    actual_spectra_eta_factors = validate_spectra_eta_factors(spectra_eta_factors)
    actual_spectra_eta_values = eta_values_from_base(actual_spectra_eta, actual_spectra_eta_factors)
    actual_transport_eta_values = eta_values_from_base(p.η, actual_spectra_eta_factors)

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
        println(f_trans, "Sweep,Superfluid_Stiffness,DC_Conductivity,Hall_Conductivity,Twist_Qy,Twist_Qy_Rho_Curv_Cos,Twist_Qy_Rho_Curv_Sin,Twist_Qy_Rho_Curv_Avg,Twist_Qy_Lambda_Diag,Twist_Qy_Rho_OffdiagCorrected")
    else
        println(f_trans, "Sweep,Superfluid_Stiffness,DC_Conductivity,Hall_Conductivity")
    end
    
    tee_println("Starting Simulation...")
    tee_println("System: $(p.Lx)x$(p.Ly), β=$(p.β), V=$(p.V), W=$(p.W), n_imp=$(p.n_imp)")
    tee_println("Config: Therm=$n_therm, Sweep=$n_measure, TransFreq=$measure_transport_freq, BinSize=$bin_size")
    tee_println("Spectra: use_twisted_spectra=$use_twisted_spectra, Ltw=$actual_spectra_Ltw, effective=$(spectra_Lx_eff)x$(spectra_Ly_eff)")
    if finite_field
        tee_println("Spectra finite-field mode: gauge_dependent_spectra=$gauge_dependent_spectra, gauge=$spectra_gauge")
    end
    tee_println("Spectra eta factors: $(actual_spectra_eta_factors)")
    tee_println("LDOS spectrum output: write_ldos_spectrum=$write_ldos_spectrum")
    if use_twisted_spectra
        tee_println("Spectra TBC: m_point_patch_half_width=$m_point_patch_half_width, spectra_eta=$actual_spectra_eta, spectra_delta_omega=$actual_spectra_delta_omega")
    end

    # --- 2. 初始化 ---
    tee_println("Initializing State...")
    state = initialize_state(p)
    cache = initialize_cache(p)
    mag_meta = magnetic_metadata(cache.magnetic)
    tee_println("Magnetic field: n_flux_sc=$(p.n_flux_sc), boundary=$(p.boundary_condition), flux_density_sc=$(mag_meta.flux_density_sc), plaquette_phase=$(mag_meta.plaquette_phase)")
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
    dos_omega_grid = use_twisted_spectra ?
                     spectra_dos_grid(p, actual_spectra_delta_omega) :
                     cache.dos_omega_grid
    if use_twisted_spectra
        Ix_pi, mx_path_kx, mx_path_ky = tbc_mx_path_metadata(p.Lx, p.Ly, actual_spectra_Ltw)
        xg_path_kx, xg_path_ky = tbc_xg_path_metadata(p.Lx, p.Ly, actual_spectra_Ltw)
        kx_grid = effective_k_grid(p.Lx, actual_spectra_Ltw)
        ky_grid = effective_k_grid(p.Ly, actual_spectra_Ltw)
        jldsave(spectra_jld_path; params=p,
                use_twisted_spectra=use_twisted_spectra,
                n_flux_sc=mag_meta.n_flux_sc,
                boundary_condition=p.boundary_condition,
                flux_density_sc=mag_meta.flux_density_sc,
                plaquette_phase=mag_meta.plaquette_phase,
                magnetic_gauge=mag_meta.magnetic_gauge,
                magnetic_pbc=mag_meta.magnetic_pbc,
                gauge_dependent_spectra=gauge_dependent_spectra,
                spectra_gauge=spectra_gauge,
                spectra_interpretation=spectra_interpretation,
                pairing_scalar_convention=p.n_flux_sc == 0 ? "bare zero-field convention" : "bare Landau-gauge diagnostic",
                pairing_scalar_gauge_invariant=p.n_flux_sc == 0,
                conductivity_convention="sigma_xx_regular_sigma_xy_kubo",
                spectra_Ltw=actual_spectra_Ltw,
                spectra_Lx_eff=spectra_Lx_eff,
                spectra_Ly_eff=spectra_Ly_eff,
                spectra_eta=actual_spectra_eta,
                multi_eta_enabled=true,
                spectra_eta_factors=actual_spectra_eta_factors,
                eta_values=actual_spectra_eta_values,
                spectra_eta_base=actual_spectra_eta,
                transport_eta_values=actual_transport_eta_values,
                transport_eta_base=p.η,
                spectra_delta_omega=actual_spectra_delta_omega,
                write_ldos_spectrum=write_ldos_spectrum,
                ldos_spectrum_grid_key="dos_omega_grid",
                omega_grid=omega_grid,
                dos_omega_grid=dos_omega_grid,
                mx_path_kx=mx_path_kx,
                mx_path_ky=mx_path_ky,
                mx_path_kx_idx=Ix_pi + 1,
                mx_path_ky_idx=collect(1:length(mx_path_ky)),
                xg_path_kx=xg_path_kx,
                xg_path_ky=xg_path_ky,
                xg_path_kx_idx=collect(1:length(xg_path_kx)),
                xg_path_ky_idx=collect(1:length(xg_path_ky)),
                kx_grid=kx_grid,
                ky_grid=ky_grid,
                m_point_patch_half_width=m_point_patch_half_width)
    else
        mx_kx_idx, mx_ky_indices, mx_kx_val, mx_ky_vals = mx_kpath(p)
        xg_kx_indices, xg_ky_indices, xg_kx_vals, xg_ky_vals = xg_kpath(p)
        jldsave(spectra_jld_path; params=p,
                use_twisted_spectra=use_twisted_spectra,
                n_flux_sc=mag_meta.n_flux_sc,
                boundary_condition=p.boundary_condition,
                flux_density_sc=mag_meta.flux_density_sc,
                plaquette_phase=mag_meta.plaquette_phase,
                magnetic_gauge=mag_meta.magnetic_gauge,
                magnetic_pbc=mag_meta.magnetic_pbc,
                gauge_dependent_spectra=gauge_dependent_spectra,
                spectra_gauge=spectra_gauge,
                spectra_interpretation=spectra_interpretation,
                pairing_scalar_convention=p.n_flux_sc == 0 ? "bare zero-field convention" : "bare Landau-gauge diagnostic",
                pairing_scalar_gauge_invariant=p.n_flux_sc == 0,
                conductivity_convention="sigma_xx_regular_sigma_xy_kubo",
                spectra_Ltw=actual_spectra_Ltw,
                spectra_Lx_eff=spectra_Lx_eff,
                spectra_Ly_eff=spectra_Ly_eff,
                spectra_eta=actual_spectra_eta,
                multi_eta_enabled=true,
                spectra_eta_factors=actual_spectra_eta_factors,
                eta_values=actual_spectra_eta_values,
                spectra_eta_base=actual_spectra_eta,
                transport_eta_values=actual_transport_eta_values,
                transport_eta_base=p.η,
                spectra_delta_omega=actual_spectra_delta_omega,
                write_ldos_spectrum=write_ldos_spectrum,
                ldos_spectrum_grid_key="dos_omega_grid",
                omega_grid=omega_grid,
                dos_omega_grid=dos_omega_grid,
                mx_path_kx=mx_kx_val,
                mx_path_ky=mx_ky_vals,
                mx_path_kx_idx=mx_kx_idx,
                mx_path_ky_idx=mx_ky_indices,
                xg_path_kx=xg_kx_vals,
                xg_path_ky=xg_ky_vals,
                xg_path_kx_idx=xg_kx_indices,
                xg_path_ky_idx=xg_ky_indices)
    end
    jldsave(pair_scatter_jld_path; params=p)

    # --- 3. 热化阶段 (Adaptive Thermalization) ---
    Nt_current = Nt_therm_init
    dt_current = calc_optimal_dt(p.β, p.V, p.mass, Nt_current)
    
    tee_println("--- Thermalization Start ---")
    tee_println("Init: Nt=$Nt_current, dt=$(round(dt_current, digits=5))")
    
    # 用于计算接受率窗口
    acc_window = 5
    recent_acc = 0
    Nt_min = Nt_measure
    Nt_max = max(40, Nt_min)
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
            
            Nt_current = adjust_thermalization_nt(Nt_current, rate; Nt_min=Nt_min, Nt_max=Nt_max)
            
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
    accum_dos_M = nothing
    accum_dos_M_patch = nothing
    accum_ldos0 = Vector{Float64}()
    accum_ldos = nothing
    accum_Ak0 = nothing
    accum_AMXpath = nothing
    accum_AXGpath = nothing
    accum_AXGnodePatch = nothing
    accum_dc_eta = Vector{Float64}()
    accum_hall_cond_eta = Vector{Float64}()
    accum_opt_eta = Matrix{Float64}(undef, 0, 0)
    accum_hall_opt_cond = Vector{ComplexF64}()
    accum_hall_opt_eta = Matrix{ComplexF64}(undef, 0, 0)
    accum_dos_eta = Matrix{Float64}(undef, 0, 0)
    accum_dos_M_eta = nothing
    accum_dos_M_patch_eta = nothing
    accum_ldos0_eta = Matrix{Float64}(undef, 0, 0)
    accum_ldos_eta = nothing
    accum_Ak0_eta = nothing
    accum_AMXpath_eta = nothing
    accum_AXGpath_eta = nothing
    accum_AXGnodePatch_eta = nothing
    diagnostic_momentum_names = finite_field && allow_gauge_dependent_spectra
    dos_M_key = diagnostic_momentum_names ? "dos_M_landau_gauge_diagnostic" : "dos_M"
    dos_M_eta_key = diagnostic_momentum_names ? "dos_M_eta_landau_gauge_diagnostic" : "dos_M_eta"
    Ak0_key = diagnostic_momentum_names ? "A_k_omega0_landau_gauge_diagnostic" : "A_k0"
    Ak0_eta_key = diagnostic_momentum_names ? "A_k_omega0_eta_landau_gauge_diagnostic" : "A_k0_eta"
    AMXpath_key = diagnostic_momentum_names ? "A_MX_path_landau_gauge_diagnostic" : "A_MX_path"
    AMXpath_eta_key = diagnostic_momentum_names ? "A_MX_path_eta_landau_gauge_diagnostic" : "A_MX_path_eta"
    AXGpath_key = diagnostic_momentum_names ? "A_XG_path_landau_gauge_diagnostic" : "A_XG_path"
    AXGpath_eta_key = diagnostic_momentum_names ? "A_XG_path_eta_landau_gauge_diagnostic" : "A_XG_path_eta"
    
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
            if write_gauge_pair_bonds_freq > 0 && i % write_gauge_pair_bonds_freq == 0
                delta_bond, pair_bond = compute_gauge_pair_bonds(cache, p, state)
                g["delta_bond_landau_gauge_covariant"] = delta_bond
                g["pair_bond_landau_gauge_covariant"] = pair_bond
            end
        end
        
        # 3. 重量级测量 (Every Freq Step)
        if i % measure_transport_freq == 0
            # 计算输运和谱
            if use_twisted_spectra
                transport_res = measure_transport_only(cache, p;
                                                       eta_values=actual_transport_eta_values,
                                                       reuse_buffers=true)
                twisted_res = measure_twisted_spectra(cache, p, state;
                                                      Ltw=actual_spectra_Ltw,
                                                      m_point_patch_half_width=m_point_patch_half_width,
                                                      spectra_eta=actual_spectra_eta,
                                                      spectra_delta_omega=actual_spectra_delta_omega,
                                                      eta_values=actual_spectra_eta_values,
                                                      reuse_buffers=false,
                                                      write_ldos_spectrum=write_ldos_spectrum)
                spec_res = SpectrumResult(transport_res.superfluid_stiffness,
                                          transport_res.dc_conductivity,
                                          transport_res.hall_conductivity,
                                          transport_res.ω_grid,
                                          transport_res.optical_conductivity,
                                          transport_res.hall_optical_conductivity,
                                          twisted_res.dos_ω_grid,
                                          twisted_res.dos,
                                          twisted_res.dos_M,
                                          twisted_res.ldos_ω0,
                                          twisted_res.A_k_ω0,
                                          twisted_res.A_MX_path,
                                          twisted_res.A_XG_path,
                                          transport_res.dc_conductivity_eta,
                                          transport_res.hall_conductivity_eta,
                                          transport_res.optical_conductivity_eta,
                                          transport_res.hall_optical_conductivity_eta,
                                          twisted_res.dos_eta,
                                          twisted_res.dos_M_eta,
                                          twisted_res.ldos_ω0_eta,
                                          twisted_res.A_k_ω0_eta,
                                          twisted_res.A_MX_path_eta,
                                          twisted_res.A_XG_path_eta,
                                          twisted_res.ldos_ω,
                                          twisted_res.ldos_ω_eta)
                spec_dos_M_patch = twisted_res.dos_M_patch
                spec_xg_node_patch = twisted_res.A_XG_node_patch
                spec_dos_M_patch_eta = twisted_res.dos_M_patch_eta
                spec_xg_node_patch_eta = twisted_res.A_XG_node_patch_eta
            else
                spec_res = measure_transport_and_spectra(cache, p;
                                                         eta_values=actual_spectra_eta_values,
                                                         reuse_buffers=true,
                                                         include_momentum_spectra=include_momentum_spectra,
                                                         write_ldos_spectrum=write_ldos_spectrum)
                spec_dos_M_patch = nothing
                spec_xg_node_patch = nothing
                spec_dos_M_patch_eta = nothing
                spec_xg_node_patch_eta = nothing
            end
            
            # A. 写入 Transport CSV (Scalars)
            if measure_twist
                twist_qy_res = measure_twist_stiffness_qy(cache, p, state;
                                                          Ax=twist_Ax, qy=twist_qy)
                line_trans = @sprintf("%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
                                      i, spec_res.superfluid_stiffness,
                                      spec_res.dc_conductivity,
                                      spec_res.hall_conductivity,
                                      twist_qy_res.qy,
                                      twist_qy_res.rho_curvature_cos,
                                      twist_qy_res.rho_curvature_sin,
                                      twist_qy_res.rho_curvature_avg,
                                      twist_qy_res.diag_correction,
                                      twist_qy_res.rho_offdiag_corrected)
            else
                line_trans = @sprintf("%d,%.6e,%.6e,%.6e\n",
                                      i, spec_res.superfluid_stiffness,
                                      spec_res.dc_conductivity,
                                      spec_res.hall_conductivity)
            end
            write(f_trans, line_trans)
            flush(f_trans)
            
            # B. 谱学数据分箱 (Binning)
            # 初始化累加器
            if bin_count == 0
                accum_opt_cond = copy(spec_res.optical_conductivity)
                accum_dos = copy(spec_res.dos)
                accum_dos_M = spec_res.dos_M === nothing ? nothing : copy(spec_res.dos_M)
                accum_dos_M_patch = spec_dos_M_patch === nothing ? nothing : copy(spec_dos_M_patch)
                accum_ldos0 = copy(spec_res.ldos_ω0)
                accum_ldos = spec_res.ldos_ω === nothing ? nothing : copy(spec_res.ldos_ω)
                accum_Ak0 = spec_res.A_k_ω0 === nothing ? nothing : copy(spec_res.A_k_ω0)
                accum_AMXpath = spec_res.A_MX_path === nothing ? nothing : copy(spec_res.A_MX_path)
                accum_AXGpath = spec_res.A_XG_path === nothing ? nothing : copy(spec_res.A_XG_path)
                accum_AXGnodePatch = spec_xg_node_patch === nothing ? nothing : copy(spec_xg_node_patch)
                accum_dc_eta = copy(spec_res.dc_conductivity_eta)
                accum_hall_cond_eta = copy(spec_res.hall_conductivity_eta)
                accum_opt_eta = copy(spec_res.optical_conductivity_eta)
                accum_hall_opt_cond = copy(spec_res.hall_optical_conductivity)
                accum_hall_opt_eta = copy(spec_res.hall_optical_conductivity_eta)
                accum_dos_eta = copy(spec_res.dos_eta)
                accum_dos_M_eta = spec_res.dos_M_eta === nothing ? nothing : copy(spec_res.dos_M_eta)
                accum_dos_M_patch_eta = spec_dos_M_patch_eta === nothing ? nothing : copy(spec_dos_M_patch_eta)
                accum_ldos0_eta = copy(spec_res.ldos_ω0_eta)
                accum_ldos_eta = spec_res.ldos_ω_eta === nothing ? nothing : copy(spec_res.ldos_ω_eta)
                accum_Ak0_eta = spec_res.A_k_ω0_eta === nothing ? nothing : copy(spec_res.A_k_ω0_eta)
                accum_AMXpath_eta = spec_res.A_MX_path_eta === nothing ? nothing : copy(spec_res.A_MX_path_eta)
                accum_AXGpath_eta = spec_res.A_XG_path_eta === nothing ? nothing : copy(spec_res.A_XG_path_eta)
                accum_AXGnodePatch_eta = spec_xg_node_patch_eta === nothing ? nothing : copy(spec_xg_node_patch_eta)
                bin_count = 1
            else
                accum_opt_cond .+= spec_res.optical_conductivity
                accum_dos .+= spec_res.dos
                accum_dc_eta .+= spec_res.dc_conductivity_eta
                accum_hall_cond_eta .+= spec_res.hall_conductivity_eta
                accum_opt_eta .+= spec_res.optical_conductivity_eta
                accum_hall_opt_cond .+= spec_res.hall_optical_conductivity
                accum_hall_opt_eta .+= spec_res.hall_optical_conductivity_eta
                accum_dos_eta .+= spec_res.dos_eta
                accum_ldos0 .+= spec_res.ldos_ω0
                accum_ldos0_eta .+= spec_res.ldos_ω0_eta
                if spec_res.ldos_ω !== nothing
                    accum_ldos === nothing && error("LDOS spectrum accumulator missing")
                    accum_ldos .+= spec_res.ldos_ω
                end
                if spec_res.ldos_ω_eta !== nothing
                    accum_ldos_eta === nothing && error("LDOS_eta spectrum accumulator missing")
                    accum_ldos_eta .+= spec_res.ldos_ω_eta
                end
                if spec_res.dos_M !== nothing
                    accum_dos_M === nothing && error("dos_M accumulator missing for momentum spectra")
                    accum_dos_M .+= spec_res.dos_M
                end
                if spec_res.dos_M_eta !== nothing
                    accum_dos_M_eta === nothing && error("dos_M_eta accumulator missing for momentum spectra")
                    accum_dos_M_eta .+= spec_res.dos_M_eta
                end
                if spec_dos_M_patch !== nothing
                    accum_dos_M_patch === nothing && error("dos_M_patch accumulator missing for TBC spectra")
                    accum_dos_M_patch .+= spec_dos_M_patch
                end
                if spec_dos_M_patch_eta !== nothing
                    accum_dos_M_patch_eta === nothing && error("dos_M_patch_eta accumulator missing for TBC spectra")
                    accum_dos_M_patch_eta .+= spec_dos_M_patch_eta
                end
                if spec_res.A_k_ω0 !== nothing
                    accum_Ak0 === nothing && error("A_k_ω0 accumulator missing for momentum spectra")
                    accum_Ak0 .+= spec_res.A_k_ω0
                end
                if spec_res.A_MX_path !== nothing
                    accum_AMXpath === nothing && error("A_MX_path accumulator missing for momentum spectra")
                    accum_AMXpath .+= spec_res.A_MX_path
                end
                if spec_res.A_XG_path !== nothing
                    accum_AXGpath === nothing && error("A_XG_path accumulator missing for momentum spectra")
                    accum_AXGpath .+= spec_res.A_XG_path
                end
                if spec_res.A_k_ω0_eta !== nothing
                    accum_Ak0_eta === nothing && error("A_k_ω0_eta accumulator missing for momentum spectra")
                    accum_Ak0_eta .+= spec_res.A_k_ω0_eta
                end
                if spec_res.A_MX_path_eta !== nothing
                    accum_AMXpath_eta === nothing && error("A_MX_path_eta accumulator missing for momentum spectra")
                    accum_AMXpath_eta .+= spec_res.A_MX_path_eta
                end
                if spec_res.A_XG_path_eta !== nothing
                    accum_AXGpath_eta === nothing && error("A_XG_path_eta accumulator missing for momentum spectra")
                    accum_AXGpath_eta .+= spec_res.A_XG_path_eta
                end
                if spec_xg_node_patch !== nothing
                    accum_AXGnodePatch === nothing && error("A_XG_node_patch accumulator missing for TBC spectra")
                    accum_AXGnodePatch .+= spec_xg_node_patch
                end
                if spec_xg_node_patch_eta !== nothing
                    accum_AXGnodePatch_eta === nothing && error("A_XG_node_patch_eta accumulator missing for TBC spectra")
                    accum_AXGnodePatch_eta .+= spec_xg_node_patch_eta
                end
                bin_count += 1
            end
            
            # 达到 Bin Size，写入 JLD2 并清空缓存
            if bin_count >= bin_size
                # 求平均
                accum_opt_cond ./= bin_count
                accum_dos ./= bin_count
                accum_dc_eta ./= bin_count
                accum_hall_cond_eta ./= bin_count
                accum_opt_eta ./= bin_count
                accum_hall_opt_cond ./= bin_count
                accum_hall_opt_eta ./= bin_count
                accum_dos_eta ./= bin_count
                accum_ldos0 ./= bin_count
                accum_ldos0_eta ./= bin_count
                if accum_ldos !== nothing
                    accum_ldos ./= bin_count
                end
                if accum_ldos_eta !== nothing
                    accum_ldos_eta ./= bin_count
                end
                if accum_dos_M !== nothing
                    accum_dos_M ./= bin_count
                end
                if accum_dos_M_eta !== nothing
                    accum_dos_M_eta ./= bin_count
                end
                if accum_dos_M_patch !== nothing
                    accum_dos_M_patch ./= bin_count
                end
                if accum_dos_M_patch_eta !== nothing
                    accum_dos_M_patch_eta ./= bin_count
                end
                if accum_Ak0 !== nothing
                    accum_Ak0 ./= bin_count
                end
                if accum_AMXpath !== nothing
                    accum_AMXpath ./= bin_count
                end
                if accum_AXGpath !== nothing
                    accum_AXGpath ./= bin_count
                end
                if accum_Ak0_eta !== nothing
                    accum_Ak0_eta ./= bin_count
                end
                if accum_AMXpath_eta !== nothing
                    accum_AMXpath_eta ./= bin_count
                end
                if accum_AXGpath_eta !== nothing
                    accum_AXGpath_eta ./= bin_count
                end
                if accum_AXGnodePatch !== nothing
                    accum_AXGnodePatch ./= bin_count
                end
                if accum_AXGnodePatch_eta !== nothing
                    accum_AXGnodePatch_eta ./= bin_count
                end
                
                # JLD2 追加写入
                # 使用 string key 来区分不同的 bin，例如 "bin_100", "bin_200" 表示到第几步的 bin
                # 注意：频繁打开关闭文件有开销，但对于 bin_size * measure_freq 步才一次的操作，这是安全的
                jldopen(spectra_jld_path, "a+") do file
                    group_name = "sweep_$i"
                    g = JLD2.Group(file, group_name)
                    g["opt_cond"] = accum_opt_cond
                    g["dos"] = accum_dos
                    g["dc_cond_eta"] = accum_dc_eta
                    g["hall_cond"] = accum_hall_cond_eta[1]
                    g["hall_cond_eta"] = accum_hall_cond_eta
                    g["opt_cond_eta"] = accum_opt_eta
                    g["hall_opt_cond"] = accum_hall_opt_cond
                    g["hall_opt_cond_eta"] = accum_hall_opt_eta
                    g["dos_eta"] = accum_dos_eta
                    if accum_dos_M !== nothing
                        g[dos_M_key] = accum_dos_M
                    end
                    if accum_dos_M_eta !== nothing
                        g[dos_M_eta_key] = accum_dos_M_eta
                    end
                    if accum_dos_M_patch !== nothing
                        g["dos_M_patch"] = accum_dos_M_patch
                    end
                    if accum_dos_M_patch_eta !== nothing
                        g["dos_M_patch_eta"] = accum_dos_M_patch_eta
                    end
                    g["LDOS_0"] = accum_ldos0
                    g["LDOS_0_eta"] = accum_ldos0_eta
                    if accum_ldos !== nothing
                        g["LDOS"] = accum_ldos
                    end
                    if accum_ldos_eta !== nothing
                        g["LDOS_eta"] = accum_ldos_eta
                    end
                    if accum_Ak0 !== nothing
                        g[Ak0_key] = accum_Ak0
                    end
                    if accum_AMXpath !== nothing
                        g[AMXpath_key] = accum_AMXpath
                    end
                    if accum_AXGpath !== nothing
                        g[AXGpath_key] = accum_AXGpath
                    end
                    if accum_Ak0_eta !== nothing
                        g[Ak0_eta_key] = accum_Ak0_eta
                    end
                    if accum_AMXpath_eta !== nothing
                        g[AMXpath_eta_key] = accum_AMXpath_eta
                    end
                    if accum_AXGpath_eta !== nothing
                        g[AXGpath_eta_key] = accum_AXGpath_eta
                    end
                    if accum_AXGnodePatch !== nothing
                        g["A_XG_node_patch"] = accum_AXGnodePatch
                    end
                    if accum_AXGnodePatch_eta !== nothing
                        g["A_XG_node_patch_eta"] = accum_AXGnodePatch_eta
                    end
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
