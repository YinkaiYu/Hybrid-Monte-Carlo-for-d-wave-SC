using Distributed
using Printf
using Random

# ---------------------------------------------------------
# 1. 自动检测并启动并行进程
# ---------------------------------------------------------
if nprocs() == 1
    # 读取 SLURM 分配的核数 (例如 4)
    slurm_ntasks = parse(Int, get(ENV, "SLURM_NTASKS", "1"))
    
    # [修改前] 留一个核给 Master
    # workers_to_add = max(0, slurm_ntasks - 1)
    
    # [修改后] 不留核！直接启动与核数相等的 Worker
    # 这样总进程数 = slurm_ntasks + 1 (Master)
    # 因为 Master 几乎不耗 CPU，所以这是安全的
    println("Master: Slurm assigned $slurm_ntasks cores. Launching $slurm_ntasks workers (Oversubscription).")
    workers_to_add = slurm_ntasks
    
    if workers_to_add > 0
        addprocs(workers_to_add)
    end
end

println("Master process $(myid()) started. Total processes: $(nprocs())")
println("Workers: $(workers())")
flush(stdout)

# ---------------------------------------------------------
# 2. 环境加载
# ---------------------------------------------------------
function find_project_root(start_dir::AbstractString)
    dir = abspath(start_dir)
    while true
        if isfile(joinpath(dir, "Project.toml"))
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not find Project.toml above $start_dir")
        dir = parent
    end
end

const PROJECT_ROOT = get(ENV, "DWAVEHMC_PROJECT_ROOT", find_project_root(@__DIR__))
@everywhere begin
    using Pkg
    Pkg.activate($PROJECT_ROOT) 
    using DwaveHMC
    using Random
    using Printf
end

# ---------------------------------------------------------
# 3. 读取参数
# ---------------------------------------------------------
# 获取 params.jl 在当前工作目录下的绝对路径
params_path = joinpath(pwd(), "params.jl")
if !isfile(params_path)
    error("params.jl not found at $params_path")
end
# 使用绝对路径 include
include(params_path)
actual_spectra_eta_factors = isdefined(@__MODULE__, :spectra_eta_factors) ?
                             Float64.(getfield(@__MODULE__, :spectra_eta_factors)) :
                             DwaveHMC.DEFAULT_SPECTRA_ETA_FACTORS
write_gauge_pair_bonds_freq = isdefined(@__MODULE__, :write_gauge_pair_bonds_freq) ? getfield(@__MODULE__, :write_gauge_pair_bonds_freq) : 0
allow_gauge_dependent_spectra = isdefined(@__MODULE__, :allow_gauge_dependent_spectra) ? getfield(@__MODULE__, :allow_gauge_dependent_spectra) : false

println("Parameters loaded. T = $(T), Total Configs = $(N_conf)")
flush(stdout)

println("Spectra options: use_twisted_spectra=$(use_twisted_spectra), spectra_Ltw=$(spectra_Ltw), spectra_eta=$(spectra_eta), spectra_delta_omega=$(spectra_delta_omega)")
println("Spectra eta factors: $(actual_spectra_eta_factors)")
println("Twist stiffness options: measure_twist=$(measure_twist), twist_Ax=$(twist_Ax), twist_qy=$(twist_qy)")
println("Magnetic options: n_flux_sc=$(p.n_flux_sc), boundary_condition=$(p.boundary_condition), write_gauge_pair_bonds_freq=$(write_gauge_pair_bonds_freq), allow_gauge_dependent_spectra=$(allow_gauge_dependent_spectra)")
flush(stdout)

# ---------------------------------------------------------
# 4. Worker 任务 (带静默模式)
# ---------------------------------------------------------
@everywhere function worker_task(seed::Int, p_base::ModelParameters, 
                                 n_therm, n_measure, 
                                 Nt_therm_init, Nt_measure, 
                                 measure_transport_freq, bin_size,
                                 spectra_Ltw, use_twisted_spectra,
                                 m_point_patch_half_width,
                                 spectra_eta, spectra_delta_omega,
                                 spectra_eta_factors,
                                 measure_twist, twist_Ax, twist_qy,
                                 allow_gauge_dependent_spectra, write_gauge_pair_bonds_freq)
    out_dir = "conf_$(seed)"
    Random.seed!(seed)
    
    # 仅在开始和结束时在 job.out 留痕
    println("Processing seed=$seed ...")
    flush(stdout)
    
    try
        # verbose=false: 详细过程写入 log 文件，不输出到 job.out
        run_simulation(p_base, out_dir; 
                       n_therm=n_therm, 
                       n_measure=n_measure, 
                       Nt_therm_init=Nt_therm_init, 
                       Nt_measure=Nt_measure,
                       measure_transport_freq=measure_transport_freq,
                       bin_size=bin_size,
                       spectra_Ltw=spectra_Ltw,
                       use_twisted_spectra=use_twisted_spectra,
                       m_point_patch_half_width=m_point_patch_half_width,
                       spectra_eta=spectra_eta,
                       spectra_delta_omega=spectra_delta_omega,
                       spectra_eta_factors=spectra_eta_factors,
                       measure_twist=measure_twist,
                       twist_Ax=twist_Ax,
                       twist_qy=twist_qy,
                       allow_gauge_dependent_spectra=allow_gauge_dependent_spectra,
                       write_gauge_pair_bonds_freq=write_gauge_pair_bonds_freq,
                       verbose=false) 
        
        return true
    catch e
        println("ERROR in seed=$seed: $e")
        return false
    end
end

# ---------------------------------------------------------
# 5. 任务分发
# ---------------------------------------------------------
results = pmap(1:N_conf) do seed
    worker_task(seed, p, 
                n_therm, n_measure, 
                Nt_therm_init, Nt_measure, 
                measure_transport_freq, bin_size,
                spectra_Ltw, use_twisted_spectra,
                m_point_patch_half_width,
                spectra_eta, spectra_delta_omega,
                actual_spectra_eta_factors,
                measure_twist, twist_Ax, twist_qy,
                allow_gauge_dependent_spectra, write_gauge_pair_bonds_freq)
end

success_count = count(results)
println("All tasks completed. Success: $(success_count)/$(N_conf)")
flush(stdout)

if success_count != N_conf
    exit(1) # 如果有任务失败，让作业状态显示为 Failed
end
