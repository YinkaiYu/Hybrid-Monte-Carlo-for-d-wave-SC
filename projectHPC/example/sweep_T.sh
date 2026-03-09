#!/bin/bash
set -e 

# ================= 配置区域 =================
# 1. 扫描参数
T_list=(0.001 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2)

# 2. 并行与作业参数
queue='fat6348'
N_NODES=1   # 每个作业申请多少个节点
N_CORES=4   # 每个作业申请多少个核
N_CONFS=8   # 每个温度跑多少个构型
PROJECT_ROOT=/home/zxli_1/yyk2025/2511_dWaveBcs/20251231_sweep-T

# 3. 物理参数
L=20
t=1.0
tp=-0.35
# 模式开关: 1=目标密度n(热化调μ), 0=固定μ
USE_TARGET_N=1
# 目标密度模式
target_n=0.85
mu_init=-1.23359
# 固定化学势模式
mu=-1.23359
W=0.0
n_imp=0.0
V=0.8
mass=1.0

# 4. 测量参数
omega_max=0.8

# 5. HMC参数
n_therm=50
n_measure=200
Nt_therm_init=26
Nt_measure=8
measure_transport_freq=1
bin_size=5

# 6. 目标密度模式下的 μ 求根参数（hybrid: secant + bracket）
mu_tune_gain=0.5
mu_tune_interval=1
mu_tune_step_max=0.08
mu_tune_tol=0.005
mu_min=-4.0
mu_max=4.0
# ===========================================

for T in ${T_list[*]}
do
    DIR_NAME="T_$T"
    
    if [ ! -d "$DIR_NAME" ]; then
        mkdir "$DIR_NAME"
    fi
    cd "$DIR_NAME"
    
    # --- 写入 params.jl ---
    # 使用 $N_CONFS 变量
    cat << EOF > params.jl
using DwaveHMC
Lx, Ly = $L, $L
t, tp = $t, $tp
W, n_imp = $W, $n_imp
V = $V
mass = $mass
T = $T
β = 1.0 / T
η = 8.0 / (Lx * Ly)
Δω = 0.2 * η
ω_max = $omega_max
n_therm = $n_therm
n_measure = $n_measure
Nt_therm_init = $Nt_therm_init
Nt_measure = $Nt_measure
measure_transport_freq = $measure_transport_freq
bin_size = $bin_size
mu_tune_gain = $mu_tune_gain
mu_tune_interval = $mu_tune_interval
mu_tune_step_max = $mu_tune_step_max
mu_tune_tol = $mu_tune_tol
mu_min = $mu_min
mu_max = $mu_max
N_conf = $N_CONFS 
EOF

    if [ "$USE_TARGET_N" -eq 1 ]; then
        cat << EOF >> params.jl
target_n = $target_n
mu_init = $mu_init
p = ModelParameters(Lx, Ly, t, tp, W, n_imp, β, V, mass;
                    target_n=target_n, μ_init=mu_init,
                    μ_tune_gain=mu_tune_gain,
                    μ_tune_interval=mu_tune_interval,
                    μ_tune_step_max=mu_tune_step_max,
                    μ_tune_tol=mu_tune_tol,
                    μ_min=mu_min, μ_max=mu_max,
                    η=η, Δω=Δω, ω_max=ω_max)
EOF
        JOB_TAG="n${target_n}"
    else
        cat << EOF >> params.jl
μ = $mu
p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, V, mass;
                    μ_tune_gain=mu_tune_gain,
                    μ_tune_interval=mu_tune_interval,
                    μ_tune_step_max=mu_tune_step_max,
                    μ_tune_tol=mu_tune_tol,
                    μ_min=mu_min, μ_max=mu_max,
                    η=η, Δω=Δω, ω_max=ω_max)
EOF
        JOB_TAG="m${mu}"
    fi

    # --- 写入 submit.slurm ---
    # 使用 $N_CORES 变量
    cat << EOF > submit.slurm
#!/bin/sh
#SBATCH -J yyk_HMC/L${L}nimp${n_imp}${JOB_TAG}T${T}
#SBATCH -p $queue
#SBATCH -N $N_NODES
#SBATCH -n $N_CORES
#SBATCH -o job.out
#SBATCH -e job.err
export LD_LIBRARY_PATH=""
export MKL_NUM_THREADS=1
export JULIA_NUM_THREADS=1
julia --project="$PROJECT_ROOT" "$PROJECT_ROOT"/run_conf.jl
EOF

    pwd
    sbatch submit.slurm
    cd ..
done
