#!/bin/bash
set -e 

# ================= 配置区域 =================
# 1. 扫描参数
T_list=(0.001 0.002 0.004 0.008 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

# 2. 并行与作业参数
queue='node6348'
N_NODES=1   # 每个作业申请多少个节点
N_CORES=4   # 每个作业申请多少个核
N_CONFS=16  # 每个温度跑多少个构型
PROJECT_ROOT=/home/zxli_1/yyk2025/2511_dWaveBcs/20251231_sweep-T

# 3. 物理参数
L=40
t=1.0
tp=-0.35
mu=-1.08
W=1.0
n_imp=0.0
J=0.8
mass=1.0

# 4. 测量参数
omega_max=4.0

# 5. HMC参数
n_therm=100
n_measure=500
Nt_therm_init=16
Nt_measure=6
measure_transport_freq=1
bin_size=5
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
t, tp, μ = $t, $tp, $mu
W, n_imp = $W, $n_imp
J = $J
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
N_conf = $N_CONFS 
p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, mass; 
                    η=η, Δω=Δω, ω_max=ω_max)
EOF

    # --- 写入 submit.slurm ---
    # 使用 $N_CORES 变量
    cat << EOF > submit.slurm
#!/bin/sh
#SBATCH -J yyk_HMC/L${L}n${n_imp}m${mu}T${T}
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