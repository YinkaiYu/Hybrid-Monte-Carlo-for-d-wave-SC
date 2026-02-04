#!/bin/bash
set -e 

# ================= 配置区域 =================
# 1. 扫描参数
T_list=(0.001 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15 0.2)

# 2. 并行与作业参数
queue='fat6348'
N_NODES=1   # 每个作业申请多少个节点
N_CORES=4   # 每个作业申请多少个核
PROJECT_ROOT=/home/zxli_1/yyk2025/2511_dWaveBcs/20251231_sweep-T

# 3. 物理参数
L=20
t=1.0
tp=-0.35
mu=-1.23359
W=0.0
n_imp=0.0
J=0.8
mass=1.0

# 4. 测量参数
omega_max=0.8

# 5. 平均场参数
Delta_MF_0=0.2
alpha=0.5
Delta_MF_tol=1e-6
Delta_MF_max_iter=2000
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
N_conf = 1
p = ModelParameters(Lx, Ly, t, tp, μ, β, J, mass;
                    W=W, n_imp=n_imp,
                    η=η, Δω=Δω, ω_max=ω_max,
                    Δ_MF_0=$Delta_MF_0, α=$alpha, Δ_MF_tol=$Delta_MF_tol, Δ_MF_max_iter=$Delta_MF_max_iter)
max_iter = $Delta_MF_max_iter
EOF

    # --- 写入 submit.slurm ---
    # 使用 $N_CORES 变量
    cat << EOF > submit.slurm
#!/bin/sh
#SBATCH -J yyk_MF/L${L}m${mu}T${T}
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
