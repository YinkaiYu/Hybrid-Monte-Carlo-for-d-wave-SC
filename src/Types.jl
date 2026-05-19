using LinearAlgebra
using Random
using SparseArrays
using FFTW        

# ---------------------------------------------------------
# 1. 模型参数 (不可变)
# ---------------------------------------------------------
"""
ModelParameters
存放所有的物理常数和系统尺寸。
使用 struct (默认不可变)，Julia 编译器能对此做极佳的优化。
"""
struct ModelParameters
    # 系统尺寸
    Lx::Int
    Ly::Int
    N::Int  # 总格点数 N = Lx * Ly
    
    # 物理参数
    t::Float64   # 近邻跃迁
    tp::Float64  # 次近邻 t'
    μ::Float64  # 固定化学势，或目标密度模式下的初始化学势
    has_target_n::Bool # 是否在热化阶段按目标电子密度调节 μ
    target_n::Float64  # 目标电子密度 n (仅 has_target_n=true 有效)

    # 无序参数
    W::Float64      # 杂质势强度
    n_imp::Float64  # 杂质浓度

    # HMC / 相互作用参数
    β::Float64   # 逆温度
    V::Float64      # 耦合常数 (benchmark-note 记号)
    mass::Float64   # HMC 虚拟质量
    μ_tune_gain::Float64    # μ 调节比例系数
    μ_tune_interval::Int    # μ 调节间隔 (sweeps)
    μ_tune_step_max::Float64 # 每次 μ 最大更新幅度
    μ_tune_tol::Float64      # |n - target_n| 容差，小于该值时停止调 μ
    μ_min::Float64          # μ 下界
    μ_max::Float64          # μ 上界
    
    # 预计算的邻居列表 (用空间换时间)
    # 存储形式：neighbor_table[site_index, direction_index]
    # 在 Julia 中，Vector{Vector{Int}} 有点慢，用 Matrix{Int} 更好 (列优先)
    nn_table::Matrix{Int}  # Nearest Neighbors (4个方向)
    nnn_table::Matrix{Int} # Next Nearest Neighbors (4个方向)

    # 光谱与输运计算参数
    η::Float64          # 展宽因子 (Broadening)
    ω_min::Float64      # 光电导频率下限
    ω_max::Float64      # 光电导频率上限
    Δω::Float64         # 频率步长
    n_ω::Int            # 频率点数
end

function Base.getproperty(p::ModelParameters, sym::Symbol)
    if sym === :J
        return getfield(p, :V)
    end
    return getfield(p, sym)
end

function _build_model_parameters(Lx::Int, Ly::Int, t, tp, μ, has_target_n::Bool, target_n, W, n_imp, β, V, mass;
                                 μ_tune_gain::Float64, μ_tune_interval::Int, μ_tune_step_max::Float64,
                                 μ_tune_tol::Float64,
                                 μ_min::Float64, μ_max::Float64,
                                 η::Float64=0.01, Δω::Float64=0.002, ω_max::Float64=4.0)
    if μ_tune_interval <= 0
        error("μ_tune_interval must be > 0")
    end
    if μ_tune_step_max <= 0
        error("μ_tune_step_max must be > 0")
    end
    if μ_tune_tol <= 0
        error("μ_tune_tol must be > 0")
    end
    if μ_min >= μ_max
        error("μ_min must be smaller than μ_max")
    end
    if has_target_n && !(0.0 <= target_n <= 2.0)
        error("target_n must be within [0, 2]")
    end

    N = Lx * Ly
    # 初始化邻居表
    # 约定方向：1: +x, 2: +y, 3: -x, 4: -y
    nn_table = zeros(Int, N, 4) 
    # 约定方向：1: +x+y, 2: -x+y, 3: -x-y, 4: +x-y
    nnn_table = zeros(Int, N, 4)

    # 辅助函数：处理周期性边界条件 (PBC)
    # x, y 是 1-based index
    function get_idx(x, y)
        x = mod1(x, Lx) # Julia自带的 mod1，结果在 [1, Lx] 之间
        y = mod1(y, Ly)
        return (y - 1) * Lx + x
    end

    for y in 1:Ly, x in 1:Lx
        i = get_idx(x, y)
        
        # --- Nearest Neighbors ---
        nn_table[i, 1] = get_idx(x + 1, y) # +x
        nn_table[i, 2] = get_idx(x, y + 1) # +y
        nn_table[i, 3] = get_idx(x - 1, y) # -x
        nn_table[i, 4] = get_idx(x, y - 1) # -y
        
        # --- Next Nearest Neighbors ---
        nnn_table[i, 1] = get_idx(x + 1, y + 1)
        nnn_table[i, 2] = get_idx(x - 1, y + 1)
        nnn_table[i, 3] = get_idx(x - 1, y - 1)
        nnn_table[i, 4] = get_idx(x + 1, y - 1)
    end
    
    ω_min = η
    n_ω = floor(Int, (ω_max - ω_min) / Δω) + 1
    
    return ModelParameters(Lx, Ly, N,
        Float64(t), Float64(tp), Float64(μ),
        Bool(has_target_n), Float64(target_n),
        Float64(W), Float64(n_imp),
        Float64(β), Float64(V), Float64(mass),
        Float64(μ_tune_gain), Int(μ_tune_interval), Float64(μ_tune_step_max), Float64(μ_tune_tol),
        Float64(μ_min), Float64(μ_max),
        nn_table, nnn_table,
        Float64(η), Float64(ω_min), Float64(ω_max), Float64(Δω), n_ω)
end

# 兼容旧接口：显式固定 μ
function ModelParameters(Lx::Int, Ly::Int, t, tp, μ, W, n_imp, β, V, mass;
                         η::Float64=0.01, Δω::Float64=0.002, ω_max::Float64=4.0,
                         μ_tune_gain::Float64=0.50, μ_tune_interval::Int=1,
                         μ_tune_step_max::Float64=0.08, μ_tune_tol::Float64=0.005,
                         μ_min::Float64=-4.0, μ_max::Float64=4.0)
    return _build_model_parameters(Lx, Ly, t, tp, μ, false, NaN, W, n_imp, β, V, mass;
                                   μ_tune_gain=μ_tune_gain, μ_tune_interval=μ_tune_interval,
                                   μ_tune_step_max=μ_tune_step_max, μ_tune_tol=μ_tune_tol,
                                   μ_min=μ_min, μ_max=μ_max,
                                   η=η, Δω=Δω, ω_max=ω_max)
end

# 新接口：μ 与 target_n 二选一
function ModelParameters(Lx::Int, Ly::Int, t, tp, W, n_imp, β, V, mass;
                         μ::Union{Nothing,Real}=nothing, target_n::Union{Nothing,Real}=nothing,
                         μ_init::Real=0.0,
                         η::Float64=0.01, Δω::Float64=0.002, ω_max::Float64=4.0,
                         μ_tune_gain::Float64=0.50, μ_tune_interval::Int=1,
                         μ_tune_step_max::Float64=0.08, μ_tune_tol::Float64=0.005,
                         μ_min::Float64=-4.0, μ_max::Float64=4.0)
    has_mu = μ !== nothing
    has_target_n = target_n !== nothing
    if has_mu == has_target_n
        error("Specify exactly one of μ or target_n.")
    end

    if has_mu
        μ_val = Float64(μ)
        return _build_model_parameters(Lx, Ly, t, tp, μ_val, false, NaN, W, n_imp, β, V, mass;
                                       μ_tune_gain=μ_tune_gain, μ_tune_interval=μ_tune_interval,
                                       μ_tune_step_max=μ_tune_step_max, μ_tune_tol=μ_tune_tol,
                                       μ_min=μ_min, μ_max=μ_max,
                                       η=η, Δω=Δω, ω_max=ω_max)
    end

    μ_val = Float64(μ_init)
    return _build_model_parameters(Lx, Ly, t, tp, μ_val, true, target_n, W, n_imp, β, V, mass;
                                   μ_tune_gain=μ_tune_gain, μ_tune_interval=μ_tune_interval,
                                   μ_tune_step_max=μ_tune_step_max, μ_tune_tol=μ_tune_tol,
                                   μ_min=μ_min, μ_max=μ_max,
                                   η=η, Δω=Δω, ω_max=ω_max)
end

# ---------------------------------------------------------
# 2. 模拟状态 (可变)
# ---------------------------------------------------------
"""
SimulationState
存放随蒙卡步演化的物理量。
使用 mutable struct。
"""
mutable struct SimulationState
    # 杂质构型 (静态无序)
    # discret_pot[i] = W or 0.0
    disorder_pot::Vector{Float64} 
    
    # 序参量场 Δ_ij
    # 我们只需要定义正方向的 bond: +x 和 +y。
    # Delta[i, 1] 对应 i -> i+x 的 bond
    # Delta[i, 2] 对应 i -> i+y 的 bond
    # 这样数组大小是 (N, 2)，内存连续，非常高效。
    Δ::Matrix{ComplexF64}
    
    # 共轭动量场 π_ij (对应 Delta)
    # 注意：在函数局部变量中尽量不要用 π，以免覆盖 Base.pi，但在 struct 字段里没问题
    π::Matrix{ComplexF64}

    # 当前 sweep 使用的有效化学势
    μ_eff::Float64
end

function initialize_state(p::ModelParameters)
    # 1. 生成无序势
    disorder_pot = zeros(Float64, p.N)
    # 随机选取 n_imp 比例的格点放置杂质
    n_sites_imp = round(Int, p.N * p.n_imp)
    imp_indices = randperm(p.N)[1:n_sites_imp] # 需要 using Random
    disorder_pot[imp_indices] .= p.W
    
    # 2. 初始化 Delta (比如随机热启动或冷启动)
    # 这里先给一个小的随机值
    Δ = (rand(ComplexF64, p.N, 2) .- (0.5 + 0.5im)) .* 0.1
    
    # 3. 初始化 Pi (置零，运行HMC时会重置)
    π = zeros(ComplexF64, p.N, 2)
    
    μ_eff = p.μ
    return SimulationState(disorder_pot, Δ, π, μ_eff)
end

# ---------------------------------------------------------
# 3. 计算缓存 (可变，核心优化)
# ---------------------------------------------------------
"""
ComputeCache
这是 Fortran 程序员最喜欢的部分。
我们在程序开始时预分配所有大矩阵，
后续计算全部使用 in-place 操作 (func!)，杜绝 calculation loop 中的 malloc。
"""
mutable struct ComputeCache
    # BdG 哈密顿量矩阵
    # 维度 2N x 2N, Hermitian
    # 注意：Julia 中 Hermitian 只是一个 wrapper，底层数据还是存放在矩阵里
    H_base::Matrix{ComplexF64} # 存储 H 的原始数据
    
    # 这是一个 wrapper，指向 H_base，告诉 LAPACK 它是厄米的
    # 我们更新时更新 H_base，计算时用 H_herm
    H_herm::Hermitian{ComplexF64, Matrix{ComplexF64}}
    
    # 对角化结果
    E_n::Vector{Float64}      # 长度 2N, Eigenvalues
    U::Matrix{ComplexF64}     # 2N x 2N, Eigenvectors
    
    # 力 F_ij 的缓存
    # 结构与 Delta 相同: (N, 2)
    forces::Matrix{ComplexF64}

    # 缓存预计算费米分布
    fermi_factors::Vector{Float64} 

    # 局域 d-wave 配对算符缓存 (长度 N)
    d_local_cache::Vector{ComplexF64}

    # 缓存用于 HMC 拒绝时的备份
    Δ_backup::Matrix{ComplexF64}
    E_n_backup::Vector{Float64}
    U_backup::Matrix{ComplexF64}
    
    # 输运计算缓存
    Jx_sparse_q0::SparseMatrixCSC{ComplexF64, Int} # 稀疏电流算符 Jx(q=0) (2N x 2N)
    Jx_sparse_qy::SparseMatrixCSC{ComplexF64, Int} # 稀疏电流算符 Jx(qx=0,qy=2π/Ly) (2N x 2N)
    J_mn::Matrix{ComplexF64}                       # 电流矩阵元 <n|Jx(q)|m> (2N x 2N, 稠密)
    temp_JU::Matrix{ComplexF64}
    
    # FFT 计划和缓存
    u_r_cache::Matrix{ComplexF64} # 用于存储 fft 前的波函数 (Lx x Ly)
    u_k_cache::Matrix{ComplexF64} # 用于存储 fft 后的波函数 (Lx x Ly)
    fft_plan::FFTW.cFFTWPlan      # 预计算的 FFT 计划

    # 站点坐标与奇偶性缓存
    x_idx::Vector{Int}
    y_idx::Vector{Int}
    parity_x::Vector{Int8}  # (-1)^x
    parity_y::Vector{Int8}  # (-1)^y

    # kx=π 路径的一维 FFT 缓存 (沿 y 方向)
    u_pi_cache::Vector{ComplexF64}
    u_pi_k_cache::Vector{ComplexF64}
    fft_plan_y::FFTW.cFFTWPlan

    # 光谱测量缓存
    omega_grid::Vector{Float64}
    sigma_omega::Vector{Float64}
    dos_omega_grid::Vector{Float64}
    dos_vals::Vector{Float64}
    dos_M_vals::Vector{Float64}
    ldos_ω0::Vector{Float64}
    ak_map::Matrix{Float64}
    ak_mx_path::Matrix{Float64}
    ak_xg_path::Matrix{Float64}
    lor_cache::Vector{Float64}
    mx_path_weights::Vector{Float64}
    xg_path_weights::Vector{Float64}
    omega_inv::Vector{Float64}
end

function initialize_cache(p::ModelParameters)
    dim = 2 * p.N
    H_base = zeros(ComplexF64, dim, dim)
    # uplo=:U 表示我们将只填充上三角部分，LAPACK 会自动处理
    H_herm = Hermitian(H_base, :U) 
    
    E_n = zeros(Float64, dim)
    U = zeros(ComplexF64, dim, dim)
    forces = zeros(ComplexF64, p.N, 2)
    fermi_factors = zeros(Float64, dim)
    d_local_cache = zeros(ComplexF64, p.N)
    Δ_backup = zeros(ComplexF64, p.N, 2)
    E_n_backup = zeros(Float64, dim)
    U_backup = zeros(ComplexF64, dim, dim)
    
    # 1. 构造稀疏电流算符 (结构不变，只初始化一次)
    # 我们将在专门的函数里填充它，这里先分配空
    Jx_sparse_q0 = spzeros(ComplexF64, dim, dim)
    Jx_sparse_qy = spzeros(ComplexF64, dim, dim)
    J_mn = zeros(ComplexF64, dim, dim)
    temp_JU = zeros(ComplexF64, dim, dim) 
    
    # 2. FFT
    # 创建一个临时的 Lx * Ly 矩阵来生成 plan
    u_r_cache = zeros(ComplexF64, p.Lx, p.Ly)
    u_k_cache = zeros(ComplexF64, p.Lx, p.Ly)
    fft_plan = plan_fft(u_k_cache) # 预规划

    x_idx = Vector{Int}(undef, p.N)
    y_idx = Vector{Int}(undef, p.N)
    parity_x = Vector{Int8}(undef, p.Lx)
    parity_y = Vector{Int8}(undef, p.Ly)
    @inbounds for x in 1:p.Lx
        parity_x[x] = iseven(x) ? Int8(1) : Int8(-1)
    end
    @inbounds for y in 1:p.Ly
        parity_y[y] = iseven(y) ? Int8(1) : Int8(-1)
    end
    @inbounds for i in 1:p.N
        x = mod1(i, p.Lx)
        y = cld(i, p.Lx)
        x_idx[i] = x
        y_idx[i] = y
    end

    u_pi_cache = zeros(ComplexF64, p.Ly)
    u_pi_k_cache = zeros(ComplexF64, p.Ly)
    fft_plan_y = plan_fft(u_pi_k_cache)

    omega_grid = collect(p.ω_min:p.Δω:p.ω_max)
    sigma_omega = zeros(Float64, length(omega_grid))
    dos_omega_grid = collect(-p.ω_max:p.Δω:p.ω_max)
    dos_vals = zeros(Float64, length(dos_omega_grid))
    dos_M_vals = zeros(Float64, length(dos_omega_grid))
    ldos_ω0 = zeros(Float64, p.N)
    ak_map = zeros(Float64, p.Lx, p.Ly)
    mx_path_len = fld(p.Ly, 2) + 1
    xg_path_len = fld(min(p.Lx, p.Ly), 2) + 1
    ak_mx_path = zeros(Float64, mx_path_len, length(dos_omega_grid))
    ak_xg_path = zeros(Float64, xg_path_len, length(dos_omega_grid))
    lor_cache = zeros(Float64, length(dos_omega_grid))
    mx_path_weights = zeros(Float64, mx_path_len)
    xg_path_weights = zeros(Float64, xg_path_len)
    omega_inv = 1.0 ./ omega_grid

    return ComputeCache(H_base, H_herm, E_n, U, forces, fermi_factors, 
                        d_local_cache, Δ_backup, E_n_backup, U_backup,
                        Jx_sparse_q0, Jx_sparse_qy, J_mn, temp_JU,
                        u_r_cache, u_k_cache, fft_plan,
                        x_idx, y_idx, parity_x, parity_y,
                        u_pi_cache, u_pi_k_cache, fft_plan_y,
                        omega_grid, sigma_omega, dos_omega_grid,
                        dos_vals, dos_M_vals, ldos_ω0, ak_map, ak_mx_path, ak_xg_path,
                        lor_cache, mx_path_weights, xg_path_weights, omega_inv)
end
