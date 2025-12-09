using LinearAlgebra
using Random

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
    μ::Float64  # 化学势
    
    # 无序参数
    W::Float64      # 杂质势强度
    n_imp::Float64  # 杂质浓度
    
    # HMC / 相互作用参数
    β::Float64   # 逆温度
    J::Float64      # 耦合常数
    dt::Float64     # MD 时间步长
    mass::Float64   # HMC 虚拟质量
    
    # 预计算的邻居列表 (用空间换时间)
    # 存储形式：neighbor_table[site_index, direction_index]
    # 在 Julia 中，Vector{Vector{Int}} 有点慢，用 Matrix{Int} 更好 (列优先)
    nn_table::Matrix{Int}  # Nearest Neighbors (4个方向)
    nnn_table::Matrix{Int} # Next Nearest Neighbors (4个方向)
end

# 构造函数：输入基本参数，自动计算 N 和邻居表
function ModelParameters(Lx::Int, Ly::Int, t, tp, μ, W, n_imp, β, J, dt, mass)
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
    
    return ModelParameters(Lx, Ly, N, 
        Float64(t), Float64(tp), Float64(μ), 
        Float64(W), Float64(n_imp), 
        Float64(β), Float64(J), Float64(dt), Float64(mass),
        nn_table, nnn_table)
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
    
    return SimulationState(disorder_pot, Δ, π)
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
    
    # 辅助向量/矩阵 (如果需要中间计算)
    # 例如计算 Force 时可能需要用到
end

function initialize_cache(p::ModelParameters)
    dim = 2 * p.N
    H_base = zeros(ComplexF64, dim, dim)
    # uplo=:U 表示我们将只填充上三角部分，LAPACK 会自动处理
    H_herm = Hermitian(H_base, :U) 
    
    E_n = zeros(Float64, dim)
    U = zeros(ComplexF64, dim, dim)
    forces = zeros(ComplexF64, p.N, 2)
    
    return ComputeCache(H_base, H_herm, E_n, U, forces)
end