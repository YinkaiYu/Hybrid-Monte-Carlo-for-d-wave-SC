using LinearAlgebra
using LogExpFunctions
using SparseArrays
using FFTW
using JLD2

"""
    compute_forces!(cache::ComputeCache, p::ModelParameters, state::SimulationState)

计算 HMC 演化所需的力 F_ij，并存储在 cache.forces 中。
公式：F_ij = -β/(2J) * ( Δ_ij - J * P_ij )
其中 P_ij = <c_i↑ c_j↓ - c_i↓ c_j↑> = -ρ_{i, j+N} - ρ_{j, i+N}
"""
function compute_forces!(cache::ComputeCache, p::ModelParameters, state::SimulationState)
    N = p.N
    forces = cache.forces
    U = cache.U
    E = cache.E_n
    f = cache.fermi_factors
    
    β_over_2J = p.β / (2 * p.J)

    # 预计算费米分布
    @inbounds @simd for n in 1:(2*N)
        # 使用库函数 logistic，自动处理溢出
        # fermi(E) = 1 / (1 + exp(βE)) = logistic(-βE)
        f[n] = logistic(-p.β * E[n])
    end
    
    # 遍历所有定义的 Bond (只遍历 +x 和 +y，与 Δ 的存储结构一致)
    # 这一段后面可以考虑做  Loop Reordering 优化，把 n 放在最外层，dir 放在最内层
    @inbounds for i in 1:N
        # 对方向进行循环：1 (+x), 2 (+y)
        for dir in 1:2
            j = p.nn_table[i, dir]
            
            # 计算 Pairing correlation P_ij
            # P_ij = -ρ_{i, j+N} - ρ_{j, i+N}
            # ρ_{u, v} = sum_n U[u, n] * f(E[n]) * conj(U[v, n])
            
            ρ_1 = zero(ComplexF64) # ρ_{i, j+N}
            ρ_2 = zero(ComplexF64) # ρ_{j, i+N}
            
            # @simd 提示编译器进行向量化优化
            @simd for n in 1:(2*N)
                
                # U[row, col] -> U[i, n]
                # 这是一个行遍历，在 Julia 中跨度较大，后面再考虑优化吧
                ρ_1 += U[i, n] * f[n] * conj(U[j+N, n])
                ρ_2 += U[j, n] * f[n] * conj(U[i+N, n])
            end
            
            P_ij = -ρ_1 - ρ_2
            
            # 计算力 F_ij
            Δ_val = state.Δ[i, dir]
            forces[i, dir] = -β_over_2J * (Δ_val - p.J * P_ij)
        end
    end
    
    return nothing
end

"""
    ObservablesResult

用于存储单次测量的结果，方便写入文件。
使用 NamedTuple 也可以，但定义结构体更清晰。
"""
struct ObservablesResult
    total_energy::Float64
    Δ_amp::Float64   # 幅度
    Δ_local::Float64 # 局域 d-wave
    Δ_global::Float64 # 全局 d-wave (绝对值)
    S_Δ::Float64     # 结构因子
    hole_conc::Float64 # 空穴浓度 p
    Δ_diff::Float64
    Δ_pair::Float64
    Δ_localpair::Float64
end

"""
    measure_observables(cache::ComputeCache, p::ModelParameters, state::SimulationState)

计算所有感兴趣的物理量。
假设调用前 H_BdG 已经对角化 (cache.E_n, cache.U 是最新的)。
"""
function measure_observables(cache::ComputeCache, p::ModelParameters, state::SimulationState)
    N = p.N
    
    # --- 1. 序参量相关 ---
    # Δ[i, 1] -> x 方向, Δ[i, 2] -> y 方向
    sum_amp = 0.0
    sum_local = 0.0
    sum_global = 0.0 + 0.0im # 复数累加
    
    @inbounds for i in 1:N
        dx = state.Δ[i, 1]
        dy = state.Δ[i, 2]
        
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # Amplitude: (|Dx| + |Dy|) / 2
        sum_amp += 0.5 * (abs_dx + abs_dy)
        
        # Local d-wave: |Dx - Dy| / 2 
        sum_local += 0.5 * abs(dx - dy)
        
        # Global d-wave (inside sum): (Dx - Dy)/2
        sum_global += 0.5 * (dx - dy)
    end
    
    val_amp = sum_amp / N
    val_local = sum_local / N
    val_global = abs(sum_global / N)
    val_S = abs2(sum_global / N) # 结构因子
    
    # --- 2. 电子/空穴浓度 ---
    # p = (1/N) * sum_{E_n > 0} ( sum_i (|u|^2 - |v|^2) ) * tanh(βEn/2)
    # 利用 cache.U 和 cache.E_n
    
    U = cache.U
    E = cache.E_n
    total_p_term = 0.0
    
    @inbounds for n in 1:(2*N)
        En = E[n]
        if En > 0
            # 计算该本征态的空间权重差 sum(|u|^2 - |v|^2)
            w_n = 0.0
            @simd for i in 1:N
                # u_{n,i} -> U[i, n]
                # v_{n,i} -> U[i+N, n]
                u2 = abs2(U[i, n])
                v2 = abs2(U[i+N, n])
                w_n += (u2 - v2)
            end
            
            # 乘以 tanh
            total_p_term += w_n * tanh(0.5 * p.β * En)
        end
    end
    
    val_hole = total_p_term / N
    
    # --- 3. 能量 (假设外部已计算，或者重新算) ---
    # 1. 费米子部分
    # E_fermion (从缓存读取)
    # - sum_{E>0} βE + 2*log1p(exp(-βE))
    E_fermion = 0.0
    @inbounds for E in cache.E_n
        if E > 0
            x = p.β * E
            E_fermion -= (x + 2.0*log1pexp(-x))
        end
    end

    # 2. 玻色子势能部分
    # E_boson = (β / 2J) * sum(|Δ|^2)
    # 使用 sum(f, itr) 极其高效，无内存分配
    coef_boson = p.β / (2 * p.J)
    E_boson = coef_boson * sum(abs2, state.Δ)

    total_energy = (E_fermion + E_boson)/N

    # ==========================================
    # Benchmark 相关的计算
    # ==========================================
    sum_diff = 0.0
    sum_pair_global = 0.0 + 0.0im # sum (P_x - P_y)/2
    sum_pair_local = 0.0 
    
    # 重新刷新 fermi factors (确保是最新的)
    @inbounds @simd for n in 1:(2*N)
        cache.fermi_factors[n] = logistic(-p.β * E[n])
    end
    f = cache.fermi_factors
    
    @inbounds for i in 1:N
        # 我们需要计算 x方向 和 y方向 的 P_{ij}
        # P_dir = < c_{i↑} c_{j↓} - c_{i↓} c_{j↑} >
        
        # --- x direction ---
        jx = p.nn_table[i, 1] # +x neighbor
        ρ_1x = zero(ComplexF64)
        ρ_2x = zero(ComplexF64)
        @simd for n in 1:(2*N)
            ρ_1x += U[i, n] * f[n] * conj(U[jx+N, n])
            ρ_2x += U[jx, n] * f[n] * conj(U[i+N, n])
        end
        P_x = -ρ_1x - ρ_2x
        
        # --- y direction ---
        jy = p.nn_table[i, 2] # +y neighbor
        ρ_1y = zero(ComplexF64)
        ρ_2y = zero(ComplexF64)
        @simd for n in 1:(2*N)
            ρ_1y += U[i, n] * f[n] * conj(U[jy+N, n])
            ρ_2y += U[jy, n] * f[n] * conj(U[i+N, n])
        end
        P_y = -ρ_1y - ρ_2y
        
        # 1. Diff: |Δ - J*P|
        # 记得 state.Δ 也是 (N, 2)
        diff_x = abs(state.Δ[i, 1] - p.J * P_x)
        diff_y = abs(state.Δ[i, 2] - p.J * P_y)
        sum_diff += (diff_x + diff_y) / 2.0 # 平均每个 bond 的偏差
        
        # 2. Pair order parameter (from fermions): J * (P_x - P_y)/2
        # 注意公式里的 J 因子
        term = p.J * 0.5 * (P_x - P_y)
        sum_pair_local += abs(term)
        sum_pair_global += term
    end
    
    val_diff = sum_diff / N
    val_pair = abs(sum_pair_global / N)
    val_localpair = sum_pair_local / N
    
    return ObservablesResult(total_energy, val_amp, val_local, val_global, val_S, val_hole,val_diff, val_pair, val_localpair)
end


# ------------------------------------------------
# 1. 初始化辅助工具
# ------------------------------------------------

"""
    build_current_operator!(cache::ComputeCache, p::ModelParameters)

构建 x 方向电流算符的稀疏矩阵 Jx。
Jx = i * sum ( t * c^dag_i c_{i+x} + t' * ... - h.c. )
注意：在 Nambu 表象下，Jx = diag(Jx_particle, Jx_particle)。
因为 Jx_hole = Jx_particle (对于实数 t)。
"""
function build_current_operator!(cache::ComputeCache, p::ModelParameters)
    N = p.N
    # 使用 Triplet 格式构建稀疏矩阵 (I, J, V)
    I_idx = Int[]
    J_idx = Int[]
    V_val = ComplexF64[]
    
    # 辅助函数：添加项 c^dag_u c_v 的系数 val
    # 即 matrix[u, v] += val
    function add_term!(u, v, val)
        push!(I_idx, u); push!(J_idx, v); push!(V_val, val)
    end
    
    # 遍历所有格点构建 Particle block (N x N)
    for i in 1:N
        # +x neighbor (Nearest Neighbor)
        j_x = p.nn_table[i, 1] 
        # Term: i * t * (c^dag_i c_j - c^dag_j c_i)
        val = im * p.t
        add_term!(i, j_x, val)      # <i|J|j>
        add_term!(j_x, i, conj(val)) # <j|J|i>
        
        # +x+y (dir=1 in nnn)
        j_xpy = p.nnn_table[i, 1]
        val_tp = im * p.tp 
        add_term!(i, j_xpy, val_tp)
        add_term!(j_xpy, i, conj(val_tp))
        
        # +x-y (dir=4 in nnn, neighbor of i is i+x-y)
        # 注意: nnn_table 定义: 1:+x+y, 2:-x+y, 3:-x-y, 4:+x-y
        j_xmy = p.nnn_table[i, 4] 
        val_tp = im * p.tp 
        add_term!(i, j_xmy, val_tp)
        add_term!(j_xmy, i, conj(val_tp))
    end
    
    # 构建 N x N 稀疏矩阵
    Jx_part = sparse(I_idx, J_idx, V_val, N, N)
    
    # 构建完整的 2N x 2N Nambu 矩阵
    # J_BdG = [ Jx_part   0       ]
    #         [ 0         Jx_part ]
    # 因为对于实数 hopping，空穴部分的电流算符矩阵元与粒子部分相同
    cache.Jx_sparse = blockdiag(Jx_part, Jx_part)
    
    return nothing
end

# ------------------------------------------------
# 2. 复杂物理量测量结构体
# ------------------------------------------------

"""
SpectrumResult
用于存储 JLD2 的重型数据
"""
struct SpectrumResult
    # 标量结果
    superfluid_stiffness::Float64
    dc_conductivity::Float64
    
    # 谱学结果 (Arrays)
    ω_grid::Vector{Float64}                 # 光电导用的 ω > 0
    optical_conductivity::Vector{Float64}   # Re σ(ω)
    dos_ω_grid::Vector{Float64}             # DOS 用的完整网格
    dos::Vector{Float64}                    # N(ω)
    dos_AN::Vector{Float64}                  # AN 点的 DOS
    
    # 动量解析的谱权重 (可选，数据量巨大，通常只存特定路径或求和)
    # 我们这里存: A(k, ω=0) (Fermi Surface) 和 DOS.
    A_k_ω0::Matrix{Float64} # 费米面谱权重
end

# ------------------------------------------------
# 3. 核心测量函数
# ------------------------------------------------

function measure_transport_and_spectra(cache::ComputeCache, p::ModelParameters)
    N = p.N
    dim = 2 * N
    β = p.β
    U = cache.U
    E = cache.E_n
    f = cache.fermi_factors # already updated in standard measure
    
    # ------------------------------------------------
    # A. 计算电流矩阵元 J_mn = <n|Jx|m>
    # ------------------------------------------------
    # 1. Temp = Jx_sparse * U  (Sparse * Dense -> Dense)
    # 2. J_mn = U' * Temp      (Dense * Dense -> Dense)
    # 这是 BLAS Level 3 操作，MKL 极快。
    
    # 注意：Jx_sparse 是常数，如果还没初始化需要初始化
    if nnz(cache.Jx_sparse) == 0
        build_current_operator!(cache, p)
    end
    
    mul!(cache.temp_JU, cache.Jx_sparse, U)
    mul!(cache.J_mn, U', cache.temp_JU) 
    
    J_mn = cache.J_mn # Alias
    
    # ------------------------------------------------
    # B. 超流刚度 ρ_s
    # ------------------------------------------------
    # 1. 抗磁项 < -Kx >
    
    val_dia = 0.0

    @inbounds for n in 1:dim
        En = E[n]
        if En > 0
            w_n = 0.0 
            @simd for i in 1:N 
                # u_{i,n} -> U[i, n] 
                # v_{i,n} -> U[i+N, n] 
                j_x = p.nn_table[i, 1] 
                j_xpy = p.nnn_table[i, 1] 
                j_xmy = p.nnn_table[i, 4] 
                w_n += p.t  * 2.0 * real( U[i+N,n]*conj(U[j_x+N,n]) - conj(U[i,n])*U[j_x,n] )
                w_n += p.tp * 2.0 * real( U[i+N,n]*conj(U[j_xpy+N,n]) - conj(U[i,n])*U[j_xpy,n] )
                w_n += p.tp * 2.0 * real( U[i+N,n]*conj(U[j_xmy+N,n]) - conj(U[i,n])*U[j_xmy,n] )
            end
            val_dia += w_n * tanh(0.5 * β * En) / N
        end
    end
    
    # 2. 顺磁项 Lambda_xx
    # sum_{n,m} (f(n) - f(m))/(Em - En) |J_nm|^2
    Lambda_xx = 0.0
    
    @inbounds for n in 1:dim
        for m in 1:dim
            diff_E = E[m] - E[n]
            diff_f = f[n] - f[m]
            
            ratio = 0.0
            if abs(diff_E) < 1e-8
                # limit E_m -> E_n
                # ratio = - f'(E_n) = beta * f[n] * (1 - f[n])
                ratio = β * f[n] * (1.0 - f[n])
            else
                ratio = diff_f / diff_E
            end
            
            Lambda_xx += ratio * abs2(J_mn[n, m])
        end
    end
    Lambda_xx /= N
    
    superfluid_stiffness = val_dia - Lambda_xx
    
    # ------------------------------------------------
    # C. 光电导与直流电导
    # ------------------------------------------------
    # Re σ(ω) = (π/Nω) * sum_{n,m} (f(n)-f(m)) |J|^2 delta(ω - (Em - En))
    # DC: ω -> 0 limit.
    
    # Grid
    ω_grid = collect(p.ω_min : p.Δω : p.ω_max)
    σ_ω = zeros(Float64, length(ω_grid))
    dc_cond = 0.0
    
    # Pre-calculate delta function broadening
    function lorentzian(x, η)
        return (1.0/π) * (η / (x^2 + η^2))
    end
    
    @inbounds for n in 1:dim
        for m in 1:dim
            Em_En = E[m] - E[n]
            J2 = abs2(J_mn[n, m])
            
            # 1. DC Conductivity
            # sum (-f') |J|^2 delta(Em - En)
            # -f' = β * f * (1-f)
            dc_cond += (β * f[n] * (1.0 - f[n])) * J2 * lorentzian(Em_En, p.η)
            
            # 2. Optical Conductivity
            fn_fm = f[n] - f[m]
            if abs(fn_fm) < 1e-12 continue end
            for (iω, ω) in enumerate(ω_grid)
                σ_ω[iω] += (fn_fm / ω) * J2 * lorentzian(ω - Em_En, p.η)
            end
        end
    end
    
    dc_cond *= (π / N)
    σ_ω .*= (π / N)
    
    # ------------------------------------------------
    # D. 态密度 (DOS) & 谱函数 A(k, 0)
    # ------------------------------------------------
    # DOS 网格：从 -ω_max 到 +ω_max (或者稍微大一点，覆盖整个能带)
    # 我们这里使用对称的区间
    dos_ω_grid = collect(-p.ω_max : p.Δω : p.ω_max)
    dos_vals = zeros(Float64, length(dos_ω_grid))
    dos_AN_vals = zeros(Float64, length(dos_ω_grid))
    
    # A(k, w=0) map
    # A(k, 0) ~ sum_n |u_n(k)|^2 delta(0 - En)
    ak_map = zeros(Float64, p.Lx, p.Ly)
    
    for n in 1:dim
        En = E[n]
        # 1. Calculate weight W_n = sum_i |u_{i,n}|^2
        # u is top half of U
        w_n = 0.0
        @simd for i in 1:N
            w_n += abs2(U[i, n])
        end
        
        # 2. Add to DOS
        # We need to cover negative energies too? Usually DOS is symmetric or plotted full.
        # Let's just plot for w in ω_grid (positive).
        # Check symmetry: E_n and -E_n.
        for (iw, w) in enumerate(dos_ω_grid)
            dos_vals[iw] += w_n * lorentzian(w - En, p.η)
        end

        # 3. DOS at Antinodal Point
        # AN point in 2D square lattice: (π, 0) or (0, π)
        # sum_{x,y} u(x,y) * (-1)^x  and  sum_{x,y} u(x,y) * (-1)^y
        sum_pi_0 = ComplexF64(0.0) # k=(pi, 0)
        sum_0_pi = ComplexF64(0.0) # k=(0, pi)
        @inbounds for i in 1:N
            # 将 i 转换为 (x, y) 坐标，1-based
            x = mod1(i, p.Lx)
            y = cld(i, p.Lx)
            val = U[i, n]
            # (-1)^x
            if iseven(x)
                sum_pi_0 += val
            else
                sum_pi_0 -= val
            end
            # (-1)^y
            if iseven(y)
                sum_0_pi += val
            else
                sum_0_pi -= val
            end
        end

        # 计算谱权重 |u_k|^2
        # 注意归一化系数 1/sqrt(N) 平方后为 1/N
        weight_AN = 0.5 * (abs2(sum_pi_0) + abs2(sum_0_pi)) / N

        # 累加到 dos_AN (使用相同的 Lorentzian 展宽)
        for (iw, w) in enumerate(dos_ω_grid)
            dos_AN_vals[iw] += weight_AN * lorentzian(w - En, p.η)
        end
        
        
        # 4. Spectral Function A(k, 0) (Fermi Surface intensity)
        # Check if En is close to 0 (within η)
        weight_at_zero = lorentzian(0.0 - En, p.η)
        
        if weight_at_zero > 1e-6
            # Perform FFT for this eigenstate
            # Copy u_{i,n} to buffer
            for i in 1:N
                # Map 1D i to 2D (x,y)
                x = mod1(i, p.Lx)
                y = cld(i, p.Lx)
                cache.u_r_cache[x, y] = U[i, n]
            end
            
            # 执行 In-place FFT
            # cache.fft_plan * cache.u_r_cache -> cache.u_k_cache
            # 使用 mul! 避免内存分配
            mul!(cache.u_k_cache, cache.fft_plan, cache.u_r_cache)
            
            # Add to map: |u_k|^2 * delta(E)
            for y in 1:p.Ly, x in 1:p.Lx
                ak_map[x, y] += abs2(cache.u_k_cache[x, y]) * weight_at_zero
            end
        end
    end
    
    dos_vals ./= N
    ak_map ./= N # Normalization of FFT
    # FFTW definition: backward fft (default) is unnormalized sum. 
    # 1/sqrt(N) factor in definition means |FFT|^2 / N.
    
    return SpectrumResult(superfluid_stiffness, dc_cond, 
                          ω_grid, σ_ω, 
                          dos_ω_grid, dos_vals, dos_AN_vals, 
                          ak_map)
end