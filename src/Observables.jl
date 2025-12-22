using LinearAlgebra
using LogExpFunctions

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

# src/Observables.jl 添加到文件末尾

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
    # [新增] Benchmark 相关的计算
    # ==========================================
    sum_diff = 0.0
    sum_pair_global = 0.0 + 0.0im # sum (P_x - P_y)/2
    
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
        sum_pair_global += term
    end
    
    val_diff = sum_diff / N
    val_pair = abs(sum_pair_global / N)
    
    return ObservablesResult(total_energy, val_amp, val_local, val_global, val_S, val_hole,val_diff, val_pair)
end