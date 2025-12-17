using LinearAlgebra
using LogExpFunctions

"""
    init_static_H!(cache::ComputeCache, p::ModelParameters, state::SimulationState)

初始化 BdG 哈密顿量的静态部分（动能 + 势能）。
这个函数只需在模拟开始（或杂质构型改变）时调用一次。
"""
function init_static_H!(cache::ComputeCache, p::ModelParameters, state::SimulationState)
    N = p.N
    H = cache.H_base # 这一步是指针传递，后面修改 H 就是修改 cache.H_base
    
    # 1. 清零 (防止有残留数据)
    fill!(H, 0.0 + 0.0im)
    
    # 2. 对角项 (势能)
    @inbounds for i in 1:N
        term = state.disorder_pot[i] - p.μ
        H[i, i] = term
        H[i+N, i+N] = -term
    end
    
    # 3. 动能项 (Hopping)
    # 只填上三角
    @inbounds for i in 1:N
        # --- Nearest Neighbors ---
        for dir in 1:4
            j = p.nn_table[i, dir]
            if j > i 
                H[i, j] = -p.t
                H[i+N, j+N] = p.t
            end
        end
        
        # --- Next Nearest Neighbors ---
        for dir in 1:4
            j = p.nnn_table[i, dir]
            if j > i
                H[i, j] = -p.tp
                H[i+N, j+N] = p.tp
            end
        end
    end
    
    return nothing
end

"""
    update_H_BdG!(cache::ComputeCache, p::ModelParameters, state::SimulationState)

只更新 BdG 哈密顿量的动态部分（配对势 Δ）。
前提：cache.H_base 已经包含了正确的静态部分。
"""
function update_H_BdG!(cache::ComputeCache, p::ModelParameters, state::SimulationState)
    N = p.N
    H = cache.H_base
    
    # 4. 配对项 (Pairing terms)
    # 填充右上角块 (1..N, N+1..2N)
    # 这里全是上三角区域，因为 row <= N < col
    # 每一个 bond (i, j) 贡献两个矩阵元:
    # (i, j+N) -> Δ_{ij}
    # (j, i+N) -> Δ_{ji} = Δ_{ij} 
    # 也即 H_BdG(Top-Right) = (1/2) * Δ_{ij} * ( c^†_{i↑} c^†_{j↓} + c^†_{j↑} c^†_{i↓} )
    
    # 我们遍历 state.Δ，它只存储了 +x (dir=1) 和 +y (dir=2) 的键
    @inbounds for i in 1:N
        # +x direction bond
        j_x = p.nn_table[i, 1] 
        val_x = 0.5 * state.Δ[i, 1]
        
        # 对应的矩阵元，直接覆盖原有数值 (Overwrite)
        H[i, j_x + N] = val_x  # (1/2) * Δ_{ij} c^†_{i↑} c^†_{j↓}
        H[j_x, i + N] = val_x  # (1/2) * Δ_{ij} c^†_{j↑} c^†_{i↓} 
        
        # +y direction bond
        j_y = p.nn_table[i, 2]
        val_y = 0.5 * state.Δ[i, 2]
        
        H[i, j_y + N] = val_y
        H[j_y, i + N] = val_y
    end

    return nothing
end

"""
    diagonalize_H_BdG!(cache::ComputeCache, p::ModelParameters)

对角化 H_BdG 并计算 HMC 能量。
H_HMC = ... - sum(log(2*cosh(beta*E/2))) ...
注意：使用标准 eigen! 进行对角化。
注意：这里只计算费米子行列式部分的贡献，玻色子项(动能+势能)在外部计算。
"""
function diagonalize_H_BdG!(cache::ComputeCache, p::ModelParameters)
    # 1. 保护原始哈密顿量
    # 因为 eigen! 会破坏输入矩阵，而我们的 cache.H_base 包含着下一时间步需要的静态项(动能等)。
    # 所以必须先将 H_base 拷贝到工作空间 U 中。
    copyto!(cache.U, cache.H_base)
    
    # 2. 对角化
    # 我们对 U 进行 Hermitian 封装，eigen! 会利用对称性加速。
    # 注意：eigen! 会返回新的 vals 和 vecs 数组 (这里会有一次内存分配)，
    # 但为了代码的稳健性，这是值得的。
    vals, vecs = eigen!(Hermitian(cache.U, :U))
    
    # 3. 将结果存回 Cache
    # vals 是实数，vecs 是复数矩阵
    copyto!(cache.E_n, vals)
    copyto!(cache.U, vecs)
    
    return nothing
end