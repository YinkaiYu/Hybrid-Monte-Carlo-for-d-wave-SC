using DwaveHMC
using LinearAlgebra
using LogExpFunctions
using SparseArrays
using Printf

# 1. 构造一个小系统
# 使用 4x4，无杂质，低温
Lx, Ly = 10, 10
t, tp, μ = 1.0, -0.35, -1.0 # 稍微调整 mu 确保有粒子
W, n_imp = 0.0, 0.0 
β, J = 1000.0, 1.6 # 低温，超导相
p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, 0.1, 1.0)

println("--- Debugging Environment ---")
println("Params: β=$β, μ=$μ")

# 2. 初始化
state = initialize_state(p)
cache = initialize_cache(p)

# 3. 强制设置一个强的 d-wave 序参量 (人工制造超导态)
# 这样我们预期的 rho_s 应该很大且为正
fill!(state.Δ, 0.0) 
# d-wave: Δx = +0.2, Δy = -0.2
for i in 1:p.N
    state.Δ[i, 1] = 0.2  # x
    state.Δ[i, 2] = -0.2 # y
end

println("State initialized with artificial d-wave order.")

# 4. 更新 Hamiltonian 并对角化
init_static_H!(cache, p, state)
update_H_BdG!(cache, p, state)
diagonalize_H_BdG!(cache, p)

N = p.N
dim = 2 * N
β = p.β
U = cache.U
E = cache.E_n
@inbounds @simd for n in 1:(2*N)
    cache.fermi_factors[n] = logistic(-p.β * E[n])
end
f = cache.fermi_factors

##### 5. 计算超流刚度部分项，逐步调试

val_dia = 0.0

for n in 1:dim
    En = E[n]
    if En > 0
        w_n = 0.0 
        for i in 1:N 
            # u_{i,n} -> U[i, n] 
            # v_{i,n} -> U[i+N, n] 
            j_x = p.nn_table[i, 1] 
            j_xpy = p.nnn_table[i, 1] 
            j_xmy = p.nnn_table[i, 4] 
            w_n += p.t  * 2.0 * real( U[i+N,n]*conj(U[j_x+N,n]) - conj(U[i,n])*U[j_x,n] )
            w_n += p.tp * 2.0 * real( U[i+N,n]*conj(U[j_xpy+N,n]) - conj(U[i,n])*U[j_xpy,n] )
            w_n += p.tp * 2.0 * real( U[i+N,n]*conj(U[j_xmy+N,n]) - conj(U[i,n])*U[j_xmy,n] )
        end
        global val_dia += w_n * tanh(0.5 * β * En) / N
    end
end

val_dia2 = 0.0

for n in 1:dim
    # spin up
    w_n = 0.0 
    for i in 1:N 
        j_x = p.nn_table[i, 1] 
        j_xpy = p.nnn_table[i, 1] 
        j_xmy = p.nnn_table[i, 4] 
        w_n += p.t  * 2.0 * real( conj(U[i,n])*U[j_x,n] )
        w_n += p.tp * 2.0 * real( conj(U[i,n])*U[j_xpy,n] )
        w_n += p.tp * 2.0 * real( conj(U[i,n])*U[j_xmy,n] )
    end
    global val_dia2 += w_n * f[n] / N
    # spin down
    w_n = 0.0
    for i in 1:N 
        j_x = p.nn_table[i, 1] 
        j_xpy = p.nnn_table[i, 1] 
        j_xmy = p.nnn_table[i, 4] 
        w_n += p.t  * 2.0 * real( U[i+N,n]*conj(U[j_x+N,n]) )
        w_n += p.tp * 2.0 * real( U[i+N,n]*conj(U[j_xpy+N,n]) )
        w_n += p.tp * 2.0 * real( U[i+N,n]*conj(U[j_xmy+N,n]) )
    end
    global val_dia2 += w_n * (1.0 - f[n]) / N
end

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

build_current_operator!(cache, p)
mul!(cache.temp_JU, cache.Jx_sparse, U)
mul!(cache.J_mn, U', cache.temp_JU) 
J_mn = cache.J_mn # Alias


# 2. 顺磁项 Lambda_xx
# sum_{n,m} (f(n) - f(m))/(Em - En) |J_nm|^2
Lambda_xx = 0.0

# Pre-calculate f factors and E diffs to optimize
# Only iterate n != m? No, if n=m, limit is beta*f*(1-f).

for n in 1:dim
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
        
        global Lambda_xx += ratio * abs2(J_mn[n, m])
        if n == m
            println(@sprintf("n=m=%d, E=%.4f, f=%.4f, ratio=%.4f, |J_nn|^2=%.4f", 
                             n, E[n], f[n], ratio, abs2(J_mn[n, m])))
        end
    end
end
Lambda_xx /= N

println(@sprintf("Diamagnetic Term < -Kx >: %.6f", val_dia))
println(@sprintf("Diamagnetic Term < -Kx >2: %.6f", val_dia2))
println(@sprintf("Paramagnetic Term Λ_xx   : %.6f", Lambda_xx))
println(@sprintf("Computed Superfluid Stiffness (rho_s): %.6f", val_dia - Lambda_xx))