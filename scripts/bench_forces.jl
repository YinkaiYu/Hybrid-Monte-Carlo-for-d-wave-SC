using BenchmarkTools
using LinearAlgebra
using Random
using Printf

# --- 1. 模拟数据环境 ---
function setup_data(N)
    # N 个格点
    dim = 2 * N
    
    # 模拟 Eigenvectors (随机复数矩阵)
    U = rand(ComplexF64, dim, dim)
    
    # 模拟 Fermi factors
    f = rand(Float64, dim)
    
    # 模拟 Neighbor table (简单的周期性一维链近似，只为了测试内存访问)
    nn_table = zeros(Int, N, 2)
    for i in 1:N
        nn_table[i, 1] = mod1(i+1, N) # +x
        nn_table[i, 2] = mod1(i-1, N) # +y (随便填一个)
    end
    
    # 模拟输出数组
    forces = zeros(ComplexF64, N, 2)
    
    # 参数
    J = 1.0
    beta_term = 0.5
    Delta = rand(ComplexF64, N, 2)
    
    return N, U, f, nn_table, forces, J, beta_term, Delta
end

# --- 2. 原始写法 (Row-major access) ---
function compute_forces_orig!(forces, U, f, nn_table, N, J, beta_term, Delta)
    fill!(forces, 0.0im)
    
    @inbounds for i in 1:N
        for dir in 1:2
            j = nn_table[i, dir]
            
            ρ_1 = zero(ComplexF64)
            ρ_2 = zero(ComplexF64)
            
            # 瓶颈在这里：对 n 遍历，跨列访问 U
            @simd for n in 1:(2*N)
                ρ_1 += U[i, n] * f[n] * conj(U[j+N, n])
                ρ_2 += U[j, n] * f[n] * conj(U[i+N, n])
            end
            
            P_ij = -ρ_1 - ρ_2
            forces[i, dir] = -beta_term * (Delta[i, dir] - J * P_ij)
        end
    end
end

# --- 3. 优化写法 (Column-major access / Loop Reordering) ---
function compute_forces_opt!(forces, U, f, nn_table, N, J, beta_term, Delta)
    # 先计算 P_ij 矩阵，或者直接累加到 forces 里
    # 为了避免创建中间大矩阵 P_ij，我们先用 Delta 初始化 forces
    # Force = -beta_term * Delta + beta_term * J * P_ij
    
    # 1. 初始化部分
    @inbounds for dir in 1:2, i in 1:N
        forces[i, dir] = -beta_term * Delta[i, dir]
    end
    
    # 2. 累加 P_ij 部分 (核心优化)
    # P_ij = - sum_n ( U[i,n]*f*U[j+N,n]* + ... )
    # 我们把 n 放在最外层！
    
    coef = -beta_term * J # 注意公式里的符号，这里做个演示
    # 实际公式: Force += (-beta_term * -J) * (- rho_1 - rho_2)
    #           Force += (beta_term * J) * (- rho_1 - rho_2)
    #           Force -= (beta_term * J) * (rho_1 + rho_2)
    c = beta_term * J 
    
    @inbounds for n in 1:(2*N)
        fn = f[n]
        
        # 内层遍历空间 i，顺着内存读 U
        for i in 1:N
            # 既然 n 固定，U[..., n] 就是连续内存
            u_i_up   = U[i, n]
            u_i_down = U[i+N, n] # 也是连续的
            
            # 对两个方向处理
            for dir in 1:2
                j = nn_table[i, dir]
                
                # 获取邻居的 U 值
                # 虽然 j 是跳跃的，但都在第 n 列，都在附近的内存页
                u_j_up   = U[j, n]
                u_j_down = U[j+N, n]
                
                # 计算贡献
                term1 = u_i_up * fn * conj(u_j_down)
                term2 = u_j_up * fn * conj(u_i_down)
                
                # 累加到 forces
                # P_ij contribution: - (term1 + term2)
                # Force contribution: - beta_term * (-J * ( - term1 - term2)) 
                #Wait, F = -C * (D - J*P). P = -rho. F = -C*D + C*J*(-rho) = -C*D - C*J*rho.
                
                forces[i, dir] -= c * (term1 + term2)
            end
        end
    end
end

# --- 4. 运行测试 ---
N = 16 * 16 # 256 sites -> Matrix size 512x512
println("Benchmarking with N=$N sites (Matrix size $(2*N)x$(2*N))...")

data = setup_data(N)
N_val, U, f, nn, F1, J, B, D = data
F2 = copy(F1)

# 预热
compute_forces_orig!(F1, U, f, nn, N_val, J, B, D)
compute_forces_opt!(F2, U, f, nn, N_val, J, B, D)

# 检查正确性
diff = maximum(abs.(F1 - F2))
println("Difference between implementations: $diff")
if diff > 1e-10
    println("WARNING: Results do not match!")
end

println("\n--- Original (Row-major inner loop) ---")
@btime compute_forces_orig!($F1, $U, $f, $nn, $N_val, $J, $B, $D)

println("\n--- Optimized (Column-major outer loop) ---")
@btime compute_forces_opt!( $F2, $U, $f, $nn, $N_val, $J, $B, $D)