using DwaveHMC
using LinearAlgebra
using Printf

# 设置一个小系统
Lx, Ly = 4, 4
β = 20.0 # 低温，接近平均场
J = 1.0
p = ModelParameters(Lx, Ly, 1.0, -0.35, -0.5, 0.0, 0.0, β, J, 0.05, 1.0) # 无杂质 W=0

s = initialize_state(p)
c = initialize_cache(p)

println("1. Initial Random Delta")
# 初始化哈密顿量
init_static_H!(c, p, s)
update_H_BdG!(c, p, s)

# 对角化
diagonalize_H_BdG!(c, p)

# 计算力
compute_forces!(c, p, s)
f_norm = norm(c.forces)
println("Force norm (Random Config): $f_norm")

println("\n2. Performing naive Mean-Field Iteration (to find self-consistent solution)")
# 我们用迭代法找一个近似的自洽解： Delta_new = J * P_ij
# 这不是 HMC，只是为了测试 Force 是否在自洽解处为零

for iter in 1:50
    # 1. 对角化当前 s.Δ 对应的 H
    update_H_BdG!(c, p, s)
    diagonalize_H_BdG!(c, p) # 这一步产出 E_n 和 U
    
    # 2. 计算 Force
    compute_forces!(c, p, s)
    current_f_norm = norm(c.forces)
    
    # 3. 根据 Force 反推 Pairing P_ij 并更新 Delta
    # F = -β/2J * (Delta - J*P) 
    # => J*P = Delta + (2J/β)*F
    # 下一步设 Delta = J*P
    
    # 更新 state.Δ
    # 简单的混合迭代: D_new = alpha * D_old + (1-alpha) * D_target
    alpha = 0.0
    factor = (2 * p.J) / p.β
    JP_target = s.Δ .+ factor .* c.forces
    s.Δ .= alpha .* s.Δ .+ (1 - alpha) .* JP_target
    
    @printf("Iter %2d: Force Norm = %.6e\n", iter, current_f_norm)
end

println("\nTest Result: As iterations proceed, Force Norm should decrease towards 0.")