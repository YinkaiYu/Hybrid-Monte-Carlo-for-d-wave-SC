using LinearAlgebra
using LogExpFunctions
using Random
using Printf

"""
    compute_total_energy(cache::ComputeCache, p::ModelParameters, state::SimulationState)

计算当前的 HMC 总哈密顿量 H_HMC。
H_HMC = E_kinetic + E_boson + E_fermion
"""
function compute_total_energy(cache::ComputeCache, p::ModelParameters, state::SimulationState)
    # 1. 费米子部分 (最耗时，通常在调用此函数前已经计算并存入 cache.E_n 了)
    # 但为了保险，这里我们假设外部负责维护 cache.E_n 的时效性，
    # 或者我们在这里显式调用 diagonalize_H_BdG!。
    # 考虑到 HMC 流程，我们在 Leapfrog 结束时会计算一次。
    # 这里我们只负责把各项加起来。
    
    # E_fermion (从缓存读取)
    # - sum_{E>0} log(2cosh(βE/2))
    # 稳定公式：log(2cosh(βE/2)) = βE/2 + log1p(exp(-βE))
    E_fermion = 0.0
    @inbounds for E in cache.E_n
        if E > 0
            x = p.β * E
            E_fermion -= (0.5*x + log1pexp(-x))
        end
    end

    # 2. 玻色子势能部分
    # E_boson = (β / 2J) * sum(|Δ|^2)
    # 使用 sum(f, itr) 极其高效，无内存分配
    coef_boson = p.β / (2 * p.J)
    E_boson = coef_boson * sum(abs2, state.Δ)

    # 3. 动能部分
    # E_kinetic = (1 / 2m) * sum(|π|^2)
    coef_kinetic = 1.0 / (2 * p.mass)
    E_kinetic = coef_kinetic * sum(abs2, state.π)

    return E_kinetic + E_boson + E_fermion
end

"""
    refresh_momentum!(state::SimulationState, p::ModelParameters)

重置共轭动量 π。
分布 P(π) ~ exp(- |π|^2 / 2m)。
# 这意味着 Re(π) 和 Im(π) 均服从方差为 m 的高斯分布：
# Re(π), Im(π) ~ Normal(0, m)  ( 标准差 σ = sqrt(m) )
"""
function refresh_momentum!(state::SimulationState, p::ModelParameters)
    # randn!(A) 会将 A 填充为标准复正态分布 (Re, Im 方差各为 0.5)
    randn!(state.π)
    
    # 我们需要的方差是 m，标准差是 sqrt(m)
    # 由于 randn 产生的实部虚部标准差是 sqrt(0.5)，
    # 所以我们需要乘以 sqrt(2 * m) 才能得到标准差 sqrt(m)
    scale = sqrt(2 * p.mass)
    state.π .*= scale
    return nothing
end

"""
    hmc_sweep!(cache::ComputeCache, p::ModelParameters, state::SimulationState; Nt::Int=10)

执行一步完整的 HMC 采样。
1. 随机生成动量
2. Leapfrog 演化 (时间步长 p.dt, 步数 Nt)
3. Metropolis 判据
"""
function hmc_sweep!(cache::ComputeCache, p::ModelParameters, state::SimulationState; Nt::Int=20)
    # --- 1. 初始化 ---
    # 注意确保当前 H_BdG 和 cache.E_n 是与当前 state.Δ 对应的
    # 通常上一步结束时已经是对应的。
    
    # 刷新动量 π_ij 复数动量
    refresh_momentum!(state, p)
    
    # 计算初始能量 H_HMC(Δ_ij,π_ij)
    H_old = compute_total_energy(cache, p, state)
    
    # 备份当前构型 Δ (以防被拒绝)
    # 注意：H_base 不需要备份，因为我们可以通过 state.Δ 和 static_H 随时重建它，
    copyto!(cache.Δ_backup, state.Δ)
    copyto!(cache.E_n_backup, cache.E_n)
    copyto!(cache.U_backup, cache.U)
    
    # --- 2. Leapfrog Integration ---
    dt = p.dt
    
    # (1) Half-step Momentum update
    # π = π + (dt/2) * F
    compute_forces!(cache, p, state)
    state.π .+= (0.5 * dt) .* cache.forces

    # 预计算系数 dt / 2m
    coef_field = dt / (2 * p.mass)
    
    # (2) Full-step loops
    for step in 1:Nt
        # A. Full-step Field update
        # Δ = Δ + dt * (π / 2m)  <-- 注意这里是 2m
        state.Δ .+= coef_field .* state.π
        
        # B. 更新哈密顿量和力
        # 这是最耗时的一步：修改 H -> 对角化 -> 算力
        update_H_BdG!(cache, p, state)
        diagonalize_H_BdG!(cache, p) # 更新能量谱
        compute_forces!(cache, p, state)  # 计算新位置的力
        
        # C. Full-step Momentum update (最后一步除外)
        # π = π + dt * F
        if step < Nt
            state.π .+= dt .* cache.forces
        end
    end
    
    # (3) Final Half-step Momentum update
    # π = π + (dt/2) * F
    state.π .+= (0.5 * dt) .* cache.forces
    
    # --- 3. Metropolis Acceptance ---
    # 计算新能量
    H_new = compute_total_energy(cache, p, state)
    
    ΔH = H_new - H_old
    
    accepted = false
    # Metropolis logic: accept if ΔH < 0 or rand < exp(-ΔH)
    if ΔH < 0 || rand() < exp(-ΔH)
        accepted = true
    else
        # Rejected: 恢复备份
        copyto!(state.Δ, cache.Δ_backup)
        copyto!(cache.E_n, cache.E_n_backup)
        copyto!(cache.U, cache.U_backup)
        # 虽然下一条链开始时会调用 update_H_BdG!，它会根据 state.Δ 重写 H_base。
        # 但为了保持数据一致性（cache 里的 H 应该对应 state.Δ），
        # 这里花极小的代价 (O(N)) 把 H_base 修复一下是好的编程习惯。
        update_H_BdG!(cache, p, state)

        # 注意：不需要恢复 forces，因为下一次使用 forces 前一定会先调用 compute_forces!
    end
    
    return accepted, ΔH
end
