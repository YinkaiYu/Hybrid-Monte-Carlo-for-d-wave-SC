using LinearAlgebra
using LogExpFunctions

@inline function set_hermitian_entry!(H::Matrix{ComplexF64}, row::Int, col::Int, val::ComplexF64)
    if row <= col
        H[row, col] = val
    else
        H[col, row] = conj(val)
    end
    return nothing
end

@inline function add_static_hopping!(H::Matrix{ComplexF64},
                                     N::Int,
                                     i::Int,
                                     j::Int,
                                     tij::Float64,
                                     phase::ComplexF64)
    set_hermitian_entry!(H, i, j, -tij * phase)
    set_hermitian_entry!(H, i + N, j + N, tij * conj(phase))
    return nothing
end

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
        term = state.disorder_pot[i] - state.μ_eff
        H[i, i] = term
        H[i+N, i+N] = -term
    end
    
    # 3. 动能项 (Hopping)
    mag = cache.magnetic
    @inbounds for i in 1:N
        add_static_hopping!(H, N, i, p.nn_table[i, 1], p.t, link_phase(mag, i, 1, 0))
        add_static_hopping!(H, N, i, p.nn_table[i, 2], p.t, link_phase(mag, i, 0, 1))
        add_static_hopping!(H, N, i, p.nnn_table[i, 1], p.tp, link_phase(mag, i, 1, 1))
        add_static_hopping!(H, N, i, p.nnn_table[i, 4], p.tp, link_phase(mag, i, 1, -1))
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
    # 也即 H_BdG(Top-Right) = Δ_{ij} * ( c^†_{i↑} c^†_{j↓} + c^†_{j↑} c^†_{i↓} )
    
    # 我们遍历 state.Δ，它只存储了 +x (dir=1) 和 +y (dir=2) 的键
    @inbounds for i in 1:N
        # +x direction bond
        j_x = p.nn_table[i, 1] 
        val_x = state.Δ[i, 1]
        
        # 对应的矩阵元，直接覆盖原有数值 (Overwrite)
        H[i, j_x + N] = val_x  # Δ_{ij} c^†_{i↑} c^†_{j↓}
        H[j_x, i + N] = val_x  # Δ_{ij} c^†_{j↑} c^†_{i↓}
        
        # +y direction bond
        j_y = p.nn_table[i, 2]
        val_y = state.Δ[i, 2]
        
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
    # 因为 eigen! 会破坏输入矩阵，而 cache.H_base 还需要用于下一步更新，
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

@inline function periodic_delta_1d(coord_i::Int, coord_j::Int, L::Int)::Int
    d = coord_j - coord_i
    if d > L / 2
        d -= L
    elseif d < -L / 2
        d += L
    end
    return d
end

"""
    build_twisted_H_BdG!(H, cache, p, state, Ax)

Build H_BdG(Ax, Δ) in H with a per-link x-direction Peierls phase on hopping
terms. Pairing terms follow update_H_BdG! exactly. This function does not
modify cache.H_base, cache.E_n, or cache.U.
"""
function build_twisted_H_BdG!(H::Matrix{ComplexF64},
                              cache::ComputeCache,
                              p::ModelParameters,
                              state::SimulationState,
                              Ax::Float64)
    p.n_flux_sc == 0 ||
        error("build_twisted_H_BdG! is not supported for finite magnetic field (n_flux_sc=$(p.n_flux_sc))")
    N = p.N
    fill!(H, 0.0 + 0.0im)

    @inbounds for i in 1:N
        term = state.disorder_pot[i] - state.μ_eff
        H[i, i] = term
        H[i + N, i + N] = -term
    end

    @inbounds for i in 1:N
        xi = cache.x_idx[i]

        for dir in 1:4
            j = p.nn_table[i, dir]
            if j > i
                dx = periodic_delta_1d(xi, cache.x_idx[j], p.Lx)
                phase = cis(Ax * dx)
                H[i, j] = -p.t * phase
                H[i + N, j + N] = p.t * conj(phase)
            end
        end

        for dir in 1:4
            j = p.nnn_table[i, dir]
            if j > i
                dx = periodic_delta_1d(xi, cache.x_idx[j], p.Lx)
                phase = cis(Ax * dx)
                H[i, j] = -p.tp * phase
                H[i + N, j + N] = p.tp * conj(phase)
            end
        end
    end

    @inbounds for i in 1:N
        j_x = p.nn_table[i, 1]
        val_x = state.Δ[i, 1]
        H[i, j_x + N] = val_x
        H[j_x, i + N] = val_x

        j_y = p.nn_table[i, 2]
        val_y = state.Δ[i, 2]
        H[i, j_y + N] = val_y
        H[j_y, i + N] = val_y
    end

    return nothing
end

@inline function probe_factor(cache::ComputeCache, i::Int, dα::Int,
                              λ::Float64, qx::Float64, qy::Float64)
    dα == 0 && return 1.0 + 0.0im
    x = cache.x_idx[i] - 1
    y = cache.y_idx[i] - 1
    θ = qx * x + qy * y
    η = (qx == 0.0 && qy == 0.0) ? 1.0 : sqrt(2.0) * cos(θ)
    return cis(λ * dα * η)
end

function build_probe_H_BdG!(H::Matrix{ComplexF64},
                            cache::ComputeCache,
                            p::ModelParameters,
                            state::SimulationState;
                            direction::Symbol=:x,
                            λ::Float64,
                            qx::Float64=0.0,
                            qy::Float64=0.0)
    N = p.N
    fill!(H, 0.0 + 0.0im)
    @inbounds for i in 1:N
        term = state.disorder_pot[i] - state.μ_eff
        H[i, i] = term
        H[i + N, i + N] = -term
    end
    mag = cache.magnetic
    @inbounds for i in 1:N
        ph = link_phase(mag, i, 1, 0) *
             probe_factor(cache, i, direction_component(direction, 1, 0), λ, qx, qy)
        add_static_hopping!(H, N, i, p.nn_table[i, 1], p.t, ph)
        ph = link_phase(mag, i, 0, 1) *
             probe_factor(cache, i, direction_component(direction, 0, 1), λ, qx, qy)
        add_static_hopping!(H, N, i, p.nn_table[i, 2], p.t, ph)
        ph = link_phase(mag, i, 1, 1) *
             probe_factor(cache, i, direction_component(direction, 1, 1), λ, qx, qy)
        add_static_hopping!(H, N, i, p.nnn_table[i, 1], p.tp, ph)
        ph = link_phase(mag, i, 1, -1) *
             probe_factor(cache, i, direction_component(direction, 1, -1), λ, qx, qy)
        add_static_hopping!(H, N, i, p.nnn_table[i, 4], p.tp, ph)
    end
    @inbounds for i in 1:N
        j_x = p.nn_table[i, 1]
        val_x = state.Δ[i, 1]
        H[i, j_x + N] = val_x
        H[j_x, i + N] = val_x
        j_y = p.nn_table[i, 2]
        val_y = state.Δ[i, 2]
        H[i, j_y + N] = val_y
        H[j_y, i + N] = val_y
    end
    return nothing
end

@inline function set_hermitian_pair!(H::Matrix{ComplexF64},
                                    row::Int,
                                    col::Int,
                                    val::ComplexF64)
    if row <= col
        H[row, col] = val
    else
        H[col, row] = conj(val)
    end
    return nothing
end

@inline function add_oriented_hopping!(H::Matrix{ComplexF64},
                                      N::Int,
                                      i::Int,
                                      j::Int,
                                      tij::Float64,
                                      phase::ComplexF64)
    set_hermitian_pair!(H, i, j, -tij * phase)
    set_hermitian_pair!(H, i + N, j + N, tij * conj(phase))
    return nothing
end

"""
    build_twisted_H_BdG_qy!(H, cache, p, state, Ax, qy, phase_shift)

Build H_BdG with a transverse finite-q vector potential on x-directed bonds:
`A_x(y) = sqrt(2) * Ax * cos(qy * (y - 1) + phase_shift)`.
Using `phase_shift=0` and `phase_shift=-π/2` gives the cosine and sine
partners used to benchmark the current-current response at qy.
"""
function build_twisted_H_BdG_qy!(H::Matrix{ComplexF64},
                                 cache::ComputeCache,
                                 p::ModelParameters,
                                 state::SimulationState,
                                 Ax::Float64,
                                 qy::Float64,
                                 phase_shift::Float64)
    p.n_flux_sc == 0 ||
        error("build_twisted_H_BdG_qy! is not supported for finite magnetic field (n_flux_sc=$(p.n_flux_sc))")
    N = p.N
    fill!(H, 0.0 + 0.0im)

    @inbounds for i in 1:N
        term = state.disorder_pot[i] - state.μ_eff
        H[i, i] = term
        H[i + N, i + N] = -term
    end

    @inbounds for i in 1:N
        # y-directed nearest-neighbor hopping is not coupled to A_x.
        add_oriented_hopping!(H, N, i, p.nn_table[i, 2], p.t, 1.0 + 0.0im)

        y = cache.y_idx[i] - 1
        local_Ax = sqrt(2.0) * Ax * cos(qy * y + phase_shift)
        phase = cis(local_Ax)

        # Match build_current_operator!: +x, +x+y, and +x-y oriented bonds.
        add_oriented_hopping!(H, N, i, p.nn_table[i, 1], p.t, phase)
        add_oriented_hopping!(H, N, i, p.nnn_table[i, 1], p.tp, phase)
        add_oriented_hopping!(H, N, i, p.nnn_table[i, 4], p.tp, phase)
    end

    @inbounds for i in 1:N
        j_x = p.nn_table[i, 1]
        val_x = state.Δ[i, 1]
        H[i, j_x + N] = val_x
        H[j_x, i + N] = val_x

        j_y = p.nn_table[i, 2]
        val_y = state.Δ[i, 2]
        H[i, j_y + N] = val_y
        H[j_y, i + N] = val_y
    end

    return nothing
end
