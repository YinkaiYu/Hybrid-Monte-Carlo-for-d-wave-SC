using DwaveHMC
using Plots
using Printf
using Statistics

# ================= 配置 =================
# 1. 稍微加大一点尺寸，更有代表性
Lx, Ly = 10, 10      
t, tp, μ = 1.0, -0.35, -1.08

# 2. 【关键】开启杂质，模拟真实工况
W, n_imp = 3.0, 0.078

β, J = 20.0, 0.8
mass = 1.0

# 3. 【关键】向左扫描，寻找崩溃点
# 我们想看 Acc 从 90% -> 60% -> 20% 的过程
Nt_list = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]

n_warmup = 100
n_measure = 100
# =========================================

# 理论周期计算
T_period = 4 * π * sqrt(mass * J / β)
L_target = T_period / 2

println("Target Trajectory Length L = $(round(L_target, digits=3))")

acc_rates = Float64[]
efficiencies = Float64[] # 存 Acc / Nt

for Nt in Nt_list
    dt = L_target / Nt
    p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, dt, mass)
    
    state = initialize_state(p)
    cache = initialize_cache(p)
    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)
    
    # 预热
    for _ in 1:n_warmup
        hmc_sweep!(cache, p, state; Nt=Nt)
    end
    
    # 测量
    accepted_count = 0
    for _ in 1:n_measure
        acc, _ = hmc_sweep!(cache, p, state; Nt=Nt)
        accepted_count += acc
    end
    
    rate = accepted_count / n_measure
    eff = rate / Nt  # 核心指标
    
    push!(acc_rates, rate)
    push!(efficiencies, eff)
    
    @printf("Nt=%2d | dt=%.3f | Acc=%.2f | Eff=%.4f\n", Nt, dt, rate, eff)
end

# ================= 画图 (双子图) =================

# 图1: 接受率
p1 = plot(Nt_list, acc_rates, 
    ylabel = "Acceptance Rate",
    label = "Acc Rate",
    marker = :circle,
    color = :blue,
    legend = :bottomright,
    ylims = (0.0, 1.1)
)
hline!(p1, [0.8], label="80% Ref", linestyle=:dash, color=:gray)

# 图2: 效率 (Acc / Nt)
# 我们找到最大值的位置，标记出来
max_eff_val, max_idx = findmax(efficiencies)
best_Nt = Nt_list[max_idx]

p2 = plot(Nt_list, efficiencies, 
    xlabel = "Number of Steps (Nt)",
    ylabel = "Efficiency (Acc / Nt)",
    label = "Efficiency",
    marker = :square,
    color = :red,
    legend = :topright
)
# 标记最高点
scatter!(p2, [best_Nt], [max_eff_val], 
    color=:green, markersize=6, label="Peak (Nt=$best_Nt)")

# 组合两张图
final_plot = plot(p1, p2, layout=(2, 1), size=(600, 800), 
                  plot_title="HMC Tuning: L = $(round(L_target, digits=2))")

savefig(final_plot, "hmc_efficiency.png")
display(final_plot)

println("\nMost efficient Nt seems to be: $best_Nt")