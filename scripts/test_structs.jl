using DwaveHMC
using Random

# 设置参数
Lx, Ly = 4, 4
t = 1.0
tp = -0.35 # 笔记中的数值
μ = -0.5   # 假设值
W = 1.0    # 笔记中的数值
n_imp = 0.1
β = 10.0
J = 1.0
dt = 0.05
mass = 1.0

println("Initializing Parameters...")
# 这里可以直接传 Int 或 Float，构造函数会自动处理
p = ModelParameters(Lx, Ly, t, tp, μ, W, n_imp, β, J, dt, mass)

println("System: $(p.Lx)x$(p.Ly), t'=$(p.tp), μ=$(p.μ), β=$(p.β)")

println("\nInitializing State...")
s = initialize_state(p)
# 注意：访问字段时用 Unicode
println("Max Disorder: $(maximum(s.disorder_pot))")
println("Δ field size: $(size(s.Δ))")
println("π field size: $(size(s.π))")

println("\nInitializing Cache...")
c = initialize_cache(p)
println("H_BdG wrapper: $(typeof(c.H_herm))")

println("Test passed!")