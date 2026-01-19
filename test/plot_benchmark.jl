using Plots
using DelimitedFiles
using Printf
using LaTeXStrings

# ==========================================
# 配置
# ==========================================
csv_file = "benchmark_beta_scan.csv"

# 自定义 X 轴刻度 (10^0 到 10^3)
my_xticks = [1, 10, 100, 1000]

# 全局画图样式设置
default(
    grid = true, 
    minorgrid = true,          # 显示次级网格 (对数坐标很需要)
    frame = :box,              # 方框围起来
    lw = 2.0,                  # 线宽
    markersize = 5,            # 散点大小
    xscale = :log10,           # X轴对数
    xticks = my_xticks,        # 强制显示 1, 10, 100, 1000
    guidefontsize = 12,        # 轴标签字体大小
    tickfontsize = 10,         # 刻度字体大小
    legendfontsize = 9,        # 图例字体大小
    margin = 5Plots.mm,         # 边距
    dpi = 300                   # 分辨率
)

# ==========================================
# 读取数据
# ==========================================
if !isfile(csv_file)
    println("Error: File '$csv_file' not found in current directory.")
    exit(1)
end

println("Reading data from $csv_file ...")

# 读取 CSV，跳过第一行表头
# data 是一个 Matrix{Any}，我们需要转换类型
raw_data, header = readdlm(csv_file, ',', header=true)

# 确保转换为 Float64
data = Float64.(raw_data)

# 根据之前的 header 顺序提取列:
# Beta, AccRate, Global, Err_Global, Pair, Err_Pair, RHS, Diff, Err_Diff
betas      = data[:, 1]
acc_rates  = data[:, 2]
avg_global = data[:, 3]
err_global = data[:, 4]
avg_pair   = data[:, 5]
err_pair   = data[:, 6]
avg_rhs    = data[:, 7]
avg_diff   = data[:, 8]
err_diff   = data[:, 9]

# ==========================================
# 画图 1: Order Parameters (Values)
# ==========================================
println("Plotting Order Parameters...")

p1 = plot(xlabel=L"$\beta$", ylabel=L"$|\Delta|$")

# 1. HMC Global
plot!(p1, betas, avg_global, yerror=err_global, 
      label=L"HMC: $|\Delta|$", marker=:circle, color=:blue)

# 2. HMC Pair (稍微错开一点点或者用不同的形状，防止重叠看不清)
plot!(p1, betas, avg_pair, yerror=err_pair, 
      label=L"HMC: $|\langle\hat{\Delta}\rangle|$", marker=:rect, color=:red, alpha=0.7)

# 3. BCS RHS (理论值)
plot!(p1, betas, avg_rhs, 
      label=L"BCS: $|\langle\hat{\Delta}\rangle|$", ls=:dash, color=:black, lw=2.5)

# 保存
savefig(p1, "plot_benchmark_values.png")


# ==========================================
# 画图 2: Consistency Check (Differences)
# ==========================================
println("Plotting Differences...")

# 计算差值
diff_GP = avg_global .- avg_pair
diff_GR = avg_global .- avg_rhs

# 误差传播: err = sqrt(err1^2 + err2^2)
# RHS 是解析解，假设无误差，所以 diff_GR 的误差就是 err_global
err_GP = sqrt.(err_global.^2 .+ err_pair.^2)

p2 = plot(xlabel="Inverse Temperature (β)", ylabel="Difference", 
          title="Consistency Check")

# 1. Global - Pair
plot!(p2, betas, diff_GP, yerror=err_GP, 
      label="Global - Pair", marker=:diamond, color=:purple)

# 2. Global - RHS
plot!(p2, betas, diff_GR, yerror=err_global, 
      label="Global - RHS", marker=:utriangle, color=:orange)

# 3. HMC Diff (Internal consistency)
plot!(p2, betas, avg_diff, yerror=err_diff, 
      label="HMC Δ_diff", marker=:hline, color=:green)

# 添加一条 y=0 的参考线
hline!(p2, [0.0], label="", color=:grey, ls=:dot)

# 保存
savefig(p2, "plot_benchmark_errors.png")


# ==========================================
# 画图 3 (额外): Acceptance Rate
# ==========================================
println("Plotting Acceptance Rates...")

p3 = plot(xlabel="Inverse Temperature (β)", ylabel="Acceptance Rate", 
          title="HMC Acceptance Rate", ylims=(0, 1.1))

plot!(p3, betas, acc_rates, 
      label="Acc Rate", marker=:star, color=:darkgreen, lw=1.5)

# 保存
savefig(p3, "plot_benchmark_acc_rate.png")

println("Done! Generated: plot_benchmark_values.png, plot_benchmark_errors.png, plot_benchmark_acc_rate.png")