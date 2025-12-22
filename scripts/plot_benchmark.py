import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 配置
# ==========================================
csv_file = "benchmark_beta_scan.csv"
output_dpi = 300  # 输出图片分辨率

# 设置出版级绘图风格
plt.rcParams.update({
    # "font.family": "serif",          # 使用衬线字体 (类似 Times New Roman)
    # "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,                 # 全局字号
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.direction": "in",         # 刻度朝内
    "ytick.direction": "in",
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "figure.figsize": (8, 6),        # 图片默认大小
    "mathtext.fontset": "cm"         # 数学公式使用 Computer Modern 字体
})

# ==========================================
# 读取数据
# ==========================================
if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found.")
    exit(1)

print(f"Reading data from {csv_file}...")
df = pd.read_csv(csv_file)

# 数据列名参考:
# Beta,AccRate,Global,Err_Global,Pair,Err_Pair,RHS,Diff,Err_Diff

# 计算额外的差值列
# Diff_GP = Global - Pair
# Diff_GR = Global - RHS
# 误差传播: sqrt(err1^2 + err2^2)
diff_gp = df['Global'] - df['Pair']
diff_gr = df['Global'] - df['RHS']
err_gp = np.sqrt(df['Err_Global']**2 + df['Err_Pair']**2)

# 定义自定义 X 轴刻度
custom_xticks = [1, 10, 100, 1000]
custom_xticklabels = [r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$"]

# ==========================================
# 图 1: Order Parameters (Values)
# ==========================================
print("Plotting Order Parameters...")
fig, ax = plt.subplots()

# 1. HMC Global
ax.errorbar(df['Beta'], df['Global'], yerr=df['Err_Global'], 
            fmt='o', capsize=3, label='HMC Global', color='tab:blue', zorder=3)

# 2. HMC Pair (稍微错开一点点 x 轴，防止重叠)
ax.errorbar(df['Beta'] * 1.05, df['Pair'], yerr=df['Err_Pair'], 
            fmt='s', capsize=3, label='HMC Pair', color='tab:red', alpha=0.8, zorder=2)

# 3. BCS RHS (理论值 - 线)
ax.plot(df['Beta'], df['RHS'], linestyle='--', color='black', label='BCS RHS', zorder=1)

# 设置坐标轴
ax.set_xscale('log')
ax.set_xlabel(r'Inverse Temperature $\beta$')
ax.set_ylabel(r'Order Parameter $|\Delta|$')
ax.set_title('Order Parameter Benchmark (Clean Limit)')

# 设置刻度
ax.set_xticks(custom_xticks)
ax.set_xticklabels(custom_xticklabels)
ax.grid(True, which="both", ls="-", alpha=0.2) # 网格

ax.legend(loc='lower right', frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig('py_benchmark_values.png', dpi=output_dpi)
plt.close()

# ==========================================
# 图 2: Consistency Check (Differences)
# ==========================================
print("Plotting Differences...")
fig, ax = plt.subplots()

# 参考线 y=0
ax.axhline(0, color='gray', linestyle=':', linewidth=1.5)

# 1. Global - Pair
ax.errorbar(df['Beta'], diff_gp, yerr=err_gp, 
            fmt='D', capsize=3, label=r'Global $-$ Pair', color='tab:purple')

# 2. Global - RHS
ax.errorbar(df['Beta'], diff_gr, yerr=df['Err_Global'], 
            fmt='^', capsize=3, label=r'Global $-$ RHS', color='tab:orange')

# 3. HMC Diff (Internal)
ax.errorbar(df['Beta'], df['Diff'], yerr=df['Err_Diff'], 
            fmt='_', capsize=3, label=r'HMC $\Delta_{\mathrm{diff}}$', color='tab:green', markersize=10)

# 设置坐标轴
ax.set_xscale('log')
ax.set_xlabel(r'Inverse Temperature $\beta$')
ax.set_ylabel('Difference')
ax.set_title('Consistency Check')

# 设置刻度
ax.set_xticks(custom_xticks)
ax.set_xticklabels(custom_xticklabels)
ax.grid(True, which="both", ls="-", alpha=0.2)

ax.legend(loc='best', frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig('py_benchmark_errors.png', dpi=output_dpi)
plt.close()

# ==========================================
# 图 3: Acceptance Rate
# ==========================================
print("Plotting Acceptance Rates...")
fig, ax = plt.subplots()

ax.plot(df['Beta'], df['AccRate'], marker='*', linestyle='-', color='tab:green', markersize=8)

ax.set_xscale('log')
ax.set_xlabel(r'Inverse Temperature $\beta$')
ax.set_ylabel('Acceptance Rate')
ax.set_title('HMC Acceptance Rate')
ax.set_ylim(-0.05, 1.05)

# 设置刻度
ax.set_xticks(custom_xticks)
ax.set_xticklabels(custom_xticklabels)
ax.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig('py_benchmark_acc_rate.png', dpi=output_dpi)
plt.close()

print("Done! Generated: py_benchmark_values.png, py_benchmark_errors.png, py_benchmark_acc_rate.png")