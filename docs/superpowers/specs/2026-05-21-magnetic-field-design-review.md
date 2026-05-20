GPT 5.5 Pro:

我基于这次 commit 的设计文档、当前 `t-V-model` 分支源码、以及 arXiv:2301.04175 的方法做静态审查。结论是：

**这份设计文档的大方向是对的，尤其是选择 magnetic PBC 而不是 OBC、把 Peierls phase 集中到一个 phase layer、禁止有限磁场下普通 FFT 动量谱，这些判断都正确。**

但我不建议按原文直接实现。需要先改几处关键点，否则最容易出错的是：Kubo current operator 的符号和 Nambu block、有限磁场下 `Superfluid_Stiffness` 的物理解释、diagonal (t') bond 的边界相位、以及 gauge-dependent pairing/spectra 输出。

下面按重要性排序。

---

## 1. 总体架构：方向正确，建议保留

设计文档的目标是：在现有 torus 几何上加入 opt-in 的有限轨道磁场路径，用 Landau gauge、Peierls phase、magnetic PBC，并优先保证 `Superfluid_Stiffness` 和 `DC_Conductivity` 的可靠性；它明确不打算第一版实现 magnetic unit cell / magnetic Bloch theorem / open boundary / Hall conductivity。这个取舍是合理的。([GitHub][1])

这和 Johnsen 那篇文章的核心做法一致：磁场通过 hopping 的 Peierls phase 进入，采用 Landau gauge (A=B(0,x,0))，并使用包含偶数个 superconducting flux quanta 的 magnetic unit cell 加 PBC 来避免边缘效应。

对你现在的程序来说，这也比 OBC 更自然。当前 `ModelParameters` 已经用 `mod1` 构造全周期的最近邻和次近邻表，`init_static_H!` 也是基于 PBC 邻居表写的。([GitHub][2])

所以第一版做：

[
x,y\text{ 都 PBC},\qquad n_{\rm flux}^{SC}=0,\pm2,\pm4,\ldots
]

这是正确方向。不要先做全 OBC。

---

## 2. 物理上最需要补的一句话：磁场下不要再叫它 BKT (T_c)

设计文档主要讲怎么实现有限磁场的 stiffness 和 transport，但没有明确说明：**有限磁场下算出来的 `Superfluid_Stiffness` 不能再直接用零场 BKT 判据定义 (T_c)。**

Johnsen 论文自己也不是在算有限温度 (T_c(B))。它明确说理论框架是 zero-temperature limit，不包含 thermal fluctuations；它用 (D_s) 判断是否还有全局超导相干，而不是用 BKT universal jump 定 (T_c)。

所以你的设计文档里应该加一个小节，明确写：

[
D_s(B,T,L)
]

在有限磁场下是一个 **field-suppressed stiffness diagnostic**，可以用来定义

[
T_{\rm stiff}(B,L):\quad D_s(B,T,L)\approx 0
]

或者

[
B_c(T,L):\quad D_s(B,T,L)\approx 0.
]

但不要写成

[
D_s(B,T_c)=2T_c/\pi.
]

零场可以用 BKT 判据；有限垂直磁场下，已经有 field-induced vortices，BKT 的 vortex–antivortex 解绑定图像不再原样适用。

这点和你的研究目标直接相关。你真正想确认的是：

[
D_s(B,T)\approx 0
]

时，regular DC resistivity 在更低温是否仍近似线性，而不是一定要定义一个严格的 (T_c(B))。

---

## 3. 最大实现风险：不能在当前 `j > i` hopping loop 上直接加 phase

当前 `init_static_H!` 的动能部分是这样做的：遍历四个 NN 方向、四个 NNN 方向，然后用 `if j > i` 只填上三角。零场下这没问题，因为 hopping 是实数，方向不重要。([GitHub][3])

但有 Peierls phase 后，方向非常重要。比如：

[
U_{i\to j}\neq U_{j\to i},\qquad U_{j\to i}=U_{i\to j}^*.
]

如果还用 `dir=1:4` 加 `if j > i`，你很容易在某些边界 bond 上不小心使用了反方向的 phase。尤其是跨 (x=L_x-1\to0) 的 magnetic boundary patch，这里最容易错。

设计文档里说第一版 common directed bonds 是：

[
+x,\quad +y,\quad +x+y,\quad +x-y.
]

这是对的。实现时应该彻底改成只遍历这四类唯一有向 bond，而不是继续用 `dir=1:4` 加 `j > i`。

我建议写成类似：

```julia
for i in 1:N
    add_hop!(H, i, nn_table[i, 1], p.t,  link_phase(mag, i, +1,  0))  # +x
    add_hop!(H, i, nn_table[i, 2], p.t,  link_phase(mag, i,  0, +1))  # +y
    add_hop!(H, i, nnn_table[i, 1], p.tp, link_phase(mag, i, +1, +1)) # +x+y
    add_hop!(H, i, nnn_table[i, 4], p.tp, link_phase(mag, i, +1, -1)) # +x-y
end
```

其中 `add_hop!` 负责填 BdG particle block 和 hole block 的 Hermitian 上三角。这样每条物理 bond 只出现一次，方向也固定。

---

## 4. Landau gauge 约定基本正确，但 diagonal boundary phase 需要更精确定义

设计文档采用：

[
U_y(x,y)=\exp(i,\alpha x),
]

[
U_x(x,y)=1\quad\text{inside},
]

[
U_x(L_x-1,y)=\exp(-i,\alpha L_x y),
]

其中

[
\alpha=\pi\frac{n_{\rm flux}^{SC}}{L_xL_y}.
]

这样每个 plaquette 的 phase product 是

[
e^{i\alpha},
]

总 flux 是

[
\alpha L_xL_y=\pi n_{\rm flux}^{SC}.
]

因此 magnetic PBC 要求 (n_{\rm flux}^{SC}) 是偶数。这个逻辑是对的，也和 superconducting flux quantum / electron Peierls phase 的关系一致。Johnsen 的 Peierls phase 公式使用 (\Phi_0^{SC}=hc/2e)，所以一个 superconducting flux quantum 对单电子 Peierls phase 是 (\pi)，不是 (2\pi)。

但是，文档里对 diagonal (t') bond 的 boundary crossing 写得还不够具体。它说 internal diagonal link 用 straight-line integral：

[
U_{dx,dy}(x,y)=
\exp\left[i,\alpha,dy\left(x+\frac{dx}{2}\right)\right],
]

然后 boundary-crossing diagonal hops 使用同一个 magnetic boundary patch。这个说法方向是对的，但实现时仍有歧义。

例如 (+x+y) bond 从

[
(L_x-1,y)\to(0,y+1)
]

穿过 (x) seam。这个 diagonal bond 的 seam patch 到底用 (y)、(y+1)、还是 (y+1/2)？不同选择差一个 phase。这个差别不是小 bug；它会影响 (t') hopping 的 gauge convention，也会影响 current operator 和 diamagnetic term。

我的建议是：不要只写一句“same patch”。应该在 `MagneticField.jl` 里实现一个统一函数：

```julia
link_phase(mag, i, dx, dy)
```

并且在文档里精确定义它对所有边界穿越的算法。然后加测试：

1. 所有 elementary plaquette 的 Wilson loop 都等于 `cis(plaquette_phase)`，包括四个角落。
2. `link_phase(i, dx, dy) == conj(link_phase(j, -dx, -dy))`，其中 (j=i+\delta)。
3. (+B) 和 (-B) 的所有 link phases 互为复共轭。
4. 对 (+x+y)、(+x-y) 跨边界 bond 单独测试解析值。

文档现在已有 plaquette 和 (\pm B) 测试计划，但 diagonal boundary crossing 的测试还应该更具体。([GitHub][1])

---

## 5. Kubo 部分是最危险的：必须从“BdG 对 probe 的导数”定义 current operator

当前代码的 current operator 是零场版本。它在 real space 中构造 (J_x(q))，包括 (+x)、(+x+y)、(+x-y) bond，然后用

[
J_{mn}=U^\dagger J U
]

计算超流刚度和电导。当前实现的 (J_x) 没有 Peierls phase，并且 Nambu block 写成 `blockdiag(Jx_part, Jx_part)`。([GitHub][4])

有限磁场下不能只把 `Jx_part` 里的 hopping 乘上 (U_{ij}) 就完事。最安全的定义是：

[
J_\eta
======

\left.
\frac{\partial H_{\rm BdG}[A^B+\lambda\eta]}{\partial\lambda}
\right|_{\lambda=0},
]

[
K_\eta
======

\left.
\frac{\partial^2 H_{\rm BdG}[A^B+\lambda\eta]}{\partial\lambda^2}
\right|_{\lambda=0}.
]

也就是说，**current operator 和 diamagnetic term 必须是同一个有限磁场 BdG Hamiltonian 对 probe vector potential 的一阶、二阶导数。**

Johnsen 的 (K_x) 和 (J_x) 里都显式包含同一个 Peierls phase (e^{i\phi_{i+\delta,i}})。 他的附录里，Kubo stiffness 的 (K_x)、(\Lambda_{xx}) 和矩阵元也都保留了 (e^{\pm i\phi})，而且求和的 bond 包括 (\delta={x,y,x\pm y,2x,2y}) 中所有有 (\delta_x\neq0) 的项。

因此我建议设计文档加一条硬要求：

**任何 analytic current operator 都必须通过 finite-difference derivative test。**

例如对一个小系统 (L=4) 或 (L=6)，随机 complex (\Delta)，随机 disorder，有限 (n_{\rm flux}^{SC}=2)，构造：

```julia
Hplus  = H[A_B + eps * eta]
Hminus = H[A_B - eps * eta]
H0     = H[A_B]

J_fd = (Hplus - Hminus) / (2eps)
K_fd = (Hplus + Hminus - 2H0) / eps^2
```

然后检查 analytic `Jx_sparse_qy` 和 analytic diamagnetic term 与 `J_fd`, `K_fd` 一致。没有这个测试，我不会信 finite-field Kubo stiffness。

这条比“smoke test stiffness finite and real”更重要。设计文档现在的 Kubo 测试只要求 current entries match `link_phase`、diamagnetic term uses same phases、结果 finite/real；这还不够。([GitHub][1])

---

## 6. `m == n` 项：不要无验证地沿用旧有限-(q_y) 规则

设计文档说，`Superfluid_Stiffness` 继续使用 finite-(q_y) transverse estimator：

[
\rho_s=\langle -K_x\rangle-\Lambda_{xx}(q_x=0,q_y=2\pi/L_y,\omega=0),
]

并且 strict `m == n` terms 仍然跳过，只对 near-degenerate `m != n` 用 derivative limit。([GitHub][1])

这和当前零场实现一致：当前 `measure_transport_only` 里顺磁项对 `m == n` 直接 `continue`，而 near-degenerate (m\neq n) 用 (\beta f(1-f)) 极限。([GitHub][4])

但是磁场、无序、涡旋背景都会更强地破坏普通平移对称性。此时 (J_\eta) 的对角矩阵元是否应当贡献，不能只靠“旧代码就是这样”决定。至少应该做两个版本的 benchmark：

1. 旧 convention：strict `m==n` 跳过。
2. full free-energy curvature convention：包含 diagonal occupation-response term。

然后用 finite-difference free-energy curvature 对比。若第一版坚持沿用旧 convention，也可以，但文档里要写清楚：这是为了和当前 zero-field transverse Kubo estimator 保持连续，而不是严格等同于任意 probe 的 full thermodynamic curvature。

这对你后面解释 (T_{\rm stiff}(B)) 很重要。

---

## 7. Pairing block 不乘 Peierls phase：这个判断是对的，但输出命名要更谨慎

设计文档说：BdG pairing block 继续使用 sampled auxiliary field `state.Δ[i,dir]`，不在 Hamiltonian 的 pairing block 里额外乘 Peierls phase；gauge-covariant pairing 只在 observable/output 层构造。这个判断我同意。([GitHub][1])

Johnsen 也是先在 BdG 里定义 bond pairing，然后在构造局域 (d)-wave pairing 时乘上 Peierls phase：

[
\Delta_i^{\pm x(y)}
===================

\Delta_{i,i\pm x(y)}
\exp(i\phi_{i,i\pm x(y)}).
]

这说明 gauge-covariant (d)-wave pairing 是 observable convention，不是把 Peierls phase 硬塞进 pairing block。

但是设计文档里说现有 `Delta_Loc`, `Delta_Glob`, `Delta_Pair`, `Delta_LocalPair`, `d_local` 在有限场下仍保留旧 bare 定义。这个地方要非常小心。当前代码里的 pairing observables 是用 bare (d_x-d_y) 或 fermionic (P_x-P_y) 做的，有限磁场下这些量是 gauge-dependent 的。([GitHub][4])

我建议：

1. 有限场下，旧字段保留可以，但 metadata 必须写：

   ```text
   pairing_scalar_convention = "bare gauge-fixed Landau-gauge diagnostic"
   pairing_scalar_gauge_invariant = false
   ```
2. 新增字段不要只叫 `pair_bond_gauge`，最好叫：

   ```text
   pair_bond_landau_gauge_covariant
   delta_bond_landau_gauge_covariant
   ```

   或者至少在 metadata 里注明 convention。
3. 第一版最好直接输出几个便宜的 gauge-covariant scalar：
   [
   \langle |\Delta_i^d| \rangle,\quad
   |\langle \Delta_i^d\rangle|,\quad
   \langle |P_i^d| \rangle,\quad
   |\langle P_i^d\rangle|.
   ]
   这样不必每次都写 (N\times2) complex bond array。
4. `write_gauge_pair_bonds` 不要只有 Bool，建议加频率：

   ```julia
   write_gauge_pair_bonds_freq::Int = 0
   ```

   `0` 表示不写，`10` 表示每 10 次 measurement 写一次。否则 (L=40), `n_measure=1000`, 多个 eta / configs 时 JLD2 文件会明显变大。

---

## 8. 普通 FFT 谱默认禁用：正确，但 `dos_M` 也要禁用或改名

设计文档说有限磁场下普通 FFT momentum spectra 是 gauge-dependent，因此默认不写 `A_k0`, `A_MX_path`, `A_XG_path`；只有用户显式 `allow_gauge_dependent_spectra=true` 时才写诊断字段。这个判断完全正确。([GitHub][1])

但当前 `measure_untwisted_spectra` 不只写 `A_k`，还写 `dos_M`、`A_MX_path`、`A_XG_path` 等普通 momentum-projected 量。当前实现里 `dos_M` 是通过 ((\pi,0))、((0,\pi)) 的普通 Fourier 权重算的。([GitHub][4])

有限磁场下，`dos_M` 也不是 gauge-invariant 的物理量。设计文档应该明确：

* DOS：可以写。
* LDOS：可以写。
* ordinary (A(k,\omega))：默认不写。
* `dos_M` / M-point projected DOS：也默认不写，或者写成 diagnostic 名称。
* MX / XG path spectra：默认不写。

否则后处理脚本很容易继续把 `dos_M` 当作物理量画图。

软件上，最好把 spectrum result 类型拆开：

```julia
GaugeInvariantSpectraResult
MomentumSpectraDiagnosticResult
```

不要在有限场下为了兼容旧 `SpectrumResult` 而写一堆 fake zeros。fake zeros 会在后处理里制造假物理结论。

---

## 9. DC conductivity 可以算，但不要直接叫 “resistivity = 1/sigma_xx”

设计文档保留 `DC_Conductivity` 为 regular longitudinal Kubo-Greenwood (\sigma_{xx})，暂时不实现 Hall conductivity。这个第一版可以接受。([GitHub][1])

但有限磁场下严格说有电导张量：

[
\hat\sigma=
\begin{pmatrix}
\sigma_{xx} & \sigma_{xy}\
-\sigma_{xy} & \sigma_{yy}
\end{pmatrix}.
]

实验纵向电阻率是 (\hat\sigma^{-1}) 的 (\rho_{xx})，不是简单 (1/\sigma_{xx})。所以第一版输出应该命名为：

```text
sigma_xx_regular
inverse_sigma_xx_proxy
```

不要在论文图里直接写：

[
\rho_{xx}=1/\sigma_{xx}
]

除非你同时证明 (\sigma_{xy}) 可以忽略。

设计文档里已经把 Hall conductivity 放在 out of scope，这没问题；但 metadata 和 plotting 脚本要避免过度解释。

---

## 10. `MagneticField.jl` phase layer 是好设计，但要注意 Julia 类型稳定性

文档提出新增 `src/MagneticField.jl`，集中提供：

```julia
validate_magnetic_field(p)
build_magnetic_cache(p)
link_phase(mag, i, dx, dy)
plaquette_phase(mag, x, y)
magnetic_metadata(mag)
```

这个设计很好。它能避免 Hamiltonian、current、diamagnetic、pairing observable 里各自手写 Landau gauge 公式。([GitHub][1])

但实现时要注意 Julia 的类型稳定性。

如果 `ComputeCache` 里写：

```julia
magnetic::Union{NoFieldCache, LandauGaugeCache}
```

然后在热循环里反复调用 `link_phase(cache.magnetic, ...)`，可能会有动态分发或 union-splitting 不稳定。更稳妥的方式是：

```julia
abstract type AbstractMagneticCache end

struct NoFieldCache <: AbstractMagneticCache
end

struct LandauGaugeCache <: AbstractMagneticCache
    Ux::Vector{ComplexF64}
    Uy::Vector{ComplexF64}
    Uxpy::Vector{ComplexF64}
    Uxmy::Vector{ComplexF64}
    ...
end

mutable struct ComputeCache{M<:AbstractMagneticCache}
    ...
    magnetic::M
end
```

不过这会让 `ComputeCache` 变成参数化类型，改动稍大。更简单、也足够高效的方案是：不在 hot loop 里调用通用 `link_phase`，而是在 `LandauGaugeCache` 里预存四个数组：

```julia
U_x[i]
U_y[i]
U_xpy[i]
U_xmy[i]
```

然后 Hamiltonian 和 current loop 直接读数组。设计文档已经说要预计算 common directed phases、避免 HMC leapfrog 或 repeated Kubo accumulation 里调用 `cis`；这点应该严格执行。([GitHub][1])

当前 (L=40) 的主要成本仍是 (2N\times2N) BdG 对角化，phase array 本身不是瓶颈；但减少 bug 比减少一点运行时间更重要。

---

## 11. 当前代码接入点：建议按这个顺序改

当前模块 include 顺序是：

```julia
include("MultiEta.jl")
include("Types.jl")
include("Hamiltonian.jl")
include("Observables.jl")
include("TwistedSpectra.jl")
include("HMC.jl")
include("Simulation.jl")
```

并没有 magnetic phase layer。([GitHub][5])

如果 `ComputeCache` 要持有 magnetic cache，那么需要处理 include 顺序。建议拆成两步：

第一步，在 `Types.jl` 前定义 magnetic cache types：

```julia
include("MagneticFieldTypes.jl")
include("Types.jl")
include("MagneticField.jl")
```

`MagneticFieldTypes.jl` 只放类型，不依赖 `ModelParameters`。`MagneticField.jl` 放依赖 `ModelParameters` 的构造函数和验证函数。

第二步，改 `initialize_cache(p)`：

```julia
mag = build_magnetic_cache(p)
return ComputeCache(..., mag, ...)
```

第三步，改 `init_static_H!`。不要沿用 `dir=1:4, if j>i` 的旧写法。只遍历四个唯一有向 bond。

第四步，改 `current_operator_matrix` 和 diamagnetic loop。当前 current operator 和 diamagnetic term 都是零场写法。([GitHub][4])

第五步，改 spectra 逻辑。有限磁场下默认只写 DOS / LDOS / transport scalars。

第六步，最后再改 HPC 脚本。当前 `sweep_T.sh` 生成的 `params.jl` 只有 (L,t,t',W,n_{\rm imp},V,\beta) 等参数，没有 magnetic 参数；`run_conf.jl` 也只打印 spectra/twist options。([GitHub][6])

---

## 12. HPC 角度：设计基本节制，但要小心输出和内存

你的当前 sweep 设置是 (L=40)、16 个构型并行、每个温度 `n_measure=1000`，transport 每 sweep 测一次。([GitHub][6])

有限磁场本身不会显著增加 BdG 对角化成本；矩阵大小还是 (2N\times2N)。主要额外成本来自：

1. complex hopping phase 让矩阵完全复数化；你现在本来已经用 `ComplexF64`，所以影响不大。
2. gauge-covariant bond output 增加 JLD2 体积。
3. 如果错误地保留 ordinary momentum spectra，会继续产生大数组。
4. 如果未来做 Hall conductivity，会多一个 current channel 和矩阵元计算。

因此第一版保持：

```julia
write_gauge_pair_bonds = false
allow_gauge_dependent_spectra = false
```

是对的。([GitHub][1])

但建议新增：

```julia
measure_gauge_pair_scalars = true
write_gauge_pair_bonds_freq = 0
```

这样默认仍能看到有限磁场下有意义的 pairing amplitude 和 vortex diagnostic，而不写巨大 bond arrays。

另外，当前 `run_conf.jl` 会在 SLURM 分配 `N_CORES=16` 时启动 16 个 workers，再加 master，总进程数是 17。([GitHub][7]) 对零场可能没事；有限磁场加上更多输出后，要注意节点内存。如果每个 worker 都持有 (H,U,J_{mn},temp_JU) 这些 (3200\times3200) complex dense matrix，单 worker 内存已经不小。建议实际跑前先做一个 `N_conf=1` 的内存记录，再决定是否仍用 16 workers。

---

## 13. 测试计划需要加强

设计文档现在的测试计划覆盖了 phase、Hermiticity、current entries、finite/real smoke tests、output fail-fast。这个基础不错。([GitHub][1])

我建议新增以下测试，优先级很高。

第一，Hamiltonian derivative test：

[
J_{\rm analytic}
\stackrel{?}{=}
\frac{H(+\epsilon)-H(-\epsilon)}{2\epsilon}.
]

这能抓住 Nambu block 符号错误、hole block conjugation 错误、diagonal (t') probe phase 错误。

第二，diamagnetic derivative test：

[
K_{\rm analytic}
\stackrel{?}{=}
\frac{H(+\epsilon)+H(-\epsilon)-2H(0)}{\epsilon^2}.
]

第三，finite-field Kubo vs finite-difference curvature test。小系统即可，不需要 production 输出。没有这个测试，不要相信 `Superfluid_Stiffness(B)`。

第四，zero-field regression test。要求：

```julia
n_flux_sc = 0
```

时 Hamiltonian、forces、transport、spectra 和旧代码一致到数值误差。这个很重要，因为你的零场结果已经是主线成果，不能被磁场分支破坏。

第五，gauge-invariant Wilson loop tests。除了 elementary plaquette，还要测 non-contractible loop：

[
\prod_y U_y(x,y)
]

在不同 (x) 的差别应当等于中间夹住的 plaquette flux。这样能抓住 torus patch 错误。

第六，post-processing missing-field test。有限磁场默认没有 `A_k0`、`A_MX_path`、`A_XG_path`、`dos_M` 时，脚本必须正常完成，而不是 silently 填零。

---

## 14. 初始化问题：低温有限磁场 HMC 可能会卡在坏的 vortex sector

设计文档没有讨论 finite-field 初态。对高温随机热启动可能没事；但你关心的是更低温，(\beta) 大，HMC 可能会在 vortex 位置、phase winding、disorder pinning 上热化很慢。

建议第一版至少支持三种初始化：

1. `:random`：保持当前随机 (\Delta)。
2. `:uniform`：均匀 (d)-wave 幅度，随机小扰动。
3. `:vortex_seeded`：根据 (n_{\rm flux}^{SC}) 放入大致的 vortex phase texture。

哪怕 `:vortex_seeded` 很粗糙，也比完全随机更容易在低温有限场下进入合理 sector。实际 production 可以用 annealing：

[
T_{\rm high}\to T_{\rm low}
]

或者

[
B=0\to B>0
]

逐步 warm start。否则你看到的低温线性电阻可能混有热化不充分的效应。

---

## 15. 我会怎样修改这份设计文档

我建议把设计文档改成下面这个版本的原则。

第一，保留：

* magnetic PBC / torus 作为第一版主路径；
* `n_flux_sc::Int` 且有限场要求偶数；
* central `MagneticField.jl`;
* Peierls phases for NN and NNN hopping;
* pairing block 不乘 Peierls phase；
* finite-field ordinary FFT spectra 默认禁用；
* Hall / cylinder / magnetic Bloch out of scope。

第二，修改或补充：

* 明确有限场下 `Superfluid_Stiffness` 不是 BKT (T_c) 判据。
* `init_static_H!` 必须改成唯一有向 bond loop，禁止继续用 `j > i` 策略。
* diagonal boundary `link_phase(i,dx,dy)` 必须给出完整算法和测试。
* Kubo current operator 必须定义为 BdG Hamiltonian 对 probe 的导数。
* 加入 finite-difference derivative test 和 curvature benchmark。
* `dos_M` 也归入 gauge-dependent spectra，默认禁用或改诊断名。
* finite-field pairing scalar 旧字段必须标注 gauge-dependent。
* 增加 gauge-covariant scalar pairing 输出，bond 输出用 frequency 控制。
* 输出 `sigma_xx_regular`，不要默认叫 physical resistivity。
* HPC 脚本 job tag 里加入 `Nv${n_flux_sc}`，避免不同磁场结果混目录。

第三，建议第一版实现路径：

```text
Phase 1: MagneticField.jl + phase tests
Phase 2: finite-field Hamiltonian + Hermiticity + zero-field regression
Phase 3: finite-field current/diamagnetic + derivative tests
Phase 4: Kubo stiffness/conductivity + finite-difference benchmark
Phase 5: spectra fail-fast + metadata + post-processing
Phase 6: production sweep script
```

不要一口气把 Hamiltonian、Kubo、spectra、HPC 全部改完。这样 debug 会非常困难。

---

## 最后判断

这份设计文档值得继续推进，但我会给它一个“**需要修改后再实现**”的评价。

最强的部分是：选择 magnetic PBC、中心化 phase layer、Peierls phase 同时进入 Hamiltonian/current/diamagnetic、禁用普通 momentum spectra。这些都很正确。

最弱的部分是：Kubo current operator 的验证不足、finite-field stiffness 的物理解释不足、diagonal boundary phase 不够精确定义、gauge-dependent 输出容易被误用。

你现在的研究问题很敏感：你要用磁场杀掉超导，然后讨论低温线性电阻。如果 `D_s(B,T)` 的定义或 current operator 有一个符号错，最后的物理结论会完全偏掉。因此第一版最重要的不是功能多，而是把这三件事做扎实：

[
\text{Hamiltonian phase convention}
\quad\Longleftrightarrow\quad
\text{current operator}
\quad\Longleftrightarrow\quad
\text{finite-difference stiffness benchmark}.
]

这三者一致后，再开始大规模 (L=40) 磁场扫温。

[1]: https://raw.githubusercontent.com/YinkaiYu/Hybrid-Monte-Carlo-for-d-wave-SC/aa3a8b65be853ec7fee1ac782407b0f7cb94c844/docs/superpowers/specs/2026-05-21-magnetic-field-design.md "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/YinkaiYu/Hybrid-Monte-Carlo-for-d-wave-SC/t-V-model/src/Types.jl "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/YinkaiYu/Hybrid-Monte-Carlo-for-d-wave-SC/t-V-model/src/Hamiltonian.jl "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/YinkaiYu/Hybrid-Monte-Carlo-for-d-wave-SC/t-V-model/src/Observables.jl "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/YinkaiYu/Hybrid-Monte-Carlo-for-d-wave-SC/t-V-model/src/DwaveHMC.jl "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/YinkaiYu/Hybrid-Monte-Carlo-for-d-wave-SC/t-V-model/projectHPC/example/sweep_T.sh "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/YinkaiYu/Hybrid-Monte-Carlo-for-d-wave-SC/t-V-model/projectHPC/run_conf.jl "raw.githubusercontent.com"
