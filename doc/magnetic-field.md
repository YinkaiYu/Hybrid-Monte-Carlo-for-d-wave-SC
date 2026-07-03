# 有限磁场约定

本文档记录当前有限轨道磁场版本的输入、规范约定和可观测量解释。第一版只支持环面上的 magnetic periodic boundary condition (magnetic PBC)，还没有实现磁性元胞、magnetic Bloch theorem 或 magnetic-translation-covariant momentum spectra。

## 磁场如何进入晶格模型

我们在 $L_x\times L_y$ 的有限晶格环面上加入均匀轨道磁场。输入参数
$n_{\rm flux}^{\rm sc}$ 表示穿过整个模拟元胞的超导磁通量子数：
$$
\Phi_{\rm tot}=n_{\rm flux}^{\rm sc}\Phi_0^{\rm sc},
\qquad
\Phi_0^{\rm sc}=\frac{hc}{2e}.
$$
因为 BdG normal hopping 中运动的是电子，一个超导磁通量子对应电子环路相位
$\pi$。因此每个 plaquette 上的电子 Peierls 相位为
$$
\alpha=\frac{\pi n_{\rm flux}^{\rm sc}}{L_xL_y}.
$$

磁场只通过 normal hopping 的 Peierls 替换进入单粒子 Hamiltonian：
$$
h_{ij}(B)=-t_{ij}U_{ij}+(w_i-\mu)\delta_{ij},
\qquad
U_{ij}=\exp(i\varphi_{ij}),
\qquad
\varphi_{ij}=\int_i^j {\bf A}\cdot d{\bf l},
$$
其中单位和符号按程序的 Peierls 相位约定吸收到 $\varphi_{ij}$ 中。BdG pairing
block 仍直接使用采样到的裸 $\Delta_{ij}$，不额外乘 Peierls 相位。

规范选择为 Landau gauge。用 0-based 坐标
$x=0,\ldots,L_x-1$、$y=0,\ldots,L_y-1$ 记格点，取
$$
A_x=0,\qquad A_y=Bx.
$$
对应的最近邻 link phase 是
$$
U_y(x,y)=e^{i\alpha x},\qquad
U_x(x,y)=1\quad (x<L_x-1).
$$
为了在有限环面上保持每个 plaquette 都有同一个磁通，跨过 $x$ 边界的 hopping
需要乘一个 magnetic PBC patch：
$$
U_x(L_x-1,y)=e^{-i\alpha L_x y}.
$$
这等价于把电子场按
$$
c_{x+L_x,y}=e^{-i\alpha L_x y}c_{x,y},
\qquad
c_{x,y+L_y}=c_{x,y}
$$
做磁周期性识别。沿 $x$、$y$ 两个方向绕环面的顺序应当给出同一个结果，因此需要
$$
e^{-i\alpha L_xL_y}=e^{-i\pi n_{\rm flux}^{\rm sc}}=1.
$$
这就是当前实现要求 $n_{\rm flux}^{\rm sc}$ 为偶数的原因。现在支持的有限磁场通量为
$$
n_{\rm flux}^{\rm sc}=\pm 2,\pm 4,\pm 6,\ldots,
$$
符号表示磁场方向；$n_{\rm flux}^{\rm sc}=0$ 是零场情形。实际运行时，非零通量必须配合
`boundary_condition=:magnetic_pbc`；零场也可以继续使用普通周期边界。奇数个超导磁通量子以及磁性元胞或 magnetic Bloch theorem
还没有实现。

## 输入参数

用户通过 `n_flux_sc::Int` 指定穿过完整 $L_x \times L_y$ 模拟元胞的超导磁通量子数，单位是 $hc/2e$。在当前模型语境下，它也就是期望的 vortex number。`n_vortices` 只是输入别名；元数据、JLD2 和后处理约定统一使用 `n_flux_sc`。

`n_flux_sc` 可以为正也可以为负，符号表示磁场方向。有限磁场时必须使用 `boundary_condition=:magnetic_pbc`，并且 magnetic PBC 要求 `n_flux_sc` 为偶数。`n_flux_sc=0` 时可以继续使用普通周期边界。

内部定义为

```julia
flux_density_sc = n_flux_sc / (Lx * Ly)
plaquette_phase = pi * flux_density_sc
```

这里刻意不用 `phi` 命名磁通密度，避免和 Peierls 相位、辅助场相位混淆。`plaquette_phase` 是电子 Peierls 相位绕单个 plaquette 的相位；由于输入磁通单位是 $hc/2e$，电子绕一圈得到的相位是 $\pi n_{\rm flux}^{\rm sc}/(L_xL_y)$。

## Landau Gauge 与 Link Phase

程序使用 0-based 坐标 $(x,y)$ 描述规范约定。Landau gauge 的最近邻 link 为

```text
U_y(x,y) = cis(plaquette_phase * x)
U_x(x,y) = 1
U_x(Lx-1,y) = cis(-plaquette_phase * Lx * y)
```

最后一行是穿过 $x$ seam 的 magnetic PBC patch。普通 $x$ 向 link 没有相位，$y$ 向 link 随 $x$ 线性变化。`n_flux_sc` 的符号直接改变 `plaquette_phase` 的符号。

次近邻 $t'$ 对角键不手写成额外特例，而是通过集中的 `link_phase` 层计算：先取直线路径上的 Peierls 积分，再按端点是否跨过 seam 乘上 magnetic PBC patch。这样最近邻、对角键和将来其它位移的 Peierls 相位都来自同一套约定。

## BdG 配对块与配对输出

BdG 非对角配对块使用采样得到的裸辅助场 $\Delta_{ij}$。程序不会再给 pairing block 额外乘 Peierls 相位。也就是说，有限磁场只通过 normal hopping 的 Peierls phase 进入 BdG 矩阵；pairing block 仍是裸的 `Delta_ij`。

已有 CSV 标量列，例如 `Delta_x - Delta_y`、`Delta_Diff`、`Delta_Pair`，保持裸 Landau-gauge diagnostic 的旧约定，目的是向后兼容。这些量在有限磁场下不应被当作规范不变量。

规范协变的 bond pairing 只写入 JLD2，且默认关闭。需要时在 `run_simulation` 中设置

```julia
write_gauge_pair_bonds_freq > 0
```

输出键为

```text
delta_bond_landau_gauge_covariant
pair_bond_landau_gauge_covariant
```

有限磁场后处理应主要使用 fermionic pairing 的 `pair_bond_landau_gauge_covariant`。辅助场版本 `delta_bond_landau_gauge_covariant` 主要用于诊断采样到的辅助场纹理。

## 超流刚度与电导

`Superfluid_Stiffness` 仍定义为横向 `xx` Meissner estimator：

$$
\rho_s=\langle -K_x\rangle-\Lambda_{xx}(q_x=0,q_y=2\pi/L_y,\omega=0).
$$

在有限磁场下，它是一个被磁场压低后的 stiffness diagnostic。不要把有限磁场的 `Superfluid_Stiffness` 当作零场 BKT universal jump 的 $T_c$ 判据。

`DC_Conductivity` 是 regular part 的 $\sigma_{xx}$，不包含超流 $\delta(\omega)$ 项。有限磁场运行如果打开 transport 或 spectra 测量，也会输出 Hall conductivity：CSV 标量列为 `Hall_Conductivity`，JLD2 谱学键为 `hall_cond`、`hall_cond_eta`、`hall_opt_cond`、`hall_opt_cond_eta`，谱后处理输出 `spectra_hall_cond.csv`。`hall_cond` / `Hall_Conductivity` 是同一有限 $\eta$ Kubo--Greenwood 表达式在 $\omega=0$ 的实部，即 $\sigma_{xy}^{\rm dc}=\mathrm{Re}\,\sigma_{xy}(0)$。

当前 `J_y` 电流算符沿程序中正向列举的 bond 求方向分量：`+y` 和 `+x+y` bond 的方向因子是 `+1`，`+x-y` bond 的方向因子是 `-1`。这与 `direction=:y` 的 `direction_component(direction, dx, dy)=dy` 约定一致；Hall tensor 使用 $J^x_{nm}J^y_{mn}$。

HPC 汇总中的 `Longitudinal_Resistivity_mean` / `Longitudinal_Resistivity_err` 由构型平均后的 $\overline{\sigma_{xx}}$ 和 $\overline{\sigma_{xy}}$ 反演得到：
$$
\rho_{xx}=
\frac{\overline{\sigma_{xx}}}
{\overline{\sigma_{xx}}^2+\overline{\sigma_{xy}}^2}.
$$
也就是说，汇总先平均 `DC_Conductivity` 和 `Hall_Conductivity`，再反演 $2\times2$ 电导张量；不保留逐构型 $\rho_{xx}$ 平均。

## 谱函数与动量空间诊断

有限轨道磁场下，普通 FFT 定义的 $A(k,\omega)$、`dos_M`、M-X 路径和 X-G 路径都依赖规范。默认不输出这些普通动量谱。

只有显式设置

```julia
allow_gauge_dependent_spectra = true
```

才会写出 Landau gauge 诊断谱，并且所有 key 都带 warning 名称：

```text
A_k_omega0_landau_gauge_diagnostic
A_MX_path_landau_gauge_diagnostic
A_XG_path_landau_gauge_diagnostic
dos_M_landau_gauge_diagnostic
```

多展宽版本也使用相同后缀：

```text
A_k_omega0_eta_landau_gauge_diagnostic
A_MX_path_eta_landau_gauge_diagnostic
A_XG_path_eta_landau_gauge_diagnostic
dos_M_eta_landau_gauge_diagnostic
```

后处理生成的 CSV 也保留 `_landau_gauge_diagnostic` 后缀。它们只适合调试 Landau-gauge 下的数值变化，不是规范不变或磁平移协变的动量分辨谱函数。

## 当前没有实现的内容

第一版没有实现以下功能：

- 磁性元胞或 magnetic Bloch theorem。
- magnetic-translation-covariant momentum spectra。
- 有限磁场下的 twisted-boundary spectra 或 twist free-energy benchmark。

如果需要论文级有限磁场动量谱，需要在 magnetic-translation-covariant spectra 基础设施补齐后再解释相关输出。

## HPC 运行建议

有限磁场低温热化通常比零场慢。原因包括 vortex position 的慢模、pairing phase texture 的重排，以及无序势下 vortex pinning 与辅助场幅度的耦合。生产扫参前建议先做短任务：

```text
N_CONFS=1
```

或使用少量 worker 的 smoke run。先检查：

- HMC acceptance 和 `dH` 是否稳定。
- 内存是否随 `write_gauge_pair_bonds_freq` 和谱输出频率可控。
- `Superfluid_Stiffness` 和 `DC_Conductivity` 是否随初始构型剧烈漂移。
- `pair_bond_landau_gauge_covariant` 输出是否存在、尺寸正确、相位纹理随 sweep 变化合理。

确认这些量稳定后，再增加构型数、worker 数和低温热化长度。
