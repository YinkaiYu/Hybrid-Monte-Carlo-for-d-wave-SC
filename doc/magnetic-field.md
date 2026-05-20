# 有限磁场约定

本文档记录当前有限轨道磁场版本的输入、规范约定和可观测量解释。第一版只支持环面上的 magnetic periodic boundary condition (magnetic PBC)，还没有实现磁性元胞、magnetic Bloch theorem、Hall conductivity 或 magnetic-translation-covariant momentum spectra。

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

`DC_Conductivity` 是 regular part 的 $\sigma_{xx}$，不包含超流 $\delta(\omega)$ 项。第一版没有 Hall conductivity，因此不能从 $\sigma_{xx}$ 和 $\sigma_{xy}$ 组成完整电阻率张量。特别是，`1 / DC_Conductivity` 只能作为粗略 proxy，不能命名为物理的 $\rho_{xx}$。

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
- Hall conductivity。
- magnetic-translation-covariant momentum spectra。
- 有限磁场下的 twisted-boundary spectra 或 twist free-energy benchmark。

如果需要论文级有限磁场动量谱或电阻率张量，需要在这些基础设施补齐后再解释相关输出。

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
