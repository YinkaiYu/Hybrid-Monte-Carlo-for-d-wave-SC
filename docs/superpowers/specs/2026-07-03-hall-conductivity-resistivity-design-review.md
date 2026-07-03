# Hall Conductivity and Resistivity Design Review

本文档审查对象是 `docs/superpowers/specs/2026-07-03-hall-conductivity-resistivity-design.md`，背景需求是：

- 在有限磁场下补充 `xy` 方向 Hall 电导计算。
- 让 `projectHPC/example/plot_stiffness.ipynb` 和 `projectHPC/example/plot_conductivity.ipynb` 可以画物理的直流纵向电阻率随温度变化。
- 避免继续把 `1 / DC_Conductivity` 当作有限磁场下的物理 `rho_xx`。

总体结论：设计文档的大方向是对的，覆盖了 current operator、transport result、JLD2 schema、postprocess、notebook 和文档更新。但不建议按原文直接实现。下面几处需要先修订，否则很容易得到能跑但物理规范不一致、数据链路不完整、或测试条件本身不成立的实现。

如果下面任何物理定义、输出 schema、notebook 预期或兼容策略不清楚，请先和用户讨论清楚，不要自行猜测后继续实现。

---

## 1. 高优先级：复数 Hall 公式和现有 sigma_xx 规范不一致

设计文档中 Hall optical tensor 写为：

```text
sigma_xy(omega) =
  i / N * sum_{n != m}
  (f_n - f_m) / (E_m - E_n)
  * Jx_nm * Jy_mn
  / (omega - (E_m - E_n) + i eta)
```

并且声称当 `alpha = beta = x` 时，实部会回到现有的正的 Lorentzian-broadened `Re sigma_xx(omega)`。

这和当前代码不一致。当前 `measure_transport_only` 中 `sigma_xx(omega)` 的实现是 Kubo-Greenwood 形式：

```julia
prefactor = (fn_fm * omega_inv[iω]) * J2
x = ω - Em_En
σ_ω_eta[iη, iω] += prefactor * lorentzian(x, eta_vals[iη])
```

也就是有限展宽下使用 `1 / omega`，而设计公式使用 `(f_n - f_m) / (E_m - E_n)` 和复数 resolvent。有限 `eta` 下这两者不严格等价。因此设计文档里的 normalization test：

```text
set alpha=beta=x in the complex tensor formula and compare real part against optical_conductivity
```

很可能失败，或者迫使实现改变现有 `optical_conductivity` 的含义。

需要先决定以下二选一：

1. 保持现有 `sigma_xx(omega)` 的 Kubo-Greenwood 展宽规范，并为 `sigma_xy` 定义一个与其相容的有限 `eta` 计算/测试方式。
2. 引入新的完整复数 optical tensor 规范，例如 `complex_sigma_xx` 和 `complex_sigma_xy`，但不要声称它逐点等同当前 `optical_conductivity`。

建议：第一版以最小改动满足电阻率需求时，可以只把 `sigma_xy^dc` 规范定义清楚，并把 frequency-dependent `sigma_xy(omega)` 标成同一 Kubo convention 下的辅助输出；不要用不成立的 `alpha=beta=x` 逐点等价测试作为门槛。

---

## 2. 高优先级：零场 Hall 为零和磁场反转测试的前提不充分

设计文档建议：

- 零场 clean 或 weak-random lattice 下 `hall_conductivity` 应接近 0。
- 匹配 random states 的 `+B` 和 `-B` 应满足 `sigma_xy(+B) ~= -sigma_xy(-B)`。

但当前仓库的 `initialize_state` 会生成复数 `Delta`，测试 helper `random_finite_field_state` 也会生成复数配对场。复数配对场本身可以破坏时间反演，因此这些对称性测试的前提并不自动成立。

测试设计需要明确：

- 零场 Hall 对称性测试应使用实 `Delta`，或显式选择时间反演不破缺的构型。
- `+B/-B` 反转测试应使用同一 disorder，并让 `Delta(-B) = conj(Delta(+B))`；如果使用实 `Delta`，则两边可以共享同一个 `Delta`。
- 如果测试使用随机复数 `Delta`，不要期待严格的 Hall 奇偶性。

---

## 3. 中优先级：Ay 有限差分测试缺少对应的 probe Hamiltonian 设计

设计文档要求对 `Ax` 和 `Ay` 都做 current-operator finite-difference derivative test。但当前 `build_probe_H_BdG!` 只给 x 方向相关 bond 加 probe factor：

- `+x`
- `+x+y`
- `+x-y`

当前 `probe_current_operator_matrix` 也只构造 x 方向 probe current。

如果要测试 `direction=:y`，设计文档需要同步要求：

- `build_probe_H_BdG!` 增加 `direction` 参数。
- `probe_factor` 或新 helper 根据 bond displacement 的 `d_alpha` 加 uniform probe phase。
- `probe_current_operator_matrix(...; direction=:y)` 和 production `current_operator_matrix(...; direction=:y)` 使用同一个 bond/displacement convention。

否则实现者会发现测试设计需要的 `Ay` probe Hamiltonian 在架构里不存在。

---

## 4. 中优先级：NaN resistivity 会破坏 HPC summary 处理

设计文档建议：

```text
If dc_conductivity <= 0 or denominator is non-finite, write NaN for Longitudinal_Resistivity.
```

但 `projectHPC/example/batch_process_csv.jl` 的 `read_conf_robust` 当前会在任意列均值出现 `NaN` 或 `Inf` 时返回 `nothing`，从而丢弃整份 CSV 的结果。这会导致：

- `Longitudinal_Resistivity_mean` 进不了 `summary_all.csv`。
- 甚至同一份 `transport.csv` 中有效的 `DC_Conductivity`、`Hall_Conductivity` 也可能被一起丢掉。

设计文档需要补充兼容策略，二选一：

1. 修改 `batch_process_csv.jl` 为逐列容错，只跳过无效列，不丢弃整份 transport。
2. `transport.csv` 中不要写裸 `NaN`，改写为空值或约定 sentinel，并确保 summary 脚本按列处理。

为了 notebook 目标，推荐修改 summary 脚本为逐列容错。否则“notebook prefer `Longitudinal_Resistivity_mean`”这条无法可靠成立。

---

## 5. 中优先级：postprocessing 范围漏了 scripts/process_spectra.jl

设计文档点名了：

- `scripts/batch_process_spectra.jl`
- `projectHPC/example/batch_process_spectra.jl`

但仓库里还有单目录处理脚本：

- `scripts/process_spectra.jl`

测试文件 `test/test_postprocess_spectra.jl` 会 include 这个脚本，并且它有和 batch processor 类似的 `collect_sweep_data`、`selected_scalar/vector`、`processed_opt_cond.csv` 输出路径。

如果要支持 `processed_hall_cond.csv` 或保持旧 fixtures 无 Hall keys 时能清理 stale Hall CSV，这个脚本也必须纳入设计。

建议把 postprocess 修改范围写成：

- `scripts/process_spectra.jl`
- `scripts/batch_process_spectra.jl`
- `projectHPC/example/batch_process_spectra.jl`
- `scripts/spectra_postprocess_utils.jl`
- `projectHPC/example/spectra_postprocess_utils.jl`

并明确两个 helper 副本都要支持 complex array 的 componentwise mean/error。

---

## 6. 中优先级：JLD2 分箱集成需要写得更具体

当前 `src/Simulation.jl` 的谱学分箱不是自动反射 `SpectrumResult` 字段，而是手工枚举：

- accumulator 初始化
- 第一个 bin copy
- 后续 bin `.+='
- 达到 `bin_size` 后除以 `bin_count`
- 写入 JLD2 group

设计文档只说在 `spectra_bins.jld2` 中添加：

- `hall_cond`
- `rho_xx`
- `hall_cond_eta`
- `rho_xx_eta`
- `hall_opt_cond`
- `hall_opt_cond_eta`

这还不够。需要明确 `Simulation.jl` 中新增对应 accumulator：

- scalar/vector accumulator: `accum_hall_cond_eta`, `accum_rho_xx_eta`
- complex vector/matrix accumulator: `accum_hall_opt_cond`, `accum_hall_opt_cond_eta`
- 写 `transport.csv` 时使用 `spec_res.hall_conductivity` 和 `spec_res.longitudinal_resistivity`
- 写 JLD2 时同步写 scalar single-eta key 和 multi-eta key

特别注意 `hall_opt_cond_eta` 是 complex array，不能用只支持 `Float64` 的统计/写 CSV helper 直接处理。

---

## 7. Notebook 设计还要覆盖现有数据来源

`projectHPC/example/plot_stiffness.ipynb` 当前直接用：

```python
R = 1 / df["DC_Conductivity_mean"]
R_err = sigma_err / sigma**2
```

`projectHPC/example/plot_conductivity.ipynb` 的 `build_dc_comparison_df` 也把 Kubo resistivity 构造成：

```python
R_dc_kubo = 1.0 / sigma_dc_kubo
```

设计文档已经写了 notebook 应 prefer `Longitudinal_Resistivity_mean`，这是对的。但还应补充：

- `plot_conductivity.ipynb` 中 `build_dc_comparison_df` 要显式读取 `Longitudinal_Resistivity_mean/err`。
- 如果存在 `Longitudinal_Resistivity_mean`，`R_dc_kubo` 应来自该列，而不是 `1 / sigma_dc_kubo`。
- 只有旧数据缺少该列时，才 fallback 到 `1 / DC_Conductivity_mean`，并在 label 中写 proxy。
- 如果要画 optical extrapolation 的 resistivity，也要明确它仍然是基于 `sigma_xx(omega)` 的 proxy，除非后续实现了 frequency-dependent tensor inversion。

---

## 8. 输出命名建议

设计文档同时使用了：

- `hall_conductivity`
- `hall_cond`
- `rho_xx`
- `longitudinal_resistivity`
- `Longitudinal_Resistivity`

这些名字都能理解，但实现前应固定一张 mapping 表，避免后处理和 notebook 写错 key。

建议：

| 层级 | sigma_xy dc | rho_xx dc | sigma_xy omega |
|---|---|---|---|
| Julia field | `hall_conductivity` | `longitudinal_resistivity` | `hall_optical_conductivity` |
| Julia eta field | `hall_conductivity_eta` | `longitudinal_resistivity_eta` | `hall_optical_conductivity_eta` |
| `transport.csv` | `Hall_Conductivity` | `Longitudinal_Resistivity` | none |
| JLD2 single eta | `hall_cond` | `rho_xx` | `hall_opt_cond` |
| JLD2 multi eta | `hall_cond_eta` | `rho_xx_eta` | `hall_opt_cond_eta` |
| HPC summary | `Hall_Conductivity_mean/err` | `Longitudinal_Resistivity_mean/err` | none |
| spectra CSV | none | none | `spectra_hall_cond.csv` |

---

## 9. 建议修改后的测试重点

保留设计文档里强测试的方向，但建议修订为：

1. `current_operator_matrix(...; direction=:x/:y)` 与对应 probe Hamiltonian 的 finite-difference derivative 一致。
2. `direction=:y` 测试必须包含 `tp != 0`，覆盖 `+x+y` 的 `+1` 和 `+x-y` 的 `-1` direction factor。
3. 零场 Hall symmetry 使用实 `Delta`。
4. `+B/-B` reversal 使用 `Delta(-B)=conj(Delta(+B))`。
5. `rho_xx` 测试直接验证：

```text
rho_xx = sigma_xx / (sigma_xx^2 + sigma_xy^2)
```

并验证 `sigma_xy = 0` 时回到 `1 / sigma_xx`。

6. Postprocess fixture 要覆盖 complex `hall_opt_cond_eta`，并验证 real/imag mean 和 standard error 分开计算。
7. 旧 JLD2 fixture 缺少 Hall keys 时，三个 processor 都应继续成功，并删除 stale Hall CSV。
8. `projectHPC/example/batch_process_csv.jl` 要有 tiny fixture，确认 `Longitudinal_Resistivity_mean/err` 能进入 `summary_all.csv`。

---

## 10. 给 Codex 的执行提醒

这份设计文档不要直接照抄实现。请先修订设计，尤其是：

1. 明确有限 `eta` 下 `sigma_xy(omega)` 和现有 `sigma_xx(omega)` 的规范关系。
2. 明确对称性测试所需的 `Delta` 条件。
3. 把 `Ay` probe Hamiltonian 纳入架构。
4. 修正 `NaN` resistivity 与 `batch_process_csv.jl` 的冲突。
5. 把 `scripts/process_spectra.jl` 和两个 `spectra_postprocess_utils.jl` 副本纳入修改范围。
6. 细化 `Simulation.jl` 分箱 accumulator 和 JLD2 写入步骤。

如果对用户真正希望 notebook 展示哪些曲线、旧数据如何标注 proxy、新数据是否必须输出 frequency-dependent Hall tensor、或 `sigma_xy` 的符号约定还有疑问，请先向用户确认。不要在这些点上自行假设。
