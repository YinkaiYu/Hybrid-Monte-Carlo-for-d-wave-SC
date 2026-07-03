# Hall Conductivity and Resistivity Design Third Review

本文档审查对象是再次修订后的：

```text
docs/superpowers/specs/2026-07-03-hall-conductivity-resistivity-design.md
```

背景需求仍然是：在有限磁场下补充 `sigma_xy` / Hall 电导输出，让 `projectHPC/example/plot_stiffness.ipynb` 和 `projectHPC/example/plot_conductivity.ipynb` 可以画物理的直流纵向电阻率随温度变化，而不是继续把 `1 / DC_Conductivity` 当作有限磁场下的物理 `rho_xx`。

总体结论：这版设计已经解决了第二轮审查中的两个最重要问题：

- 统计约定已经明确为 **先平均 conductivity tensor，再求逆得到 `rho_xx`**。
- notebook fallback 已明确限制为旧数据；新数据有 Hall 列但 `rho_xx` 无效时，不应静默退回 `1 / sigma_xx` proxy。

我没有发现新的阻塞性物理定义问题。下面是实现前仍建议补清楚的细节，主要是测试覆盖和脚本范围，避免实现时遗漏边界路径。

---

## 1. 中优先级：顶层 summary 脚本是否纳入范围仍需明确

当前设计只要求更新：

```text
projectHPC/example/batch_process_csv.jl
```

这对用户当前点名的两个 `projectHPC/example/*.ipynb` 是够用的。但仓库里还有两个顶层 summary 脚本：

```text
scripts/batch_csv_summary_T.jl
scripts/batch_csv_summary_beta.jl
```

它们也读取 `transport.csv` 并生成 `summary_all.csv` 风格输出。设计现在没有说这两条路径是：

1. 明确 out of scope，只保证 HPC example notebook；
2. 还是也要同步支持 `Hall_Conductivity_mean` 和 derived `Longitudinal_Resistivity_mean`。

建议在设计文档里加一句明确范围。否则后续有人用顶层 `scripts/batch_csv_summary_T.jl` 处理有限磁场数据时，可能仍然得不到物理 `rho_xx` 列。

如果目标只限 `projectHPC/example/plot_stiffness.ipynb` 和 `projectHPC/example/plot_conductivity.ipynb`，可以明确写：

```text
Top-level scripts/batch_csv_summary_*.jl remain legacy helpers in this implementation.
Only projectHPC/example/batch_process_csv.jl is guaranteed to derive rho_xx.
```

---

## 2. 中优先级：summary 测试应验证数值公式和误差传播，不只验证列存在

设计已经要求：

- `Longitudinal_Resistivity_mean` 从 `DC_Conductivity` 和 `Hall_Conductivity` 的均值推导。
- `Longitudinal_Resistivity_err` 用 per-conf covariance 的 delta method。

这很好，但测试描述目前主要说：

```text
confirming Longitudinal_Resistivity_mean/err is derived from DC_Conductivity and Hall_Conductivity means
```

建议把 fixture 设计写得更具体，避免实现只测列存在。

推荐 synthetic fixture 至少包含两个或三个 `conf_*`，每个 `transport.csv` 有可手算的 per-conf means，例如：

```text
conf_001: sigma_xx = 2, sigma_xy = 1
conf_002: sigma_xx = 4, sigma_xy = 3
```

测试应断言：

```text
xbar = mean([2, 4])
ybar = mean([1, 3])
rho = xbar / (xbar^2 + ybar^2)
```

并且 `Longitudinal_Resistivity_err` 与设计中的梯度/covariance 公式一致。这样可以防止实现不小心：

- 平均了 raw `Longitudinal_Resistivity` 旧列；
- 用 `1 / DC_Conductivity_mean`；
- 忽略 `sigma_xy`；
- 或把 covariance of samples 和 covariance of sample mean 混淆。

---

## 3. 低优先级：selected eta scalar Hall 输出仍有轻微歧义

设计里写：

```text
If a selected eta factor is requested, also write selected scalar DC Hall files if useful for analysis, or fold them into the existing selected DC output format.
```

这不是阻塞项，但实现时容易出现不一致：`spectra_dc_cond.csv` 已经只写 `DC_Conductivity`，如果再加 Hall，可能有两种做法：

1. 扩展现有 selected scalar 文件：

```text
eta_factor,DC_Conductivity,DC_Error,Hall_Conductivity,Hall_Error
```

2. 新增一个 Hall scalar 文件：

```text
spectra_hall_dc_cond.csv
```

建议设计明确“第一版不要求 selected eta scalar Hall 输出”，或者明确文件名和 header。否则 postprocess 测试和 notebook 可能各按不同假设实现。

由于用户的核心 notebook 温度曲线来自 `summary_all.csv`，第一版可以把 selected eta scalar Hall 明确标为 out of scope，仅保证：

- `transport.csv` / `summary_all.csv` 的默认 eta scalar；
- `spectra_hall_cond.csv` 的 frequency-dependent Hall。

---

## 4. 低优先级：`Longitudinal_Resistivity_n_finite_conf` 的列归类要和现有 writer 对齐

设计建议可选写：

```text
Longitudinal_Resistivity_n_finite_conf
```

这对 notebook 诊断有用。但当前 `projectHPC/example/batch_process_csv.jl` 的 final header 逻辑把只以 `_mean` / `_err` 结尾的列归到 data columns，其它 key 会按 param columns 处理。

如果实现要写这个诊断列，建议同时更新 header 分组逻辑，或者把列名/分类约定写清楚。否则它可能出现在参数列区域，不影响数值但会让 `summary_all.csv` 的列组织比较怪。

这不是必须项；如果不打算第一版输出 `n_finite_conf`，可以在设计里删掉 “optionally write” 或标成后续增强。

---

## 5. 低优先级：targeted verification commands 可以再加一个 HPC/summary 覆盖

设计的 targeted commands 是：

```bash
julia --project test/test_magnetic_field.jl
julia --project test/test_postprocess_spectra.jl
julia --project test/test_simulation_tbc.jl
```

如果 `projectHPC/example/batch_process_csv.jl` 的 fixture 放进 `test/test_postprocess_spectra.jl`，这组命令可以覆盖主要改动。但如果 summary fixture 放进新测试文件或 `test/test_hpc_scripts.jl`，targeted list 也要同步更新。

建议在设计里写明 summary fixture 放在哪里。例如：

```text
Add the batch_process_csv.jl fixture to test/test_postprocess_spectra.jl.
```

这样 targeted verification list 就保持准确。

---

## 结论

这版设计已经可以作为实现基础。实现前只需要补清楚：

1. 顶层 `scripts/batch_csv_summary_*.jl` 是否 out of scope。
2. `batch_process_csv.jl` 的 summary fixture 要验证实际 `rho_xx` 数值和 covariance error。
3. selected eta scalar Hall 文件是否 out of scope 或固定 header。
4. 可选 `n_finite_conf` 列是否真的输出，以及如何归类。

这些都是工程执行层面的细节，不再是主要物理定义阻塞项。Codex 如果对这些范围选择还有不确定，应先和用户确认再实现。
