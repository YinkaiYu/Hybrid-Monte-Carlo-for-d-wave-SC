# Hall Conductivity and Resistivity Design Second Review

本文档审查对象是修订后的：

```text
docs/superpowers/specs/2026-07-03-hall-conductivity-resistivity-design.md
```

背景需求仍然是：在有限磁场下补充 `sigma_xy` / Hall 电导输出，让 `projectHPC/example/plot_stiffness.ipynb` 和 `projectHPC/example/plot_conductivity.ipynb` 可以画物理的直流纵向电阻率随温度变化，而不是继续把 `1 / DC_Conductivity` 当作有限磁场下的物理 `rho_xx`。

总体结论：修订版已经解决了上一轮审查中的多数关键问题，尤其是：

- 不再声称 Hall tensor 公式在有限 `eta` 下与现有 `optical_conductivity` 逐点等价。
- 补上了 `direction=:y` 的 probe Hamiltonian / derivative-test 需求。
- 补上了 `scripts/process_spectra.jl`、两个 `spectra_postprocess_utils.jl` 副本和 `projectHPC/example/batch_process_csv.jl`。
- 明确了 complex Hall arrays 要做 real/imag componentwise statistics。
- 明确 notebook 应优先使用 `Longitudinal_Resistivity_mean/err`，旧数据才 fallback 到 `1 / sigma_xx` proxy。

但我仍不建议不加修改就开始实现。下面几个问题需要先修订设计或和用户确认。

如果下面任一点的物理定义、统计 estimator 或 notebook 展示策略不清楚，请先和用户讨论清楚，不要自行假设。

---

## 1. 高优先级：`rho_xx` 的 ensemble estimator 仍未定义清楚

修订版设计定义了单次测量的张量反演公式：

```text
rho_xx = sigma_xx / (sigma_xx^2 + sigma_xy^2)
```

并要求 `transport.csv` 每行写：

```text
Hall_Conductivity,Longitudinal_Resistivity
```

这意味着后续 `projectHPC/example/batch_process_csv.jl` 会对 `Longitudinal_Resistivity` 做普通均值，得到：

```text
mean(rho_xx(config))
```

但用户在 notebook 中之前实际画的是：

```text
1 / mean(sigma_xx)
```

有限磁场下自然也可能期望画：

```text
mean(sigma_xx) / (mean(sigma_xx)^2 + mean(sigma_xy)^2)
```

这两个 estimator 一般不相等：

```text
mean[ sigma_xx / (sigma_xx^2 + sigma_xy^2) ]
!=
mean(sigma_xx) / (mean(sigma_xx)^2 + mean(sigma_xy)^2)
```

这不是实现细节，而是物理统计定义。尤其当前数据链路会跨 sweep、conf、温度点做平均；如果不先定义清楚，`Longitudinal_Resistivity_mean` 的含义会变得含混。

建议先和用户确认要画哪一种：

1. **逐构型反演再平均**：把 `rho_xx` 当作一个非线性 observable，`summary_all.csv` 中的 `Longitudinal_Resistivity_mean` 就是 `mean(rho_xx(config))`。
2. **先平均电导再反演**：`summary_all.csv` 保存 `DC_Conductivity_mean` 和 `Hall_Conductivity_mean`，notebook 再计算 `rho_xx_from_mean_sigma`。
3. **两者都输出**：raw `transport.csv` 保留逐测量 `Longitudinal_Resistivity`，summary 或 notebook 同时给出 `Longitudinal_Resistivity_mean` 和 `Longitudinal_Resistivity_FromMeanConductivity`，图例明确区分。

如果目标是给 collaborator 看温度曲线，我建议第一版至少输出并标注两者，避免后续因为 averaging convention 返工。

---

## 2. 高优先级：`NaN`/缺列时 notebook fallback 仍可能误用旧 proxy

修订版已经要求 `batch_process_csv.jl` 做 columnwise finite-value tolerant processing，这是对的。但它同时保留：

```text
raw outputs 中 rho_xx 无效时写 NaN
```

以及 notebook：

```text
缺少 Longitudinal_Resistivity_mean 时 fallback 到 1 / DC_Conductivity_mean，只给旧数据使用。
```

这里仍有一个边界情况：新数据可能已经有 `Hall_Conductivity_mean`，但因为某些温度点 `rho_xx` raw column 全是 `NaN` 或被 summary 脚本逐列跳过，`summary_all.csv` 中没有 `Longitudinal_Resistivity_mean`。如果 notebook 只按“是否存在 `Longitudinal_Resistivity_mean`”判断，就会误以为这是旧数据，并退回 `1 / DC_Conductivity_mean` proxy。

建议设计明确 notebook fallback 条件：

- 只有当 `Hall_Conductivity_mean` 和 `Longitudinal_Resistivity_mean` 都不存在时，才认为是旧数据并使用 proxy。
- 如果 `Hall_Conductivity_mean` 存在但 `Longitudinal_Resistivity_mean` 不存在或无 finite 值，应显示“physical rho_xx unavailable/invalid”，不要静默 fallback。
- 更稳妥的做法是在 `summary_all.csv` 中保留 `Longitudinal_Resistivity_mean` 列，即使某些温度点为空或 `NaN`，并额外写 `Longitudinal_Resistivity_n_finite`。

这个条件需要写进 notebook 设计，否则最终图可能仍然画出用户明确不想要的有限磁场 proxy。

---

## 3. 中优先级：`sigma_xy^dc` 必须明确独立计算 omega=0，不要复用第一个 omega grid 点

修订版写了：

```text
sigma_xy^dc = Re sigma_xy(0)
```

这是正确的。但当前代码的 `omega_grid` 从 `p.omega_min = p.eta` 开始，不包含 `omega=0`。测试设计里也提到比较 `hall_optical_conductivity[1]` 时只是在“同一 grid”上比较。

实现文档应显式写一句：

- `hall_conductivity_eta[i_eta]` 必须用 `omega=0` 的公式单独计算。
- `hall_optical_conductivity_eta[i_eta, 1]` 是 `omega_grid[1]` 处的 optical response，不能当作 dc Hall。

否则实现时很容易直接取 `hall_optical_conductivity[1]` 作为 `Hall_Conductivity`。

---

## 4. 中优先级：`build_current_operator!` 的 direction/cache 行为还需要定死

修订版新增了：

```julia
current_operator_matrix(cache, p; direction=:x, qx=0.0, qy=0.0)
build_current_operator!(cache, p; direction=:x, qx=0.0, qy=0.0, store=:q0)
```

并在 cache 中只新增：

```julia
Jy_sparse_q0
```

这基本够用，因为 Hall 只需要 `Jy(q=0)`，stiffness 仍只需要 `Jx(q_y)`。但需要明确 `build_current_operator!` 的行为：

- `direction=:x, store=:q0` 写 `cache.Jx_sparse_q0`。
- `direction=:x, store=:qy` 写 `cache.Jx_sparse_qy`。
- `direction=:y, store=:q0` 写 `cache.Jy_sparse_q0`。
- `direction=:y, store=:qy` 当前不支持时应报错，除非真的新增 `Jy_sparse_qy`。

当前设计没有说 `direction=:y, store=:qy` 怎么办。建议明确报错，避免后续有人误以为 Jy 的 finite-q cache 已经存在。

---

## 5. 中优先级：summary error bar 的样本数定义要和列级过滤一致

修订版要求 `batch_process_csv.jl`：

```text
Error estimates should use the number of finite samples for each observable, not real_n_conf.
```

方向正确，但建议再具体一点。当前 summary 脚本的统计层级是：

1. 每个 `conf_*` 目录内部先对 `transport.csv` 的 sweep 求均值。
2. 再跨 conf 对这些 per-conf means 求温度点均值和 error。

列级过滤后，每个 observable 的有效 conf 数可能不同。因此建议输出：

- `Observable_mean`
- `Observable_err`
- 可选 `Observable_n_finite_conf`

并且 error 应按该 observable 的有效 conf 数计算：

```text
std(per_conf_means_finite) / sqrt(n_finite_conf)
```

如果 `n_finite_conf == 1`，error 写 0 或 NaN 需要统一约定。当前设计只说“use number of finite samples”，还不够避免实现歧义。

---

## 6. 低优先级：optical rho 的误差传播暂时没有定义

修订版要求 conductivity notebook 对 optical extrapolation：

```text
fit Re sigma_xx(omega)
fit Re sigma_xy(omega)
rho_xx_optical = sigma_xx_fit / (sigma_xx_fit^2 + sigma_xy_fit^2)
```

这满足主要需求。但如果图里要显示 optical `rho_xx` 的 error bar，目前没有定义误差传播。可以第一版不画 optical rho error bar，但设计应明确：

- direct Kubo scalar `Longitudinal_Resistivity_err` 来自 summary。
- optical extrapolated `rho_xx_optical` 第一版不画 error bar，或只作为 diagnostic 曲线。
- 如果要画 error bar，需要 bootstrap/jackknife 或至少线性误差传播，并同时使用 `Re_Error` of `sigma_xx` 和 `sigma_xy`。

这不是阻塞实现的问题，但建议在 notebook 文字和图例中避免让 optical extrapolation 看起来和 direct scalar 一样可靠。

---

## 7. 建议更新后的实现前 checklist

在开始代码实现前，建议 Codex 先把设计文档补上以下决定：

1. 用户确认 `rho_xx(T)` 的 averaging convention：逐构型反演再平均、先平均电导再反演、还是两者都输出。
2. notebook fallback 条件：只有旧数据才允许 `1 / sigma_xx` proxy；新数据 Hall 列存在但 rho 无效时不要静默 fallback。
3. `hall_conductivity_eta` 明确单独用 `omega=0` 计算，不能取 optical grid 第一点。
4. `build_current_operator!(direction=:y, store=:qy)` 明确报错或补 cache。
5. `batch_process_csv.jl` 的 per-observable `n_finite_conf`、single-sample error 约定明确。
6. optical extrapolated `rho_xx` 是否画 error bar明确；不画就标成 diagnostic。

修订版总体已经接近可执行；上述第 1 和第 2 点建议先和用户确认，因为它们直接影响最终 notebook 里的电阻率曲线含义。
