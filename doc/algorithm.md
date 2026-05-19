# HMC 算法笔记

当前程序的输入 `V` 是 t-V 模型的物理耦合；内部有效配对耦合为
$$
g_{\rm pair}=\frac{V}{2}.
$$
辅助场 $\Delta_{ij}$ 直接作为 BdG 矩阵的非对角 gap。

## HMC 作用量

对固定辅助场构型，程序使用
$$
H_{\rm HMC}=
\frac{1}{2m}\sum_{\langle ij\rangle}|\pi_{ij}|^2
+\frac{\beta}{g_{\rm pair}}\sum_{\langle ij\rangle}|\Delta_{ij}|^2
-\sum_{E_n>0}
\left[
\beta E_n+2\log\left(1+e^{-\beta E_n}\right)
\right].
$$
写成输入参数 `V` 就是
$$
H_{\rm HMC}=
\frac{1}{2m}\sum_{\langle ij\rangle}|\pi_{ij}|^2
+\frac{2\beta}{V}\sum_{\langle ij\rangle}|\Delta_{ij}|^2
+S_f(\Delta).
$$

其中 $E_n$ 是当前 $H_{\rm BdG}(\Delta)$ 的本征值。费米子项只对正能量求和，并使用 `log1pexp` 避免低温溢出。

## Force

令
$$
P_{ij}=
\left\langle c_{i\uparrow}c_{j\downarrow}
-c_{i\downarrow}c_{j\uparrow}\right\rangle.
$$
HMC force 为
$$
F_{ij}
=-\frac{\beta}{g_{\rm pair}}\left(\Delta_{ij}-g_{\rm pair}P_{ij}\right)
=-\frac{2\beta}{V}\left(\Delta_{ij}-\frac{V}{2}P_{ij}\right).
$$
因此低温 saddle 条件是
$$
\Delta_{ij}-\frac{V}{2}P_{ij}=0.
$$

## Leapfrog

程序中的复数动量和辅助场按如下方式演化：
$$
\dot\Delta_{ij}=\frac{\pi_{ij}}{2m},
\qquad
\dot\pi_{ij}=F_{ij}.
$$
单次 HMC sweep 流程为：

```text
刷新复数动量 π
计算初始 H_HMC(Δ, π)
半步更新 π
重复 Nt 次：
    整步更新 Δ
    更新 H_BdG 并对角化
    计算 force
    除最后一步外整步更新 π
半步更新 π
计算末态 H_HMC(Δ', π')
按 exp[-(H_final-H_initial)] 做 Metropolis 接受/拒绝
```

`calc_optimal_dt(β, V, mass, Nt)` 使用同一个内部耦合 $g_{\rm pair}=V/2$ 来估计步长：
$$
\delta t
=\frac{1}{2N_t}\,2\pi\sqrt{\frac{m g_{\rm pair}}{\beta}}
=\frac{\pi}{N_t}\sqrt{\frac{mV}{2\beta}}.
$$

## 实现检查点

- `compute_total_energy` 和 `measure_observables` 中的玻色项系数是 $2\beta/V$。
- `compute_forces!` 使用 $g_{\rm pair}=V/2$。
- `Delta_Diff` 检查 $|\Delta-(V/2)P|$。
- `Delta_Pair` 和 `Delta_LocalPair` 是由费米子配对期望值乘以 $g_{\rm pair}$ 得到的 BdG gap 估计。
- stiffness、conductivity、DOS、谱函数只依赖 BdG 矩阵，不额外引入 pairing 归一化因子。
