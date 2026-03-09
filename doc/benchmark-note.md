下面的笔记对应的文献
Phys. Rev. Lett. **94**, 217001
Phys. Rev. B **84**, 024522 
Phys. Rev. B **105**, 054501

## 模型（decouple之后）

$$
\hat{H}=\hat{H}_{\mathrm{BdG}}+H_{\mathrm{cl}}
$$

$$
\hat{H}_{\mathrm{BdG}}
=
-\sum_{ij\sigma} t_{ij}\,(c_{i\sigma}^\dagger c_{j\sigma} + \text{h.c.})
- \mu\sum_{i\sigma}\, c_{i\sigma}^\dagger c_{i\sigma}
-\sum_{\langle ij\rangle}
\big(
\Delta_{ij}^*(c_{i\uparrow} c_{j\downarrow} - c_{i\downarrow} c_{j\uparrow})
+ \Delta_{ij}(c_{i\downarrow}^\dagger c_{j\uparrow}^\dagger - c_{i\uparrow}^\dagger c_{j\downarrow}^\dagger)
\big).
$$
$$
H_{\mathrm{cl}}=\frac{1}{V} \sum_{\langle ij\rangle} |\Delta_{ij}|^2
$$
## HMC程序

我们要用HMC来完成对复数值场 $\{\Delta_{ij}\}$ 的经典蒙卡采样.

在程序中我们需要定义的是 Nambu basis 下的 BdG 哈密顿量矩阵 $H_{\mathrm{BdG}}$，它与 BdG 哈密顿量的关系如下：
$$
\hat{H}_{\mathrm{BdG}}\equiv \Psi^\dagger H_{\mathrm{BdG}} \Psi \,,
$$
$$
\Psi=
\begin{pmatrix}
 \vec{c}_\uparrow\\
 \vec{c}_\downarrow^\dagger
\end{pmatrix}
\,, \quad
H_{\mathrm{BdG}}=
\begin{pmatrix}
 h & \Delta\\
 \Delta^\dagger & -h^*
\end{pmatrix}
\,,
$$
$$
h_{ij}=-t_{ij}-\mu\delta_{ij} \,.
$$

---

我们的 HMC 依赖于以下两行核心公式：
$$
H_{\mathrm{HMC}}=
\frac{1}{2m}\sum_{\braket{ij}}|\pi_{ij}|^2
+\frac{\beta}{V}\sum_{\braket{ij}}|\Delta_{ij}|^2
-\mathrm{tr}\, \mathrm{ln}\, \left(1+e^{-\beta H_{\mathrm{BdG}}}\right)
\,.
$$
$$
F_{ij}=-\frac{\beta}{V}\left(\Delta_{ij}-V\braket{ c_{i\uparrow} c_{j\downarrow} - c_{i\downarrow} c_{j\uparrow} }\right) \,.
$$
这里可以看出一个重要的物理意义，在零温极限，也即 $\beta\gg V$ 时，我们的经典蒙卡就回到自洽平均场的结果： $\Delta_{ij}-V\braket{ c_{i\uparrow} c_{j\downarrow} - c_{i\downarrow} c_{j\uparrow}}=0$，这一点可以用于检验程序的正确性。

在 HMC 程序中，哈密顿量演化的运动方程是：
$$
\frac{\partial \Delta_{ij}}{\partial t} = \frac{\partial H_{\mathrm{HMC}}}{\partial \pi_{ij}^*} = \frac{\pi_{ij}}{2m}
$$
$$
\frac{\partial \pi_{ij}}{\partial t} = -\frac{\partial H_{\mathrm{HMC}}}{\partial \Delta_{ij}^*} = F_{ij}
$$
在程序中这通过 Leapfrog 积分实现，注意需要小心处理 $\frac{1}{2m}$ 这种系数，以及注意 $\Delta_{ij}, \pi_{ij}$ 都是复数， 在 Julia 实现中，我们直接使用复数类型来储存这些数组。这里的求导我们使用了自洽的 Wirtinger calculus 规则。

完整的算法流程（伪代码）如下：

```psedo-code
HMC主程序:
	初始化 Δ_ij 复数场
	循环 Nsweep 次:
		初始化 π_ij 复数动量
		计算能量 H_HMC(Δ_ij,π_ij)
		分子动力学演化 Δ_ij,π_ij -> Δ'_ij,π'_ij
		计算能量 H'_HMC(Δ'_ij,π'_ij)
		Metropolis 更新 R=e^{-(H'_HMC-H_HMC)}

计算能量 H_HMC(Δ_ij,π_ij):
	对角化 H_BdG(Δ_ij) 得到 E_n %其实在计算力时已经做了这一步对角化
	H_HMC = 1/(2m) \sum_<ij> |π_ij|^2 %可以提前生成好近邻bond列表，求和时遍历它即可
		+ β/V \sum_<ij> |Δ_ij|^2 
		- \sum_(E_n>0) (β*E_n + 2*log(1+exp(-β*E_n))) %使用log1pexp或者log1p函数避免数值不稳定

分子动力学演化 Δ_ij,π_ij -> Δ'_ij,π'_ij :
	计算力 F_ij(Δ_ij)
	半步动量更新 π_ij = π_ij + δt/2 F_ij
	循环 Nt 次:
		整步场更新 Δ_ij = Δ_ij + δt π_ij / (2m) %这里需要注意系数
		计算力 F_ij(Δ_ij)
		整步动量更新 π_ij = π_ij + δt F_ij %最后一步除外
	半步动量更新 π_ij = π_ij + δt/2 F_ij

计算力 F_ij(Δ_ij) :
	计算密度矩阵 ρ=(e^(β*H_BdG)+1)^(-1) %其实要用对角化来算，而不是求逆
	F_ij = -β/V * ( Δ_ij - V <c_i↑ c_j↓ - c_i↓ c_j↑> ) %从密度矩阵中读出期望值
```
