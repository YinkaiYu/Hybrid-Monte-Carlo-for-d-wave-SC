## HMC程序笔记

每个 $\{\Delta_{ij}\}$ 构型的BdG哈密顿量：
$$
\hat{H}_{\mathrm{BdG}}
=
-\sum_{ij\sigma} t_{ij}\,(c_{i\sigma}^\dagger c_{j\sigma} + \text{h.c.})
+ \sum_{i\sigma} (w_i - \mu)\, c_{i\sigma}^\dagger c_{i\sigma}
-\frac12 \sum_{\langle ij\rangle}
\big(
\Delta_{ij}^*(c_{i\uparrow} c_{j\downarrow} - c_{i\downarrow} c_{j\uparrow})
+ \Delta_{ij}(c_{i\downarrow}^\dagger c_{j\uparrow}^\dagger - c_{i\uparrow}^\dagger c_{j\downarrow}^\dagger)
\big).
$$
where $t_{ij}$ is the hopping integral between sites $i$ and $j$ on a square lattice which we take (to produce a cuprate-like Fermi surface) to be $t_{ij}=1$ between nearest-neighbor sites, $t_{ij}=-0.35$ between second-neighbor sites, and $t_{ij}=0$ for all further neighbors. The random potentials, $w_j$, represent the effects of disorder:  on a randomly chosen fraction $n_{\rm imp}$ of sites we set $w_j=w>0$, with $w_j=0$ on all other sites.

我们要用HMC来完成对复数值场 $\{\Delta_{ij}\}$ 的经典蒙卡采样.

在程序中我们需要定义的是 Nambu basis 下的 BdG 哈密顿量矩阵 $H_{\mathrm{BdG}}$，它与 BdG 哈密顿量的关系如下：
$$
\hat{H}_{\mathrm{BdG}}\equiv\frac12 \Psi^\dagger H_{\mathrm{BdG}} \Psi \,,
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
h_{ij}=-t_{ij}+(w_i-\mu)\delta_{ij} \,.
$$
我们的 HMC 依赖于以下两行核心公式：
$$
H_{\mathrm{HMC}}=
\frac{1}{2m}\sum_{\braket{ij}}|\pi_{ij}|^2
+\frac{\beta}{2J}\sum_{\braket{ij}}|\Delta_{ij}|^2
- \mathrm{tr}\, \mathrm{ln}\, \left(1+e^{-\beta H_{\mathrm{BdG}}}\right)
\,.
$$
$$
F_{ij}=-\frac{\beta}{2J}\left(\Delta_{ij}-J\braket{ c_{i\uparrow} c_{j\downarrow} - c_{i\downarrow} c_{j\uparrow} }\right) \,.
$$
这里可以看出一个重要的物理意义，在零温极限，也即 $\beta\gg J$ 时，我们的经典蒙卡就回到自洽平均场的结果： $\Delta_{ij}-J\braket{ c_{i\uparrow} c_{j\downarrow} - c_{i\downarrow} c_{j\uparrow}}=0$，这一点可以用于检验程序的正确性。

在 HMC 程序中，哈密顿量演化的运动方程是：
$$
\frac{\partial \Delta_{ij}}{\partial t} = \frac{\partial H_{\mathrm{HMC}}}{\partial \pi_{ij}^*} = \frac{\pi_{ij}}{2m}
$$
$$
\frac{\partial \pi_{ij}}{\partial t} = -\frac{\partial H_{\mathrm{HMC}}}{\partial \Delta_{ij}^*} = F_{ij}
$$
在程序中这通过 Leapfrog 积分实现，注意需要小心处理 $\frac{1}{2m}$ 这种系数，以及注意 $\Delta_{ij}, \pi_{ij}$ 都是复数， 在 Julia 实现中，我们直接使用复数类型来储存这些数组。这里的求导我们使用了自洽的 Wirtinger calculus 规则，在形式上和实数的运动方程有所不同，但都是自洽的。（若为实数场，常见的 HMC 约定其实是  $F_{ij}=-\frac{\beta}{J}\left(\Delta_{ij}-J\braket{ c_{i\uparrow} c_{j\downarrow} - c_{i\downarrow} c_{j\uparrow} }\right) \,, \frac{\partial \Delta_{ij}}{\partial t}= \frac{\pi_{ij}}{m}\,.$）

还有，在生成动量的高斯分布时，也需要注意小心处理 $\frac{1}{2m}$ 这种系数。通常，我们可以设置 $m=1$ 直接作为动量高斯分布的方差。 

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
		+ β/(2J) \sum_<ij> |Δ_ij|^2 
		- \sum_(E_n>0) 2*log(2*cosh(β*E_n/2)) %使用log1pexp或者log1p函数避免数值不稳定

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
	F_ij = -β/(2J) ( Δ_ij - J <c_i↑ c_j↓ - c_i↓ c_j↑> ) %从密度矩阵中读出期望值
```

在实现时，需要注意：
- 在计算密度矩阵时，不要直接使用矩阵求逆，而应当对角化 `H_BdG` 矩阵，得到然后用本征值、本征向量、费米分布函数来构造。而且本征值其实在计算能量时也可以复用。
- 实际上不需要算出并储存完整的密度矩阵，只需要计算我们需要的 `<c_i↑ c_j↓ - c_i↓ c_j↑>` 矩阵元就够了。
- 善用Julia的向量操作。
- `H_BdG` 要显式封装为厄米矩阵，便于对角化。注意不要反复构造 `H_BdG` ，提前预分配好内存，更新时原地更新，减少GC压力。
- 矩阵运算能使用 LAPACK/BLAS 就尽量用。不要低估 Garbage Collection 压力。在性能上 `BLAS`  几乎永远是最佳的，除非 Julia 自带含有`!`的函数（in-place functions）。
- 包括数组/矩阵切片也是，尽量使用 `view` 而不是直接用Julia的默认切片，以减小GC压力。
- 若程序中含有大量矩阵向量运算，可以提前申请中间量的内存，重复利用，从而降低 allocation 和 GC 压力。
- 可以同时跑多条马尔可夫链，使用MPI并行，或者 `Distributed `标准库。
- 近邻、次近邻列表可以提前构建好，包括pairing项的index列表，也提前构建好。
- 由于 `H_BdG` 的粒子空穴对称性，其本征值是正负成对出现的，可以只对正能量求和，也许更数值稳定。建议使用log1pexp或者log1p函数避免数值不稳定。