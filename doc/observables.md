## 物理可观测量

Pair field 的幅度
$$
\Delta_{\text{amplitude}} = \frac{1}{N} \sum_i \frac{|\Delta_{i, i+\hat{x}}| + |\Delta_{i, i+\hat{y}}|}{2}
$$
局部 d-wave
$$
\Delta_{\text{local}} = \frac{1}{N} \sum_i \frac{\left|\Delta_{i, i+\hat{x}} - \Delta_{i, i+\hat{y}} \right|}{2}
$$
整体 d-wave
$$
\Delta_{\text{global}} = \left|\frac{1}{N} \sum_i \frac{\Delta_{i, i+\hat{x}} - \Delta_{i, i+\hat{y}}}{2} \right|
$$
d-wave 结构因子
$$
S_{\Delta} = \left|\frac{1}{N} \sum_i \frac{\Delta_{i, i+\hat{x}} - \Delta_{i, i+\hat{y}}}{2} \right|^2
$$
还需测量电子浓度 $n$ 和空穴浓度 $p$ ，具体计算方法如下：
根据已经对角化得到的 $H_{\text{BdG}}$ 本征值和本征态：
$$
E_n, \quad
U_n = \begin{pmatrix}
u_{n, 1} \\
\vdots \\
u_{n, N} \\
v_{n, 1} \\
\vdots \\
v_{n, N}
\end{pmatrix}
$$
局域电子数为
$$
n_i = \langle c_{i\uparrow}^\dagger c_{i\uparrow} \rangle + \langle c_{i\downarrow}^\dagger c_{i\downarrow} \rangle= \sum_{n=1}^{2N} \left[ |u_{n,i}|^2 f(E_n) + |v_{n,i}|^2 (1 - f(E_n)) \right]
$$
其中 $f(E)$ 是费米-狄拉克分布函数：$f(E) = \frac{1}{e^{\beta E} + 1}$
于是，平均电子数密度（电子浓度）为：
$$
n=\frac1N\sum_{i=1}^{N} \sum_{n=1}^{2N} \left[ |u_{n,i}|^2 f(E_n) + |v_{n,i}|^2 (1 - f(E_n)) \right]
$$
空穴浓度为：
$$p=1-n$$
或者，利用 $H_{\text{BdG}}$ 的粒子空穴对称性，以上计算还可以简化为：
$$
p=\frac1N\sum_{i=1}^{N}\sum_{E_n>0}\left(|u_{n,i}|^2-|v_{n,i}|^2\right) \tanh\left(\frac{\beta E_n}{2}\right)
$$
$$n=1-p$$
除了物理可观测量外，为了方便调试，还需计算蒙卡接受率。

## 超流刚度、电导、态密度与谱函数的测量

在每个蒙卡构型中，对角化得到的 $H_{\text{BdG}}$ 本征值和本征态：
$$
E_n, \quad
U_n = \begin{pmatrix}
u_{1,n} \\
\vdots \\
u_{N,n} \\
v_{1,n} \\
\vdots \\
v_{N,n}
\end{pmatrix}
$$
以下的物理量都基于这些信息，结合费米分布函数 $f(E_n)=1/(e^{\beta E_n}+1)$ 来计算。

很有用的关系：
$$\braket{c_{i,\uparrow}^\dagger c_{j,\uparrow}}=\sum_{n=1}^{2N}u_{i,n}^* u_{j,n} f(E_n)$$
$$\braket{c_{i,\downarrow}^\dagger c_{j,\downarrow}}=\sum_{n=1}^{2N}v_{i,n} v_{j,n}^* \left(1-f(E_n)\right)$$
根据 BdG 哈密顿量的对称性，还有：
$$\braket{c_{i,\uparrow}^\dagger c_{j,\uparrow}}+\braket{c_{i,\downarrow}^\dagger c_{j,\downarrow}}=\sum_{E_n>0}\left(v_{i,n} v_{j,n}^*-u_{i,n}^* u_{j,n}\right) \tanh\left(\frac{\beta E_n}{2}\right)$$

### 超流刚度

正式计算和文献对比时，本文档中的超流刚度指 Meissner stiffness 的静态横向响应：
$$
\rho_s=\braket{-K_x}-\Lambda_{xx}(q_x=0,q_y\to0,\omega=0)
$$
这里电流方向为 $x$，横向动量沿 $y$。在有限尺寸 $L_x\times L_y$ 周期格子上，程序使用最小非零横向动量
$$
q_x=0,\qquad q_y=\frac{2\pi}{L_y}.
$$
这与 Scalapino--White--Zhang、Xiang--Wheatley 以及相关 finite-size BdG 文献中的 Meissner stiffness 定义一致。直接取 $q=0$ 对应的是 uniform twist / flux response，是另一个有限尺寸 estimator，不能和这个 transverse estimator 逐构型混用。

其中抗磁项：
$$\braket{-K_x}=\frac1N\sum_{i,\sigma}\left[\,t\braket{c_{i,\sigma}^\dagger c_{i+\hat{x},\sigma}}\,+\,t'\braket{c_{i,\sigma}^\dagger c_{i+\hat{x}+\hat{y},\sigma}}\,+\,t'\braket{c_{i,\sigma}^\dagger c_{i+\hat{x}-\hat{y},\sigma}}\,+\text{h.c.}\,\right]$$
流流关联（顺磁项）：
$$\Lambda_{xx}(q,\omega=0) = \frac{1}{N} \sum_{n\ne m}^{2N} \frac{f(E_n) - f(E_m)}{E_m - E_n} |J^x_{nm}(q)|^2$$
$$J^x_{nm}(q)=\bra{n}J^x(q)\ket{m}\,,\quad J^x(q)=\text{i}\,\sum_{i,\sigma}\left[\,t\,c_{i,\sigma}^\dagger c_{i+\hat{x},\sigma}\,+\,t'\,c_{i,\sigma}^\dagger c_{i+\hat{x}+\hat{y},\sigma}\,+\,t'\,c_{i,\sigma}^\dagger c_{i+\hat{x}-\hat{y},\sigma}\,-\text{h.c.}\,\right] \text{e}^{\text{i} q \cdot r_i}$$

在具体的程序计算中，抗磁项可以直接显式手写循环来算。顺磁项使用稀疏电流矩阵和 `BLAS` 计算矩阵元。正式的 finite-$q_y$ Kubo 求和显式跳过严格的 $m=n$ 项；但对于 $m\ne n$ 的近简并本征态，仍需使用稳定极限：
$$ \lim_{E_m \to E_n} \frac{f(E_n) - f(E_m)}{E_m - E_n} = -f'(E_n) =\beta f(E_n) [1 - f(E_n)]$$

也就是说，正式输出 `Superfluid_Stiffness` 的定义是：
$$
\rho_s^{T}(L)=
\braket{-K_x}-
\Lambda_{xx}\left(q_x=0,q_y=\frac{2\pi}{L_y},\omega=0\right),
\qquad
\Lambda_{xx}:\sum_{m\ne n}.
$$
这里 $T$ 表示 transverse response。做论文级有限尺寸分析时，应以这个量为准，并在需要时随 $L_y$ 做 $q_y\to0$ 外推。

### Twist 自由能差分 benchmark

除了正式 Kubo 公式，也可以对 Peierls twist 后的 BdG 自由能做有限差分，作为数值 benchmark。这个测量需要额外对角化，生产运行中默认不打开；在 `run_simulation` 中需显式设置 `measure_twist=true`。

单构型的费米子有效作用量取为
$$
S_f(A)=-\sum_{E_n(A)>0}\left[\beta E_n(A)+2\log\left(1+e^{-\beta E_n(A)}\right)\right].
$$
对固定辅助场构型的 uniform twist，可定义
$$
s_1=\frac{S_f(A)-S_f(-A)}{2A},\qquad
s_2=\frac{S_f(A)+S_f(-A)-2S_f(0)}{A^2},
$$
单构型曲率为
$$
\rho_{\mathrm{twist,curv}}=\frac{s_2}{\beta N}.
$$
如果要从 twist 自由能得到 ensemble estimator，需要在采样后用
$$
\rho_{\mathrm{twist,full}}=
\frac{\overline{s_2}-\left(\overline{s_1^2}-\overline{s_1}^2\right)}{\beta N}.
$$

如果要验证 uniform twist 的有限差分实现，应和 $q=0$ 的静态 Kubo 曲率比较，并包含 $m=n$ 极限项：
$$
\Lambda^{\mathrm{static}}_{xx}(q=0)=
\frac{1}{N}\sum_{m,n}R_{mn}|J^x_{nm}(0)|^2,
$$
其中
$$
R_{mn}=
\begin{cases}
\dfrac{f(E_n)-f(E_m)}{E_m-E_n},&E_m\ne E_n,\\[0.8em]
\beta f(E_n)\left[1-f(E_n)\right],&E_m=E_n.
\end{cases}
$$
这个 $m=n$ 项来自能级移动导致的 occupation response，有限差分自由能曲率会自动包含它。因此，uniform twist 不应拿来和正式的 finite-$q_y$、$m\ne n$ Kubo stiffness 逐构型比较。

为了做 finite-$q_y$ 层面的调试，可以使用横向调制的 x 方向矢势
$$
A_x(y)=\sqrt{2}A\cos\left(q_y(y-1)+\phi\right),
\qquad q_y=\frac{2\pi}{L_y}.
$$
分别取 $\phi=0$ 和 $\phi=-\pi/2$ 得到 cosine/sine 两个曲率，再取平均：
$$
\rho_{\mathrm{twist},q_y}=\frac12\left(\rho_{\cos}+\rho_{\sin}\right).
$$
这对应 `transport.csv` 中显式打开 `measure_twist=true` 后输出的
`Twist_Qy`、`Twist_Qy_Rho_Curv_Cos`、`Twist_Qy_Rho_Curv_Sin`、`Twist_Qy_Rho_Curv_Avg`。

有限差分 twist 曲率是自由能的 full curvature，会包含与严格 $m=n$ 极限相对应的对角响应。因此：

| 名称 | 动量 | 严格 $m=n$ 项 | 用途 |
| --- | --- | --- | --- |
| `Superfluid_Stiffness` | $q_x=0,\ q_y=2\pi/L_y$ | 跳过；保留 $m\ne n$ 近简并极限 | 正式 finite-size Meissner stiffness，与文献对比 |
| `Twist_Qy_Rho_Curv_Avg` | 横向调制 $q_y=2\pi/L_y$ | 有限差分自动包含 | 调试 full curvature；可和包含 $m=n$ 的 Kubo 诊断量比较 |
| uniform twist curvature | $q=0$ uniform flux | 有限差分自动包含 | 调试 helicity/twist stiffness；可和 $q=0$ 且包含 $m=n$ 的静态 Kubo 比较 |

如果需要把 twist 结果画到正式 `Superfluid_Stiffness` 的同一张图上，应明确标注它是 full curvature 或做对角项修正。对 finite-$q_y$ 的情况，若
$$
\Lambda_{\mathrm{diag}}(q_y)=
\frac{1}{N}\sum_n \beta f(E_n)\left[1-f(E_n)\right]|J^x_{nn}(q_y)|^2,
$$
则可构造调试量
$$
\rho_{\mathrm{twist,offdiag}}=
\rho_{\mathrm{twist},q_y}+\Lambda_{\mathrm{diag}}(q_y),
$$
用于和 `Superfluid_Stiffness` 的 $m\ne n$ Kubo 定义做逐构型对比。这个量只是诊断工具，不作为正式 observable。
对应的 CSV 列为 `Twist_Qy_Lambda_Diag` 和 `Twist_Qy_Rho_OffdiagCorrected`。

### 光电导与直流电导

流流关联
$$\Lambda_{xx}(q=0,\omega) = \frac{1}{N} \sum_{n\ne m}^{2N} \frac{f(E_n) - f(E_m)}{E_m - E_n-\omega-\text{i}\eta} |J^x_{nm}(0)|^2$$
根据 $\frac{1}{x - i\eta} = P(\frac{1}{x}) + i\pi\delta(x)$ 得 
$$ \text{Im } \Lambda_{xx}(\omega) = \pi \frac{1}{N} \sum_{n\ne m}^{2N} (f(E_n) - f(E_m)) |J^x_{nm}(0)|^2 \delta(E_m - E_n - \omega) $$
光电导定义为 $\text{Re}\,\sigma_{xx}(\omega)=\frac{1}{\text{i}\omega}\Lambda_{xx}(q=0,\omega)=\frac{1}{\omega}\text{Im}\,\Lambda_{xx}(q=0,\omega)$，也即：
$$ \text{Re}\,\sigma_{xx}(\omega) = \frac{\pi}{N \omega} \sum_{n\ne m}^{2N} (f(E_n) - f(E_m)) |J^x_{nm}(0)|^2 \delta(\omega - (E_m - E_n)) $$
取直流极限为（Kubo–Greenwood公式）：
$$ \sigma_{\text{DC}}=\text{Re}\,\sigma_{xx}(\omega\to0)=\frac{\pi}{N} \sum_{n\ne m}^{2N} \left( -f'(E_n) \right) |J^x_{nm}(0)|^2 \delta(E_m - E_n) $$
其中 $-f'(E_n) =\beta f(E_n) [1 - f(E_n)]$ 。注意这只包含了 regular part，没有包含超流导致的发散项 $\pi \rho_s \delta(\omega)$ 。

数值上取 $\delta$ 函数为：
$$ \delta(E) = \frac{1}{\pi} \frac{\eta}{E^2 + \eta^2} $$
其中 $\eta$ 是手动设置的小量。可以取 $\eta\approx W/N$，也即平均能级间距，其中 $W$ 是能带宽度。

在程序中，我们既要计算 $\sigma_{\text{DC}}$ 也要计算 $\text{Re}\,\sigma_{xx}(\omega)$。其中 $\omega\in[\eta,t]$，间隔为手动设置的 $\Delta\omega$。

### 态密度与谱函数

态密度（DOS）
$$ N(\omega) = \frac{1}{N} \sum_{n=1}^{2N} \left( \sum_{i=1}^{N} |u_{i,n}|^2 \right) \delta(\omega - E_n) $$
其中谱权重为 $W_n\equiv \sum_{i=1}^{N} |u_{i,n}|^2$ 。为了方便后续灵活处理，其实还可以输出整个 $W_n$ 数组和 $E_n$ 数组。

谱函数
$$ A(\mathbf{k}, \omega) = \sum_{n=1}^{2N} |\tilde{u}_{n}(\mathbf{k})|^2 \delta(\omega - E_n) $$
其中动量空间的波函数在程序中用 FFT 计算
$$ \tilde{u}_{n}(\mathbf{k}) = \frac{1}{\sqrt{N}} \sum_{j=1}^{N} e^{-i \mathbf{k} \cdot \mathbf{r}_j} u_{j,n} $$
同样，也可以输出完整的谱权重 $W_{n}(\mathbf{k}) = |\tilde{u}_{n}(\mathbf{k})|^2$ 数组。

### 赝能隙

为了看 pseudogap，我们关注 antinode 处的谱函数 
$$A_{\mathrm{AN}}(\omega)\equiv \frac12 \left[A(\mathbf{k}=(\pi,0), \omega)+A(\mathbf{k}=(0,\pi), \omega)\right]$$
以及局部的 d-wave 配对强度
$$
\Delta_{\text{localpair}} = \frac{J}{N} \sum_i 
\left| \frac{\braket{ c_{i\uparrow} c_{i+\hat{x}\downarrow} - c_{i\downarrow} c_{i+\hat{x}\uparrow}} - \braket{ c_{i\uparrow} c_{i+\hat{y}\downarrow} - c_{i\downarrow} c_{i+\hat{y}\uparrow}}}{2} \right|
$$

## Binder ratio

之前的测量是这样的
$$
|\Delta|_{\text{global}} = \overline{\frac{J}{N} \left|\sum_i 
\frac{\braket{ c_{i\uparrow} c_{i+\hat{x}\downarrow} - c_{i\downarrow} c_{i+\hat{x}\uparrow}} - \braket{ c_{i\uparrow} c_{i+\hat{y}\downarrow} - c_{i\downarrow} c_{i+\hat{y}\uparrow}}}{2} \right|}
$$
$$
|\Delta|_{\text{local}} = \overline{\frac{J}{N} \sum_i 
\left| \frac{\braket{ c_{i\uparrow} c_{i+\hat{x}\downarrow} - c_{i\downarrow} c_{i+\hat{x}\uparrow}} - \braket{ c_{i\uparrow} c_{i+\hat{y}\downarrow} - c_{i\downarrow} c_{i+\hat{y}\uparrow}}}{2} \right|}
$$
其中 $\overline{\cdot}$ 表示对辅助场的蒙卡平均，$\braket{\cdot}$ 表示费米子算符求迹，$\sum_i$ 表示对 $N$  个空间格点求和。

下面定义 Binder ratio。

设复随机变量 $z = \mu + \delta z$ 服从复高斯分布，均值为 $\mu$，总方差为 $\sigma^2$。那么 $|z|^2$ 的期望值为 $|\mu|^2+\sigma^2$;  $|z|^4$ 的期望值为 $|\mu|^4+4|\mu|^2\sigma^2+2\sigma^4$.

记配对算符为
$$
d_i = \frac{J}{2}\left( c_{i\uparrow} c_{i+\hat{x}\downarrow} - c_{i\downarrow} c_{i+\hat{x}\uparrow} -  c_{i\uparrow} c_{i+\hat{y}\downarrow} - c_{i\downarrow} c_{i+\hat{y}\uparrow} \right)
$$
$$
D = \frac{1}{N}\sum_i d_i
$$
定义 Binder ratio 
$$
B_{\text{global}}=1-\frac{\overline{|\braket{D}|^4}}{2\,\left(\overline{|\braket{D}|^2}\right)^2}
$$
$$
B_{\text{local}} = 1-\frac{\overline{\frac1N\sum_{i=1}^N \left| \braket{d_i} \right|^4}}{2 \left( \overline{\frac1N\sum_{i=1}^N \left| \braket{d_i} \right|^2} \right)^2}
$$
有配对时 $B=0.5$；无配对时 $B=0$.

## 相位关联

为了与 Phys. Rev. Lett. **94**, 217001 和 Phys. Rev. B **84**, 024522 做比对，我们添加几个可观测量，以显示配对场（被蒙卡采样的经典场）的相位关联：
$$
S(l_x,l_y)=\frac1N\sum_i \braket{e^{\mathrm{i}\theta_i^x}e^{-\mathrm{i}\theta_{i+l}^x}}=\frac1N\sum_i\left\langle\mathrm{arg}\left(\Delta_{i,i+\hat{x}}\Delta_{i+l,i+l+\hat{x}}^*\right)\right\rangle
$$
$$
F(l_x,l_y)=\frac1N\sum_i \braket{e^{\mathrm{i}\theta_i^x}e^{-\mathrm{i}\theta_{i+l}^y}}=\frac1N\sum_i\left\langle\mathrm{arg}\left(\Delta_{i,i+\hat{y}}\Delta_{i+l,i+l+\hat{y}}^*\right)\right\rangle
$$
文献 Phys. Rev. Lett. **94**, 217001 计算的是 $S(\frac{L}{2},\frac{L}{2})$, $S(\frac{L}{2},0)$, $F(0,0)$
文献 Phys. Rev. B **84**, 024522 计算的是 $S(\frac{L}{2},0)$
我们也需要把这些量输出出来。
