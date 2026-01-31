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

定义超流刚度：
$$
\rho_s=\braket{-K_x}-\Lambda_{xx}(q_x=0,q_y\to0,\omega=0)
$$
其中抗磁项：
$$\braket{-K_x}=\frac1N\sum_{i,\sigma}\left[\,t\braket{c_{i,\sigma}^\dagger c_{i+\hat{x},\sigma}}\,+\,t'\braket{c_{i,\sigma}^\dagger c_{i+\hat{x}+\hat{y},\sigma}}\,+\,t'\braket{c_{i,\sigma}^\dagger c_{i+\hat{x}-\hat{y},\sigma}}\,+\text{h.c.}\,\right]$$
流流关联（顺磁项）：
$$\Lambda_{xx}(q,\omega=0) = \frac{1}{N} \sum_{n\ne m}^{2N} \frac{f(E_n) - f(E_m)}{E_m - E_n} |J^x_{nm}(q)|^2$$
$$J^x_{nm}(q)=\bra{n}J^x(q)\ket{m}\,,\quad J^x(q)=\text{i}\,\sum_{i,\sigma}\left[\,t\,c_{i,\sigma}^\dagger c_{i+\hat{x},\sigma}\,+\,t'\,c_{i,\sigma}^\dagger c_{i+\hat{x}+\hat{y},\sigma}\,+\,t'\,c_{i,\sigma}^\dagger c_{i+\hat{x}-\hat{y},\sigma}\,-\text{h.c.}\,\right] \text{e}^{\text{i} q \cdot r_i}$$
数值上，在计算超流刚度顺磁项时，取 $q_x=0, q_y=\frac{2\pi}{L_y}$ .

在具体的程序计算中，抗磁项可以直接显示手写循环来算。而顺磁项需要定义稀疏矩阵用 `BLAS` 来做稀疏矩阵乘法。此外，当 $E_m\approx E_n$ 时还需注意数值稳定性： 
$$ \lim_{E_m \to E_n} \frac{f(E_n) - f(E_m)}{E_m - E_n} = -f'(E_n) =\beta f(E_n) [1 - f(E_n)]$$

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