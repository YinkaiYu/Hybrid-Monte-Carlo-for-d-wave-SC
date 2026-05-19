# t-V 模型理论口径

当前程序模拟最近邻 singlet pairing 通道的 t-V 有效模型。输入参数 `V` 是物理耦合，也对应 Phys. Rev. B **84**, 024522 中图注使用的 $V$。程序中采样的 $\Delta_{ij}$ 是直接进入 BdG 矩阵的 gap 矩阵元。

## BdG 哈密顿量

对每个静态辅助场构型 $\{\Delta_{ij}\}$，程序使用
$$
\hat H_{\rm BdG}(\Delta)=
-\sum_{ij\sigma} t_{ij}(c^\dagger_{i\sigma}c_{j\sigma}+{\rm h.c.})
+\sum_{i\sigma}(w_i-\mu)c^\dagger_{i\sigma}c_{i\sigma}
-\sum_{\langle ij\rangle}
\left[
\Delta_{ij}^{*}(c_{i\uparrow}c_{j\downarrow}-c_{i\downarrow}c_{j\uparrow})
+\Delta_{ij}(c^\dagger_{j\downarrow}c^\dagger_{i\uparrow}
-c^\dagger_{j\uparrow}c^\dagger_{i\downarrow})
\right].
$$

在 Nambu basis
$$
\Psi=
\begin{pmatrix}
\vec c_\uparrow\\
\vec c_\downarrow^\dagger
\end{pmatrix}
$$
下，矩阵形式为
$$
H_{\rm BdG}=
\begin{pmatrix}
h & \Delta\\
\Delta^\dagger & -h^*
\end{pmatrix},
\qquad
h_{ij}=-t_{ij}+(w_i-\mu)\delta_{ij}.
$$

这里的 BdG 非对角块直接是 $\Delta$，没有额外的 $1/2$。

## 辅助场权重

当前程序的经典场权重为
$$
S_{\Delta}=\frac{2\beta}{V}\sum_{\langle ij\rangle}|\Delta_{ij}|^2.
$$
为了避免在代码中反复写 $V/2$，定义内部有效配对耦合
$$
g_{\rm pair}\equiv \frac{V}{2}.
$$
于是
$$
S_{\Delta}=\frac{\beta}{g_{\rm pair}}\sum_{\langle ij\rangle}|\Delta_{ij}|^2.
$$

令
$$
P_{ij}\equiv
\left\langle c_{i\uparrow}c_{j\downarrow}
-c_{i\downarrow}c_{j\uparrow}\right\rangle,
$$
低温 saddle/self-consistency 条件为
$$
\Delta_{ij}-g_{\rm pair}P_{ij}=0,
\qquad
\text{即}\qquad
\Delta_{ij}-\frac{V}{2}P_{ij}=0.
$$

代码中 `pairing_coupling(p)` 返回 $g_{\rm pair}=V/2$。HMC force、能量、自洽残差、fermionic pair observable 和推荐步长都使用这个内部耦合。

## t-J 口径换算

主分支的 t-J 程序使用另一种辅助场归一化：
$$
H_{\rm BdG}^{(J)}=
\begin{pmatrix}
h & \Delta_J/2\\
\Delta_J^\dagger/2 & -h^*
\end{pmatrix},
\qquad
S_{\Delta}^{(J)}=\frac{\beta}{2J}\sum|\Delta_J|^2.
$$
它的自洽方程写作
$$
\Delta_J-JP=0.
$$

与当前 t-V 程序比较时使用
$$
J=V,\qquad
\Delta_J=2\Delta_{\rm tV}.
$$
在这个变量变换下，两边的 BdG 矩阵和辅助场权重相同；stiffness、conductivity、DOS、谱函数等只依赖 BdG 矩阵的物理量也应一致。
