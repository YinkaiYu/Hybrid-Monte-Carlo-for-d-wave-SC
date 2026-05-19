# Notation 说明

当前程序采用 t-V 口径。输入参数 `V` 是物理耦合，也就是 Phys. Rev. B **84**, 024522 图注中的 $V$。

## 当前程序的变量

程序采样的 $\Delta_{ij}$ 直接进入 BdG 矩阵：
$$
H_{\rm BdG}=
\begin{pmatrix}
h & \Delta\\
\Delta^\dagger & -h^*
\end{pmatrix}.
$$

辅助场权重为
$$
S_\Delta=\frac{2\beta}{V}\sum_{\langle ij\rangle}|\Delta_{ij}|^2.
$$
定义内部有效配对耦合
$$
g_{\rm pair}=\frac{V}{2},
$$
则
$$
S_\Delta=\frac{\beta}{g_{\rm pair}}\sum_{\langle ij\rangle}|\Delta_{ij}|^2.
$$

自洽方程和 force 分别为
$$
\Delta_{ij}-g_{\rm pair}P_{ij}=0,
$$
$$
F_{ij}=-\frac{\beta}{g_{\rm pair}}\left(\Delta_{ij}-g_{\rm pair}P_{ij}\right).
$$

## 与 t-J 口径的换算

t-J 口径常把 BdG 非对角块写成 $\Delta_J/2$，并使用
$$
S_\Delta^{(J)}=\frac{\beta}{2J}\sum|\Delta_J|^2,
\qquad
\Delta_J-JP=0.
$$

与当前程序比较时使用
$$
J=V,
\qquad
\Delta_J=2\Delta_{\rm tV}.
$$
因此直接比较辅助场幅度时要记住 $\Delta_J$ 比当前程序的 BdG gap 大 2 倍；比较 stiffness、conductivity、DOS、谱函数等 BdG 观测量时则不需要额外缩放。
