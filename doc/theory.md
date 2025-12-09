## 模型

The model we use to describe the superconducting state contains hopping terms, interaction terms, and disorder potential terms. The Hamiltonian is
$$
H = -\sum_{ij\sigma} t_{ij}\,(c_{i\sigma}^\dagger c_{j\sigma} + \text{h.c.})
\;+\; \sum_{i\sigma} (w_i - \mu)\, c_{i\sigma}^\dagger c_{i\sigma}
\;+\; J \sum_{\langle ij \rangle} \vec{S}_i \cdot \vec{S}_j .
$$
其中 $w_i$ 就是 disorder potential term.
## Hubbard–Stratonovich Transformation

考虑 d-wave 通道的路径积分
$$
\mathcal{Z}
= \int \mathcal{D}|\Delta|\,\mathcal{D}\theta \;
\exp\!\Bigg(
-\int_{0}^{\beta} d\tau
\Big[
-\sum_{ij,\sigma} t_{ij}\,(c_{i\sigma}^\dagger c_{j\sigma} + \text{h.c.})
+ \sum_{i\sigma} (w_i - \mu)\, c_{i\sigma}^\dagger c_{i\sigma}
$$
$$
\qquad\qquad
-\frac12 \sum_{\langle ij\rangle} |\Delta_{ij}|\,
\big(
e^{-i\theta_{ij}}(c_{i\uparrow} c_{j\downarrow} - c_{i\downarrow} c_{j\uparrow})
+ e^{i\theta_{ij}}(c_{i\downarrow}^\dagger c_{j\uparrow}^\dagger - c_{i\uparrow}^\dagger c_{j\downarrow}^\dagger)
\big)
+ \frac{1}{2J} \sum_{\langle ij\rangle} |\Delta_{ij}|^2
\Big]
\Bigg).
$$

## Bogoliubov–de Gennes Approximation

鞍点近似，忽略虚时方向的量子涨落，取

$$
\Delta_{ij} = J\,\big\langle c_{i\uparrow} c_{j\downarrow} - c_{i\downarrow} c_{j\uparrow} \big\rangle
$$
BdG哈密顿量为：
$$
\hat{H}_{\mathrm{BdG}}
=
-\sum_{ij} t_{ij} (c_i^\dagger c_j + \text{h.c.})
+ \sum_{i\sigma} (w_i - \mu)\, c_{i\sigma}^\dagger c_{i\sigma}
$$
$$
\qquad\qquad
-\frac12 \sum_{\langle ij\rangle} |\Delta_{ij}|\,
\big(
e^{-i\theta_{ij}}(c_{i\uparrow} c_{j\downarrow} - c_{i\downarrow} c_{j\uparrow})
+ e^{i\theta_{ij}}(c_{i\downarrow}^\dagger c_{j\uparrow}^\dagger - c_{i\uparrow}^\dagger c_{j\downarrow}^\dagger)
\big).
$$

配分函数为
$$
\mathcal{Z}
= \int \mathcal{D}|\Delta|\,\mathcal{D}\theta \;
\exp\!\left( -\frac{\beta}{2J} \sum_{\langle ij\rangle} |\Delta_{ij}|^2 \right)
\operatorname{Tr}\exp(-\beta \hat{H}_{\mathrm{BdG}})
$$
$$
= \int \mathcal{D}|\Delta|\,\mathcal{D}\theta \;
\exp\!\left( -\frac{\beta}{2J} \sum_{\langle ij\rangle} |\Delta_{ij}|^2 \right)
\det\!\left( 1 + e^{-\beta H_{\mathrm{BdG}}} \right)^{\frac12}
.
$$
接下来考虑做经典蒙卡，采样 $|\Delta_{ij}|$ 和 $\theta_{ij}$