using LinearAlgebra
using InteractiveUtils # 用于查看地址

N = 1000
# 创建一个复数矩阵
A = rand(ComplexF64, N, N)
# 构造 Hermitian wrapper
H_wrap = Hermitian(A, :U)

println("Original Matrix Memory Address: ", pointer(H_wrap.data))

# 执行 eigen!
# Codex 说 vecs 会指向和 H_wrap.data 相同的地址
vals, vecs = eigen!(H_wrap)

println("Eigenvectors Memory Address:    ", pointer(vecs))

println("Are they the same? ", pointer(H_wrap.data) == pointer(vecs))