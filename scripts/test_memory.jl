using LinearAlgebra
using InteractiveUtils # 用于查看地址
using BenchmarkTools
using LinearAlgebra.LAPACK

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

# 检查数值是否一致 (即便地址不同，看看数值有无被写入同一内容)
max_diff = maximum(abs.(vecs .- H_wrap.data))
println("Max |vecs - H_wrap.data|: ", max_diff)

# 只接收一个返回值时，eigen! 依然会计算并返回一个 Eigen 对象
A2 = copy(A)
alloc_one = @allocated begin
    eig_obj_tmp = eigen!(Hermitian(copy(A2), :U))
    # 只为测分配，不保留结果
    nothing
end
eig_obj = eigen!(Hermitian(A2, :U))
println("Single-return eigen!: pointer(eig_obj.vectors) == pointer(A2)? ",
    pointer(eig_obj.vectors) == pointer(A2))
println("Max |eig_obj.vectors - A2|: ", maximum(abs.(eig_obj.vectors .- A2)))

# 比较返回两个值时的分配量
A3 = copy(A)
alloc_two = @allocated begin
    vals_tmp, vecs_tmp = eigen!(Hermitian(copy(A3), :U))
    nothing
end

# 比较eigen而非eigen!的分配量
A3 = copy(A)
alloc_eigen = @allocated begin
    vals_tmp, vecs_tmp = eigen(Hermitian(copy(A3), :U))
    nothing
end

# 比较LAPACK函数的分配量
A4 = copy(A)
alloc_lapack = @allocated begin
    vals_tmp, info = LAPACK.syevd!('V', 'U', A4)
    nothing
end

# 检查数值是否一致
vals, vecs = eigen!(H_wrap)
vals_lapack, info = LAPACK.syevd!('V', 'U', H_wrap.data)
max_diff_vals = maximum(abs.(vals .- vals_lapack))
max_diff_vecs = maximum(abs.(vecs .- H_wrap.data))
println("Max |vals - vals_lapack|: ", max_diff_vals)
println("Max |vecs - vecs_lapack|: ", max_diff_vecs)

println("Allocations (bytes) two-return eigen!: ", alloc_two)
println("Allocations (bytes) single-return eigen!: ", alloc_one)
println("Allocations (bytes) eigen (non-destructive): ", alloc_eigen)
println("Allocations (bytes) LAPACK.syevd!: ", alloc_lapack)
