using DwaveHMC
using LinearAlgebra
using BenchmarkTools

Lx, Ly = 6, 6
p = ModelParameters(Lx, Ly, 1.0, -0.35, -0.5, 1.0, 0.1, 10.0, 1.0, 0.05, 1.0)
s = initialize_state(p)
c = initialize_cache(p)
println("System Size: $(p.N) sites, Matrix Dim: $(2*p.N)")

println("Initializing Static H...")
init_static_H!(c, p, s)  # <--- 必须先调用这个
println("Updating Dynamic H...")
update_H_BdG!(c, p, s)

println("\nBenchmarking update_H_BdG! (Pairing update only)...")
@btime update_H_BdG!($c, $p, $s)

println("\nBenchmarking energy compute (eigen! version)...")
@btime compute_fermion_energy!($c, $p)