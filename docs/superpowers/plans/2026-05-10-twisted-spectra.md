# Twisted Spectra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in twisted-boundary-condition spectral measurements with `dos_AN_patch`, while preserving current default simulation and output behavior.

**Architecture:** First make the test entry point reliable with a default `runtests.jl` that excludes legacy benchmark scripts. Then add TBC as a spectra-only measurement in `src/TwistedSpectra.jl`, refactor the existing heavy measurement into transport and untwisted spectra helpers, and let `run_simulation` choose the old or TBC spectra path by parameters.

**Tech Stack:** Julia, `Test`, `LinearAlgebra`, `FFTW`, `JLD2`, existing `DwaveHMC` module APIs.

---

## File Structure

- Create `test/runtests.jl`: stable default test entry for `Pkg.test()`.
- Create `test/test_core_smoke.jl`: small deterministic core smoke tests replacing stale constructor/HMC smoke scripts for the default suite.
- Create `test/test_twisted_spectra.jl`: TBC Hamiltonian, measurement, normalization, and supercell-equivalence tests.
- Create `test/test_simulation_tbc.jl`: tiny TBC-enabled `run_simulation` output test.
- Create `test/benchmark_twisted_spectra.jl`: opt-in timing/allocation benchmark, not included by `runtests.jl`.
- Create `src/TwistedSpectra.jl`: TBC result type, TBC Hamiltonian builder, effective-grid helpers, antinode patch selection, and `measure_twisted_spectra`.
- Modify `src/DwaveHMC.jl`: include `TwistedSpectra.jl` and export `measure_twisted_spectra`.
- Modify `src/Observables.jl`: split heavy measurement into internal transport and untwisted spectra helpers without changing `measure_transport_and_spectra` behavior.
- Modify `src/Simulation.jl`: add TBC parameters, metadata, branching, and optional `dos_AN_patch` accumulation/writing.
- Modify `scripts/process_spectra.jl`, `scripts/batch_process_spectra.jl`, and `projectHPC/example/batch_process_spectra.jl`: read effective grid metadata and optional `dos_AN_patch` compatibly.
- Modify `doc/observables.md`: document spectra TBC, `dos_AN_patch`, and the fact that TBC is not used for HMC or transport.

### Task 1: Default Test Harness

**Files:**
- Create: `test/runtests.jl`
- Create: `test/test_core_smoke.jl`

- [ ] **Step 1: Add stable `Pkg.test()` entry**

Create `test/runtests.jl`:

```julia
using Test

@testset "DwaveHMC default test suite" begin
    include("test_core_smoke.jl")
    include("test_twist_stiffness.jl")

    if get(ENV, "DWAVEHMC_RUN_SIMULATION_TESTS", "0") == "1"
        include("test_simulation.jl")
    end
end
```

Rationale: `test/` currently contains benchmarks and historical smoke scripts. The default suite should include deterministic tests only; the existing `test_simulation.jl` remains opt-in because it is a longer end-to-end run.

- [ ] **Step 2: Add deterministic core smoke tests**

Create `test/test_core_smoke.jl`:

```julia
using Test
using Random
using LinearAlgebra
using DwaveHMC

@testset "Core model/cache/HMC smoke" begin
    Random.seed!(1234)

    p = ModelParameters(4, 4, 1.0, -0.35, -0.5, 0.0, 0.0, 5.0, 1.0, 1.0;
                        η=0.25, Δω=0.25, ω_max=2.0)
    state = initialize_state(p)
    cache = initialize_cache(p)

    @test p.N == 16
    @test size(state.Δ) == (p.N, 2)
    @test size(state.π) == (p.N, 2)
    @test size(cache.H_base) == (2p.N, 2p.N)
    @test size(cache.U) == (2p.N, 2p.N)
    @test length(cache.E_n) == 2p.N
    @test size(cache.ak_map) == (p.Lx, p.Ly)

    init_static_H!(cache, p, state)
    H_static = Matrix(Hermitian(cache.H_base, :U))
    @test isapprox(H_static, H_static'; atol=1e-12, rtol=1e-12)

    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)
    @test issorted(cache.E_n)
    @test all(isfinite, cache.E_n)
    @test isfinite(compute_total_energy(cache, p, state))

    compute_forces!(cache, p, state)
    @test all(z -> isfinite(real(z)) && isfinite(imag(z)), cache.forces)

    Random.seed!(5678)
    acc, dH = hmc_sweep!(cache, p, state; Nt=1, dt=1.0e-3)
    @test acc isa Bool
    @test isfinite(dH)
    @test all(z -> isfinite(real(z)) && isfinite(imag(z)), state.Δ)
end
```

- [ ] **Step 3: Verify the new default suite**

Run:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Expected: PASS. If `test_twist_stiffness.jl` is too slow locally, run the targeted harness first:

```bash
julia --project test/runtests.jl
```

Expected: PASS.

- [ ] **Step 4: Commit test harness**

```bash
git add test/runtests.jl test/test_core_smoke.jl
git commit -m "补齐默认测试入口"
```

### Task 2: TBC Tests First

**Files:**
- Create: `test/test_twisted_spectra.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Add TBC tests to default harness**

Modify `test/runtests.jl` to include the TBC test file after the existing stiffness test:

```julia
using Test

@testset "DwaveHMC default test suite" begin
    include("test_core_smoke.jl")
    include("test_twist_stiffness.jl")
    include("test_twisted_spectra.jl")

    if get(ENV, "DWAVEHMC_RUN_SIMULATION_TESTS", "0") == "1"
        include("test_simulation.jl")
    end
end
```

- [ ] **Step 2: Create TBC test helpers**

Create `test/test_twisted_spectra.jl` with these imports and helpers:

```julia
using Test
using Random
using LinearAlgebra
using DwaveHMC

@inline test_site_index_xy(x::Int, y::Int, Lx::Int, Ly::Int) =
    (mod1(y, Ly) - 1) * Lx + mod1(x, Lx)

function test_set_hermitian_pair!(H::Matrix{ComplexF64},
                                  row::Int,
                                  col::Int,
                                  val::ComplexF64)
    if row <= col
        H[row, col] = val
    else
        H[col, row] = conj(val)
    end
    return nothing
end

function test_add_hop!(H::Matrix{ComplexF64},
                       N::Int,
                       i::Int,
                       j::Int,
                       h::ComplexF64)
    test_set_hermitian_pair!(H, i, j, h)
    test_set_hermitian_pair!(H, i + N, j + N, -conj(h))
    return nothing
end

function build_repeated_supercell_H!(H::Matrix{ComplexF64},
                                     p::ModelParameters,
                                     state::SimulationState,
                                     Ltw::Int)
    Lx_eff = p.Lx * Ltw
    Ly_eff = p.Ly * Ltw
    N_eff = Lx_eff * Ly_eff
    fill!(H, 0.0 + 0.0im)

    @inbounds for y in 1:Ly_eff, x in 1:Lx_eff
        i_eff = test_site_index_xy(x, y, Lx_eff, Ly_eff)
        i_small = test_site_index_xy(x, y, p.Lx, p.Ly)

        onsite = state.disorder_pot[i_small] - state.μ_eff
        H[i_eff, i_eff] = onsite
        H[i_eff + N_eff, i_eff + N_eff] = -onsite

        for (dx, dy, tt) in ((1, 0, p.t), (0, 1, p.t),
                             (1, 1, p.tp), (1, -1, p.tp))
            j_eff = test_site_index_xy(x + dx, y + dy, Lx_eff, Ly_eff)
            test_add_hop!(H, N_eff, i_eff, j_eff, -tt + 0.0im)
        end

        jx_eff = test_site_index_xy(x + 1, y, Lx_eff, Ly_eff)
        jy_eff = test_site_index_xy(x, y + 1, Lx_eff, Ly_eff)
        H[i_eff, jx_eff + N_eff] = state.Δ[i_small, 1]
        H[jx_eff, i_eff + N_eff] = state.Δ[i_small, 1]
        H[i_eff, jy_eff + N_eff] = state.Δ[i_small, 2]
        H[jy_eff, i_eff + N_eff] = state.Δ[i_small, 2]
    end

    return nothing
end

function setup_tbc_fixture(; Lx::Int=4, Ly::Int=4)
    Random.seed!(20260510)
    p = ModelParameters(Lx, Ly, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                        η=0.25, Δω=0.25, ω_max=2.0)
    state = initialize_state(p)
    cache = initialize_cache(p)

    @inbounds for i in 1:p.N
        state.disorder_pot[i] = 0.03 * sin(0.37 * i)
        state.Δ[i, 1] = 0.11 * cis(0.19 * i)
        state.Δ[i, 2] = -0.08 * cis(0.23 * i + 0.4)
    end

    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)
    return p, state, cache
end
```

- [ ] **Step 3: Add failing tests for low-level TBC helpers**

Append:

```julia
@testset "TBC Hamiltonian builder" begin
    p, state, cache = setup_tbc_fixture()
    dim = 2 * p.N
    H = zeros(ComplexF64, dim, dim)

    for nx in 0:1, ny in 0:1
        qx = 2π * nx / 2
        qy = 2π * ny / 2
        DwaveHMC.build_tbc_H_BdG!(H, p, state, qx, qy)
        Hfull = Matrix(Hermitian(H, :U))
        @test isapprox(Hfull, Hfull'; atol=1e-12, rtol=1e-12)
        @test all(isfinite, real.(Hfull))
        @test all(isfinite, imag.(Hfull))
    end

    DwaveHMC.build_tbc_H_BdG!(H, p, state, 0.0, 0.0)
    @test isapprox(Matrix(Hermitian(H, :U)),
                   Matrix(Hermitian(cache.H_base, :U));
                   atol=1e-12, rtol=1e-12)
end
```

- [ ] **Step 4: Add failing supercell-equivalence test**

Append:

```julia
@testset "TBC sectors reproduce repeated supercell spectrum" begin
    p, state, _ = setup_tbc_fixture(Lx=3, Ly=3)
    Ltw = 2
    dim_small = 2 * p.N
    dim_super = 2 * p.N * Ltw^2

    sector_vals = Float64[]
    H_sector = zeros(ComplexF64, dim_small, dim_small)
    for nx in 0:Ltw-1, ny in 0:Ltw-1
        qx = 2π * nx / Ltw
        qy = 2π * ny / Ltw
        DwaveHMC.build_tbc_H_BdG!(H_sector, p, state, qx, qy)
        append!(sector_vals, eigvals(Hermitian(H_sector, :U)))
    end

    H_super = zeros(ComplexF64, dim_super, dim_super)
    build_repeated_supercell_H!(H_super, p, state, Ltw)
    super_vals = eigvals(Hermitian(H_super, :U))

    sort!(sector_vals)
    sort!(super_vals)
    @test length(sector_vals) == length(super_vals)
    @test isapprox(sector_vals, super_vals; atol=1e-9, rtol=1e-9)
end
```

- [ ] **Step 5: Verify tests fail for missing TBC builder**

Run:

```bash
julia --project test/test_twisted_spectra.jl
```

Expected: FAIL with `UndefVarError: build_tbc_H_BdG! not defined`.

- [ ] **Step 6: Leave failing tests uncommitted**

Do not commit the red state. Task 3 commits these tests together with the builder once the low-level tests pass.

### Task 3: TBC Helpers And Hamiltonian Builder

**Files:**
- Create: `src/TwistedSpectra.jl`
- Modify: `src/DwaveHMC.jl`

- [ ] **Step 1: Include and export TBC measurement API**

Modify `src/DwaveHMC.jl`:

```julia
export measure_observables, measure_transport_and_spectra
export measure_twisted_spectra
export run_simulation, calc_optimal_dt

include("Types.jl")
include("Hamiltonian.jl")
include("Observables.jl")
include("TwistedSpectra.jl")
include("HMC.jl")
include("Simulation.jl")
```

- [ ] **Step 2: Create helper/result skeleton**

Create `src/TwistedSpectra.jl`:

```julia
using LinearAlgebra
using FFTW

struct TwistedSpectraResult
    dos_ω_grid::Vector{Float64}
    dos::Vector{Float64}
    dos_AN::Vector{Float64}
    dos_AN_patch::Vector{Float64}
    A_k_ω0::Matrix{Float64}
    A_kpath::Matrix{Float64}
    kx_grid::Vector{Float64}
    ky_grid::Vector{Float64}
    kpath_kx::Float64
    kpath_ky::Vector{Float64}
    Ltw::Int
    antinode_patch_half_width::Float64
end

@inline lorentzian_spectra(x::Float64, η::Float64) =
    (1.0 / π) * (η / (x * x + η * η))

@inline site_index_xy(x::Int, y::Int, Lx::Int, Ly::Int) =
    (mod1(y, Ly) - 1) * Lx + mod1(x, Lx)

@inline function boundary_winding(x::Int, y::Int, dx::Int, dy::Int,
                                  Lx::Int, Ly::Int)
    wx = x + dx > Lx ? 1 : (x + dx < 1 ? -1 : 0)
    wy = y + dy > Ly ? 1 : (y + dy < 1 ? -1 : 0)
    return wx, wy
end

@inline function tbc_phase(x::Int, y::Int, dx::Int, dy::Int,
                           Lx::Int, Ly::Int,
                           qx::Float64, qy::Float64)
    wx, wy = boundary_winding(x, y, dx, dy, Lx, Ly)
    return cis(-(qx * wx + qy * wy))
end

@inline function set_tbc_hermitian_pair!(H::Matrix{ComplexF64},
                                         row::Int,
                                         col::Int,
                                         val::ComplexF64)
    if row <= col
        H[row, col] = val
    else
        H[col, row] = conj(val)
    end
    return nothing
end

@inline function add_tbc_hop!(H::Matrix{ComplexF64},
                              N::Int,
                              i::Int,
                              j::Int,
                              h::ComplexF64)
    set_tbc_hermitian_pair!(H, i, j, h)
    set_tbc_hermitian_pair!(H, i + N, j + N, -conj(h))
    return nothing
end
```

- [ ] **Step 3: Implement effective-grid helpers**

Append to `src/TwistedSpectra.jl`:

```julia
@inline function effective_index_to_twist_fft(I::Int, L::Int, Ltw::Int)
    n = mod(-I, Ltw)
    m = div(I + n, Ltw)
    return mod(n, Ltw), mod(m, L)
end

@inline function twist_fft_to_effective_index(m::Int, n::Int, L::Int, Ltw::Int)
    return mod(m * Ltw - n, L * Ltw)
end

function effective_k_grid(L::Int, Ltw::Int)
    Leff = L * Ltw
    vals = Vector{Float64}(undef, Leff)
    @inbounds for I in 0:Leff-1
        k = 2π * I / Leff
        vals[I + 1] = k > π ? k - 2π : k
    end
    return vals
end

@inline function periodic_k_distance(a::Float64, b::Float64)
    d = abs(a - b)
    return min(d, 2π - d)
end

function antinode_patch_mask(kx_grid::Vector{Float64},
                             ky_grid::Vector{Float64},
                             half_width::Float64)
    mask = falses(length(kx_grid), length(ky_grid))
    count = 0
    @inbounds for ix in eachindex(kx_grid), iy in eachindex(ky_grid)
        kx = kx_grid[ix]
        ky = ky_grid[iy]
        near_pi_0 = periodic_k_distance(kx, π) <= half_width &&
                    periodic_k_distance(ky, 0.0) <= half_width
        near_0_pi = periodic_k_distance(kx, 0.0) <= half_width &&
                    periodic_k_distance(ky, π) <= half_width
        if near_pi_0 || near_0_pi
            mask[ix, iy] = true
            count += 1
        end
    end
    return mask, count
end
```

- [ ] **Step 4: Implement TBC Hamiltonian builder**

Append:

```julia
function build_tbc_H_BdG!(H::Matrix{ComplexF64},
                          p::ModelParameters,
                          state::SimulationState,
                          qx::Float64,
                          qy::Float64)
    N = p.N
    Lx = p.Lx
    Ly = p.Ly
    fill!(H, 0.0 + 0.0im)

    @inbounds for y in 1:Ly, x in 1:Lx
        i = site_index_xy(x, y, Lx, Ly)

        onsite = state.disorder_pot[i] - state.μ_eff
        H[i, i] = onsite
        H[i + N, i + N] = -onsite

        for (dx, dy, tt) in ((1, 0, p.t), (0, 1, p.t),
                             (1, 1, p.tp), (1, -1, p.tp))
            j = site_index_xy(x + dx, y + dy, Lx, Ly)
            ph = tbc_phase(x, y, dx, dy, Lx, Ly, qx, qy)
            add_tbc_hop!(H, N, i, j, -tt * ph)
        end

        jx = site_index_xy(x + 1, y, Lx, Ly)
        phx = tbc_phase(x, y, 1, 0, Lx, Ly, qx, qy)
        valx = state.Δ[i, 1] * phx
        H[i, jx + N] = valx
        H[jx, i + N] = valx

        jy = site_index_xy(x, y + 1, Lx, Ly)
        phy = tbc_phase(x, y, 0, 1, Lx, Ly, qx, qy)
        valy = state.Δ[i, 2] * phy
        H[i, jy + N] = valy
        H[jy, i + N] = valy
    end

    return nothing
end
```

- [ ] **Step 5: Run low-level tests**

Run:

```bash
julia --project test/test_twisted_spectra.jl
```

Expected: PASS for the low-level Hamiltonian and supercell tests.

- [ ] **Step 6: Commit builder**

```bash
git add src/DwaveHMC.jl src/TwistedSpectra.jl test/runtests.jl test/test_twisted_spectra.jl
git commit -m "添加TBC谱函数哈密顿量构造"
```

### Task 4: TBC Measurement

**Files:**
- Modify: `src/TwistedSpectra.jl`
- Modify: `test/test_twisted_spectra.jl`

- [ ] **Step 1: Add failing measurement regression and shape tests**

Append to `test/test_twisted_spectra.jl`:

```julia
@testset "Twisted spectra measurement" begin
    p, state, cache = setup_tbc_fixture()

    base = measure_transport_and_spectra(cache, p; reuse_buffers=false)
    H_before = copy(cache.H_base)
    E_before = copy(cache.E_n)
    U_before = copy(cache.U)

    tw1 = measure_twisted_spectra(cache, p, state;
                                  Ltw=1,
                                  antinode_patch_half_width=0.0,
                                  reuse_buffers=false)

    @test cache.H_base == H_before
    @test cache.E_n == E_before
    @test cache.U == U_before

    @test size(tw1.A_k_ω0) == size(base.A_k_ω0)
    @test size(tw1.A_kpath) == size(base.A_kpath)
    @test isapprox(tw1.dos, base.dos; atol=1e-8, rtol=1e-8)
    @test isapprox(tw1.dos_AN, base.dos_AN; atol=1e-8, rtol=1e-8)
    @test isapprox(tw1.A_k_ω0, base.A_k_ω0; atol=1e-8, rtol=1e-8)
    @test isapprox(tw1.A_kpath, base.A_kpath; atol=1e-8, rtol=1e-8)
    @test isapprox(tw1.dos_AN_patch, tw1.dos_AN; atol=1e-8, rtol=1e-8)

    tw2 = measure_twisted_spectra(cache, p, state; Ltw=2, reuse_buffers=false)
    @test size(tw2.A_k_ω0) == (p.Lx * 2, p.Ly * 2)
    @test size(tw2.A_kpath, 1) == fld(p.Ly * 2, 2) + 1
    @test size(tw2.A_kpath, 2) == length(cache.dos_omega_grid)
    @test length(tw2.kx_grid) == p.Lx * 2
    @test length(tw2.ky_grid) == p.Ly * 2
    @test all(isfinite, tw2.dos)
    @test all(isfinite, tw2.dos_AN)
    @test all(isfinite, tw2.dos_AN_patch)
    @test all(isfinite, tw2.A_k_ω0)
    @test all(isfinite, tw2.A_kpath)
end
```

Run:

```bash
julia --project test/test_twisted_spectra.jl
```

Expected: FAIL with `UndefVarError: measure_twisted_spectra not defined`.

- [ ] **Step 2: Add exact antinode and path helper functions**

Append to `src/TwistedSpectra.jl`:

```julia
function exact_effective_point(L_eff::Int, fraction::Float64, name::String)
    I = round(Int, fraction * L_eff)
    if !isapprox(I, fraction * L_eff; atol=1e-12, rtol=0.0)
        error("Effective grid cannot represent exact $name point")
    end
    return mod(I, L_eff)
end

function tbc_kpath_metadata(Lx::Int, Ly::Int, Ltw::Int)
    Lx_eff = Lx * Ltw
    Ly_eff = Ly * Ltw
    Ix_pi = exact_effective_point(Lx_eff, 0.5, "kx=π")
    ky_count = fld(Ly_eff, 2) + 1
    kx = 2π * Ix_pi / Lx_eff
    kx = kx > π ? kx - 2π : kx
    ky = Vector{Float64}(undef, ky_count)
    @inbounds for idx in 1:ky_count
        I = idx - 1
        ky[idx] = 2π * I / Ly_eff
    end
    return Ix_pi, kx, ky
end
```

- [ ] **Step 3: Implement `measure_twisted_spectra`**

Append:

```julia
function measure_twisted_spectra(cache::ComputeCache,
                                 p::ModelParameters,
                                 state::SimulationState;
                                 Ltw::Int=2,
                                 antinode_patch_half_width::Float64=π / max(p.Lx, p.Ly),
                                 reuse_buffers::Bool=false)
    Ltw <= 0 && error("Ltw must be positive")
    antinode_patch_half_width < 0 && error("antinode_patch_half_width must be nonnegative")

    N = p.N
    dim = 2 * N
    Lx = p.Lx
    Ly = p.Ly
    Lx_eff = Lx * Ltw
    Ly_eff = Ly * Ltw

    dos_ω_grid = reuse_buffers ? cache.dos_omega_grid : copy(cache.dos_omega_grid)
    nω = length(dos_ω_grid)
    dos_vals = zeros(Float64, nω)
    dos_AN_vals = zeros(Float64, nω)
    dos_AN_patch_vals = zeros(Float64, nω)
    A_k0 = zeros(Float64, Lx_eff, Ly_eff)
    _, kpath_kx, kpath_ky = tbc_kpath_metadata(Lx, Ly, Ltw)
    A_kpath = zeros(Float64, length(kpath_ky), nω)
    lor_cache = zeros(Float64, nω)

    kx_grid = effective_k_grid(Lx, Ltw)
    ky_grid = effective_k_grid(Ly, Ltw)
    patch_mask, patch_count = antinode_patch_mask(kx_grid, ky_grid,
                                                  antinode_patch_half_width)
    patch_count == 0 && error("Antinodal patch contains no effective momentum points")

    Ix_pi = exact_effective_point(Lx_eff, 0.5, "kx=π")
    Iy_pi = exact_effective_point(Ly_eff, 0.5, "ky=π")
    Ix_zero = 0
    Iy_zero = 0

    nx_pi, mx_pi = effective_index_to_twist_fft(Ix_pi, Lx, Ltw)
    ny_zero, my_zero = effective_index_to_twist_fft(Iy_zero, Ly, Ltw)
    nx_zero, mx_zero = effective_index_to_twist_fft(Ix_zero, Lx, Ltw)
    ny_pi, my_pi = effective_index_to_twist_fft(Iy_pi, Ly, Ltw)

    Htw = zeros(ComplexF64, dim, dim)
    Uwork = similar(Htw)
    Etw = zeros(Float64, dim)

    @inbounds for nx in 0:Ltw-1, ny in 0:Ltw-1
        qx = 2π * nx / Ltw
        qy = 2π * ny / Ltw

        build_tbc_H_BdG!(Htw, p, state, qx, qy)
        copyto!(Uwork, Htw)
        vals, vecs = eigen!(Hermitian(Uwork, :U))
        copyto!(Etw, vals)

        for n in 1:dim
            En = Etw[n]
            w_n = 0.0
            @simd for i in 1:N
                w_n += abs2(vecs[i, n])
            end

            for iw in eachindex(dos_ω_grid)
                lor_cache[iw] = lorentzian_spectra(dos_ω_grid[iw] - En, p.η)
                dos_vals[iw] += w_n * lor_cache[iw]
            end

            for y in 1:Ly, x in 1:Lx
                i = (y - 1) * Lx + x
                ph = cis(qx * (x - 1) / Lx + qy * (y - 1) / Ly)
                cache.u_r_cache[x, y] = vecs[i, n] * ph
            end
            mul!(cache.u_k_cache, cache.fft_plan, cache.u_r_cache)

            exact_weight = 0.0
            if nx == nx_pi && ny == ny_zero
                exact_weight += 0.5 * abs2(cache.u_k_cache[mx_pi + 1, my_zero + 1]) / N
            end
            if nx == nx_zero && ny == ny_pi
                exact_weight += 0.5 * abs2(cache.u_k_cache[mx_zero + 1, my_pi + 1]) / N
            end
            if exact_weight > 0
                for iw in eachindex(dos_ω_grid)
                    dos_AN_vals[iw] += exact_weight * lor_cache[iw]
                end
            end

            patch_weight = 0.0
            weight_at_zero = lorentzian_spectra(-En, p.η)
            for my in 0:Ly-1, mx in 0:Lx-1
                Ix = twist_fft_to_effective_index(mx, nx, Lx, Ltw)
                Iy = twist_fft_to_effective_index(my, ny, Ly, Ltw)
                wk = abs2(cache.u_k_cache[mx + 1, my + 1]) / N

                if patch_mask[Ix + 1, Iy + 1]
                    patch_weight += wk
                end

                if weight_at_zero > 1e-6
                    A_k0[Ix + 1, Iy + 1] += abs2(cache.u_k_cache[mx + 1, my + 1]) * weight_at_zero
                end
            end
            patch_weight /= patch_count
            for iw in eachindex(dos_ω_grid)
                dos_AN_patch_vals[iw] += patch_weight * lor_cache[iw]
            end

            if nx == nx_pi
                for my in 0:Ly-1
                    Iy = twist_fft_to_effective_index(my, ny, Ly, Ltw)
                    if Iy <= fld(Ly_eff, 2)
                        wk = abs2(cache.u_k_cache[mx_pi + 1, my + 1]) / N
                        path_idx = Iy + 1
                        for iw in eachindex(dos_ω_grid)
                            A_kpath[path_idx, iw] += wk * lor_cache[iw]
                        end
                    end
                end
            end
        end
    end

    dos_vals ./= (N * Ltw^2)
    A_k0 ./= N

    return TwistedSpectraResult(
        dos_ω_grid,
        reuse_buffers ? dos_vals : copy(dos_vals),
        reuse_buffers ? dos_AN_vals : copy(dos_AN_vals),
        reuse_buffers ? dos_AN_patch_vals : copy(dos_AN_patch_vals),
        reuse_buffers ? A_k0 : copy(A_k0),
        reuse_buffers ? A_kpath : copy(A_kpath),
        kx_grid,
        ky_grid,
        kpath_kx,
        kpath_ky,
        Ltw,
        antinode_patch_half_width,
    )
end
```

- [ ] **Step 4: Run TBC measurement tests**

Run:

```bash
julia --project test/test_twisted_spectra.jl
```

Expected: PASS. If `A_kpath` differs for `Ltw=1`, inspect FFT phase and index mapping; do not relax tolerances until the mapping is understood.

- [ ] **Step 5: Run full default suite**

Run:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Expected: PASS.

- [ ] **Step 6: Commit TBC measurement**

```bash
git add src/TwistedSpectra.jl test/test_twisted_spectra.jl
git commit -m "实现TBC谱函数测量"
```

### Task 5: Split Transport And Untwisted Spectra Helpers

**Files:**
- Modify: `src/Observables.jl`
- Modify: `test/test_twisted_spectra.jl`

- [ ] **Step 1: Add internal result structs**

In `src/Observables.jl`, near `SpectrumResult`, add:

```julia
struct TransportResult
    superfluid_stiffness::Float64
    dc_conductivity::Float64
    ω_grid::Vector{Float64}
    optical_conductivity::Vector{Float64}
end

struct SpectraOnlyResult
    dos_ω_grid::Vector{Float64}
    dos::Vector{Float64}
    dos_AN::Vector{Float64}
    A_k_ω0::Matrix{Float64}
    A_kpath::Matrix{Float64}
end
```

- [ ] **Step 2: Extract transport helper**

Create:

```julia
function measure_transport_only(cache::ComputeCache,
                                p::ModelParameters;
                                reuse_buffers::Bool=false)
```

Move the current `measure_transport_and_spectra` code from the beginning through the end of optical conductivity normalization into this helper. The helper must end with:

```julia
    if reuse_buffers
        return TransportResult(superfluid_stiffness, dc_cond, ω_grid, σ_ω)
    end

    return TransportResult(superfluid_stiffness, dc_cond,
                           copy(ω_grid), copy(σ_ω))
end
```

Keep these details exactly from the current function:

```julia
@inbounds @simd for n in 1:dim
    f[n] = logistic(-β * E[n])
end
```

and both current-operator paths:

```julia
if nnz(cache.Jx_sparse_qy) == 0
    build_current_operator!(cache, p; qx=0.0, qy=qy, store=:qy)
end

if nnz(cache.Jx_sparse_q0) == 0
    build_current_operator!(cache, p; qx=0.0, qy=0.0, store=:q0)
end
```

- [ ] **Step 3: Extract untwisted spectra helper**

Create:

```julia
function measure_untwisted_spectra(cache::ComputeCache,
                                   p::ModelParameters;
                                   reuse_buffers::Bool=false)
```

Move the current DOS, `dos_AN`, `A_k_ω0`, and `A_kpath` block into this helper. The helper must begin by binding the same spectral caches:

```julia
N = p.N
Lx = p.Lx
Ly = p.Ly
dim = 2 * N
U = cache.U
E = cache.E_n
dos_ω_grid = cache.dos_omega_grid
dos_vals = cache.dos_vals
dos_AN_vals = cache.dos_AN_vals
ak_map = cache.ak_map
ak_path = cache.ak_path
lor_cache = cache.lor_cache
kpath_weights = cache.kpath_weights
x_idx = cache.x_idx
y_idx = cache.y_idx
parity_x = cache.parity_x
parity_y = cache.parity_y
```

The helper must end with:

```julia
    if reuse_buffers
        return SpectraOnlyResult(dos_ω_grid, dos_vals, dos_AN_vals,
                                 ak_map, ak_path)
    end

    return SpectraOnlyResult(copy(dos_ω_grid), copy(dos_vals), copy(dos_AN_vals),
                             copy(ak_map), copy(ak_path))
end
```

- [ ] **Step 4: Recompose public API**

Replace the body of `measure_transport_and_spectra` with:

```julia
function measure_transport_and_spectra(cache::ComputeCache,
                                       p::ModelParameters;
                                       reuse_buffers::Bool=false)
    transport = measure_transport_only(cache, p; reuse_buffers=reuse_buffers)
    spectra = measure_untwisted_spectra(cache, p; reuse_buffers=reuse_buffers)

    return SpectrumResult(transport.superfluid_stiffness,
                          transport.dc_conductivity,
                          transport.ω_grid,
                          transport.optical_conductivity,
                          spectra.dos_ω_grid,
                          spectra.dos,
                          spectra.dos_AN,
                          spectra.A_k_ω0,
                          spectra.A_kpath)
end
```

- [ ] **Step 5: Add helper equivalence test**

Append to `test/test_twisted_spectra.jl`:

```julia
@testset "Transport/spectra helper split preserves public result" begin
    p, _, cache = setup_tbc_fixture()
    full = measure_transport_and_spectra(cache, p; reuse_buffers=false)
    trans = DwaveHMC.measure_transport_only(cache, p; reuse_buffers=false)
    spec = DwaveHMC.measure_untwisted_spectra(cache, p; reuse_buffers=false)

    @test full.superfluid_stiffness == trans.superfluid_stiffness
    @test full.dc_conductivity == trans.dc_conductivity
    @test full.ω_grid == trans.ω_grid
    @test full.optical_conductivity == trans.optical_conductivity
    @test full.dos_ω_grid == spec.dos_ω_grid
    @test full.dos == spec.dos
    @test full.dos_AN == spec.dos_AN
    @test full.A_k_ω0 == spec.A_k_ω0
    @test full.A_kpath == spec.A_kpath
end
```

- [ ] **Step 6: Run regression tests**

Run:

```bash
julia --project test/test_twisted_spectra.jl
julia --project -e 'using Pkg; Pkg.test()'
```

Expected: PASS.

- [ ] **Step 7: Commit helper split**

```bash
git add src/Observables.jl test/test_twisted_spectra.jl
git commit -m "拆分输运和谱函数测量"
```

### Task 6: Simulation Integration

**Files:**
- Modify: `src/Simulation.jl`
- Create: `test/test_simulation_tbc.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Add TBC keyword arguments**

Modify `run_simulation` signature:

```julia
function run_simulation(p::ModelParameters, out_dir::String;
                        n_therm::Int=100,
                        n_measure::Int=500,
                        Nt_therm_init::Int=10,
                        Nt_measure::Int=5,
                        measure_transport_freq::Int=1,
                        bin_size::Int=5,
                        measure_twist::Bool=false,
                        twist_Ax::Float64=1.0e-3,
                        twist_qy::Float64=2π / p.Ly,
                        spectra_Ltw::Int=1,
                        use_twisted_spectra::Bool=spectra_Ltw > 1,
                        antinode_patch_half_width::Float64=π / max(p.Lx, p.Ly),
                        verbose::Bool=true)
```

At the beginning of the function body, after output directory setup, add:

```julia
if spectra_Ltw <= 0
    error("spectra_Ltw must be positive")
end
if antinode_patch_half_width < 0
    error("antinode_patch_half_width must be nonnegative")
end
```

- [ ] **Step 2: Write TBC metadata**

Replace current `jldsave(spectra_jld_path; ...)` metadata block with:

```julia
omega_grid = cache.omega_grid
dos_omega_grid = cache.dos_omega_grid
spectra_Lx_eff = p.Lx * (use_twisted_spectra ? spectra_Ltw : 1)
spectra_Ly_eff = p.Ly * (use_twisted_spectra ? spectra_Ltw : 1)

if use_twisted_spectra
    _, kpath_kx, kpath_ky = tbc_kpath_metadata(p.Lx, p.Ly, spectra_Ltw)
    jldsave(spectra_jld_path;
            params=p,
            spectra_Ltw=spectra_Ltw,
            spectra_Lx_eff=spectra_Lx_eff,
            spectra_Ly_eff=spectra_Ly_eff,
            antinode_patch_half_width=antinode_patch_half_width,
            omega_grid=omega_grid,
            dos_omega_grid=dos_omega_grid,
            kpath_kx=kpath_kx,
            kpath_ky=kpath_ky)
else
    kx_idx, ky_indices, kx_val, ky_vals = antinode_kpath(p)
    jldsave(spectra_jld_path;
            params=p,
            spectra_Ltw=1,
            spectra_Lx_eff=p.Lx,
            spectra_Ly_eff=p.Ly,
            omega_grid=omega_grid,
            dos_omega_grid=dos_omega_grid,
            kpath_kx=kx_val,
            kpath_ky=ky_vals,
            kpath_kx_idx=kx_idx,
            kpath_ky_idx=ky_indices)
end
```

Add startup logging after current config logs:

```julia
tee_println("Spectra TBC: enabled=$(use_twisted_spectra), Ltw=$(spectra_Ltw), effective=$(spectra_Lx_eff)x$(spectra_Ly_eff)")
```

- [ ] **Step 3: Branch heavy measurement**

Replace:

```julia
spec_res = measure_transport_and_spectra(cache, p; reuse_buffers=true)
```

with:

```julia
dos_AN_patch_res = nothing
if use_twisted_spectra
    transport = measure_transport_only(cache, p; reuse_buffers=true)
    tw = measure_twisted_spectra(cache, p, state;
                                 Ltw=spectra_Ltw,
                                 antinode_patch_half_width=antinode_patch_half_width,
                                 reuse_buffers=false)
    spec_res = SpectrumResult(transport.superfluid_stiffness,
                              transport.dc_conductivity,
                              transport.ω_grid,
                              transport.optical_conductivity,
                              tw.dos_ω_grid,
                              tw.dos,
                              tw.dos_AN,
                              tw.A_k_ω0,
                              tw.A_kpath)
    dos_AN_patch_res = tw.dos_AN_patch
else
    spec_res = measure_transport_and_spectra(cache, p; reuse_buffers=true)
end
```

- [ ] **Step 4: Accumulate optional `dos_AN_patch`**

Near the spectral accumulators, add:

```julia
accum_dos_AN_patch = Vector{Float64}()
```

In the `bin_count == 0` branch, add:

```julia
if dos_AN_patch_res !== nothing
    accum_dos_AN_patch = copy(dos_AN_patch_res)
end
```

In the accumulation branch, add:

```julia
if dos_AN_patch_res !== nothing
    accum_dos_AN_patch .+= dos_AN_patch_res
end
```

Before writing the JLD2 group, add:

```julia
if !isempty(accum_dos_AN_patch)
    accum_dos_AN_patch ./= bin_count
end
```

Inside the JLD2 group write block, add:

```julia
if !isempty(accum_dos_AN_patch)
    g["dos_AN_patch"] = accum_dos_AN_patch
end
```

- [ ] **Step 5: Add TBC simulation test**

Create `test/test_simulation_tbc.jl`:

```julia
using Test
using Random
using JLD2
using DwaveHMC

@testset "run_simulation TBC output" begin
    Random.seed!(2468)
    p = ModelParameters(4, 4, 1.0, -0.35, -0.5, 0.0, 0.0, 6.0, 1.0, 1.0;
                        η=0.5, Δω=0.5, ω_max=2.0)

    out_dir = joinpath("data", "test_tbc_spectra_enabled_L4")
    isdir(out_dir) && rm(out_dir; recursive=true, force=true)

    run_simulation(p, out_dir;
                   n_therm=0,
                   n_measure=1,
                   Nt_therm_init=2,
                   Nt_measure=1,
                   measure_transport_freq=1,
                   bin_size=1,
                   spectra_Ltw=2,
                   use_twisted_spectra=true,
                   verbose=false)

    spectra_path = joinpath(out_dir, "spectra_bins.jld2")
    @test isfile(spectra_path)

    jldopen(spectra_path, "r") do file
        @test file["spectra_Ltw"] == 2
        @test file["spectra_Lx_eff"] == 8
        @test file["spectra_Ly_eff"] == 8
        @test haskey(file, "kpath_kx")
        @test haskey(file, "kpath_ky")
        @test haskey(file, "sweep_1")

        g = file["sweep_1"]
        @test size(g["A_k0"]) == (8, 8)
        @test size(g["A_kpath"], 1) == 5
        @test haskey(g, "dos_AN_patch")
        @test length(g["dos_AN_patch"]) == length(g["dos"])
        @test all(isfinite, g["dos_AN_patch"])
    end
end
```

- [ ] **Step 6: Include TBC simulation test in default suite**

Modify `test/runtests.jl`:

```julia
using Test

@testset "DwaveHMC default test suite" begin
    include("test_core_smoke.jl")
    include("test_twist_stiffness.jl")
    include("test_twisted_spectra.jl")
    include("test_simulation_tbc.jl")

    if get(ENV, "DWAVEHMC_RUN_SIMULATION_TESTS", "0") == "1"
        include("test_simulation.jl")
    end
end
```

- [ ] **Step 7: Run integration tests**

Run:

```bash
julia --project test/test_simulation_tbc.jl
julia --project -e 'using Pkg; Pkg.test()'
```

Expected: PASS. Confirm TBC-disabled runs do not write `dos_AN_patch` by running the default part of `test_simulation.jl` manually if needed.

- [ ] **Step 8: Commit simulation integration**

```bash
git add src/Simulation.jl test/runtests.jl test/test_simulation_tbc.jl
git commit -m "接入可选TBC谱函数输出"
```

### Task 7: Post-Processing Compatibility

**Files:**
- Modify: `scripts/process_spectra.jl`
- Modify: `scripts/batch_process_spectra.jl`
- Modify: `projectHPC/example/batch_process_spectra.jl`

- [ ] **Step 1: Update single-directory processor metadata reads**

In `scripts/process_spectra.jl`, read metadata with fallback:

```julia
spectra_Ltw = haskey(file, "spectra_Ltw") ? file["spectra_Ltw"] : 1
Lx_eff = haskey(file, "spectra_Lx_eff") ? file["spectra_Lx_eff"] : params.Lx
Ly_eff = haskey(file, "spectra_Ly_eff") ? file["spectra_Ly_eff"] : params.Ly
dos_omega_grid = haskey(file, "dos_omega_grid") ? file["dos_omega_grid"] :
                 collect(-params.ω_max : params.Δω : params.ω_max)
```

Add collection for optional patch:

```julia
list_dos_AN_patch = Vector{Vector{Float64}}()
```

When reading each group:

```julia
if haskey(g, "dos_AN_patch")
    push!(list_dos_AN_patch, g["dos_AN_patch"])
end
```

After `mean_dos_AN`, compute:

```julia
mean_dos_AN_patch, err_dos_AN_patch =
    isempty(list_dos_AN_patch) ? (nothing, nothing) : calc_stats(list_dos_AN_patch)
```

Write `processed_dos_AN_patch.csv` only when present:

```julia
if mean_dos_AN_patch !== nothing
    output_dos_AN_patch = joinpath(target_dir, "processed_dos_AN_patch.csv")
    open(output_dos_AN_patch, "w") do io
        println(io, "omega,DOS_AN_patch,Error")
        for i in 1:length(mean_dos_AN_patch)
            @printf(io, "%.6f,%.6f,%.6f\n",
                    dos_omega_grid[i], mean_dos_AN_patch[i], err_dos_AN_patch[i])
        end
    end
end
```

For `processed_ak0.csv`, replace `params.Lx, params.Ly` with `Lx_eff, Ly_eff`.

- [ ] **Step 2: Update batch processor**

Apply the same metadata and optional `dos_AN_patch` logic to `scripts/batch_process_spectra.jl`.

For the `A_k0` loop, use:

```julia
for x in 1:Lx_eff
    for y in 1:Ly_eff
        kx = 2π * (x - 1) / Lx_eff
        ky = 2π * (y - 1) / Ly_eff
        if kx > π kx -= 2π end
        if ky > π ky -= 2π end
        @printf(io, "%d,%d,%.6f,%.6f,%.6f,%.6f\n",
                x, y, kx, ky, mean_ak[x, y], err_ak[x, y])
    end
end
```

- [ ] **Step 3: Update HPC example processor**

In `projectHPC/example/batch_process_spectra.jl`, extend `process_single_config` to read patch arrays:

```julia
sum_dos_AN_patch = haskey(g1, "dos_AN_patch") ? copy(g1["dos_AN_patch"]) : nothing
```

During accumulation:

```julia
if sum_dos_AN_patch !== nothing
    if !haskey(g, "dos_AN_patch")
        return nothing
    end
    sum_dos_AN_patch .+= g["dos_AN_patch"]
end
```

Return it:

```julia
dos_AN_patch=sum_dos_AN_patch === nothing ? nothing : (sum_dos_AN_patch ./ count)
```

In `process_T_directory`, collect `samples_dos_AN_patch`, compute stats when nonempty, and write `spectra_dos_AN_patch.csv`.

Use `spectra_Lx_eff` and `spectra_Ly_eff` metadata for `spectra_ak0.csv`.

- [ ] **Step 4: Run syntax checks**

Run:

```bash
julia --project -e 'include("scripts/process_spectra.jl")'
```

Expected: It may error if the hard-coded target data directory is missing; syntax and load-time package errors should be fixed. For safer syntax-only checks, run:

```bash
julia --project -e 'Meta.parseall(read("scripts/process_spectra.jl", String)); Meta.parseall(read("scripts/batch_process_spectra.jl", String)); Meta.parseall(read("projectHPC/example/batch_process_spectra.jl", String)); println("parse ok")'
```

Expected: prints `parse ok`.

- [ ] **Step 5: Commit post-processing compatibility**

```bash
git add scripts/process_spectra.jl scripts/batch_process_spectra.jl projectHPC/example/batch_process_spectra.jl
git commit -m "兼容TBC谱函数后处理"
```

### Task 8: Opt-In Benchmark

**Files:**
- Create: `test/benchmark_twisted_spectra.jl`

- [ ] **Step 1: Add benchmark script**

Create `test/benchmark_twisted_spectra.jl`:

```julia
using Random
using BenchmarkTools
using DwaveHMC

function setup_benchmark_fixture(L::Int)
    Random.seed!(13579)
    p = ModelParameters(L, L, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                        η=8.0 / (L * L), Δω=4.0 / (L * L), ω_max=3.0)
    state = initialize_state(p)
    cache = initialize_cache(p)
    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)
    return p, state, cache
end

L = parse(Int, get(ENV, "DWAVEHMC_TBC_BENCH_L", "6"))
Ltw_values = parse.(Int, split(get(ENV, "DWAVEHMC_TBC_BENCH_LTW", "1,2,4"), ","))

p, state, cache = setup_benchmark_fixture(L)

println("Twisted spectra benchmark")
println("L=$(p.Lx)x$(p.Ly), dim=$(2p.N), omega_points=$(length(cache.dos_omega_grid))")

for Ltw in Ltw_values
    println()
    println("Ltw=$Ltw, effective=$(p.Lx * Ltw)x$(p.Ly * Ltw), sectors=$(Ltw^2)")
    result = @benchmark measure_twisted_spectra($cache, $p, $state; Ltw=$Ltw, reuse_buffers=false) samples=3 evals=1
    display(result)
end
```

- [ ] **Step 2: Run benchmark manually**

Run:

```bash
julia --project test/benchmark_twisted_spectra.jl
```

Expected: prints timing/allocation summaries for `Ltw=1,2,4`. This file is not included in `test/runtests.jl`.

- [ ] **Step 3: Commit benchmark**

```bash
git add test/benchmark_twisted_spectra.jl
git commit -m "添加TBC谱函数benchmark"
```

### Task 9: Documentation And Final Verification

**Files:**
- Modify: `doc/observables.md`

- [ ] **Step 1: Document TBC spectra behavior**

Add a subsection under `### 态密度与谱函数`:

```markdown
#### Twisted-boundary spectra

谱函数的 twisted-boundary-condition (TBC) 测量只作为后处理使用，不参与 HMC
采样、力、接受率、输运或超流刚度。启用 `run_simulation(...; spectra_Ltw=Ltw,
use_twisted_spectra=true)` 后，程序在每个热化后的构型上对
`Ltw^2` 个 twist sector 分别构造和对角化原始小晶胞 BdG 矩阵，然后把
小晶胞 FFT index 与 twist index 合并成有效动量网格：

$$
L_x^{\mathrm{eff}} = L_x L_{\mathrm{tw}},\quad
L_y^{\mathrm{eff}} = L_y L_{\mathrm{tw}}.
$$

默认 `spectra_Ltw=1` 且 `use_twisted_spectra=false`，因此旧的谱函数输出和
数组形状不变。

`dos_AN` 保持旧定义，即精确 antinode 点
$(\pi,0)$ 与 $(0,\pi)$ 的平均。TBC 额外输出 `dos_AN_patch`，它在有效动量
网格上对两个 antinode 附近的 patch 做平均，patch 半宽由
`antinode_patch_half_width` 控制，默认是 `π / max(Lx, Ly)`。
```

- [ ] **Step 2: Run targeted verification**

Run:

```bash
julia --project test/test_twisted_spectra.jl
julia --project test/test_simulation_tbc.jl
julia --project -e 'using Pkg; Pkg.test()'
```

Expected: all pass.

- [ ] **Step 3: Run opt-in legacy simulation test with small settings**

Run:

```bash
DWAVEHMC_RUN_SIMULATION_TESTS=1 \
HMC_TEST_N_THERM=1 \
HMC_TEST_N_MEASURE=2 \
HMC_TEST_NT_THERM=2 \
HMC_TEST_NT_MEASURE=1 \
HMC_TEST_TRANS_FREQ=1 \
HMC_TEST_BIN_SIZE=1 \
julia --project test/runtests.jl
```

Expected: PASS. This confirms the legacy end-to-end simulation file still works when explicitly requested.

- [ ] **Step 4: Check git diff and status**

Run:

```bash
git diff --stat
git status --short
```

Expected: only intended source, test, script, benchmark, and documentation files are modified. The reference file `doc/tbc_spectra_codex_instructions.md` can remain untracked unless the user explicitly wants it committed.

- [ ] **Step 5: Commit docs and final state**

```bash
git add doc/observables.md
git commit -m "记录TBC谱函数测量约定"
```

After this task, provide the user with:

- list of commits made
- verification commands and results
- benchmark command and summary
- note that TBC is opt-in and default output remains compatible
