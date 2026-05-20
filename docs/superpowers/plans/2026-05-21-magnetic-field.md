# Magnetic Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reliable finite orbital magnetic field support with magnetic PBC, Landau-gauge Peierls phases, gauge-covariant pairing diagnostics, and finite-field-safe transport/spectra output.

**Architecture:** Introduce a centralized magnetic phase layer and route Hamiltonian, Kubo current/diamagnetic terms, and gauge-pair output through it. Implement in layers with derivative tests before production Kubo usage, and keep finite-field momentum spectra opt-in diagnostics only.

**Tech Stack:** Julia, DwaveHMC structs/functions, LinearAlgebra, SparseArrays, JLD2, existing `Test` suite and script-based post-processing.

---

## File Structure

- Create `src/MagneticFieldTypes.jl`: small magnetic cache type definitions that `Types.jl` can reference without depending on `ModelParameters`.
- Create `src/MagneticField.jl`: validation, cache construction, link-phase lookup, metadata, and test-only helper formulas.
- Modify `src/DwaveHMC.jl`: include magnetic files before Hamiltonian/observables.
- Modify `src/Types.jl`: add `n_flux_sc` and `boundary_condition` to `ModelParameters`; add constructor alias keyword `n_vortices`; parameterize or extend `ComputeCache` to hold the magnetic cache.
- Modify `src/Hamiltonian.jl`: build normal hopping from unique directed bonds and add finite-field probe-Hamiltonian helpers for derivative tests.
- Modify `src/Observables.jl`: route current operators, diamagnetic terms, transport, and gauge-pair helpers through the magnetic phase layer.
- Modify `src/Simulation.jl`: add finite-field validation, metadata, spectra gating, gauge-pair output frequency, and fail-fast checks.
- Modify `src/TwistedSpectra.jl`: reject finite-field TBC spectra explicitly.
- Modify `scripts/spectra_postprocess_utils.jl`, `scripts/process_spectra.jl`, `scripts/batch_process_spectra.jl`, `projectHPC/example/spectra_postprocess_utils.jl`, and `projectHPC/example/batch_process_spectra.jl`: treat `dos_M` and momentum spectra as optional.
- Modify `projectHPC/run_conf.jl` and `projectHPC/example/sweep_T.sh`: pass/print magnetic parameters and add `Nv${n_flux_sc}` to output naming.
- Add `test/test_magnetic_field.jl`: phase, Hamiltonian, Kubo derivative, and output behavior tests.
- Modify `test/runtests.jl`, `test/test_simulation_tbc.jl`, `test/test_postprocess_spectra.jl`, and `test/test_hpc_scripts.jl`: include finite-field tests and optional spectra handling.
- Add `doc/magnetic-field.md`; update `doc/theory.md`, `doc/observables.md`, and `README.md` with finite-field discoverability links.

---

### Task 1: Magnetic Parameters and Phase Layer

**Files:**
- Create: `src/MagneticFieldTypes.jl`
- Create: `src/MagneticField.jl`
- Modify: `src/DwaveHMC.jl`
- Modify: `src/Types.jl`
- Create: `test/test_magnetic_field.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write failing magnetic validation and phase tests**

Create `test/test_magnetic_field.jl` with:

```julia
using Test
using LinearAlgebra
using Random
using JLD2
using DwaveHMC

function magnetic_test_parameters(; Lx=4, Ly=4, n_flux_sc=0,
                                  n_vortices=nothing,
                                  boundary_condition=:periodic)
    return ModelParameters(Lx, Ly, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                           η=0.25, Δω=0.25, ω_max=2.0,
                           n_flux_sc=n_flux_sc,
                           n_vortices=n_vortices,
                           boundary_condition=boundary_condition)
end

function site_index(x::Int, y::Int, Lx::Int)
    return (y - 1) * Lx + x
end

@testset "Magnetic field validation" begin
    p0 = magnetic_test_parameters()
    mag0 = DwaveHMC.build_magnetic_cache(p0)
    @test mag0 isa DwaveHMC.NoFieldCache
    @test DwaveHMC.link_phase(mag0, 1, 1, 0) == 1.0 + 0.0im
    @test DwaveHMC.link_phase(mag0, 1, 0, 1) == 1.0 + 0.0im

    p_plus = magnetic_test_parameters(n_flux_sc=2, boundary_condition=:magnetic_pbc)
    mag_plus = DwaveHMC.build_magnetic_cache(p_plus)
    @test mag_plus isa DwaveHMC.LandauGaugeCache
    @test mag_plus.n_flux_sc == 2
    @test mag_plus.flux_density_sc == 2 / 16
    @test mag_plus.plaquette_phase ≈ π * 2 / 16

    p_minus = magnetic_test_parameters(n_flux_sc=-2, boundary_condition=:magnetic_pbc)
    mag_minus = DwaveHMC.build_magnetic_cache(p_minus)
    for i in 1:p_plus.N, (dx, dy) in ((1, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (0, -1))
        @test DwaveHMC.link_phase(mag_minus, i, dx, dy) ≈ conj(DwaveHMC.link_phase(mag_plus, i, dx, dy))
    end

    p_alias = magnetic_test_parameters(n_vortices=-2, boundary_condition=:magnetic_pbc)
    @test p_alias.n_flux_sc == -2

    @test_throws ErrorException magnetic_test_parameters(n_flux_sc=1, boundary_condition=:magnetic_pbc)
    @test_throws ErrorException magnetic_test_parameters(n_flux_sc=2, boundary_condition=:periodic)
    @test_throws ErrorException magnetic_test_parameters(n_flux_sc=2, n_vortices=-2,
                                                         boundary_condition=:magnetic_pbc)
end

@testset "Landau gauge link phases" begin
    p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2, boundary_condition=:magnetic_pbc)
    mag = DwaveHMC.build_magnetic_cache(p)
    α = mag.plaquette_phase
    @test DwaveHMC.plaquette_phase(mag, 1, 1) ≈ α

    for y0 in 0:(p.Ly - 1), x0 in 0:(p.Lx - 1)
        i = site_index(x0 + 1, y0 + 1, p.Lx)
        ux = DwaveHMC.link_phase(mag, i, 1, 0)
        iy = site_index(mod(x0 + 1, p.Lx) + 1, y0 + 1, p.Lx)
        uy_right = DwaveHMC.link_phase(mag, iy, 0, 1)
        ix_up = site_index(x0 + 1, mod(y0 + 1, p.Ly) + 1, p.Lx)
        ux_up_reverse = DwaveHMC.link_phase(mag, ix_up, 1, 0)
        uy = DwaveHMC.link_phase(mag, i, 0, 1)
        wilson = ux * uy_right * conj(ux_up_reverse) * conj(uy)
        @test wilson ≈ cis(α)
    end

    i_boundary = site_index(p.Lx, 2, p.Lx) # zero-based (Lx-1, 1)
    @test DwaveHMC.link_phase(mag, i_boundary, 1, 0) ≈ cis(-α * p.Lx * 1)
    @test DwaveHMC.link_phase(mag, i_boundary, 1, 1) ≈
          cis(α * (p.Lx - 0.5)) * cis(-α * p.Lx * 2)
    @test DwaveHMC.link_phase(mag, i_boundary, 1, -1) ≈
          cis(-α * (p.Lx - 0.5)) * cis(-α * p.Lx * 0)
end

@testset "Landau gauge reverse links" begin
    p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2, boundary_condition=:magnetic_pbc)
    mag = DwaveHMC.build_magnetic_cache(p)
    for y0 in 0:(p.Ly - 1), x0 in 0:(p.Lx - 1), (dx, dy) in ((1, 0), (0, 1), (1, 1), (1, -1))
        i = site_index(x0 + 1, y0 + 1, p.Lx)
        xj = mod(x0 + dx, p.Lx)
        yj = mod(y0 + dy, p.Ly)
        j = site_index(xj + 1, yj + 1, p.Lx)
        @test DwaveHMC.link_phase(mag, i, dx, dy) ≈ conj(DwaveHMC.link_phase(mag, j, -dx, -dy))
    end
end
```

Modify `test/runtests.jl` so the default suite includes the new file first:

```julia
include("test_magnetic_field.jl")
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
julia --project test/test_magnetic_field.jl
```

Expected: fail with `UndefKeywordError: keyword argument n_flux_sc not assigned` or `UndefVarError: build_magnetic_cache not defined`.

- [ ] **Step 3: Add magnetic cache type definitions**

Create `src/MagneticFieldTypes.jl`:

```julia
abstract type AbstractMagneticCache end

struct NoFieldCache <: AbstractMagneticCache
    Lx::Int
    Ly::Int
    n_flux_sc::Int
    flux_density_sc::Float64
    plaquette_phase::Float64
end

struct LandauGaugeCache <: AbstractMagneticCache
    Lx::Int
    Ly::Int
    n_flux_sc::Int
    flux_density_sc::Float64
    plaquette_phase::Float64
    U_x::Vector{ComplexF64}
    U_y::Vector{ComplexF64}
    U_xpy::Vector{ComplexF64}
    U_xmy::Vector{ComplexF64}
end
```

Modify `src/DwaveHMC.jl` include order:

```julia
include("MultiEta.jl")
include("MagneticFieldTypes.jl")
include("Types.jl")
include("MagneticField.jl")
include("Hamiltonian.jl")
```

- [ ] **Step 4: Add magnetic fields to `ModelParameters`**

In `src/Types.jl`, add fields after `nnn_table`:

```julia
    n_flux_sc::Int
    boundary_condition::Symbol
```

Extend `_build_model_parameters` with keyword defaults:

```julia
                                 η::Float64=0.01, Δω::Float64=0.002, ω_max::Float64=4.0,
                                 n_flux_sc::Int=0,
                                 n_vortices::Union{Nothing,Int}=nothing,
                                 boundary_condition::Symbol=:periodic)
```

Before constructing `ModelParameters`, validate:

```julia
    actual_n_flux_sc = n_vortices === nothing ? n_flux_sc : Int(n_vortices)
    if n_vortices !== nothing && n_flux_sc != 0 && n_flux_sc != actual_n_flux_sc
        error("n_flux_sc and n_vortices must match when both are provided")
    end
    if actual_n_flux_sc == 0
        boundary_condition in (:periodic, :magnetic_pbc) ||
            error("boundary_condition must be :periodic or :magnetic_pbc")
    else
        boundary_condition === :magnetic_pbc ||
            error("Finite n_flux_sc requires boundary_condition=:magnetic_pbc")
        iseven(actual_n_flux_sc) ||
            error("magnetic PBC requires even n_flux_sc")
    end
```

Pass the new fields into the constructor:

```julia
        nn_table, nnn_table,
        Int(actual_n_flux_sc), Symbol(boundary_condition),
        Float64(η), Float64(ω_min), Float64(ω_max), Float64(Δω), n_ω)
```

Add `n_flux_sc::Int=0`, `n_vortices::Union{Nothing,Int}=nothing`, and `boundary_condition::Symbol=:periodic` to both public `ModelParameters(...)` constructors and forward them to `_build_model_parameters`.

- [ ] **Step 5: Implement `src/MagneticField.jl`**

Create `src/MagneticField.jl`:

```julia
@inline function _site_xy0(i::Int, Lx::Int)
    return mod1(i, Lx) - 1, cld(i, Lx) - 1
end

function validate_magnetic_field(p::ModelParameters)
    if p.n_flux_sc == 0
        p.boundary_condition in (:periodic, :magnetic_pbc) ||
            error("boundary_condition must be :periodic or :magnetic_pbc")
    else
        p.boundary_condition === :magnetic_pbc ||
            error("Finite n_flux_sc requires boundary_condition=:magnetic_pbc")
        iseven(p.n_flux_sc) ||
            error("magnetic PBC requires even n_flux_sc")
    end
    return nothing
end

@inline function _landau_link_phase(Lx::Int, Ly::Int, α::Float64,
                                    x0::Int, y0::Int, dx::Int, dy::Int)
    wx = fld(x0 + dx, Lx)
    y_end = y0 + dy
    raw = cis(α * dy * (x0 + 0.5 * dx))
    patch = cis(-α * wx * Lx * y_end)
    return raw * patch
end

function build_magnetic_cache(p::ModelParameters)
    validate_magnetic_field(p)
    flux_density_sc = p.n_flux_sc / p.N
    α = π * flux_density_sc
    if p.n_flux_sc == 0
        return NoFieldCache(p.Lx, p.Ly, 0, 0.0, 0.0)
    end

    U_x = Vector{ComplexF64}(undef, p.N)
    U_y = Vector{ComplexF64}(undef, p.N)
    U_xpy = Vector{ComplexF64}(undef, p.N)
    U_xmy = Vector{ComplexF64}(undef, p.N)
    @inbounds for i in 1:p.N
        x0, y0 = _site_xy0(i, p.Lx)
        U_x[i] = _landau_link_phase(p.Lx, p.Ly, α, x0, y0, 1, 0)
        U_y[i] = _landau_link_phase(p.Lx, p.Ly, α, x0, y0, 0, 1)
        U_xpy[i] = _landau_link_phase(p.Lx, p.Ly, α, x0, y0, 1, 1)
        U_xmy[i] = _landau_link_phase(p.Lx, p.Ly, α, x0, y0, 1, -1)
    end
    return LandauGaugeCache(p.Lx, p.Ly, p.n_flux_sc, flux_density_sc, α,
                            U_x, U_y, U_xpy, U_xmy)
end

@inline link_phase(::NoFieldCache, i::Int, dx::Int, dy::Int) = 1.0 + 0.0im

@inline function link_phase(mag::LandauGaugeCache, i::Int, dx::Int, dy::Int)
    if dx == 1 && dy == 0
        return mag.U_x[i]
    elseif dx == 0 && dy == 1
        return mag.U_y[i]
    elseif dx == 1 && dy == 1
        return mag.U_xpy[i]
    elseif dx == 1 && dy == -1
        return mag.U_xmy[i]
    else
        x0, y0 = _site_xy0(i, mag.Lx)
        return _landau_link_phase(mag.Lx, mag.Ly, mag.plaquette_phase, x0, y0, dx, dy)
    end
end

@inline plaquette_phase(mag::AbstractMagneticCache, x::Int, y::Int) =
    getfield(mag, :plaquette_phase)

function magnetic_metadata(mag::AbstractMagneticCache)
    return (n_flux_sc=mag.n_flux_sc,
            flux_density_sc=mag.flux_density_sc,
            plaquette_phase=mag.plaquette_phase,
            magnetic_gauge=mag.n_flux_sc == 0 ? "none" : "Landau gauge",
            magnetic_pbc=mag.n_flux_sc != 0)
end
```

- [ ] **Step 6: Run phase tests and the default suite subset**

Run:

```bash
julia --project test/test_magnetic_field.jl
julia --project test/test_core_smoke.jl
```

Expected: both pass.

- [ ] **Step 7: Commit magnetic phase layer**

```bash
git add src/MagneticFieldTypes.jl src/MagneticField.jl src/DwaveHMC.jl src/Types.jl test/test_magnetic_field.jl test/runtests.jl
git commit -m "添加磁场相位层"
```

---

### Task 2: ComputeCache Integration and Directed Hamiltonian Hopping

**Files:**
- Modify: `src/Types.jl`
- Modify: `src/Hamiltonian.jl`
- Modify: `test/test_magnetic_field.jl`
- Test: `test/test_hamiltonian.jl`, `test/test_forces.jl`

- [ ] **Step 1: Add failing Hamiltonian phase tests**

Append to `test/test_magnetic_field.jl`:

```julia
@testset "Magnetic Hamiltonian hopping phases" begin
    p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2, boundary_condition=:magnetic_pbc)
    state = initialize_state(p)
    cache = initialize_cache(p)
    fill!(state.Δ, 0.0 + 0.0im)

    init_static_H!(cache, p, state)
    H = cache.H_base
    Hfull = Matrix(Hermitian(H, :U))
    N = p.N
    i = site_index(p.Lx, 2, p.Lx)
    jx = p.nn_table[i, 1]
    jxpy = p.nnn_table[i, 1]
    jxmy = p.nnn_table[i, 4]

    ph_x = DwaveHMC.link_phase(cache.magnetic, i, 1, 0)
    ph_xpy = DwaveHMC.link_phase(cache.magnetic, i, 1, 1)
    ph_xmy = DwaveHMC.link_phase(cache.magnetic, i, 1, -1)
    @test Hfull[i, jx] ≈ -p.t * ph_x
    @test Hfull[i, jxpy] ≈ -p.tp * ph_xpy
    @test Hfull[i, jxmy] ≈ -p.tp * ph_xmy
    @test Hfull[i + N, jx + N] ≈ p.t * conj(ph_x)
    @test Hfull ≈ Hfull'
end
```

Run:

```bash
julia --project test/test_magnetic_field.jl
```

Expected: fail because `cache.magnetic` does not exist or `init_static_H!` still uses unphased hopping.

- [ ] **Step 2: Add magnetic cache to `ComputeCache`**

In `src/Types.jl`, change the cache definition to:

```julia
mutable struct ComputeCache{M<:AbstractMagneticCache}
```

Add the field as the final `ComputeCache` field, after `omega_inv`:

```julia
    magnetic::M
```

In `initialize_cache(p::ModelParameters)`, build the magnetic cache before returning:

```julia
    magnetic = build_magnetic_cache(p)
```

Pass `magnetic` as the final `ComputeCache(...)` constructor argument, immediately after `omega_inv`. Keep all existing field names otherwise unchanged.

- [ ] **Step 3: Replace static hopping with unique directed bonds**

In `src/Hamiltonian.jl`, add helpers before `init_static_H!`:

```julia
@inline function set_hermitian_entry!(H::Matrix{ComplexF64}, row::Int, col::Int, val::ComplexF64)
    if row <= col
        H[row, col] = val
    else
        H[col, row] = conj(val)
    end
    return nothing
end

@inline function add_static_hopping!(H::Matrix{ComplexF64},
                                     N::Int,
                                     i::Int,
                                     j::Int,
                                     tij::Float64,
                                     phase::ComplexF64)
    set_hermitian_entry!(H, i, j, -tij * phase)
    set_hermitian_entry!(H, i + N, j + N, tij * conj(phase))
    return nothing
end
```

Replace the old `dir in 1:4` plus `if j > i` hopping loops in `init_static_H!` with:

```julia
    mag = cache.magnetic
    @inbounds for i in 1:N
        add_static_hopping!(H, N, i, p.nn_table[i, 1], p.t, link_phase(mag, i, 1, 0))
        add_static_hopping!(H, N, i, p.nn_table[i, 2], p.t, link_phase(mag, i, 0, 1))
        add_static_hopping!(H, N, i, p.nnn_table[i, 1], p.tp, link_phase(mag, i, 1, 1))
        add_static_hopping!(H, N, i, p.nnn_table[i, 4], p.tp, link_phase(mag, i, 1, -1))
    end
```

This intentionally avoids `j > i` for finite-field hopping.

- [ ] **Step 4: Run Hamiltonian and force tests**

Run:

```bash
julia --project test/test_magnetic_field.jl
julia --project test/test_hamiltonian.jl
julia --project test/test_forces.jl
```

Expected: all pass. If the old zero-field Hamiltonian tests fail, inspect only the static hopping section; zero-field phase arrays must be `1 + 0im`.

- [ ] **Step 5: Commit directed Hamiltonian hopping**

```bash
git add src/Types.jl src/Hamiltonian.jl test/test_magnetic_field.jl
git commit -m "统一磁场哈密顿量跃迁相位"
```

---

### Task 3: Probe Hamiltonian and Kubo Derivative Tests

**Files:**
- Modify: `src/Hamiltonian.jl`
- Modify: `src/Observables.jl`
- Modify: `test/test_magnetic_field.jl`

- [ ] **Step 1: Add failing derivative tests**

Append to `test/test_magnetic_field.jl`:

```julia
function random_finite_field_state(p)
    Random.seed!(20260521)
    state = initialize_state(p)
    state.disorder_pot .= randn(p.N) .* 0.05
    state.Δ .= (randn(p.N, 2) .+ im .* randn(p.N, 2)) .* 0.03
    return state
end

@testset "Kubo operators match Hamiltonian derivatives" begin
    p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2, boundary_condition=:magnetic_pbc)
    state = random_finite_field_state(p)
    cache = initialize_cache(p)
    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)

    dim = 2 * p.N
    Hplus = zeros(ComplexF64, dim, dim)
    Hminus = zeros(ComplexF64, dim, dim)
    H0 = zeros(ComplexF64, dim, dim)
    eps = 1.0e-6

    for qy in (0.0, 2π / p.Ly)
        DwaveHMC.build_probe_H_BdG!(Hplus, cache, p, state; λ=eps, qx=0.0, qy=qy)
        DwaveHMC.build_probe_H_BdG!(Hminus, cache, p, state; λ=-eps, qx=0.0, qy=qy)
        DwaveHMC.build_probe_H_BdG!(H0, cache, p, state; λ=0.0, qx=0.0, qy=qy)

        J_fd = (Matrix(Hermitian(Hplus, :U)) - Matrix(Hermitian(Hminus, :U))) ./ (2eps)
        K_fd = (Matrix(Hermitian(Hplus, :U)) + Matrix(Hermitian(Hminus, :U)) -
                2 .* Matrix(Hermitian(H0, :U))) ./ (eps^2)

        J_an = Matrix(DwaveHMC.current_operator_matrix(cache, p; qx=0.0, qy=qy))
        K_an = DwaveHMC.diamagnetic_operator_matrix(cache, p; qx=0.0, qy=qy)

        @test norm(J_an - J_fd) / max(norm(J_fd), 1.0) < 1.0e-6
        @test norm(K_an - K_fd) / max(norm(K_fd), 1.0) < 5.0e-4
    end
end
```

Run:

```bash
julia --project test/test_magnetic_field.jl
```

Expected: fail with `UndefVarError: build_probe_H_BdG! not defined` or derivative mismatch.

- [ ] **Step 2: Implement probe Hamiltonian builder**

In `src/Hamiltonian.jl`, add:

```julia
@inline function probe_factor(cache::ComputeCache, i::Int, dx::Int, λ::Float64, qx::Float64, qy::Float64)
    dx == 0 && return 1.0 + 0.0im
    x = cache.x_idx[i] - 1
    y = cache.y_idx[i] - 1
    θ = qx * x + qy * y
    η = (qx == 0.0 && qy == 0.0) ? 1.0 : sqrt(2.0) * cos(θ)
    return cis(λ * dx * η)
end

function build_probe_H_BdG!(H::Matrix{ComplexF64},
                            cache::ComputeCache,
                            p::ModelParameters,
                            state::SimulationState;
                            λ::Float64,
                            qx::Float64=0.0,
                            qy::Float64=0.0)
    N = p.N
    fill!(H, 0.0 + 0.0im)
    @inbounds for i in 1:N
        term = state.disorder_pot[i] - state.μ_eff
        H[i, i] = term
        H[i + N, i + N] = -term
    end
    mag = cache.magnetic
    @inbounds for i in 1:N
        ph = link_phase(mag, i, 1, 0) * probe_factor(cache, i, 1, λ, qx, qy)
        add_static_hopping!(H, N, i, p.nn_table[i, 1], p.t, ph)
        ph = link_phase(mag, i, 0, 1)
        add_static_hopping!(H, N, i, p.nn_table[i, 2], p.t, ph)
        ph = link_phase(mag, i, 1, 1) * probe_factor(cache, i, 1, λ, qx, qy)
        add_static_hopping!(H, N, i, p.nnn_table[i, 1], p.tp, ph)
        ph = link_phase(mag, i, 1, -1) * probe_factor(cache, i, 1, λ, qx, qy)
        add_static_hopping!(H, N, i, p.nnn_table[i, 4], p.tp, ph)
    end
    @inbounds for i in 1:N
        j_x = p.nn_table[i, 1]
        val_x = state.Δ[i, 1]
        H[i, j_x + N] = val_x
        H[j_x, i + N] = val_x
        j_y = p.nn_table[i, 2]
        val_y = state.Δ[i, 2]
        H[i, j_y + N] = val_y
        H[j_y, i + N] = val_y
    end
    return nothing
end
```

This probe builder is for derivative tests and diagnostics. It uses the existing `sqrt(2) * cos(qy*y)` finite-`q_y` real-probe normalization for nonzero probe momentum so the finite-difference Hamiltonian remains Hermitian and the normalization matches `measure_twist_stiffness_qy`.

- [ ] **Step 3: Implement analytic current and diamagnetic matrices from the same probe convention**

In `src/Observables.jl`, replace `current_operator_matrix` internals with directed bonds and add a finite-field diamagnetic matrix helper:

```julia
@inline function probe_weight(cache::ComputeCache, i::Int, qx::Float64, qy::Float64)
    x = cache.x_idx[i] - 1
    y = cache.y_idx[i] - 1
    θ = qx * x + qy * y
    return (qx == 0.0 && qy == 0.0) ? 1.0 : sqrt(2.0) * cos(θ)
end

@inline function add_sparse_hermitian_pair!(I_idx::Vector{Int},
                                            J_idx::Vector{Int},
                                            V_val::Vector{ComplexF64},
                                            row::Int,
                                            col::Int,
                                            val::ComplexF64)
    push!(I_idx, row); push!(J_idx, col); push!(V_val, val)
    push!(I_idx, col); push!(J_idx, row); push!(V_val, conj(val))
    return nothing
end

@inline function add_current_derivative_bond!(I_idx::Vector{Int},
                                              J_idx::Vector{Int},
                                              V_val::Vector{ComplexF64},
                                              cache::ComputeCache,
                                              N::Int,
                                              i::Int,
                                              j::Int,
                                              tij::Float64,
                                              dx::Int,
                                              dy::Int,
                                              qx::Float64,
                                              qy::Float64)
    η = probe_weight(cache, i, qx, qy)
    u = link_phase(cache.magnetic, i, dx, dy)
    d1 = -im * tij * η * u
    add_sparse_hermitian_pair!(I_idx, J_idx, V_val, i, j, d1)
    add_sparse_hermitian_pair!(I_idx, J_idx, V_val, i + N, j + N, -conj(d1))
    return nothing
end

function current_operator_matrix(cache::ComputeCache,
                                 p::ModelParameters;
                                 qx::Float64=0.0,
                                 qy::Float64=0.0)
    N = p.N
    I_idx = Int[]
    J_idx = Int[]
    V_val = ComplexF64[]
    sizehint!(I_idx, 12 * N)
    sizehint!(J_idx, 12 * N)
    sizehint!(V_val, 12 * N)

    @inbounds for i in 1:N
        add_current_derivative_bond!(I_idx, J_idx, V_val, cache, N,
                                     i, p.nn_table[i, 1], p.t, 1, 0, qx, qy)
        add_current_derivative_bond!(I_idx, J_idx, V_val, cache, N,
                                     i, p.nnn_table[i, 1], p.tp, 1, 1, qx, qy)
        add_current_derivative_bond!(I_idx, J_idx, V_val, cache, N,
                                     i, p.nnn_table[i, 4], p.tp, 1, -1, qx, qy)
    end

    return sparse(I_idx, J_idx, V_val, 2 * N, 2 * N)
end

@inline function add_dense_hermitian_pair!(M::Matrix{ComplexF64},
                                           row::Int,
                                           col::Int,
                                           val::ComplexF64)
    M[row, col] = val
    M[col, row] = conj(val)
    return nothing
end

@inline function add_diamagnetic_bond!(K::Matrix{ComplexF64},
                                       cache::ComputeCache,
                                       N::Int,
                                       i::Int,
                                       j::Int,
                                       tij::Float64,
                                       dx::Int,
                                       dy::Int,
                                       qx::Float64,
                                       qy::Float64)
    η = probe_weight(cache, i, qx, qy)
    u = link_phase(cache.magnetic, i, dx, dy)
    d2 = tij * η^2 * u
    add_dense_hermitian_pair!(K, i, j, d2)
    add_dense_hermitian_pair!(K, i + N, j + N, -conj(d2))
    return nothing
end

function diamagnetic_operator_matrix(cache::ComputeCache,
                                     p::ModelParameters;
                                     qx::Float64=0.0,
                                     qy::Float64=0.0)
    N = p.N
    K = zeros(ComplexF64, 2 * N, 2 * N)
    @inbounds for i in 1:N
        add_diamagnetic_bond!(K, cache, N, i, p.nn_table[i, 1], p.t, 1, 0, qx, qy)
        add_diamagnetic_bond!(K, cache, N, i, p.nnn_table[i, 1], p.tp, 1, 1, qx, qy)
        add_diamagnetic_bond!(K, cache, N, i, p.nnn_table[i, 4], p.tp, 1, -1, qx, qy)
    end
    return K
end
```

Keep `build_current_operator!` storing sparse matrices in `cache.Jx_sparse_q0` and `cache.Jx_sparse_qy`.

- [ ] **Step 4: Run derivative tests**

Run:

```bash
julia --project test/test_magnetic_field.jl
```

Expected: derivative tests pass. If only the current test fails by a global sign, choose the sign that matches `build_probe_H_BdG!` and update the Kubo comments accordingly; `Lambda` uses `abs2`, but the derivative test fixes the convention.

- [ ] **Step 5: Commit Kubo derivative infrastructure**

```bash
git add src/Hamiltonian.jl src/Observables.jl test/test_magnetic_field.jl
git commit -m "校验磁场Kubo算符导数"
```

---

### Task 4: Transport Measurement and Stiffness Diagnostics

**Files:**
- Modify: `src/Observables.jl`
- Modify: `test/test_magnetic_field.jl`
- Test: `test/test_twist_stiffness.jl`

- [ ] **Step 1: Add finite-field transport smoke and curvature diagnostic tests**

Append to `test/test_magnetic_field.jl`:

```julia
@testset "Finite-field transport is finite and diagnostic curvature is available" begin
    p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2, boundary_condition=:magnetic_pbc)
    state = random_finite_field_state(p)
    cache = initialize_cache(p)
    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)

    res = measure_transport_only(cache, p; eta_values=[p.η], reuse_buffers=false)
    @test isfinite(res.superfluid_stiffness)
    @test isfinite(res.dc_conductivity)
    @test length(res.optical_conductivity) == length(cache.omega_grid)

    K0 = DwaveHMC.diamagnetic_operator_matrix(cache, p; qx=0.0, qy=0.0)
    dia_from_matrix = 0.0
    for n in 1:(2 * p.N)
        if cache.E_n[n] > 0
            ψn = @view cache.U[:, n]
            dia_from_matrix += -real(dot(ψn, K0 * ψn)) * tanh(0.5 * p.β * cache.E_n[n]) / p.N
        end
    end
    @test DwaveHMC.diamagnetic_expectation_x(cache, p) ≈ dia_from_matrix atol=1.0e-10

    diag = DwaveHMC.measure_full_curvature_diagnostic(cache, p, state; Ax=1.0e-4, qy=2π / p.Ly)
    @test isfinite(diag.rho_full_curvature)
    @test isfinite(diag.lambda_diag)
end
```

Run:

```bash
julia --project test/test_magnetic_field.jl
```

Expected: fail with `UndefVarError: measure_full_curvature_diagnostic not defined` or transport phase errors.

- [ ] **Step 2: Route diamagnetic expectation through phased bonds**

In `measure_transport_only`, replace the manual zero-field `val_dia` loop with a helper:

```julia
function diamagnetic_expectation_x(cache::ComputeCache, p::ModelParameters)
    N = p.N
    dim = 2 * N
    β = p.β
    U = cache.U
    E = cache.E_n
    mag = cache.magnetic
    total = 0.0
    @inbounds for n in 1:dim
        En = E[n]
        if En > 0
            w_n = 0.0
            @simd for i in 1:N
                j = p.nn_table[i, 1]
                ph = link_phase(mag, i, 1, 0)
                w_n += 2.0 * real(p.t * ph * (U[i+N,n] * conj(U[j+N,n]) - conj(U[i,n]) * U[j,n]))
                j = p.nnn_table[i, 1]
                ph = link_phase(mag, i, 1, 1)
                w_n += 2.0 * real(p.tp * ph * (U[i+N,n] * conj(U[j+N,n]) - conj(U[i,n]) * U[j,n]))
                j = p.nnn_table[i, 4]
                ph = link_phase(mag, i, 1, -1)
                w_n += 2.0 * real(p.tp * ph * (U[i+N,n] * conj(U[j+N,n]) - conj(U[i,n]) * U[j,n]))
            end
            total += w_n * tanh(0.5 * β * En) / N
        end
    end
    return total
end
```

Then set:

```julia
val_dia = diamagnetic_expectation_x(cache, p)
```

- [ ] **Step 3: Add full-curvature diagnostic helper**

In `src/Observables.jl`, add:

```julia
struct FullCurvatureDiagnostic
    qy::Float64
    rho_full_curvature::Float64
    lambda_diag::Float64
end

function measure_full_curvature_diagnostic(cache::ComputeCache,
                                           p::ModelParameters,
                                           state::SimulationState;
                                           Ax::Float64=1.0e-4,
                                           qy::Float64=2π / p.Ly)
    H = zeros(ComplexF64, 2 * p.N, 2 * p.N)
    build_probe_H_BdG!(H, cache, p, state; λ=0.0, qx=0.0, qy=qy)
    S0 = fermion_logdet_action_from_eigs(eigvals!(Hermitian(H, :U)), p.β)
    build_probe_H_BdG!(H, cache, p, state; λ=Ax, qx=0.0, qy=qy)
    Splus = fermion_logdet_action_from_eigs(eigvals!(Hermitian(H, :U)), p.β)
    build_probe_H_BdG!(H, cache, p, state; λ=-Ax, qx=0.0, qy=qy)
    Sminus = fermion_logdet_action_from_eigs(eigvals!(Hermitian(H, :U)), p.β)
    rho = (Splus + Sminus - 2S0) / (Ax^2 * p.β * p.N)
    lambda_diag = measure_kubo_diag_correction_qy(cache, p; qy=qy)
    return FullCurvatureDiagnostic(qy, rho, lambda_diag)
end
```

This is a diagnostic benchmark, not the production `Superfluid_Stiffness` column.

- [ ] **Step 4: Run transport tests**

Run:

```bash
julia --project test/test_magnetic_field.jl
julia --project test/test_twist_stiffness.jl
```

Expected: both pass. The twist test remains zero-field and guards that existing diagnostics still work.

- [ ] **Step 5: Commit finite-field transport phase routing**

```bash
git add src/Observables.jl test/test_magnetic_field.jl
git commit -m "接入磁场Kubo输运相位"
```

---

### Task 5: Gauge-Covariant Pair Bond Output

**Files:**
- Modify: `src/Observables.jl`
- Modify: `src/Simulation.jl`
- Modify: `test/test_magnetic_field.jl`

- [ ] **Step 1: Add failing pair-bond output tests**

Append to `test/test_magnetic_field.jl`:

```julia
@testset "Gauge-covariant pair bond output" begin
    mktempdir() do out_dir
        p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2, boundary_condition=:magnetic_pbc)
        Random.seed!(20260521)
        run_simulation(p, out_dir;
                       n_therm=0,
                       n_measure=2,
                       Nt_measure=1,
                       measure_transport_freq=2,
                       bin_size=1,
                       write_gauge_pair_bonds_freq=1,
                       allow_gauge_dependent_spectra=false,
                       verbose=false)
        jldopen(joinpath(out_dir, "pairing_scatter.jld2"), "r") do file
            @test haskey(file, "sweep_1/delta_bond_landau_gauge_covariant")
            @test haskey(file, "sweep_1/pair_bond_landau_gauge_covariant")
            @test size(file["sweep_1/delta_bond_landau_gauge_covariant"]) == (p.N, 2)
            @test size(file["sweep_1/pair_bond_landau_gauge_covariant"]) == (p.N, 2)
        end
    end
end
```

Run:

```bash
julia --project test/test_magnetic_field.jl
```

Expected: fail with `UndefKeywordError: write_gauge_pair_bonds_freq not assigned`.

- [ ] **Step 2: Add pair-bond computation helper**

In `src/Observables.jl`, add:

```julia
function compute_gauge_pair_bonds(cache::ComputeCache, p::ModelParameters, state::SimulationState)
    N = p.N
    U = cache.U
    E = cache.E_n
    f = cache.fermi_factors
    g_pair = pairing_coupling(p)
    @inbounds @simd for n in 1:(2 * N)
        f[n] = logistic(-p.β * E[n])
    end
    delta_bond = Matrix{ComplexF64}(undef, N, 2)
    pair_bond = Matrix{ComplexF64}(undef, N, 2)
    @inbounds for i in 1:N
        for dir in 1:2
            j = p.nn_table[i, dir]
            rho_1 = zero(ComplexF64)
            rho_2 = zero(ComplexF64)
            @simd for n in 1:(2 * N)
                rho_1 += U[i, n] * f[n] * conj(U[j+N, n])
                rho_2 += U[j, n] * f[n] * conj(U[i+N, n])
            end
            P_ij = -rho_1 - rho_2
            dx = dir == 1 ? 1 : 0
            dy = dir == 1 ? 0 : 1
            ph = link_phase(cache.magnetic, i, dx, dy)
            delta_bond[i, dir] = state.Δ[i, dir] * ph
            pair_bond[i, dir] = g_pair * P_ij * ph
        end
    end
    return delta_bond, pair_bond
end
```

The fermionic `pair_bond` is the primary physical object for finite-field post-processing; `delta_bond` is retained as an auxiliary-field diagnostic.

- [ ] **Step 3: Wire `write_gauge_pair_bonds_freq` through simulation**

In `run_simulation`, add keyword:

```julia
write_gauge_pair_bonds_freq::Int=0
```

Validate:

```julia
write_gauge_pair_bonds_freq >= 0 || error("write_gauge_pair_bonds_freq must be nonnegative")
```

Do not change the existing scalar pairing CSV columns such as `Δ_x - Δ_y`, `Delta_Diff`, `Delta_Pair`, or `Delta_LocalPair`. They remain the zero-field-compatible bare convention; the new gauge-covariant finite-field data is written only to JLD2 when `write_gauge_pair_bonds_freq > 0`.

Inside the existing `jldopen(pair_scatter_jld_path, "a+")` block, after writing `d_local`, add:

```julia
if write_gauge_pair_bonds_freq > 0 && i % write_gauge_pair_bonds_freq == 0
    delta_bond, pair_bond = compute_gauge_pair_bonds(cache, p, state)
    g["delta_bond_landau_gauge_covariant"] = delta_bond
    g["pair_bond_landau_gauge_covariant"] = pair_bond
end
```

- [ ] **Step 4: Run pair output tests**

Run:

```bash
julia --project test/test_magnetic_field.jl
```

Expected: pass.

- [ ] **Step 5: Commit pair bond output**

```bash
git add src/Observables.jl src/Simulation.jl test/test_magnetic_field.jl
git commit -m "写出规范协变配对bond"
```

---

### Task 6: Finite-Field Spectra Gating and Optional Post-Processing

**Files:**
- Modify: `src/Observables.jl`
- Modify: `src/Simulation.jl`
- Modify: `src/TwistedSpectra.jl`
- Modify: `scripts/spectra_postprocess_utils.jl`
- Modify: `scripts/process_spectra.jl`
- Modify: `scripts/batch_process_spectra.jl`
- Modify: `projectHPC/example/spectra_postprocess_utils.jl`
- Modify: `projectHPC/example/batch_process_spectra.jl`
- Modify: `test/test_simulation_tbc.jl`
- Modify: `test/test_postprocess_spectra.jl`

- [ ] **Step 1: Add failing finite-field spectra gating tests**

Add to `test/test_simulation_tbc.jl`:

```julia
@testset "finite magnetic field disables gauge-dependent spectra by default" begin
    mktempdir() do out_dir
        p = ModelParameters(4, 4, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                            η=0.25, Δω=0.25, ω_max=2.0,
                            n_flux_sc=2, boundary_condition=:magnetic_pbc)
        spectra_path = run_tiny_spectra_simulation(p, out_dir; use_twisted_spectra=false)
        jldopen(spectra_path, "r") do file
            @test file["n_flux_sc"] == 2
            @test file["gauge_dependent_spectra"] == false
            g = file["sweep_1"]
            @test haskey(g, "dos")
            @test haskey(g, "LDOS_0")
            @test !haskey(g, "dos_M")
            @test !haskey(g, "A_k0")
            @test !haskey(g, "A_MX_path")
            @test !haskey(g, "A_XG_path")
            @test !haskey(g, "dos_M_landau_gauge_diagnostic")
            @test !haskey(g, "A_k_omega0_landau_gauge_diagnostic")
        end
    end
end

@testset "finite magnetic field diagnostic spectra use warning names" begin
    mktempdir() do out_dir
        p = ModelParameters(4, 4, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                            η=0.25, Δω=0.25, ω_max=2.0,
                            n_flux_sc=2, boundary_condition=:magnetic_pbc)
        spectra_path = run_tiny_spectra_simulation(p, out_dir;
                                                   use_twisted_spectra=false,
                                                   allow_gauge_dependent_spectra=true)
        jldopen(spectra_path, "r") do file
            @test file["gauge_dependent_spectra"] == true
            @test file["spectra_gauge"] == "Landau gauge"
            @test occursin("diagnostic", file["spectra_interpretation"])
            g = file["sweep_1"]
            @test haskey(g, "dos_M_landau_gauge_diagnostic")
            @test haskey(g, "A_k_omega0_landau_gauge_diagnostic")
            @test haskey(g, "A_MX_path_landau_gauge_diagnostic")
            @test haskey(g, "A_XG_path_landau_gauge_diagnostic")
            @test !haskey(g, "dos_M")
            @test !haskey(g, "A_k0")
        end
    end
end

@testset "finite magnetic field rejects incompatible twist features" begin
    mktempdir() do out_dir
        p = ModelParameters(4, 4, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                            η=0.25, Δω=0.25, ω_max=2.0,
                            n_flux_sc=2, boundary_condition=:magnetic_pbc)
        @test_throws ErrorException run_simulation(p, out_dir;
                                                   n_therm=0, n_measure=0,
                                                   use_twisted_spectra=true,
                                                   spectra_Ltw=2,
                                                   verbose=false)
        @test_throws ErrorException run_simulation(p, out_dir;
                                                   n_therm=0, n_measure=0,
                                                   measure_twist=true,
                                                   verbose=false)
        state = initialize_state(p)
        cache = initialize_cache(p)
        init_static_H!(cache, p, state)
        update_H_BdG!(cache, p, state)
        diagonalize_H_BdG!(cache, p)
        @test_throws ErrorException measure_twisted_spectra(cache, p, state; Ltw=2)
    end
end
```

Run:

```bash
julia --project test/test_simulation_tbc.jl
```

Expected: fail because finite-field spectra still write old momentum fields or keywords are missing.

- [ ] **Step 2: Make spectra result momentum fields optional**

In `src/Observables.jl`, change `SpectrumResult` and `SpectraOnlyResult` momentum fields to allow `nothing`:

```julia
dos_M::Union{Nothing,Vector{Float64}}
A_k_ω0::Union{Nothing,Matrix{Float64}}
A_MX_path::Union{Nothing,Matrix{Float64}}
A_XG_path::Union{Nothing,Matrix{Float64}}
dos_M_eta::Union{Nothing,Matrix{Float64}}
A_k_ω0_eta::Union{Nothing,Array{Float64,3}}
A_MX_path_eta::Union{Nothing,Array{Float64,3}}
A_XG_path_eta::Union{Nothing,Array{Float64,3}}
```

Update constructors so zero-field calls still pass concrete arrays and finite-field calls can pass `nothing`.

- [ ] **Step 3: Add `include_momentum_spectra` to untwisted spectra measurement**

Change:

```julia
function measure_untwisted_spectra(cache::ComputeCache, p::ModelParameters;
                                   eta_values::AbstractVector{<:Real}=Float64[p.η],
                                   reuse_buffers::Bool=false)
```

to:

```julia
function measure_untwisted_spectra(cache::ComputeCache, p::ModelParameters;
                                   eta_values::AbstractVector{<:Real}=Float64[p.η],
                                   include_momentum_spectra::Bool=true,
                                   reuse_buffers::Bool=false)
```

When `include_momentum_spectra == false`, keep DOS and LDOS accumulation, skip `dos_M`, `A_k_ω0`, `A_MX_path`, and `A_XG_path`, and return `nothing` for those fields. Do not allocate fake zero arrays.

Also change `measure_transport_and_spectra` to accept and forward the same keyword:

```julia
function measure_transport_and_spectra(cache::ComputeCache, p::ModelParameters;
                                       eta_values::AbstractVector{<:Real}=Float64[p.η],
                                       include_momentum_spectra::Bool=true,
                                       reuse_buffers::Bool=false)
    transport = measure_transport_only(cache, p; eta_values=eta_values,
                                       reuse_buffers=reuse_buffers)
    spectra = measure_untwisted_spectra(cache, p;
                                        eta_values=eta_values,
                                        include_momentum_spectra=include_momentum_spectra,
                                        reuse_buffers=reuse_buffers)
    return SpectrumResult(transport.superfluid_stiffness,
                          transport.dc_conductivity,
                          transport.ω_grid,
                          transport.optical_conductivity,
                          spectra.dos_ω_grid,
                          spectra.dos,
                          spectra.dos_M,
                          spectra.ldos_ω0,
                          spectra.A_k_ω0,
                          spectra.A_MX_path,
                          spectra.A_XG_path,
                          transport.dc_conductivity_eta,
                          transport.optical_conductivity_eta,
                          spectra.dos_eta,
                          spectra.dos_M_eta,
                          spectra.ldos_ω0_eta,
                          spectra.A_k_ω0_eta,
                          spectra.A_MX_path_eta,
                          spectra.A_XG_path_eta)
end
```

- [ ] **Step 4: Add simulation fail-fast and metadata**

In `run_simulation`, add keyword:

```julia
allow_gauge_dependent_spectra::Bool=false
```

Add validation after spectra options are computed:

```julia
finite_field = p.n_flux_sc != 0
if finite_field && use_twisted_spectra
    error("use_twisted_spectra is not supported for finite magnetic field")
end
if finite_field && measure_twist
    error("measure_twist is not supported for finite magnetic field")
end
include_momentum_spectra = !finite_field || allow_gauge_dependent_spectra
```

Pass `include_momentum_spectra` to `measure_transport_and_spectra` and `measure_untwisted_spectra`. Write metadata:

```julia
mag_meta = magnetic_metadata(cache.magnetic)
gauge_dependent_spectra = finite_field && allow_gauge_dependent_spectra
spectra_gauge = finite_field ? "Landau gauge" : "none"
spectra_interpretation = gauge_dependent_spectra ?
    "diagnostic only; not a gauge-invariant momentum-resolved spectral function" :
    "gauge-invariant or not momentum-resolved"
```

In `src/TwistedSpectra.jl`, add the same direct-call guard at the start of `measure_twisted_spectra` after the keyword validation begins:

```julia
if p.n_flux_sc != 0
    error("measure_twisted_spectra is not supported for finite magnetic field")
end
```

- [ ] **Step 5: Skip missing momentum arrays during binning and JLD2 writes**

In the bin accumulator section of `run_simulation`, only initialize/add/divide/write momentum arrays when `spec_res.A_k_ω0 !== nothing`. For example:

```julia
if spec_res.A_k_ω0 !== nothing
    accum_Ak0 = copy(spec_res.A_k_ω0)
else
    accum_Ak0 = nothing
end
```

At JLD2 write time:

```julia
if spec_res.A_k_ω0 !== nothing
    if gauge_dependent_spectra
        g["A_k_omega0_landau_gauge_diagnostic"] = accum_Ak0
        g["A_MX_path_landau_gauge_diagnostic"] = accum_AMXpath
        g["A_XG_path_landau_gauge_diagnostic"] = accum_AXGpath
        g["A_k_omega0_eta_landau_gauge_diagnostic"] = accum_Ak0_eta
        g["A_MX_path_eta_landau_gauge_diagnostic"] = accum_AMXpath_eta
        g["A_XG_path_eta_landau_gauge_diagnostic"] = accum_AXGpath_eta
    else
        g["A_k0"] = accum_Ak0
        g["A_MX_path"] = accum_AMXpath
        g["A_XG_path"] = accum_AXGpath
        g["A_k0_eta"] = accum_Ak0_eta
        g["A_MX_path_eta"] = accum_AMXpath_eta
        g["A_XG_path_eta"] = accum_AXGpath_eta
    end
end
if spec_res.dos_M !== nothing
    if gauge_dependent_spectra
        g["dos_M_landau_gauge_diagnostic"] = accum_dos_M
        g["dos_M_eta_landau_gauge_diagnostic"] = accum_dos_M_eta
    else
        g["dos_M"] = accum_dos_M
        g["dos_M_eta"] = accum_dos_M_eta
    end
end
```

Use diagnostic names only when `allow_gauge_dependent_spectra=true` and `p.n_flux_sc != 0`.

- [ ] **Step 6: Make post-processing momentum fields optional**

In `scripts/process_spectra.jl` and `scripts/batch_process_spectra.jl`, push `dos_M`, `A_k0`, `A_MX_path`, and `A_XG_path` only when the keys exist. Use this pattern:

```julia
ak_multi_key = haskey(g, "A_k_omega0_eta_landau_gauge_diagnostic") ? "A_k_omega0_eta_landau_gauge_diagnostic" : "A_k0_eta"
ak_key = haskey(g, "A_k_omega0_landau_gauge_diagnostic") ? "A_k_omega0_landau_gauge_diagnostic" : "A_k0"
mx_multi_key = haskey(g, "A_MX_path_eta_landau_gauge_diagnostic") ? "A_MX_path_eta_landau_gauge_diagnostic" : "A_MX_path_eta"
mx_key = haskey(g, "A_MX_path_landau_gauge_diagnostic") ? "A_MX_path_landau_gauge_diagnostic" : "A_MX_path"
xg_multi_key = haskey(g, "A_XG_path_eta_landau_gauge_diagnostic") ? "A_XG_path_eta_landau_gauge_diagnostic" : "A_XG_path_eta"
xg_key = haskey(g, "A_XG_path_landau_gauge_diagnostic") ? "A_XG_path_landau_gauge_diagnostic" : "A_XG_path"

has_momentum = haskey(g, ak_key) || haskey(g, ak_multi_key)
if has_momentum
    push!(list_ak, selected_matrix(g, ak_multi_key, ak_key, eta_idx))
    push!(list_mx_path, selected_matrix(g, mx_multi_key, mx_key, eta_idx))
    push!(list_xg_path, selected_matrix(g, xg_multi_key, xg_key, eta_idx))
end
dos_m_multi_key = haskey(g, "dos_M_eta_landau_gauge_diagnostic") ? "dos_M_eta_landau_gauge_diagnostic" : "dos_M_eta"
dos_m_key = haskey(g, "dos_M_landau_gauge_diagnostic") ? "dos_M_landau_gauge_diagnostic" : "dos_M"
if haskey(g, dos_m_key) || haskey(g, dos_m_multi_key)
    push!(list_dos_M, selected_vector(g, dos_m_multi_key, dos_m_key, eta_idx))
end
```

Only write `processed_dos_M.csv`, `processed_ak0.csv`, `processed_MX_path.csv`, `processed_XG_path.csv`, `processed_dos_AN.csv`, `processed_dos_node.csv`, and path peak summaries when the relevant samples are present. Otherwise `rm(...; force=true)` those outputs.

Mirror the same optional behavior in `projectHPC/example/batch_process_spectra.jl`.

- [ ] **Step 7: Run simulation and post-processing tests**

Run:

```bash
julia --project test/test_simulation_tbc.jl
julia --project test/test_postprocess_spectra.jl
```

Expected: both pass.

- [ ] **Step 8: Commit spectra gating**

```bash
git add src/Observables.jl src/Simulation.jl src/TwistedSpectra.jl scripts/spectra_postprocess_utils.jl scripts/process_spectra.jl scripts/batch_process_spectra.jl projectHPC/example/spectra_postprocess_utils.jl projectHPC/example/batch_process_spectra.jl test/test_simulation_tbc.jl test/test_postprocess_spectra.jl
git commit -m "禁用有限磁场动量谱默认输出"
```

---

### Task 7: Simulation Metadata and HPC Parameters

**Files:**
- Modify: `src/Simulation.jl`
- Modify: `projectHPC/run_conf.jl`
- Modify: `projectHPC/example/sweep_T.sh`
- Modify: `projectHPC/example/batch_process_csv.jl`
- Modify: `test/test_hpc_scripts.jl`
- Modify: `test/test_magnetic_field.jl`

- [ ] **Step 1: Add failing metadata tests**

Append to `test/test_magnetic_field.jl`:

```julia
@testset "Finite-field metadata is written" begin
    mktempdir() do out_dir
        p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=-2, boundary_condition=:magnetic_pbc)
        run_simulation(p, out_dir;
                       n_therm=0,
                       n_measure=1,
                       Nt_measure=1,
                       measure_transport_freq=1,
                       bin_size=1,
                       verbose=false)
        jldopen(joinpath(out_dir, "spectra_bins.jld2"), "r") do file
            @test file["n_flux_sc"] == -2
            @test file["boundary_condition"] == :magnetic_pbc
            @test file["flux_density_sc"] == -2 / 16
            @test file["plaquette_phase"] ≈ -π * 2 / 16
            @test file["magnetic_gauge"] == "Landau gauge"
            @test file["magnetic_pbc"] == true
            @test file["pairing_scalar_convention"] == "bare Landau-gauge diagnostic"
            @test file["pairing_scalar_gauge_invariant"] == false
            @test file["conductivity_convention"] == "sigma_xx_regular"
        end
    end
end
```

Run:

```bash
julia --project test/test_magnetic_field.jl
```

Expected: fail because metadata keys are missing.

- [ ] **Step 2: Write metadata in `run_simulation`**

In the JLD2 initialization blocks in `src/Simulation.jl`, add:

```julia
mag_meta = magnetic_metadata(cache.magnetic)
```

and include the fields:

```julia
n_flux_sc=mag_meta.n_flux_sc,
boundary_condition=p.boundary_condition,
flux_density_sc=mag_meta.flux_density_sc,
plaquette_phase=mag_meta.plaquette_phase,
magnetic_gauge=mag_meta.magnetic_gauge,
magnetic_pbc=mag_meta.magnetic_pbc,
gauge_dependent_spectra=gauge_dependent_spectra,
spectra_gauge=spectra_gauge,
spectra_interpretation=spectra_interpretation,
pairing_scalar_convention=p.n_flux_sc == 0 ? "bare zero-field convention" : "bare Landau-gauge diagnostic",
pairing_scalar_gauge_invariant=p.n_flux_sc == 0,
conductivity_convention="sigma_xx_regular",
```

Also log:

```julia
tee_println("Magnetic field: n_flux_sc=$(p.n_flux_sc), boundary=$(p.boundary_condition), flux_density_sc=$(mag_meta.flux_density_sc), plaquette_phase=$(mag_meta.plaquette_phase)")
```

- [ ] **Step 3: Pass parameters through HPC runner**

In `projectHPC/run_conf.jl`, define defaults after `include(params_path)`:

```julia
write_gauge_pair_bonds_freq = isdefined(@__MODULE__, :write_gauge_pair_bonds_freq) ?
                              getfield(@__MODULE__, :write_gauge_pair_bonds_freq) : 0
allow_gauge_dependent_spectra = isdefined(@__MODULE__, :allow_gauge_dependent_spectra) ?
                                getfield(@__MODULE__, :allow_gauge_dependent_spectra) : false
```

Print:

```julia
println("Magnetic options: n_flux_sc=$(p.n_flux_sc), boundary=$(p.boundary_condition), write_gauge_pair_bonds_freq=$(write_gauge_pair_bonds_freq), allow_gauge_dependent_spectra=$(allow_gauge_dependent_spectra)")
```

Add the two new arguments to `worker_task` and pass them into `run_simulation`.

- [ ] **Step 4: Add magnetic options to example sweep**

In `projectHPC/example/sweep_T.sh`, add near spectra/twist defaults:

```bash
n_flux_sc=0
boundary_condition=:periodic
write_gauge_pair_bonds_freq=0
allow_gauge_dependent_spectra=false
```

Include `Nv${n_flux_sc}` in output root/job tags. In generated `params.jl`, pass:

```julia
p = ModelParameters(Lx, Ly, t, tp, W, n_imp, β, V, mass;
                    target_n=target_n,
                    μ_init=mu_init,
                    η=eta,
                    Δω=delta_omega,
                    ω_max=omega_max,
                    n_flux_sc=n_flux_sc,
                    boundary_condition=boundary_condition)
write_gauge_pair_bonds_freq = $write_gauge_pair_bonds_freq
allow_gauge_dependent_spectra = $allow_gauge_dependent_spectra
```

Use the same keyword additions in the fixed-`μ` branch.

- [ ] **Step 5: Run HPC script tests**

Run:

```bash
julia --project test/test_hpc_scripts.jl
julia --project test/test_magnetic_field.jl
```

Expected: both pass.

- [ ] **Step 6: Commit metadata and HPC integration**

```bash
git add src/Simulation.jl projectHPC/run_conf.jl projectHPC/example/sweep_T.sh projectHPC/example/batch_process_csv.jl test/test_hpc_scripts.jl test/test_magnetic_field.jl
git commit -m "接入有限磁场HPC参数"
```

---

### Task 8: Documentation

**Files:**
- Create: `doc/magnetic-field.md`
- Modify: `doc/theory.md`
- Modify: `doc/observables.md`
- Modify: `README.md`

- [ ] **Step 1: Write finite-field documentation**

Create `doc/magnetic-field.md`:

````markdown
# 有限磁场约定

本程序第一版只支持 torus 上的 magnetic periodic boundary condition。用户使用
`n_flux_sc::Int` 指定穿过整个 `Lx x Ly` 计算胞的 `hc/2e` 超导磁通量子数。
它也是预期 vortex 数。构造器也接受等价别名 `n_vortices::Int`，但内部和
输出 metadata 统一记录为 `n_flux_sc`。`n_flux_sc` 可以为正或负；正负号表示磁场方向。

magnetic PBC 要求 `n_flux_sc` 为偶数。程序内部使用

```julia
flux_density_sc = n_flux_sc / (Lx * Ly)
plaquette_phase = pi * flux_density_sc
```

`flux_density_sc` 不命名为 `phi`，避免和 Peierls link phase 混淆。

坐标按 0-based 解释。Landau gauge link 约定为

```text
U_y(x,y) = cis(plaquette_phase * x)
U_x(x,y) = 1
U_x(Lx-1,y) = cis(-plaquette_phase * Lx * y)
```

对角 `t'` bond 使用直线 Peierls 积分和同一个 endpoint seam patch。有限磁场下
BdG pairing block 仍使用采样的裸 `Delta_ij`，不额外乘 Peierls phase。
既有 CSV 标量列（例如 `Delta_x - Delta_y`、`Delta_Diff`、`Delta_Pair`）
仍保持裸 convention 以兼容旧脚本。规范协变 pairing 只在 observable 或 JLD2
输出层构造，默认不写；设置 `write_gauge_pair_bonds_freq > 0` 时写出
`delta_bond_landau_gauge_covariant` 和 `pair_bond_landau_gauge_covariant`。
有限磁场后处理应优先使用 fermionic `pair_bond_landau_gauge_covariant`；
`delta_bond_landau_gauge_covariant` 是辅助场诊断。

`Superfluid_Stiffness` 是 transverse `xx` Meissner estimator：

```text
rho_s = <-K_x> - Lambda_xx(qx=0, qy=2pi/Ly, omega=0)
```

有限 `q_y` current 使用与 `measure_twist_stiffness_qy` 一致的
`sqrt(2) * cos(q_y y)` real-probe normalization；strict `m == n` 项仍按当前
production Kubo convention 跳过，full free-energy curvature 只作为单独诊断。
有限垂直磁场下它是 field-suppressed stiffness diagnostic，不是零场 BKT
universal jump 的 `T_c` 判据。

`DC_Conductivity` 是 regular `sigma_xx`，不包含超流 delta 函数。第一版没有
Hall conductivity，因此 `1 / sigma_xx` 只能作为 proxy，不能直接称为物理
`rho_xx`。

普通 FFT 的 `A(k,omega)`、`dos_M`、MX/XG path 在有限轨道磁场下是
gauge-dependent。默认不输出；只有 `allow_gauge_dependent_spectra=true` 时
才作为 Landau-gauge diagnostic 输出，并使用
`A_k_omega0_landau_gauge_diagnostic`、`A_MX_path_landau_gauge_diagnostic`、
`A_XG_path_landau_gauge_diagnostic` 和 `dos_M_landau_gauge_diagnostic`
这类带 warning 的 JLD2 key。

第一版没有实现 magnetic unit cell、magnetic Bloch theorem、Hall conductivity
或 magnetic-translation-covariant momentum spectra。

有限磁场低温热化可能比零场慢，因为 vortex 位置、pairing phase texture 和
无序势需要共同松弛。正式 HPC 扫描前先做 `N_CONFS=1` 或小 worker 数的短跑，
检查接受率、内存占用、`Superfluid_Stiffness`、`DC_Conductivity` 和
`pair_bond_landau_gauge_covariant` 是否稳定，再扩展到完整并行作业。
````

- [ ] **Step 2: Link docs from existing theory and observables docs**

In `doc/theory.md`, add a short section after the BdG Hamiltonian section:

```markdown
## 有限轨道磁场

有限磁场的实现和符号约定见 `doc/magnetic-field.md`。核心约定是磁场只通过
normal hopping 的 Peierls phase 进入，BdG pairing block 仍使用裸辅助场
`\Delta_{ij}`。
```

In `doc/observables.md`, add near the stiffness section:

```markdown
有限磁场下，`Superfluid_Stiffness` 仍是 transverse `xx` Meissner estimator，
不是 BKT universal jump 判据。`DC_Conductivity` 是 regular `sigma_xx`；
没有计算 Hall 分量时不要把 `1 / sigma_xx` 直接解释为物理纵向电阻率。
有限磁场 pairing 和谱函数的规范约定见 `doc/magnetic-field.md`。
```

In `README.md`, add:

```markdown
Finite-field conventions are documented in `doc/magnetic-field.md`.
```

- [ ] **Step 3: Run doc-adjacent tests**

Run:

```bash
julia --project test/test_core_smoke.jl
```

Expected: pass.

- [ ] **Step 4: Commit docs**

```bash
git add doc/magnetic-field.md doc/theory.md doc/observables.md README.md
git commit -m "补充有限磁场文档"
```

---

### Task 9: Full Verification and Cleanup

**Files:**
- All modified files

- [ ] **Step 1: Run targeted fast tests**

Run:

```bash
julia --project test/test_magnetic_field.jl
julia --project test/test_hamiltonian.jl
julia --project test/test_forces.jl
julia --project test/test_pairing_convention.jl
julia --project test/test_twist_stiffness.jl
julia --project test/test_simulation_tbc.jl
julia --project test/test_postprocess_spectra.jl
julia --project test/test_hpc_scripts.jl
```

Expected: all pass.

- [ ] **Step 2: Run full default suite**

Run:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Expected: pass. If runtime is excessive, capture the completed targeted tests and the point where the full suite stopped.

- [ ] **Step 3: Run one finite-field smoke simulation manually**

Run:

```bash
julia --project -e '
using DwaveHMC, Random, JLD2
Random.seed!(20260521)
p = ModelParameters(4, 4, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                    η=0.25, Δω=0.25, ω_max=2.0,
                    n_flux_sc=2, boundary_condition=:magnetic_pbc)
out = mktempdir()
run_simulation(p, out; n_therm=0, n_measure=2, Nt_measure=1,
               measure_transport_freq=1, bin_size=1,
               write_gauge_pair_bonds_freq=1, verbose=false)
println(out)
println(read(joinpath(out, "transport.csv"), String))
jldopen(joinpath(out, "spectra_bins.jld2"), "r") do f
    @assert f["n_flux_sc"] == 2
    @assert !haskey(f["sweep_1"], "A_k0")
end
'
```

Expected: command exits successfully and prints finite `Superfluid_Stiffness` and `DC_Conductivity` rows.

- [ ] **Step 4: Inspect allocations-risk areas**

Run:

```bash
rg -n "cis\\(|link_phase\\(|write_gauge_pair_bonds_freq|allow_gauge_dependent_spectra|dos_M_landau" src test scripts projectHPC doc
```

Expected:

- `cis(` appears in magnetic cache construction, probe diagnostics, and existing twist/TBC code, not inside repeated finite-field Kubo inner loops except through precomputed arrays or explicit diagnostics.
- `allow_gauge_dependent_spectra` appears in simulation and metadata paths.
- no finite-field default path writes fake `A_k0`, `A_MX_path`, `A_XG_path`, or `dos_M`.

- [ ] **Step 5: Inspect git status and commit cleanup edits**

If verification requires small cleanup edits:

```bash
git status --short
git add src/MagneticFieldTypes.jl src/MagneticField.jl src/DwaveHMC.jl src/Types.jl src/Hamiltonian.jl src/Observables.jl src/Simulation.jl src/TwistedSpectra.jl scripts/spectra_postprocess_utils.jl scripts/process_spectra.jl scripts/batch_process_spectra.jl projectHPC/run_conf.jl projectHPC/example/spectra_postprocess_utils.jl projectHPC/example/batch_process_spectra.jl projectHPC/example/batch_process_csv.jl projectHPC/example/sweep_T.sh test/test_magnetic_field.jl test/runtests.jl test/test_simulation_tbc.jl test/test_postprocess_spectra.jl test/test_hpc_scripts.jl doc/magnetic-field.md doc/theory.md doc/observables.md README.md
git commit -m "清理有限磁场验证问题"
```

If no cleanup is needed, do not create an empty commit.
