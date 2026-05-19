# Multi-eta Spectra Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store exact multi-`η` spectra/DC results in JLD2 for `η, 2η, 4η, 8η, 16η, 32η, 64η`, while default CSV workflows keep using `eta_factor = 1`.

**Architecture:** Add a small `MultiEta.jl` helper module area for validation and factor selection, then thread validated `eta_values` through transport, untwisted spectra, TBC spectra, simulation binning, and post-processing. TBC spectra use `spectra_eta` as their base broadening, while scalar transport/DC keeps `p.η` as its base broadening. Existing single-`η` fields remain first-slice compatibility fields; multi-`η` arrays use `η` as dimension 1.

**Tech Stack:** Julia, DwaveHMC structs/functions, JLD2, existing script-based post-processing, `Test`.

---

## File Structure

- Create `src/MultiEta.jl`: default factors, validation, factor lookup, and small slicing helpers shared by simulation and scripts.
- Modify `src/DwaveHMC.jl`: include `MultiEta.jl` before measurement/simulation code.
- Modify `src/Observables.jl`: extend result structs and implement multi-`η` transport plus untwisted spectra accumulation.
- Modify `src/TwistedSpectra.jl`: extend TBC result struct and implement multi-`η` TBC spectra accumulation.
- Modify `src/Simulation.jl`: add `spectra_eta_factors`, write metadata, accumulate/write multi-`η` JLD2 fields, keep old CSV fields as factor-1 data.
- Modify `scripts/spectra_postprocess_utils.jl`: add `eta_factor` index helpers and field selection helpers for old/new JLD2 data.
- Modify `scripts/process_spectra.jl` and `scripts/batch_process_spectra.jl`: add `eta_factor` keyword and use selected slices when exporting standard CSV names.
- Modify `projectHPC/example/spectra_postprocess_utils.jl` and `projectHPC/example/batch_process_spectra.jl`: mirror post-processing selection for ensemble summaries.
- Modify `test/test_simulation_tbc.jl`: validate JLD2 metadata, multi-`η` fields, first-slice compatibility, and invalid factor lists.
- Modify `test/test_postprocess_spectra.jl`: validate selected `eta_factor` export and old-format compatibility.

---

### Task 1: Multi-eta Validation Helpers

**Files:**
- Create: `src/MultiEta.jl`
- Modify: `src/DwaveHMC.jl`
- Test: `test/test_simulation_tbc.jl`

- [ ] **Step 1: Write failing validation tests**

Add this testset near the top of `test/test_simulation_tbc.jl`, after `run_tiny_spectra_simulation`:

```julia
@testset "spectra eta factor validation" begin
    @test DwaveHMC.validate_spectra_eta_factors([1, 2, 4]) == [1.0, 2.0, 4.0]
    @test DwaveHMC.validate_spectra_eta_factors((1, 2, 4, 8)) == [1.0, 2.0, 4.0, 8.0]
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors(Float64[])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([0, 1, 2])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([1, -2, 4])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([1, NaN, 4])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([1, Inf, 4])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([2, 4, 8])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([2, 1, 4])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([1, 2, 2])

    factors = [1.0, 2.0, 4.0, 8.0]
    @test DwaveHMC.eta_factor_index(factors, 1) == 1
    @test DwaveHMC.eta_factor_index(factors, 4) == 3
    @test_throws ErrorException DwaveHMC.eta_factor_index(factors, 16)
end
```

- [ ] **Step 2: Run validation tests and verify failure**

Run:

```bash
julia --project test/test_simulation_tbc.jl
```

Expected: fail with `UndefVarError: validate_spectra_eta_factors not defined`.

- [ ] **Step 3: Implement `src/MultiEta.jl`**

Create `src/MultiEta.jl` with:

```julia
const DEFAULT_SPECTRA_ETA_FACTORS = Float64[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
const ETA_FACTOR_ATOL = 1.0e-12

function validate_spectra_eta_factors(factors)::Vector{Float64}
    vals = Float64.(collect(factors))
    isempty(vals) && error("spectra_eta_factors must be non-empty")
    all(isfinite, vals) || error("spectra_eta_factors must be finite")
    all(vals .> 0.0) || error("spectra_eta_factors must be positive")
    isapprox(vals[1], 1.0; atol=ETA_FACTOR_ATOL, rtol=0.0) ||
        error("spectra_eta_factors must start with 1")

    @inbounds for i in eachindex(vals)
        for j in firstindex(vals):(i - 1)
            if isapprox(vals[i], vals[j]; atol=ETA_FACTOR_ATOL, rtol=0.0)
                error("spectra_eta_factors must not contain duplicate factors")
            end
        end
    end

    return vals
end

function eta_factor_index(factors, eta_factor)::Int
    factor = Float64(eta_factor)
    @inbounds for i in eachindex(factors)
        if isapprox(Float64(factors[i]), factor; atol=ETA_FACTOR_ATOL, rtol=0.0)
            return i
        end
    end
    error("eta_factor=$factor not found. Available factors: $(collect(factors))")
end

eta_values_from_base(base_eta::Real, factors::AbstractVector{<:Real}) =
    Float64(base_eta) .* Float64.(factors)
```

- [ ] **Step 4: Include helpers**

Modify `src/DwaveHMC.jl` so includes start with:

```julia
include("MultiEta.jl")
include("Types.jl")
include("Hamiltonian.jl")
```

Do not export these helpers; tests and scripts should call them as `DwaveHMC.validate_spectra_eta_factors` and `DwaveHMC.eta_factor_index`.

- [ ] **Step 5: Run validation tests and verify pass**

Run:

```bash
julia --project test/test_simulation_tbc.jl
```

Expected: validation testset passes; simulation field assertions are added in Task 5.

- [ ] **Step 6: Commit**

```bash
git add src/MultiEta.jl src/DwaveHMC.jl test/test_simulation_tbc.jl
git commit -m "添加eta因子校验"
```

---

### Task 2: Multi-eta Result Structs

**Files:**
- Modify: `src/Observables.jl`
- Modify: `src/TwistedSpectra.jl`

- [ ] **Step 1: Extend result structs in `src/Observables.jl`**

Change `SpectrumResult`, `TransportResult`, and `SpectraOnlyResult` to include multi-`η` fields while preserving existing field names:

```julia
struct SpectrumResult
    superfluid_stiffness::Float64
    dc_conductivity::Float64
    ω_grid::Vector{Float64}
    optical_conductivity::Vector{Float64}
    dos_ω_grid::Vector{Float64}
    dos::Vector{Float64}
    dos_M::Vector{Float64}
    ldos_ω0::Vector{Float64}
    A_k_ω0::Matrix{Float64}
    A_MX_path::Matrix{Float64}
    A_XG_path::Matrix{Float64}
    dc_conductivity_eta::Vector{Float64}
    optical_conductivity_eta::Matrix{Float64}
    dos_eta::Matrix{Float64}
    dos_M_eta::Matrix{Float64}
    ldos_ω0_eta::Matrix{Float64}
    A_k_ω0_eta::Array{Float64, 3}
    A_MX_path_eta::Array{Float64, 3}
    A_XG_path_eta::Array{Float64, 3}
end

struct TransportResult
    superfluid_stiffness::Float64
    dc_conductivity::Float64
    ω_grid::Vector{Float64}
    optical_conductivity::Vector{Float64}
    dc_conductivity_eta::Vector{Float64}
    optical_conductivity_eta::Matrix{Float64}
end

struct SpectraOnlyResult
    dos_ω_grid::Vector{Float64}
    dos::Vector{Float64}
    dos_M::Vector{Float64}
    ldos_ω0::Vector{Float64}
    A_k_ω0::Matrix{Float64}
    A_MX_path::Matrix{Float64}
    A_XG_path::Matrix{Float64}
    dos_eta::Matrix{Float64}
    dos_M_eta::Matrix{Float64}
    ldos_ω0_eta::Matrix{Float64}
    A_k_ω0_eta::Array{Float64, 3}
    A_MX_path_eta::Array{Float64, 3}
    A_XG_path_eta::Array{Float64, 3}
end
```

- [ ] **Step 2: Extend `TwistedSpectraResult`**

In `src/TwistedSpectra.jl`, add multi-`η` fields before metadata:

```julia
struct TwistedSpectraResult
    dos_ω_grid::Vector{Float64}
    dos::Vector{Float64}
    dos_M::Vector{Float64}
    dos_M_patch::Vector{Float64}
    ldos_ω0::Vector{Float64}
    A_k_ω0::Matrix{Float64}
    A_MX_path::Matrix{Float64}
    A_XG_path::Matrix{Float64}
    A_XG_node_patch::Matrix{Float64}
    dos_eta::Matrix{Float64}
    dos_M_eta::Matrix{Float64}
    dos_M_patch_eta::Matrix{Float64}
    ldos_ω0_eta::Matrix{Float64}
    A_k_ω0_eta::Array{Float64, 3}
    A_MX_path_eta::Array{Float64, 3}
    A_XG_path_eta::Array{Float64, 3}
    A_XG_node_patch_eta::Array{Float64, 3}
    kx_grid::Vector{Float64}
    ky_grid::Vector{Float64}
    mx_path_kx::Float64
    mx_path_ky::Vector{Float64}
    xg_path_kx::Vector{Float64}
    xg_path_ky::Vector{Float64}
    Ltw::Int
    m_point_patch_half_width::Float64
    spectra_eta::Float64
    spectra_delta_omega::Float64
end
```

- [ ] **Step 3: Run tests and capture constructor failures**

Run:

```bash
julia --project test/test_twisted_spectra.jl
```

Expected: fail at constructors that still pass the old argument list.

- [ ] **Step 4: Defer commit until constructors compile**

Do not commit this task alone if tests fail. The commit happens in Task 4 after all constructors compile.

---

### Task 3: Multi-eta Transport and Untwisted Spectra

**Files:**
- Modify: `src/Observables.jl`
- Test: `test/test_twisted_spectra.jl`
- Test: `test/test_simulation_tbc.jl`

- [ ] **Step 1: Add focused direct-measurement tests**

In `test/test_twisted_spectra.jl`, add a testset that calls the untwisted measurement with two factors and checks first-slice compatibility:

```julia
@testset "untwisted multi eta first slice compatibility" begin
    Random.seed!(20260520)
    p = ModelParameters(4, 4, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                        η=0.25, Δω=0.25, ω_max=2.0)
    state = initialize_state(p)
    cache = initialize_cache(p)
    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)

    eta_values = [p.η, 2p.η]
    spec = measure_transport_and_spectra(cache, p; eta_values=eta_values)

    @test length(spec.dc_conductivity_eta) == 2
    @test size(spec.optical_conductivity_eta) == (2, length(spec.ω_grid))
    @test size(spec.dos_eta) == (2, length(spec.dos_ω_grid))
    @test size(spec.A_MX_path_eta, 1) == 2
    @test spec.dc_conductivity == spec.dc_conductivity_eta[1]
    @test spec.optical_conductivity == vec(spec.optical_conductivity_eta[1, :])
    @test spec.dos == vec(spec.dos_eta[1, :])
    @test spec.A_MX_path == spec.A_MX_path_eta[1, :, :]
end
```

- [ ] **Step 2: Run direct-measurement test and verify failure**

Run:

```bash
julia --project test/test_twisted_spectra.jl
```

Expected: fail because `measure_transport_and_spectra` has no `eta_values` keyword.

- [ ] **Step 3: Update `measure_transport_only` signature and accumulation**

Change the signature to:

```julia
function measure_transport_only(cache::ComputeCache, p::ModelParameters;
                                eta_values::AbstractVector{<:Real}=Float64[p.η],
                                reuse_buffers::Bool=false)
```

Replace scalar `σ_ω`/`dc_cond` accumulation for conductivity with:

```julia
eta_vals = Float64.(eta_values)
nη = length(eta_vals)
σ_ω_eta = zeros(Float64, nη, length(ω_grid))
dc_cond_eta = zeros(Float64, nη)

@inbounds for n in 1:dim
    fprime = β * f[n] * (1.0 - f[n])
    for m in 1:dim
        m == n && continue
        Em_En = E[m] - E[n]
        J2 = abs2(J_mn[n, m])

        for iη in 1:nη
            dc_cond_eta[iη] += fprime * J2 * lorentzian(Em_En, eta_vals[iη])
        end

        fn_fm = f[n] - f[m]
        if abs(fn_fm) < 1e-12
            continue
        end
        for iω in eachindex(ω_grid)
            coef = fn_fm * omega_inv[iω] * J2
            x = ω_grid[iω] - Em_En
            for iη in 1:nη
                σ_ω_eta[iη, iω] += coef * lorentzian(x, eta_vals[iη])
            end
        end
    end
end

dc_cond_eta .*= (π / N)
σ_ω_eta .*= (π / N)
copyto!(σ_ω, view(σ_ω_eta, 1, :))
dc_cond = dc_cond_eta[1]
```

Return:

```julia
if reuse_buffers
    return TransportResult(superfluid_stiffness, dc_cond, ω_grid, σ_ω,
                           dc_cond_eta, σ_ω_eta)
end

return TransportResult(superfluid_stiffness, dc_cond, copy(ω_grid), copy(σ_ω),
                       copy(dc_cond_eta), copy(σ_ω_eta))
```

- [ ] **Step 4: Update `measure_untwisted_spectra` signature and arrays**

Change the signature to:

```julia
function measure_untwisted_spectra(cache::ComputeCache, p::ModelParameters;
                                   eta_values::AbstractVector{<:Real}=Float64[p.η],
                                   reuse_buffers::Bool=false)
```

At the start of the spectra section allocate multi-`η` arrays:

```julia
eta_vals = Float64.(eta_values)
nη = length(eta_vals)
nω = length(dos_ω_grid)
dos_eta = zeros(Float64, nη, nω)
dos_M_eta = zeros(Float64, nη, nω)
ldos_eta = zeros(Float64, nη, N)
ak_eta = zeros(Float64, nη, Lx, Ly)
ak_mx_eta = zeros(Float64, nη, n_mx_path, nω)
ak_xg_eta = zeros(Float64, nη, n_xg_path, nω)
```

For each eigenstate, replace the single `lor_cache` update with a per-factor loop:

```julia
for iη in 1:nη
    η = eta_vals[iη]
    for iw in eachindex(dos_ω_grid)
        lor_cache[iw] = lorentzian(dos_ω_grid[iw] - En, η)
    end

    for iw in eachindex(dos_ω_grid)
        lor_val = lor_cache[iw]
        dos_eta[iη, iw] += w_n * lor_val
        dos_M_eta[iη, iw] += weight_M * lor_val
        for k in 1:n_mx_path
            ak_mx_eta[iη, k, iw] += mx_path_weights[k] * lor_val
        end
        for k in 1:n_xg_path
            ak_xg_eta[iη, k, iw] += xg_path_weights[k] * lor_val
        end
    end

    weight_at_zero = lorentzian(-En, η)
    for i in 1:N
        ldos_eta[iη, i] += abs2(U[i, n]) * weight_at_zero
    end
    if weight_at_zero > 1e-6
        for y in 1:Ly, x in 1:Lx
            ak_eta[iη, x, y] += abs2(cache.u_k_cache[x, y]) * weight_at_zero
        end
    end
end
```

After the eigenstate loop, normalize all factors and copy factor 1 into old buffers:

```julia
dos_eta ./= N
ak_eta ./= N
copyto!(dos_vals, view(dos_eta, 1, :))
copyto!(dos_M_vals, view(dos_M_eta, 1, :))
copyto!(ldos_ω0, view(ldos_eta, 1, :))
copyto!(ak_map, view(ak_eta, 1, :, :))
copyto!(ak_mx_path, view(ak_mx_eta, 1, :, :))
copyto!(ak_xg_path, view(ak_xg_eta, 1, :, :))
```

Return the new fields in `SpectraOnlyResult`.

- [ ] **Step 5: Update `measure_transport_and_spectra`**

Change signature:

```julia
function measure_transport_and_spectra(cache::ComputeCache, p::ModelParameters;
                                       eta_values::AbstractVector{<:Real}=Float64[p.η],
                                       reuse_buffers::Bool=false)
```

Call both sub-measurements with the same `eta_values`, then construct `SpectrumResult` with old fields and all multi-`η` fields:

```julia
transport = measure_transport_only(cache, p; eta_values=eta_values, reuse_buffers=reuse_buffers)
spectra = measure_untwisted_spectra(cache, p; eta_values=eta_values, reuse_buffers=reuse_buffers)

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
```

- [ ] **Step 6: Run direct measurement tests**

Run:

```bash
julia --project test/test_twisted_spectra.jl
```

Expected: pass existing tests and new untwisted multi-`η` compatibility test.

- [ ] **Step 7: Commit**

```bash
git add src/Observables.jl test/test_twisted_spectra.jl
git commit -m "支持非TBC多eta测量"
```

---

### Task 4: Multi-eta TBC Spectra

**Files:**
- Modify: `src/TwistedSpectra.jl`
- Test: `test/test_twisted_spectra.jl`

- [ ] **Step 1: Add TBC direct-measurement test**

In `test/test_twisted_spectra.jl`, add:

```julia
@testset "TBC multi eta first slice compatibility" begin
    Random.seed!(20260520)
    p = ModelParameters(4, 4, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                        η=0.25, Δω=0.25, ω_max=2.0)
    state = initialize_state(p)
    cache = initialize_cache(p)

    tw = measure_twisted_spectra(cache, p, state;
                                 Ltw=2,
                                 spectra_eta=p.η,
                                 spectra_delta_omega=p.Δω,
                                 eta_values=[p.η, 2p.η])

    @test size(tw.dos_eta, 1) == 2
    @test size(tw.A_MX_path_eta, 1) == 2
    @test size(tw.A_XG_node_patch_eta, 1) == 2
    @test tw.dos == vec(tw.dos_eta[1, :])
    @test tw.dos_M_patch == vec(tw.dos_M_patch_eta[1, :])
    @test tw.A_MX_path == tw.A_MX_path_eta[1, :, :]
    @test tw.A_XG_node_patch == tw.A_XG_node_patch_eta[1, :, :]
end
```

- [ ] **Step 2: Run TBC test and verify failure**

Run:

```bash
julia --project test/test_twisted_spectra.jl
```

Expected: fail because `measure_twisted_spectra` has no `eta_values` keyword.

- [ ] **Step 3: Update `measure_twisted_spectra` signature**

Change signature to include `eta_values`:

```julia
function measure_twisted_spectra(cache::ComputeCache,
                                 p::ModelParameters,
                                 state::SimulationState;
                                 Ltw::Int=2,
                                 m_point_patch_half_width::Float64=π / max(p.Lx, p.Ly),
                                 spectra_eta::Float64=p.η,
                                 spectra_delta_omega::Float64=p.Δω,
                                 eta_values::AbstractVector{<:Real}=Float64[spectra_eta],
                                 reuse_buffers::Bool=false)
```

Normalize and validate local consistency:

```julia
eta_vals = Float64.(eta_values)
isempty(eta_vals) && error("eta_values must be non-empty")
all(isfinite, eta_vals) || error("eta_values must be finite")
all(eta_vals .> 0.0) || error("eta_values must be positive")
nη = length(eta_vals)
```

- [ ] **Step 4: Allocate multi-`η` TBC arrays**

After current single arrays are created, add:

```julia
dos_eta = zeros(Float64, nη, nω)
dos_M_eta = zeros(Float64, nη, nω)
dos_M_patch_eta = zeros(Float64, nη, nω)
ldos_eta = zeros(Float64, nη, N)
A_k0_eta = zeros(Float64, nη, Lx_eff, Ly_eff)
A_MX_path_eta = zeros(Float64, nη, length(mx_path_ky), nω)
A_XG_path_eta = zeros(Float64, nη, length(xg_path_kx), nω)
A_XG_node_patch_eta = zeros(Float64, nη, length(xg_path_kx), nω)
```

- [ ] **Step 5: Replace single-`η` Lorentzian accumulation with factor loop**

Inside the eigenstate loop, after FFT weights are available, accumulate for each factor:

```julia
for iη in 1:nη
    η = eta_vals[iη]
    for iw in eachindex(dos_ω_grid)
        lor_cache[iw] = lorentzian_spectra(dos_ω_grid[iw] - En, η)
        dos_eta[iη, iw] += w_n * lor_cache[iw]
    end

    if has_pi_0_sector || has_0_pi_sector
        for iw in eachindex(dos_ω_grid)
            dos_M_eta[iη, iw] += exact_weight * lor_cache[iw]
        end
    end

    weight_at_zero = lorentzian_spectra(-En, η)
    for i in 1:N
        ldos_eta[iη, i] += abs2(vecs[i, n]) * weight_at_zero
    end

    patch_weight = 0.0
    for my in 0:Ly-1, mx in 0:Lx-1
        Ix = twist_fft_to_effective_index(mx, nx, Lx, Ltw)
        Iy = twist_fft_to_effective_index(my, ny, Ly, Ltw)
        wk = abs2(cache.u_k_cache[mx + 1, my + 1]) / N
        if patch_mask[Ix + 1, Iy + 1]
            patch_weight += wk
        end
        if weight_at_zero > 1e-6
            A_k0_eta[iη, Ix + 1, Iy + 1] += abs2(cache.u_k_cache[mx + 1, my + 1]) * weight_at_zero
        end
    end
    patch_weight /= patch_count
    for iw in eachindex(dos_ω_grid)
        dos_M_patch_eta[iη, iw] += patch_weight * lor_cache[iw]
    end

    if nx == nx_pi
        for my in 0:Ly-1
            Iy = twist_fft_to_effective_index(my, ny, Ly, Ltw)
            if Iy <= fld(Ly_eff, 2)
                wk = abs2(cache.u_k_cache[mx_pi + 1, my + 1]) / N
                path_idx = Iy + 1
                for iw in eachindex(dos_ω_grid)
                    A_MX_path_eta[iη, path_idx, iw] += wk * lor_cache[iw]
                end
            end
        end
    end

    for path_idx in eachindex(xg_path_kx)
        if nx == xg_nx[path_idx] && ny == xg_ny[path_idx]
            wk = abs2(cache.u_k_cache[xg_mx[path_idx] + 1, xg_my[path_idx] + 1]) / N
            for iw in eachindex(dos_ω_grid)
                A_XG_path_eta[iη, path_idx, iw] += wk * lor_cache[iw]
            end
        end
    end

    for (path_idx, mx_term, my_term, weight_factor) in xg_patch_terms_by_sector[sector_idx]
        wk = weight_factor * abs2(cache.u_k_cache[mx_term + 1, my_term + 1]) / N
        for iw in eachindex(dos_ω_grid)
            A_XG_node_patch_eta[iη, path_idx, iw] += wk * lor_cache[iw]
        end
    end
end
```

Keep the FFT and sector bookkeeping outside the factor loop.

- [ ] **Step 6: Normalize all factors and populate compatibility fields**

After sector loops:

```julia
dos_eta ./= (N * Ltw^2)
ldos_eta ./= Ltw^2
A_k0_eta ./= N

dos_vals = vec(copy(@view dos_eta[1, :]))
dos_M_vals = vec(copy(@view dos_M_eta[1, :]))
dos_M_patch_vals = vec(copy(@view dos_M_patch_eta[1, :]))
ldos_ω0 = vec(copy(@view ldos_eta[1, :]))
A_k0 = copy(@view A_k0_eta[1, :, :])
A_MX_path = copy(@view A_MX_path_eta[1, :, :])
A_XG_path = copy(@view A_XG_path_eta[1, :, :])
A_XG_node_patch = copy(@view A_XG_node_patch_eta[1, :, :])
```

Return all new fields in `TwistedSpectraResult`.

- [ ] **Step 7: Run TBC measurement tests**

Run:

```bash
julia --project test/test_twisted_spectra.jl
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add src/TwistedSpectra.jl test/test_twisted_spectra.jl
git commit -m "支持TBC多eta谱函数"
```

---

### Task 5: Simulation Integration and JLD2 Binning

**Files:**
- Modify: `src/Simulation.jl`
- Test: `test/test_simulation_tbc.jl`

- [ ] **Step 1: Extend simulation integration tests**

In the default untwisted test in `test/test_simulation_tbc.jl`, add metadata and field checks inside the JLD2 block:

```julia
@test file["multi_eta_enabled"] == true
@test file["spectra_eta_factors"] == DwaveHMC.DEFAULT_SPECTRA_ETA_FACTORS
@test file["eta_values"] == 0.25 .* DwaveHMC.DEFAULT_SPECTRA_ETA_FACTORS
@test file["spectra_eta_base"] == 0.25
```

Inside `g = file["sweep_1"]`, add:

```julia
for key in ("dc_cond_eta", "opt_cond_eta", "dos_eta", "dos_M_eta",
            "LDOS_0_eta", "A_k0_eta", "A_MX_path_eta", "A_XG_path_eta")
    @test haskey(g, key)
end
@test length(g["dc_cond_eta"]) == 7
@test size(g["opt_cond_eta"], 1) == 7
@test size(g["dos_eta"], 1) == 7
@test size(g["A_MX_path_eta"], 1) == 7
@test g["opt_cond"] == vec(g["opt_cond_eta"][1, :])
@test g["dos"] == vec(g["dos_eta"][1, :])
@test g["A_MX_path"] == g["A_MX_path_eta"][1, :, :]
```

In the TBC test, add:

```julia
@test haskey(g, "dos_M_patch_eta")
@test haskey(g, "A_XG_node_patch_eta")
@test size(g["dos_M_patch_eta"], 1) == 7
@test size(g["A_XG_node_patch_eta"], 1) == 7
@test g["dos_M_patch"] == vec(g["dos_M_patch_eta"][1, :])
@test g["A_XG_node_patch"] == g["A_XG_node_patch_eta"][1, :, :]
```

Add invalid factor integration tests:

```julia
@testset "run_simulation rejects invalid spectra eta factors" begin
    mktempdir() do out_dir
        p = tiny_simulation_parameters()
        @test_throws ErrorException run_simulation(p, out_dir;
                                                   n_therm=0,
                                                   n_measure=0,
                                                   spectra_eta_factors=[2, 4, 8],
                                                   verbose=false)
        @test !isfile(joinpath(out_dir, "spectra_bins.jld2"))
    end
end
```

- [ ] **Step 2: Run simulation integration tests and verify failure**

Run:

```bash
julia --project test/test_simulation_tbc.jl
```

Expected: fail because `run_simulation` has no `spectra_eta_factors` keyword and JLD2 lacks multi-`η` fields.

- [ ] **Step 3: Add keyword and validation**

In `run_simulation` docstring and signature, add:

```julia
spectra_eta_factors=DEFAULT_SPECTRA_ETA_FACTORS,
```

After `actual_spectra_eta` validation:

```julia
actual_spectra_eta_factors = validate_spectra_eta_factors(spectra_eta_factors)
actual_eta_values = eta_values_from_base(actual_spectra_eta, actual_spectra_eta_factors)
```

Log:

```julia
tee_println("Spectra eta factors: $(actual_spectra_eta_factors)")
```

- [ ] **Step 4: Write JLD2 metadata**

In both TBC and non-TBC `jldsave` calls, add:

```julia
multi_eta_enabled=true,
spectra_eta_factors=actual_spectra_eta_factors,
eta_values=actual_eta_values,
spectra_eta_base=actual_spectra_eta,
```

- [ ] **Step 5: Pass `eta_values` into measurements**

For TBC path:

```julia
transport_res = measure_transport_only(cache, p;
                                       eta_values=actual_eta_values,
                                       reuse_buffers=true)
twisted_res = measure_twisted_spectra(cache, p, state;
                                      Ltw=actual_spectra_Ltw,
                                      m_point_patch_half_width=m_point_patch_half_width,
                                      spectra_eta=actual_spectra_eta,
                                      spectra_delta_omega=actual_spectra_delta_omega,
                                      eta_values=actual_eta_values,
                                      reuse_buffers=false)
```

For non-TBC path:

```julia
spec_res = measure_transport_and_spectra(cache, p;
                                         eta_values=actual_eta_values,
                                         reuse_buffers=true)
```

When constructing `SpectrumResult` from TBC pieces, pass transport and twisted multi-`η` fields.

- [ ] **Step 6: Add multi-`η` accumulators**

Near existing accumulators, add:

```julia
accum_dc_eta = Vector{Float64}()
accum_opt_eta = Matrix{Float64}(undef, 0, 0)
accum_dos_eta = Matrix{Float64}(undef, 0, 0)
accum_dos_M_eta = Matrix{Float64}(undef, 0, 0)
accum_dos_M_patch_eta = nothing
accum_ldos0_eta = Matrix{Float64}(undef, 0, 0)
accum_Ak0_eta = Array{Float64, 3}(undef, 0, 0, 0)
accum_AMXpath_eta = Array{Float64, 3}(undef, 0, 0, 0)
accum_AXGpath_eta = Array{Float64, 3}(undef, 0, 0, 0)
accum_AXGnodePatch_eta = nothing
```

In the `bin_count == 0` branch:

```julia
accum_dc_eta = copy(spec_res.dc_conductivity_eta)
accum_opt_eta = copy(spec_res.optical_conductivity_eta)
accum_dos_eta = copy(spec_res.dos_eta)
accum_dos_M_eta = copy(spec_res.dos_M_eta)
accum_dos_M_patch_eta = spec_dos_M_patch_eta === nothing ? nothing : copy(spec_dos_M_patch_eta)
accum_ldos0_eta = copy(spec_res.ldos_ω0_eta)
accum_Ak0_eta = copy(spec_res.A_k_ω0_eta)
accum_AMXpath_eta = copy(spec_res.A_MX_path_eta)
accum_AXGpath_eta = copy(spec_res.A_XG_path_eta)
accum_AXGnodePatch_eta = spec_xg_node_patch_eta === nothing ? nothing : copy(spec_xg_node_patch_eta)
```

In the accumulation branch, add matching `.+=` operations for every multi-`η` accumulator.

In the normalization branch, add matching `./= bin_count` operations.

- [ ] **Step 7: Write multi-`η` JLD2 fields**

Inside the `jldopen` write block, add:

```julia
g["dc_cond_eta"] = accum_dc_eta
g["opt_cond_eta"] = accum_opt_eta
g["dos_eta"] = accum_dos_eta
g["dos_M_eta"] = accum_dos_M_eta
if accum_dos_M_patch_eta !== nothing
    g["dos_M_patch_eta"] = accum_dos_M_patch_eta
end
g["LDOS_0_eta"] = accum_ldos0_eta
g["A_k0_eta"] = accum_Ak0_eta
g["A_MX_path_eta"] = accum_AMXpath_eta
g["A_XG_path_eta"] = accum_AXGpath_eta
if accum_AXGnodePatch_eta !== nothing
    g["A_XG_node_patch_eta"] = accum_AXGnodePatch_eta
end
```

Keep existing old field writes unchanged.

- [ ] **Step 8: Run simulation integration tests**

Run:

```bash
julia --project test/test_simulation_tbc.jl
```

Expected: pass.

- [ ] **Step 9: Commit**

```bash
git add src/Simulation.jl test/test_simulation_tbc.jl
git commit -m "写入多eta谱学JLD2"
```

---

### Task 6: Local Post-processing Eta Selection

**Files:**
- Modify: `scripts/spectra_postprocess_utils.jl`
- Modify: `scripts/process_spectra.jl`
- Modify: `scripts/batch_process_spectra.jl`
- Test: `test/test_postprocess_spectra.jl`

- [ ] **Step 1: Extend synthetic fixture with multi-`η` fields**

In `write_synthetic_spectra` in `test/test_postprocess_spectra.jl`, add keyword:

```julia
multi_eta=true,
```

After metadata writes, add:

```julia
if multi_eta
    file["multi_eta_enabled"] = true
    file["spectra_eta_factors"] = [1.0, 2.0, 4.0]
    file["eta_values"] = [0.125, 0.25, 0.5]
    file["spectra_eta_base"] = 0.125
end
```

Inside each sweep, after old fields, add:

```julia
if multi_eta
    stack_eta_vector(base) = vcat(reshape(base .* 1.0, 1, :),
                                  reshape(base .* 2.0, 1, :),
                                  reshape(base .* 4.0, 1, :))
    stack_eta_matrix(base) = permutedims(cat(base, base .* 2.0, base .* 4.0; dims=3),
                                         (3, 1, 2))

    opt_base = [1.0, 2.0] .+ offset .+ sweep
    dos_base = [3.0, 4.0, 5.0] .+ offset .+ sweep
    dos_M_base = [6.0, 7.0, 8.0] .+ offset .+ sweep
    ldos_base = collect(1.0:4.0) .+ offset .+ sweep
    ak_base = reshape(collect(1.0:prod(effective)), effective) .+ offset .+ sweep
    mx_base = mx_path .+ offset .+ sweep
    xg_base = xg_path .+ offset .+ sweep
    node_patch_base = xg_node_patch .+ offset .+ sweep

    file["$prefix/dc_cond_eta"] = [100.0, 200.0, 400.0] .+ offset .+ sweep
    file["$prefix/opt_cond_eta"] = stack_eta_vector(opt_base)
    file["$prefix/dos_eta"] = stack_eta_vector(dos_base)
    file["$prefix/dos_M_eta"] = stack_eta_vector(dos_M_base)
    if use_twisted_spectra
        file["$prefix/dos_M_patch_eta"] = stack_eta_vector([10.0, 20.0, 30.0] .+ offset .+ sweep)
    end
    file["$prefix/LDOS_0_eta"] = stack_eta_vector(ldos_base)
    file["$prefix/A_k0_eta"] = stack_eta_matrix(ak_base)
    file["$prefix/A_MX_path_eta"] = stack_eta_matrix(mx_base)
    file["$prefix/A_XG_path_eta"] = stack_eta_matrix(xg_base)
    if use_twisted_spectra
        file["$prefix/A_XG_node_patch_eta"] = stack_eta_matrix(node_patch_base)
    end
end
```

- [ ] **Step 2: Add post-processing selection tests**

Add:

```julia
@testset "process_spectra.jl selects requested eta factor" begin
    mktempdir() do root
        target_dir = joinpath(root, PROCESS_TARGET_REL)
        write_synthetic_spectra(target_dir; nsweeps=1)
        Base.invokelatest(ProcessSpectraScript.process_spectra_directory, target_dir; eta_factor=4)

        @test first_data_value(joinpath(target_dir, "processed_dos.csv"), 2) == 16.0
        @test first_data_value(joinpath(target_dir, "processed_dos_AN.csv"), 2) == 32.0
    end
end

@testset "process_spectra.jl rejects old data for non-default eta" begin
    mktempdir() do root
        target_dir = joinpath(root, PROCESS_TARGET_REL)
        write_synthetic_spectra(target_dir; nsweeps=1, multi_eta=false)
        @test_throws ErrorException Base.invokelatest(ProcessSpectraScript.process_spectra_directory,
                                                      target_dir;
                                                      eta_factor=4)
    end
end
```

- [ ] **Step 3: Run post-processing tests and verify failure**

Run:

```bash
julia --project test/test_postprocess_spectra.jl
```

Expected: fail because post-processing functions do not accept `eta_factor`.

- [ ] **Step 4: Add selection helpers**

In `scripts/spectra_postprocess_utils.jl`, add:

```julia
function selected_eta_index(file, eta_factor)
    factor = Float64(eta_factor)
    if haskey(file, "spectra_eta_factors")
        return DwaveHMC.eta_factor_index(collect(file["spectra_eta_factors"]), factor)
    end
    isapprox(factor, 1.0; atol=DwaveHMC.ETA_FACTOR_ATOL, rtol=0.0) ||
        error("Old spectra file has no multi-eta data; only eta_factor=1 is available")
    return 1
end

function selected_vector(group, multi_key::AbstractString, old_key::AbstractString, eta_idx::Int)
    if haskey(group, multi_key)
        return vec(group[multi_key][eta_idx, :])
    end
    eta_idx == 1 || error("Missing $multi_key for selected eta factor")
    return group[old_key]
end

function selected_matrix(group, multi_key::AbstractString, old_key::AbstractString, eta_idx::Int)
    if haskey(group, multi_key)
        return group[multi_key][eta_idx, :, :]
    end
    eta_idx == 1 || error("Missing $multi_key for selected eta factor")
    return group[old_key]
end
```

- [ ] **Step 5: Update local processors**

In both `scripts/process_spectra.jl` and `scripts/batch_process_spectra.jl`:

Change function signatures:

```julia
function collect_sweep_data(file; eta_factor=1)
```

At the top:

```julia
eta_idx = selected_eta_index(file, eta_factor)
```

Inside sweep loop, replace pushes:

```julia
push!(list_opt, selected_vector(g, "opt_cond_eta", "opt_cond", eta_idx))
push!(list_dos, selected_vector(g, "dos_eta", "dos", eta_idx))
push!(list_dos_M, selected_vector(g, "dos_M_eta", "dos_M", eta_idx))
if haskey(g, "dos_M_patch") || haskey(g, "dos_M_patch_eta")
    push!(list_dos_M_patch, selected_vector(g, "dos_M_patch_eta", "dos_M_patch", eta_idx))
end
if haskey(g, "LDOS_0") || haskey(g, "LDOS_0_eta")
    push!(list_ldos0, selected_vector(g, "LDOS_0_eta", "LDOS_0", eta_idx))
end
push!(list_ak, selected_matrix(g, "A_k0_eta", "A_k0", eta_idx))
push!(list_mx_path, selected_matrix(g, "A_MX_path_eta", "A_MX_path", eta_idx))
push!(list_xg_path, selected_matrix(g, "A_XG_path_eta", "A_XG_path", eta_idx))
```

Change processing function signatures:

```julia
function process_spectra_directory(target_dir::AbstractString=target_dir; eta_factor=1)
function process_single_directory(target_dir; eta_factor=1)
function process_batch_spectra_root(root_dir::AbstractString=root_dir; eta_factor=1)
```

Pass `eta_factor` through each call.

In each `main`, parse:

```julia
eta_factor = parse(Float64, get(ENV, "DWAVEHMC_SPECTRA_ETA_FACTOR", "1"))
```

- [ ] **Step 6: Run local post-processing tests**

Run:

```bash
julia --project test/test_postprocess_spectra.jl
```

Expected: local post-processing tests pass; HPC tests may still fail until Task 7 mirrors helpers.

- [ ] **Step 7: Commit**

```bash
git add scripts/spectra_postprocess_utils.jl scripts/process_spectra.jl scripts/batch_process_spectra.jl test/test_postprocess_spectra.jl
git commit -m "后处理选择eta因子"
```

---

### Task 7: HPC Post-processing Eta Selection

**Files:**
- Modify: `projectHPC/example/spectra_postprocess_utils.jl`
- Modify: `projectHPC/example/batch_process_spectra.jl`
- Test: `test/test_postprocess_spectra.jl`

- [ ] **Step 1: Add HPC selected-factor test**

In `test/test_postprocess_spectra.jl`, add:

```julia
@testset "projectHPC batch processor selects requested eta factor" begin
    mktempdir() do root
        t_dir = joinpath(root, "T_0.10")
        write_synthetic_spectra(joinpath(t_dir, "conf_001"); offset=0.0, nsweeps=1)
        write_synthetic_spectra(joinpath(t_dir, "conf_002"); offset=10.0, nsweeps=1)
        Base.invokelatest(HPCProcessSpectraScript.process_T_directory, t_dir; eta_factor=4)

        @test first_data_value(joinpath(t_dir, "spectra_dos.csv"), 2) == 36.0
        @test first_data_value(joinpath(t_dir, "spectra_dos_AN.csv"), 2) == 52.0
    end
end
```

- [ ] **Step 2: Run HPC post-processing test and verify failure**

Run:

```bash
julia --project test/test_postprocess_spectra.jl
```

Expected: fail because `process_T_directory` does not accept `eta_factor`.

- [ ] **Step 3: Mirror helper functions**

Add the same `selected_eta_index`, `selected_vector`, and `selected_matrix` helper functions from Task 6 to `projectHPC/example/spectra_postprocess_utils.jl`.

- [ ] **Step 4: Update `process_single_config`**

Change signature:

```julia
function process_single_config(jld_path; eta_factor=1)
```

After reading metadata and first sweep group:

```julia
eta_idx = selected_eta_index(file, eta_factor)
```

Initialize sums using selected helpers:

```julia
sum_opt = copy(selected_vector(g1, "opt_cond_eta", "opt_cond", eta_idx))
sum_dos = copy(selected_vector(g1, "dos_eta", "dos", eta_idx))
sum_dos_M = copy(selected_vector(g1, "dos_M_eta", "dos_M", eta_idx))
sum_ak = copy(selected_matrix(g1, "A_k0_eta", "A_k0", eta_idx))
sum_mx_path = copy(selected_matrix(g1, "A_MX_path_eta", "A_MX_path", eta_idx))
sum_xg_path = copy(selected_matrix(g1, "A_XG_path_eta", "A_XG_path", eta_idx))
has_ldos0 = haskey(g1, "LDOS_0") || haskey(g1, "LDOS_0_eta")
sum_ldos0 = has_ldos0 ? copy(selected_vector(g1, "LDOS_0_eta", "LDOS_0", eta_idx)) : nothing
has_node_patch = haskey(g1, "A_XG_node_patch") || haskey(g1, "A_XG_node_patch_eta")
node_source_key = has_node_patch ? "A_XG_node_patch" : "A_XG_path"
node_multi_key = has_node_patch ? "A_XG_node_patch_eta" : "A_XG_path_eta"
sum_node_path = copy(selected_matrix(g1, node_multi_key, node_source_key, eta_idx))
has_patch = haskey(g1, "dos_M_patch") || haskey(g1, "dos_M_patch_eta")
sum_dos_M_patch = has_patch ? copy(selected_vector(g1, "dos_M_patch_eta", "dos_M_patch", eta_idx)) : nothing
```

In the sweep loop, add selected helper values to sums and keep existing consistency checks adapted to old-or-new field presence.

- [ ] **Step 5: Thread `eta_factor` through HPC directory processing**

Change signatures:

```julia
function process_T_directory(dir_path; eta_factor=1)
function main()
```

Call:

```julia
res = process_single_config(jld_path; eta_factor=eta_factor)
```

In `main`, parse:

```julia
eta_factor = parse(Float64, get(ENV, "DWAVEHMC_SPECTRA_ETA_FACTOR", "1"))
```

and pass it into `process_T_directory`.

- [ ] **Step 6: Run post-processing tests**

Run:

```bash
julia --project test/test_postprocess_spectra.jl
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add projectHPC/example/spectra_postprocess_utils.jl projectHPC/example/batch_process_spectra.jl test/test_postprocess_spectra.jl
git commit -m "HPC后处理选择eta因子"
```

---

### Task 8: Verification and Allocation Sanity

**Files:**
- Test: `test/test_twisted_spectra.jl`
- Test: `test/test_simulation_tbc.jl`
- Test: `test/test_postprocess_spectra.jl`

- [ ] **Step 1: Run focused tests**

Run:

```bash
julia --project test/test_twisted_spectra.jl
```

Expected: pass.

Run:

```bash
julia --project test/test_simulation_tbc.jl
```

Expected: pass.

Run:

```bash
julia --project test/test_postprocess_spectra.jl
```

Expected: pass.

- [ ] **Step 2: Run full default suite**

Run:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Expected: pass.

- [ ] **Step 3: Run allocation sanity check**

Run:

```bash
julia --project -e 'using Random, DwaveHMC; Random.seed!(20260520); p=ModelParameters(4,4,1.0,-0.35,-0.5,0.0,0.0,8.0,1.0,1.0; η=0.25, Δω=0.25, ω_max=2.0); state=initialize_state(p); cache=initialize_cache(p); init_static_H!(cache,p,state); update_H_BdG!(cache,p,state); diagonalize_H_BdG!(cache,p); measure_transport_and_spectra(cache,p; eta_values=0.25 .* [1.0,2.0,4.0], reuse_buffers=true); bytes=@allocated measure_transport_and_spectra(cache,p; eta_values=0.25 .* [1.0,2.0,4.0], reuse_buffers=true); println(bytes)'
```

Expected: command completes and prints a finite integer. Record the number in the final implementation summary. If it is unexpectedly huge for `4x4`, inspect accidental array comprehensions or temporary container creation inside inner loops before merging.

- [ ] **Step 4: Inspect git state**

Run:

```bash
git status --short
```

Expected: clean, or only intentional uncommitted changes that are being prepared for the next commit.

- [ ] **Step 5: Final commit if verification changed tests or docs**

If verification required any small fixes, commit them:

```bash
git add src scripts projectHPC test
git commit -m "验证多eta谱学输出"
```

If no files changed during verification, do not create an empty commit.

---

## Self-review Notes

- Spec coverage: validation, JLD2 metadata, multi-`η` arrays, first-slice compatibility, default CSV behavior, post-processing selection, old-data rejection, HPC skip behavior, and allocation sanity are each mapped to tasks.
- Type consistency: field names use `dc_cond_eta` in JLD2 and `dc_conductivity_eta` in Julia result structs; all spectra array names use the existing old field name plus `_eta`.
- Scope: one implementation plan is enough because measurement, simulation writing, and post-processing selection share one data contract and must be verified together.
