# Hall Conductivity and Longitudinal Resistivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add finite-field Hall conductivity output and use the averaged conductivity tensor to derive physical longitudinal resistivity for the HPC temperature notebooks.

**Architecture:** Make current operators direction-aware, compute `sigma_xy` from the documented off-diagonal Kubo tensor, and keep `sigma_xx` semantics unchanged. Persist Hall scalar and optical outputs through simulation and spectra postprocessing, then derive `Longitudinal_Resistivity_mean/err` only in the HPC summary layer by averaging conductivities first and inverting once.

**Tech Stack:** Julia core library (`src/*.jl`), JLD2/DelimitedFiles/Glob postprocessing scripts, Julia `Test` suites, and Python/Jupyter notebook JSON source cells edited in place.

---

## File Structure

- `src/Types.jl`: add `Jy_sparse_q0` to `ComputeCache` and initialize it.
- `src/Hamiltonian.jl`: add `direction` support to `build_probe_H_BdG!` and make probe phases depend on the selected bond displacement component.
- `src/Observables.jl`: add direction-aware current builders, Hall fields in `TransportResult` / `SpectrumResult`, and Hall Kubo accumulation in `measure_transport_only`.
- `src/Simulation.jl`: write `Hall_Conductivity` in `transport.csv` and Hall arrays in `spectra_bins.jld2`.
- `scripts/process_spectra.jl`, `scripts/batch_process_spectra.jl`, `scripts/spectra_postprocess_utils.jl`: postprocess complex Hall optical spectra for single-directory and top-level batch flows.
- `projectHPC/example/batch_process_spectra.jl`, `projectHPC/example/spectra_postprocess_utils.jl`: postprocess complex Hall optical spectra for HPC `T_*/conf_*` ensembles.
- `projectHPC/example/batch_process_csv.jl`: derive `Longitudinal_Resistivity_mean/err` from paired per-conf means of `DC_Conductivity` and `Hall_Conductivity`.
- `projectHPC/example/plot_stiffness.ipynb`, `projectHPC/example/plot_conductivity.ipynb`: prefer physical resistivity columns and use Hall optical data when available.
- `doc/observables.md`, `doc/magnetic-field.md`: document formulas, conventions, outputs, and old-proxy caveat removal.
- `test/test_magnetic_field.jl`: current-operator derivative tests, direct Hall formula regression, zero-field and field-reversal checks.
- `test/test_twisted_spectra.jl`: result shape and compatibility checks for Hall eta fields.
- `test/test_simulation_tbc.jl`: simulation output schema checks.
- `test/test_postprocess_spectra.jl`: synthetic Hall JLD2 fixtures and postprocessor assertions.
- `test/test_hpc_scripts.jl`: HPC summary fixture and notebook source checks.

## Task 1: Direction-Aware Current Operators

**Files:**
- Modify: `src/Types.jl`
- Modify: `src/Hamiltonian.jl`
- Modify: `src/Observables.jl`
- Test: `test/test_magnetic_field.jl`

- [ ] **Step 1: Write failing derivative and cache tests**

Add this helper after `random_finite_field_state(p)` in `test/test_magnetic_field.jl`:

```julia
function real_finite_field_state(p; seed=20260704)
    Random.seed!(seed)
    state = initialize_state(p)
    state.disorder_pot .= randn(p.N) .* 0.05
    state.Δ .= randn(p.N, 2) .* 0.03 .+ 0.0im
    state.π .= 0.0 + 0.0im
    return state
end
```

Replace the existing `"Kubo operators match Hamiltonian derivatives"` testset body with direction-aware coverage:

```julia
@testset "Kubo operators match Hamiltonian derivatives" begin
    for (Lx, Ly) in ((4, 4), (2, 2))
        p = magnetic_test_parameters(Lx=Lx, Ly=Ly, n_flux_sc=2,
                                     boundary_condition=:magnetic_pbc)
        state = random_finite_field_state(p)
        cache = initialize_cache(p)
        init_static_H!(cache, p, state)
        update_H_BdG!(cache, p, state)

        dim = 2 * p.N
        Hplus = zeros(ComplexF64, dim, dim)
        Hminus = zeros(ComplexF64, dim, dim)
        H0 = zeros(ComplexF64, dim, dim)
        eps = 1.0e-6

        for direction in (:x, :y)
            qy_values = direction === :x ? (0.0, 2π / p.Ly) : (0.0,)
            for qy in qy_values
                DwaveHMC.build_probe_H_BdG!(Hplus, cache, p, state;
                                            direction=direction, λ=eps, qx=0.0, qy=qy)
                DwaveHMC.build_probe_H_BdG!(Hminus, cache, p, state;
                                            direction=direction, λ=-eps, qx=0.0, qy=qy)
                DwaveHMC.build_probe_H_BdG!(H0, cache, p, state;
                                            direction=direction, λ=0.0, qx=0.0, qy=qy)

                J_fd = (Matrix(Hermitian(Hplus, :U)) - Matrix(Hermitian(Hminus, :U))) ./ (2eps)
                K_fd = (Matrix(Hermitian(Hplus, :U)) + Matrix(Hermitian(Hminus, :U)) -
                        2 .* Matrix(Hermitian(H0, :U))) ./ (eps^2)

                J_an = Matrix(DwaveHMC.probe_current_operator_matrix(cache, p;
                                                                      direction=direction,
                                                                      qx=0.0, qy=qy))
                K_an = DwaveHMC.diamagnetic_operator_matrix(cache, p;
                                                            direction=direction,
                                                            qx=0.0, qy=qy)

                @test norm(J_an - J_fd) / max(norm(J_fd), 1.0) < 1.0e-6
                @test norm(K_an - K_fd) / max(norm(K_fd), 1.0) < 5.0e-4
            end
        end
    end
end
```

Extend `"Production Kubo current uses finite-field Nambu phases"` with:

```julia
J_prod_y = Matrix(DwaveHMC.current_operator_matrix(cache, p; direction=:y, qx=0.0, qy=0.0))
J_probe_y = Matrix(DwaveHMC.probe_current_operator_matrix(cache, p; direction=:y, qx=0.0, qy=0.0))
@test J_prod_y ≈ -J_probe_y atol=1.0e-12 rtol=1.0e-12

DwaveHMC.build_current_operator!(cache, p; direction=:y, qx=0.0, qy=0.0, store=:q0)
@test nnz(cache.Jy_sparse_q0) > 0
@test_throws ErrorException DwaveHMC.build_current_operator!(cache, p; direction=:y,
                                                             qx=0.0, qy=2π / p.Ly,
                                                             store=:qy)
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
julia --project test/test_magnetic_field.jl
```

Expected before implementation: failure because `direction` keyword is unsupported and `cache.Jy_sparse_q0` does not exist.

- [ ] **Step 3: Implement direction-aware probe/current operators**

In `src/Types.jl`, add `Jy_sparse_q0` immediately after `Jx_sparse_qy` in `ComputeCache`, initialize it with `spzeros(ComplexF64, dim, dim)`, and pass it to the `ComputeCache(...)` constructor:

```julia
Jy_sparse_q0::SparseMatrixCSC{ComplexF64, Int} # 稀疏电流算符 Jy(q=0) (2N x 2N)
```

In `src/Observables.jl`, add this helper near `probe_weight`:

```julia
@inline function direction_component(direction::Symbol, dx::Int, dy::Int)
    direction === :x && return dx
    direction === :y && return dy
    error("Unsupported current direction: $direction")
end
```

Update `add_probe_current_derivative_bond!`, `probe_current_operator_matrix`, `add_kubo_current_bond!`, `current_operator_matrix`, `add_diamagnetic_bond!`, `diamagnetic_operator_matrix`, and `build_current_operator!` so each accepts `direction::Symbol=:x`, computes `dα = direction_component(direction, dx, dy)`, skips bonds with `dα == 0`, and multiplies first derivatives by `dα` and second derivatives by `dα^2`. The current-bond loops must include the kinetic bonds `+x`, `+y`, `+x+y`, and `+x-y`; the skip rule leaves exactly the intended three bonds for each direction.

Use this cache policy in `build_current_operator!`:

```julia
function build_current_operator!(cache::ComputeCache, p::ModelParameters;
                                 direction::Symbol=:x,
                                 qx::Float64=0.0,
                                 qy::Float64=0.0,
                                 store::Symbol=:q0)
    J_sparse = current_operator_matrix(cache, p; direction=direction, qx=qx, qy=qy)

    if direction === :x && store === :q0
        cache.Jx_sparse_q0 = J_sparse
    elseif direction === :x && store === :qy
        cache.Jx_sparse_qy = J_sparse
    elseif direction === :y && store === :q0
        cache.Jy_sparse_q0 = J_sparse
    elseif direction === :y && store === :qy
        error("Jy(qy) cache is not part of the Hall transport design")
    else
        error("Unknown current-operator cache tag: $store")
    end
    return nothing
end
```

In `src/Hamiltonian.jl`, change `probe_factor` to accept the selected displacement component:

```julia
@inline function probe_factor(cache::ComputeCache, i::Int, dα::Int,
                              λ::Float64, qx::Float64, qy::Float64)
    dα == 0 && return 1.0 + 0.0im
    x = cache.x_idx[i] - 1
    y = cache.y_idx[i] - 1
    θ = qx * x + qy * y
    η = (qx == 0.0 && qy == 0.0) ? 1.0 : sqrt(2.0) * cos(θ)
    return cis(λ * dα * η)
end
```

Add `direction::Symbol=:x` to `build_probe_H_BdG!`, compute the component for each kinetic bond, and apply the probe factor to all four kinetic bonds.

- [ ] **Step 4: Run the task test**

Run:

```bash
julia --project test/test_magnetic_field.jl
```

Expected after implementation: current-operator derivative tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/Types.jl src/Hamiltonian.jl src/Observables.jl test/test_magnetic_field.jl
git commit -m "添加方向相关电流算符"
```

## Task 2: Hall Kubo Measurement

**Files:**
- Modify: `src/Observables.jl`
- Test: `test/test_magnetic_field.jl`
- Test: `test/test_twisted_spectra.jl`

- [ ] **Step 1: Write failing Hall physics tests**

Add this manual formula helper to `test/test_magnetic_field.jl`:

```julia
function manual_hall_tensor(cache, p; eta=p.η, omega=0.0)
    dim = 2 * p.N
    U = cache.U
    E = cache.E_n
    β = p.β
    Jx = DwaveHMC.current_operator_matrix(cache, p; direction=:x, qx=0.0, qy=0.0)
    Jy = DwaveHMC.current_operator_matrix(cache, p; direction=:y, qx=0.0, qy=0.0)
    Jx_mn = U' * (Jx * U)
    Jy_mn = U' * (Jy * U)
    total = 0.0 + 0.0im
    for n in 1:dim, m in 1:dim
        m == n && continue
        diff_E = E[m] - E[n]
        f_n = logistic(-β * E[n])
        f_m = logistic(-β * E[m])
        ratio = abs(diff_E) < 1.0e-8 ? β * f_n * (1.0 - f_n) : (f_n - f_m) / diff_E
        total += ratio * Jx_mn[n, m] * Jy_mn[m, n] / (omega - diff_E + im * eta)
    end
    return im * total / p.N
end
```

Add these testsets after the finite-field transport test:

```julia
@testset "Hall Kubo formula matches direct implementation" begin
    p = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2, boundary_condition=:magnetic_pbc)
    state = real_finite_field_state(p)
    cache = initialize_cache(p)
    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)

    res = DwaveHMC.measure_transport_only(cache, p; eta_values=[p.η, 2p.η],
                                          reuse_buffers=false)
    manual_dc = manual_hall_tensor(cache, p; eta=p.η, omega=0.0)
    manual_first = manual_hall_tensor(cache, p; eta=p.η, omega=cache.omega_grid[1])

    @test res.hall_conductivity ≈ real(manual_dc) atol=1.0e-10 rtol=1.0e-10
    @test res.hall_conductivity_eta[1] ≈ real(manual_dc) atol=1.0e-10 rtol=1.0e-10
    @test res.hall_optical_conductivity[1] ≈ manual_first atol=1.0e-10 rtol=1.0e-10
    @test length(res.hall_conductivity_eta) == 2
    @test size(res.hall_optical_conductivity_eta) == (2, length(res.ω_grid))
    @test cache.omega_grid[1] == p.η
end

@testset "Hall response respects zero-field and field-reversal symmetries" begin
    p0 = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=0, boundary_condition=:periodic)
    state0 = real_finite_field_state(p0)
    cache0 = initialize_cache(p0)
    init_static_H!(cache0, p0, state0)
    update_H_BdG!(cache0, p0, state0)
    diagonalize_H_BdG!(cache0, p0)
    res0 = DwaveHMC.measure_transport_only(cache0, p0; reuse_buffers=false)
    @test abs(res0.hall_conductivity) < 1.0e-8
    @test maximum(abs.(res0.hall_optical_conductivity)) < 1.0e-8

    p_plus = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=2,
                                      boundary_condition=:magnetic_pbc)
    p_minus = magnetic_test_parameters(Lx=4, Ly=4, n_flux_sc=-2,
                                       boundary_condition=:magnetic_pbc)
    state_plus = real_finite_field_state(p_plus)
    state_minus = initialize_state(p_minus)
    state_minus.disorder_pot .= state_plus.disorder_pot
    state_minus.Δ .= state_plus.Δ
    state_minus.π .= 0.0 + 0.0im

    cache_plus = initialize_cache(p_plus)
    cache_minus = initialize_cache(p_minus)
    for (cache, p, state) in ((cache_plus, p_plus, state_plus),
                              (cache_minus, p_minus, state_minus))
        init_static_H!(cache, p, state)
        update_H_BdG!(cache, p, state)
        diagonalize_H_BdG!(cache, p)
    end
    res_plus = DwaveHMC.measure_transport_only(cache_plus, p_plus; reuse_buffers=false)
    res_minus = DwaveHMC.measure_transport_only(cache_minus, p_minus; reuse_buffers=false)
    @test res_plus.dc_conductivity ≈ res_minus.dc_conductivity atol=1.0e-8 rtol=1.0e-6
    @test res_plus.hall_conductivity ≈ -res_minus.hall_conductivity atol=1.0e-8 rtol=1.0e-5
end
```

In `test/test_twisted_spectra.jl`, extend the existing multi-eta assertions around `measure_transport_and_spectra`:

```julia
@test length(spec.hall_conductivity_eta) == 2
@test size(spec.hall_optical_conductivity_eta) == (2, length(spec.ω_grid))
@test spec.hall_conductivity == spec.hall_conductivity_eta[1]
@test spec.hall_optical_conductivity == vec(spec.hall_optical_conductivity_eta[1, :])
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
julia --project test/test_magnetic_field.jl
julia --project test/test_twisted_spectra.jl
```

Expected before implementation: failures because Hall fields are missing.

- [ ] **Step 3: Implement result fields and Hall accumulation**

In `src/Observables.jl`, extend `TransportResult`:

```julia
struct TransportResult
    superfluid_stiffness::Float64
    dc_conductivity::Float64
    hall_conductivity::Float64
    ω_grid::Vector{Float64}
    optical_conductivity::Vector{Float64}
    hall_optical_conductivity::Vector{ComplexF64}
    dc_conductivity_eta::Vector{Float64}
    hall_conductivity_eta::Vector{Float64}
    optical_conductivity_eta::Matrix{Float64}
    hall_optical_conductivity_eta::Matrix{ComplexF64}
end
```

Extend `SpectrumResult` with matching Hall scalar/vector fields immediately after the existing longitudinal scalar/vector fields, and update the compatibility constructor so old callers receive `hall_conductivity = 0.0`, `hall_optical_conductivity = zeros(ComplexF64, length(ω_grid))`, `hall_conductivity_eta = [0.0]`, and a one-row complex zero matrix.

In `measure_transport_only`, keep the existing stiffness and longitudinal `sigma_xx` code unchanged. After `Jx(q=0)` matrix elements are computed and before `cache.J_mn` is reused for `Jy`, preserve them:

```julia
Jx_mn = copy(cache.J_mn)
```

Build `Jy(q=0)`, compute its eigenbasis matrix elements, and accumulate:

```julia
if nnz(cache.Jy_sparse_q0) == 0
    build_current_operator!(cache, p; direction=:y, qx=0.0, qy=0.0, store=:q0)
end
mul!(cache.temp_JU, cache.Jy_sparse_q0, U)
mul!(cache.J_mn, U', cache.temp_JU)
Jy_mn = cache.J_mn

hall_dc_eta_complex = zeros(ComplexF64, nη)
hall_opt_eta = zeros(ComplexF64, nη, length(ω_grid))
@inbounds for n in 1:dim, m in 1:dim
    m == n && continue
    diff_E = E[m] - E[n]
    f_n = f[n]
    f_m = f[m]
    ratio = abs(diff_E) < 1.0e-8 ? β * f_n * (1.0 - f_n) : (f_n - f_m) / diff_E
    current_product = Jx_mn[n, m] * Jy_mn[m, n]
    @simd for iη in 1:nη
        hall_dc_eta_complex[iη] += im * ratio * current_product / (-diff_E + im * eta_vals[iη])
    end
    for (iω, ω) in enumerate(ω_grid)
        @simd for iη in 1:nη
            hall_opt_eta[iη, iω] += im * ratio * current_product / (ω - diff_E + im * eta_vals[iη])
        end
    end
end
hall_dc_eta_complex ./= N
hall_opt_eta ./= N
hall_cond_eta = real.(hall_dc_eta_complex)
hall_cond = hall_cond_eta[1]
hall_opt = vec(hall_opt_eta[1, :])
```

Return these fields from both `reuse_buffers=true` and copying branches. `sigma_xx` arrays remain `Float64`; Hall optical arrays are `ComplexF64`.

- [ ] **Step 4: Run the task tests**

Run:

```bash
julia --project test/test_magnetic_field.jl
julia --project test/test_twisted_spectra.jl
```

Expected after implementation: Hall formula and shape tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/Observables.jl test/test_magnetic_field.jl test/test_twisted_spectra.jl
git commit -m "计算霍尔电导Kubo张量"
```

## Task 3: Simulation Output Schema

**Files:**
- Modify: `src/Simulation.jl`
- Test: `test/test_simulation_tbc.jl`

- [ ] **Step 1: Write failing schema test**

Add this testset near the output-schema tests in `test/test_simulation_tbc.jl`:

```julia
@testset "simulation writes Hall transport schema" begin
    p = tiny_simulation_parameters()
    mktempdir() do out_dir
        Random.seed!(20260704)
        run_simulation(p, out_dir;
                       n_therm=0,
                       n_measure=1,
                       Nt_measure=1,
                       measure_transport_freq=1,
                       bin_size=1,
                       include_momentum_spectra=false,
                       verbose=false)

        transport_header = split(strip(readline(joinpath(out_dir, "transport.csv"))), ",")
        @test "Hall_Conductivity" in transport_header
        @test !("Longitudinal_Resistivity" in transport_header)

        jldopen(joinpath(out_dir, "spectra_bins.jld2"), "r") do file
            @test file["conductivity_convention"] == "sigma_xx_regular_sigma_xy_kubo"
            @test haskey(file, "sweep_1/hall_cond")
            @test haskey(file, "sweep_1/hall_cond_eta")
            @test haskey(file, "sweep_1/hall_opt_cond")
            @test haskey(file, "sweep_1/hall_opt_cond_eta")
            @test !haskey(file, "sweep_1/rho_xx")
            @test file["sweep_1/hall_opt_cond"] isa Vector{ComplexF64}
            @test file["sweep_1/hall_opt_cond_eta"] isa Matrix{ComplexF64}
        end
    end
end
```

- [ ] **Step 2: Run failing schema test**

Run:

```bash
julia --project test/test_simulation_tbc.jl
```

Expected before implementation: missing `Hall_Conductivity` and Hall JLD2 keys.

- [ ] **Step 3: Implement simulation writers and accumulators**

In both `jldsave` metadata calls in `src/Simulation.jl`, change:

```julia
conductivity_convention="sigma_xx_regular"
```

to:

```julia
conductivity_convention="sigma_xx_regular_sigma_xy_kubo"
```

Update `transport.csv` headers:

```julia
println(f_trans, "Sweep,Superfluid_Stiffness,DC_Conductivity,Hall_Conductivity")
```

and, when twist diagnostics are enabled:

```julia
println(f_trans, "Sweep,Superfluid_Stiffness,DC_Conductivity,Hall_Conductivity,Twist_Qy,Twist_Qy_Rho_Curv_Cos,Twist_Qy_Rho_Curv_Sin,Twist_Qy_Rho_Curv_Avg,Twist_Qy_Lambda_Diag,Twist_Qy_Rho_OffdiagCorrected")
```

When constructing `SpectrumResult` from `transport_res` plus twisted spectra, include the new Hall fields in the same order defined in Task 2.

Add accumulators next to the existing conductivity accumulators:

```julia
accum_hall_cond_eta = Vector{Float64}()
accum_hall_opt_cond = Vector{ComplexF64}()
accum_hall_opt_eta = Matrix{ComplexF64}(undef, 0, 0)
```

Initialize, add, divide, and write them alongside `accum_dc_eta` and `accum_opt_eta`:

```julia
accum_hall_cond_eta = copy(spec_res.hall_conductivity_eta)
accum_hall_opt_cond = copy(spec_res.hall_optical_conductivity)
accum_hall_opt_eta = copy(spec_res.hall_optical_conductivity_eta)
```

```julia
accum_hall_cond_eta .+= spec_res.hall_conductivity_eta
accum_hall_opt_cond .+= spec_res.hall_optical_conductivity
accum_hall_opt_eta .+= spec_res.hall_optical_conductivity_eta
```

```julia
accum_hall_cond_eta ./= bin_count
accum_hall_opt_cond ./= bin_count
accum_hall_opt_eta ./= bin_count
```

```julia
g["hall_cond"] = accum_hall_cond_eta[1]
g["hall_cond_eta"] = accum_hall_cond_eta
g["hall_opt_cond"] = accum_hall_opt_cond
g["hall_opt_cond_eta"] = accum_hall_opt_eta
```

Write `transport.csv` rows with `spec_res.hall_conductivity` immediately after `spec_res.dc_conductivity`.

- [ ] **Step 4: Run schema test**

Run:

```bash
julia --project test/test_simulation_tbc.jl
```

Expected after implementation: simulation schema test passes.

- [ ] **Step 5: Commit**

```bash
git add src/Simulation.jl test/test_simulation_tbc.jl
git commit -m "输出霍尔电导数据"
```

## Task 4: Spectra Hall Postprocessing

**Files:**
- Modify: `scripts/spectra_postprocess_utils.jl`
- Modify: `projectHPC/example/spectra_postprocess_utils.jl`
- Modify: `scripts/process_spectra.jl`
- Modify: `scripts/batch_process_spectra.jl`
- Modify: `projectHPC/example/batch_process_spectra.jl`
- Test: `test/test_postprocess_spectra.jl`

- [ ] **Step 1: Write failing postprocessing tests**

Extend `write_synthetic_spectra` in `test/test_postprocess_spectra.jl` with keyword `include_hall=true`. For each sweep, when `include_hall`, write:

```julia
hall_base = ComplexF64[1.0 + 10.0im, 2.0 + 20.0im] .+ (offset + sweep)
file["$prefix/hall_cond"] = 10.0 + offset + sweep
file["$prefix/hall_opt_cond"] = hall_base
if multi_eta
    file["$prefix/hall_cond_eta"] = [10.0, 20.0, 40.0] .+ offset .+ sweep
    file["$prefix/hall_opt_cond_eta"] = vcat(reshape(hall_base .* 1.0, 1, :),
                                             reshape(hall_base .* 2.0, 1, :),
                                             reshape(hall_base .* 4.0, 1, :))
end
```

Add `"processed_hall_cond.csv"` to stale single-directory cleanup checks and `"spectra_hall_cond.csv"` to HPC stale cleanup checks.

Add assertions to the existing single-directory testsets:

```julia
@test header(joinpath(target_dir, "processed_hall_cond.csv")) ==
      "omega,Re_Sigma_xy,Re_Error,Im_Sigma_xy,Im_Error"
@test first_data_value(joinpath(target_dir, "processed_hall_cond.csv"), 2) == 2.0
@test first_data_value(joinpath(target_dir, "processed_hall_cond.csv"), 4) == 10.0
```

For top-level batch `T_0.10`, assert:

```julia
@test header(joinpath(target_dir, "spectra_hall_cond.csv")) ==
      "omega,Re_Sigma_xy,Re_Error,Im_Sigma_xy,Im_Error"
@test first_data_value(joinpath(target_dir, "spectra_hall_cond.csv"), 2) == 2.0
@test first_data_value(joinpath(target_dir, "spectra_hall_cond.csv"), 4) == 10.0
```

For HPC `T_0.10` with offsets `0.0` and `10.0`, assert:

```julia
@test header(joinpath(t_dir, "spectra_hall_cond.csv")) ==
      "omega,Re_Sigma_xy,Re_Error,Im_Sigma_xy,Im_Error"
@test first_data_value(joinpath(t_dir, "spectra_hall_cond.csv"), 2) == 7.0
@test first_data_value(joinpath(t_dir, "spectra_hall_cond.csv"), 4) == 10.0
```

Add an old-file test with `include_hall=false`, touch stale Hall CSVs before processing, and assert the stale files are removed.

- [ ] **Step 2: Run failing postprocessing test**

Run:

```bash
julia --project test/test_postprocess_spectra.jl
```

Expected before implementation: Hall CSV files are missing or stale files remain.

- [ ] **Step 3: Implement complex stats and writers**

In both `spectra_postprocess_utils.jl` files, add:

```julia
function calc_complex_stats(data_list)
    n_samples = length(data_list)
    if n_samples == 0
        return nothing, nothing, nothing
    end
    raw_shape = size(data_list[1])
    sum_val = zeros(ComplexF64, raw_shape)
    sum_re_sq = zeros(Float64, raw_shape)
    sum_im_sq = zeros(Float64, raw_shape)
    for d in data_list
        sum_val .+= d
        sum_re_sq .+= real.(d) .^ 2
        sum_im_sq .+= imag.(d) .^ 2
    end
    mean_val = sum_val ./ n_samples
    var_re = max.((sum_re_sq ./ n_samples) .- (real.(mean_val) .^ 2), 0.0)
    var_im = max.((sum_im_sq ./ n_samples) .- (imag.(mean_val) .^ 2), 0.0)
    return mean_val, sqrt.(var_re ./ n_samples), sqrt.(var_im ./ n_samples)
end

function write_complex_series_csv(path, header, grid, mean_values, err_re, err_im)
    open(path, "w") do io
        println(io, header)
        for i in eachindex(mean_values)
            @printf(io, "%.6f,%.6e,%.6e,%.6e,%.6e\n",
                    grid[i], real(mean_values[i]), err_re[i],
                    imag(mean_values[i]), err_im[i])
        end
    end
end
```

In `scripts/process_spectra.jl` and `scripts/batch_process_spectra.jl`, add `list_hall_opt = Vector{Vector{ComplexF64}}()` in `collect_sweep_data`; push `selected_vector(g, "hall_opt_cond_eta", "hall_opt_cond", eta_idx)` when Hall keys exist; return `hall_opt=list_hall_opt`.

Write or remove the Hall CSV:

```julia
if length(data.hall_opt) == data.count
    mean_hall, err_hall_re, err_hall_im = calc_complex_stats(data.hall_opt)
    write_complex_series_csv(joinpath(target_dir, "processed_hall_cond.csv"),
                             "omega,Re_Sigma_xy,Re_Error,Im_Sigma_xy,Im_Error",
                             meta.omega_grid, mean_hall, err_hall_re, err_hall_im)
else
    rm(joinpath(target_dir, "processed_hall_cond.csv"); force=true)
end
```

For top-level batch processing in `scripts/batch_process_spectra.jl`, write `"spectra_hall_cond.csv"` and remove it when Hall keys are absent.

In `projectHPC/example/batch_process_spectra.jl`, include `"spectra_hall_cond.csv"` in `SPECTRA_OUTPUT_FILES`, add `hall_opt` to `process_single_config` results and compatibility signatures, collect `samples_hall_opt`, compute `calc_complex_stats`, and write:

```julia
write_complex_series_csv(joinpath(dir_path, "spectra_hall_cond.csv"),
                         "omega,Re_Sigma_xy,Re_Error,Im_Sigma_xy,Im_Error",
                         omega_grid, final_hall, err_hall_re, err_hall_im)
```

If not every valid configuration has Hall data, remove `spectra_hall_cond.csv`.

- [ ] **Step 4: Run postprocessing test**

Run:

```bash
julia --project test/test_postprocess_spectra.jl
```

Expected after implementation: Hall CSVs are produced with componentwise statistics; old fixtures remove stale Hall CSVs.

- [ ] **Step 5: Commit**

```bash
git add scripts/process_spectra.jl scripts/batch_process_spectra.jl scripts/spectra_postprocess_utils.jl projectHPC/example/batch_process_spectra.jl projectHPC/example/spectra_postprocess_utils.jl test/test_postprocess_spectra.jl
git commit -m "后处理霍尔光电导谱"
```

## Task 5: HPC Summary and Notebook Resistivity Logic

**Files:**
- Modify: `projectHPC/example/batch_process_csv.jl`
- Modify: `projectHPC/example/plot_stiffness.ipynb`
- Modify: `projectHPC/example/plot_conductivity.ipynb`
- Test: `test/test_hpc_scripts.jl`

- [ ] **Step 1: Write failing HPC summary fixture**

Add this helper and testset to `test/test_hpc_scripts.jl`:

```julia
using DelimitedFiles
using Statistics

function write_small_csv(path, content)
    mkpath(dirname(path))
    write(path, content)
end

function summary_value(path, column)
    data, header = readdlm(path, ',', header=true)
    names = string.(vec(header))
    idx = findfirst(==(column), names)
    idx === nothing && error("missing column $column")
    return Float64(data[1, idx])
end

@testset "HPC CSV summary derives longitudinal resistivity from averaged conductivity tensor" begin
    mktempdir() do tmp
        t_dir = joinpath(tmp, "T_0.10")
        write_small_csv(joinpath(t_dir, "params.jl"), """
using DwaveHMC
T = 0.10
β = 10.0
Lx = 2
Ly = 2
""")
        for (conf, sx, sy, raw_rho) in (("conf_001", 2.0, 1.0, 999.0),
                                        ("conf_002", 4.0, 3.0, 888.0))
            cdir = joinpath(t_dir, conf)
            write_small_csv(joinpath(cdir, "observables.csv"), """
Sweep,Energy,D2,D4,Avg_d2,Avg_d4
1,1.0,2.0,3.0,2.0,3.0
""")
            write_small_csv(joinpath(cdir, "transport.csv"), """
Sweep,Superfluid_Stiffness,DC_Conductivity,Hall_Conductivity,Longitudinal_Resistivity,Bad_Column
1,0.5,$sx,$sy,$raw_rho,NaN
""")
        end

        cmd = julia_cmd("--project=$(REPO_ROOT)",
                        joinpath(REPO_ROOT, "projectHPC", "example", "batch_process_csv.jl");
                        dir=tmp,
                        env=["DWAVEHMC_ANALYSIS_DIR" => tmp])
        @test success(cmd)

        summary = joinpath(tmp, "summary_all.csv")
        x = [2.0, 4.0]
        y = [1.0, 3.0]
        xbar = mean(x)
        ybar = mean(y)
        D = xbar^2 + ybar^2
        expected_rho = xbar / D
        gx = (ybar^2 - xbar^2) / D^2
        gy = -2.0 * xbar * ybar / D^2
        Cxx = var(x) / length(x)
        Cxy = cov(x, y) / length(x)
        Cyy = var(y) / length(y)
        expected_err = sqrt(max(gx^2 * Cxx + 2.0 * gx * gy * Cxy + gy^2 * Cyy, 0.0))

        @test summary_value(summary, "DC_Conductivity_mean") ≈ xbar
        @test summary_value(summary, "Hall_Conductivity_mean") ≈ ybar
        @test summary_value(summary, "Longitudinal_Resistivity_mean") ≈ expected_rho
        @test summary_value(summary, "Longitudinal_Resistivity_err") ≈ expected_err
        @test summary_value(summary, "Longitudinal_Resistivity_mean") != mean([999.0, 888.0])

        summary_header = readline(summary)
        @test !occursin("Longitudinal_Resistivity_n_finite_conf", summary_header)
        @test !occursin("Bad_Column_mean", summary_header)
    end
end
```

Add notebook source checks:

```julia
@testset "notebooks prefer tensor resistivity inputs" begin
    stiffness_nb = read_repo_file("projectHPC", "example", "plot_stiffness.ipynb")
    conductivity_nb = read_repo_file("projectHPC", "example", "plot_conductivity.ipynb")
    @test occursin("Longitudinal_Resistivity_mean", stiffness_nb)
    @test occursin("Hall_Conductivity_mean", stiffness_nb)
    @test occursin("old-data proxy", stiffness_nb)
    @test occursin("Longitudinal_Resistivity_mean", conductivity_nb)
    @test occursin("spectra_hall_cond.csv", conductivity_nb)
    @test occursin("sigma_xy", conductivity_nb)
end
```

- [ ] **Step 2: Run failing HPC script test**

Run:

```bash
julia --project test/test_hpc_scripts.jl
```

Expected before implementation: summary omits derived physical resistivity and notebooks do not mention Hall inputs.

- [ ] **Step 3: Implement columnwise summary and delta-method resistivity**

In `projectHPC/example/batch_process_csv.jl`, change `read_conf_robust` so it computes finite means column by column:

```julia
names_out = String[]
means_out = Float64[]
for idx in indices
    vals = Float64[]
    for raw in data[:, idx]
        value = try
            Float64(raw)
        catch
            NaN
        end
        isfinite(value) && push!(vals, value)
    end
    if !isempty(vals)
        push!(names_out, string(col_names[idx]))
        push!(means_out, mean(vals))
    end
end
isempty(names_out) && return nothing, nothing
return names_out, means_out
```

Add a derived-resistivity helper:

```julia
function longitudinal_resistivity_stats(pairs)
    n = length(pairs)
    n == 0 && return nothing
    xs = [p[1] for p in pairs]
    ys = [p[2] for p in pairs]
    x = mean(xs)
    y = mean(ys)
    D = x^2 + y^2
    if !isfinite(D) || D == 0.0
        return (mean=NaN, err=NaN)
    end
    rho = x / D
    n == 1 && return (mean=rho, err=0.0)
    gx = (y^2 - x^2) / D^2
    gy = -2.0 * x * y / D^2
    Cxx = var(xs) / n
    Cxy = cov(xs, ys) / n
    Cyy = var(ys) / n
    variance = gx^2 * Cxx + 2.0 * gx * gy * Cxy + gy^2 * Cyy
    return (mean=rho, err=sqrt(max(variance, 0.0)))
end
```

Inside each temperature loop, maintain:

```julia
rho_pairs = Tuple{Float64, Float64}[]
```

When a `transport.csv` was read, build `t_conf_map` from `t_names/t_vals`; after pushing generic observables, append a pair only when both conductivities are finite:

```julia
if haskey(t_conf_map, "DC_Conductivity") && haskey(t_conf_map, "Hall_Conductivity")
    sx = t_conf_map["DC_Conductivity"]
    sy = t_conf_map["Hall_Conductivity"]
    if isfinite(sx) && isfinite(sy)
        push!(rho_pairs, (sx, sy))
    end
end
```

After generic mean/error columns are written into `row`, call `longitudinal_resistivity_stats(rho_pairs)`. If it returns data, set and register:

```julia
row["Longitudinal_Resistivity_mean"] = rho_stats.mean
row["Longitudinal_Resistivity_err"] = rho_stats.err
push!(all_keys, "Longitudinal_Resistivity_mean")
push!(all_keys, "Longitudinal_Resistivity_err")
```

This intentionally overwrites any old raw `Longitudinal_Resistivity_mean` that the generic loop might have produced.

- [ ] **Step 4: Implement notebook source changes**

Patch `projectHPC/example/plot_stiffness.ipynb` source cells so the resistivity cell uses this policy:

```python
rho_mean_col = "Longitudinal_Resistivity_mean"
rho_err_col = "Longitudinal_Resistivity_err"
hall_mean_col = "Hall_Conductivity_mean"
mask = df["T"] > T_cut
T = df.loc[mask, "T"]

if rho_mean_col in df.columns:
    R = df.loc[mask, rho_mean_col]
    R_err = df.loc[mask, rho_err_col] if rho_err_col in df.columns else None
    R_label = r"$\rho_{xx}$"
elif hall_mean_col in df.columns and "DC_Conductivity_mean" in df.columns:
    sx = df.loc[mask, "DC_Conductivity_mean"].to_numpy(dtype=float)
    sy = df.loc[mask, hall_mean_col].to_numpy(dtype=float)
    denom = sx * sx + sy * sy
    R = pd.Series(np.where(np.isfinite(denom) & (denom != 0.0), sx / denom, np.nan),
                  index=df.index[mask])
    R_err = None
    R_label = r"$\rho_{xx}$ from mean $\sigma$ tensor"
else:
    sigma = df.loc[mask, "DC_Conductivity_mean"]
    sigma_err = df.loc[mask, "DC_Conductivity_err"] if "DC_Conductivity_err" in df.columns else None
    R = 1.0 / sigma
    R_err = sigma_err / (sigma**2) if sigma_err is not None else None
    R_label = r"$1/\sigma_{xx}$ old-data proxy"
```

Patch `projectHPC/example/plot_conductivity.ipynb` so:

- `build_dc_comparison_df` reads `Longitudinal_Resistivity_mean/err` explicitly for `R_dc_kubo`.
- If Hall columns exist but summary resistivity is absent, it computes `sigma_xx/(sigma_xx^2 + sigma_xy^2)` from mean conductivities.
- It falls back to `1 / sigma_dc_kubo` only when both Hall and longitudinal resistivity columns are absent, and labels that row as an old-data proxy.
- Add `read_opt_cond(data_dir, filename="spectra_hall_cond.csv")` usage for Hall spectra and fit `Re_Sigma_xy` on the same low-frequency window as `Re_Sigma`.
- Compute `R_dc_optical = sigma_xx_fit / (sigma_xx_fit^2 + sigma_xy_fit^2)` when Hall spectra are present; otherwise use the old optical proxy label.

Use `apply_patch` for notebook JSON source lines where possible. If JSON structure makes a tiny line patch unsafe, use a short Python `json` transform that changes only `cell["source"]` arrays and preserves outputs.

- [ ] **Step 5: Run HPC script test**

Run:

```bash
julia --project test/test_hpc_scripts.jl
```

Expected after implementation: summary fixture and notebook source checks pass.

- [ ] **Step 6: Commit**

```bash
git add projectHPC/example/batch_process_csv.jl projectHPC/example/plot_stiffness.ipynb projectHPC/example/plot_conductivity.ipynb test/test_hpc_scripts.jl
git commit -m "汇总纵向电阻率"
```

## Task 6: Documentation and Full Verification

**Files:**
- Modify: `doc/observables.md`
- Modify: `doc/magnetic-field.md`

- [ ] **Step 1: Update formulas and output documentation**

In `doc/observables.md`, add formulas for:

```tex
\sigma_{xy}(\omega)
=\frac{i}{N}
\sum_{n\ne m}
\frac{f_n-f_m}{E_m-E_n}
\frac{J^x_{nm}J^y_{mn}}
{\omega-(E_m-E_n)+i\eta}
```

```tex
\sigma_{xy}^{\rm dc}=\mathrm{Re}\,\sigma_{xy}(0)
```

```tex
\rho_{xx}=
\frac{\overline{\sigma_{xx}}}
{\overline{\sigma_{xx}}^2+\overline{\sigma_{xy}}^2}
```

State that `DC_Conductivity` remains regular `sigma_xx` and excludes the superfluid delta peak. List `Hall_Conductivity`, `hall_cond`, `hall_opt_cond`, `spectra_hall_cond.csv`, and `Longitudinal_Resistivity_mean/err`.

In `doc/magnetic-field.md`, remove or replace the old warning that Hall conductivity is not implemented. Document the `J_y` bond convention: `+y`, `+x+y` with factor `+1`, and `+x-y` with factor `-1`.

- [ ] **Step 2: Run targeted verification**

Run:

```bash
julia --project test/test_magnetic_field.jl
julia --project test/test_twisted_spectra.jl
julia --project test/test_simulation_tbc.jl
julia --project test/test_postprocess_spectra.jl
julia --project test/test_hpc_scripts.jl
```

Expected: all targeted tests pass.

- [ ] **Step 3: Run full verification**

Run:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Expected: full test suite passes. If runtime is too high in the current environment, keep the targeted command output and record the full-suite limitation in the final report.

- [ ] **Step 4: Commit docs and any verification-only fixes**

```bash
git add doc/observables.md doc/magnetic-field.md
git status --short
git commit -m "补充霍尔电导公式文档"
```

Only commit if there are staged documentation or verification-fix changes.
