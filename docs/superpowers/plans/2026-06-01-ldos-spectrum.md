# LDOS Spectrum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional LDOS(omega) spectrum output controlled by `write_ldos_spectrum=false`.

**Architecture:** Extend existing spectra result structs with optional LDOS spectrum arrays, compute them only when requested, and carry them through simulation binning into JLD2. Reuse existing `dos_omega_grid` and multi-eta post-processing helpers for CSV export.

**Tech Stack:** Julia, JLD2, existing DwaveHMC spectra code, shell SLURM examples, Jupyter notebook JSON.

---

## File Structure

- Modify `src/Observables.jl`: optional untwisted LDOS spectrum computation and result fields.
- Modify `src/TwistedSpectra.jl`: optional TBC LDOS spectrum computation and result fields.
- Modify `src/Simulation.jl`: keyword, metadata, bin accumulation, JLD2 writes.
- Modify `projectHPC/run_conf.jl`: optional params read and forwarding.
- Modify `scripts/spectra_postprocess_utils.jl`: 3D eta-first selector and LDOS spectrum CSV writer.
- Modify `scripts/process_spectra.jl` and `scripts/batch_process_spectra.jl`: collect and export LDOS spectra.
- Modify `projectHPC/example/spectra_postprocess_utils.jl` and `projectHPC/example/batch_process_spectra.jl`: HPC ensemble support.
- Modify `projectHPC/example/sweep_T.sh`: default false parameter.
- Create `projectHPC/example/plot_ldos_spectrum.ipynb`: plotting notebook.
- Modify tests in `test/test_simulation_tbc.jl`, `test/test_postprocess_spectra.jl`, and `test/test_hpc_scripts.jl`.

### Task 1: Red Tests

- [ ] Add simulation tests asserting default absence and enabled presence of `LDOS` and `LDOS_eta`.
- [ ] Add synthetic post-processing fixture data with `LDOS_eta` shaped `(eta, site, omega)`.
- [ ] Add assertions for `processed_ldos.csv` and `spectra_ldos.csv` row counts and headers.
- [ ] Add HPC script assertions for `write_ldos_spectrum=false` and run-conf forwarding.
- [ ] Run targeted tests and verify they fail on missing feature:

```bash
julia --project test/test_simulation_tbc.jl
julia --project test/test_postprocess_spectra.jl
julia --project test/test_hpc_scripts.jl
```

### Task 2: Core Measurement

- [ ] Add optional `ldos_ω_eta::Union{Nothing,Array{Float64,3}}` and `ldos_ω::Union{Nothing,Matrix{Float64}}` fields to spectra result structs.
- [ ] Add `write_ldos_spectrum::Bool=false` keyword to `measure_untwisted_spectra` and `measure_twisted_spectra`.
- [ ] Accumulate site-resolved Lorentzian weights over all omega only when the flag is true.
- [ ] Preserve existing `LDOS_0` semantics and normalizations.
- [ ] Run the simulation test and verify it passes.

### Task 3: Simulation Binning

- [ ] Add `write_ldos_spectrum::Bool=false` keyword to `run_simulation`.
- [ ] Write metadata `write_ldos_spectrum` and `ldos_spectrum_grid_key`.
- [ ] Forward the flag into untwisted and TBC spectra measurements.
- [ ] Lazily allocate, sum, average, and write `LDOS` / `LDOS_eta` only when present.
- [ ] Run the simulation test again.

### Task 4: Post-Processing

- [ ] Add `selected_site_omega_matrix` helper for `(eta, site, omega)` data.
- [ ] Add `write_ldos_spectrum_csv`.
- [ ] Collect `LDOS_eta` / `LDOS` in single-directory scripts and write `processed_ldos.csv`.
- [ ] Collect per-config LDOS spectra in HPC batch post-processing and write `spectra_ldos.csv`.
- [ ] Run post-processing tests.

### Task 5: HPC Examples And Notebook

- [ ] Update `projectHPC/run_conf.jl`.
- [ ] Update `projectHPC/example/sweep_T.sh`.
- [ ] Add `projectHPC/example/plot_ldos_spectrum.ipynb`.
- [ ] Run HPC script tests.

### Task 6: Verification

- [ ] Run targeted tests:

```bash
julia --project test/test_simulation_tbc.jl
julia --project test/test_postprocess_spectra.jl
julia --project test/test_hpc_scripts.jl
```

- [ ] Run `git diff --check`.
- [ ] Inspect `git diff --stat`.
