# Hall Conductivity and Longitudinal Resistivity Design

## Goal

Add Hall conductivity output to finite-field transport measurements so the notebooks can plot the physical longitudinal resistivity

$$
\rho_{xx}^{\rm dc}=\frac{\sigma_{xx}^{\rm dc}}{(\sigma_{xx}^{\rm dc})^2+(\sigma_{xy}^{\rm dc})^2}
$$

instead of the finite-field proxy \(1/\sigma_{xx}^{\rm dc}\). The implementation should also output the frequency-dependent Hall response \(\sigma_{xy}(\omega)\), and the conductivity notebook should use both \(\sigma_{xx}(\omega)\) and \(\sigma_{xy}(\omega)\) when it builds low-frequency optical extrapolations of \(\rho_{xx}\).

## Current State

The existing transport path computes:

- `DC_Conductivity`: regular \( \sigma_{xx}^{\rm dc} \)
- `optical_conductivity`: \( \mathrm{Re}\,\sigma_{xx}(\omega) \)
- `dc_cond_eta` / `opt_cond_eta` in `spectra_bins.jld2`
- postprocessed `spectra_opt_cond.csv` with columns `omega,Re_Sigma,Error`

Finite-field documentation currently warns that `1 / DC_Conductivity` is only a proxy because Hall conductivity is missing.

## Physical Definitions

For the current direction \(\alpha\in\{x,y\}\), define the \(q=0\) current operator \(J^\alpha\) as the Peierls derivative of the kinetic BdG Hamiltonian with respect to a uniform vector potential \(A_\alpha\). Landau-gauge link phases are included exactly as in the Hamiltonian.

For the diagonal channel, keep the existing regular Kubo-Greenwood convention:

$$
\mathrm{Re}\,\sigma_{xx}(\omega)
=\frac{\pi}{N\omega}
\sum_{n\ne m}(f_n-f_m)|J^x_{nm}|^2
\delta_\eta\!\left(\omega-(E_m-E_n)\right),
$$

and

$$
\sigma_{xx}^{\rm dc}
=\frac{\pi}{N}\sum_{n\ne m}\beta f_n(1-f_n)|J^x_{nm}|^2
\delta_\eta(E_m-E_n).
$$

For the Hall channel, compute the complex optical tensor from the same eigenbasis:

$$
\sigma_{xy}(\omega)
=\frac{i}{N}
\sum_{n\ne m}
\frac{f_n-f_m}{E_m-E_n}
\frac{J^x_{nm}J^y_{mn}}
{\omega-(E_m-E_n)+i\eta}.
$$

Here \(f_n=f(E_n)\), \(J^\alpha_{nm}=\langle n|J^\alpha|m\rangle\), and strict \(n=m\) terms are skipped. This is a complex Kubo tensor convention for the Hall channel. At finite \(\eta\), its \(\alpha=\beta=x\) specialization is not identical point-by-point to the existing Kubo-Greenwood `optical_conductivity` output, because the existing diagonal channel uses \(1/\omega\) in the broadened spectral weight while this tensor formula uses \((E_m-E_n)^{-1}\). Therefore the implementation must not change the existing \(\mathrm{Re}\,\sigma_{xx}(\omega)\) semantics and must not test finite-\(\eta\) equality between those two conventions. For near-degenerate \(m\ne n\), use the same stable limit as the static stiffness path:

$$
\frac{f_n-f_m}{E_m-E_n}\rightarrow \beta f_n(1-f_n).
$$

The DC Hall conductivity is the zero-frequency value of the same broadened expression:

$$
\sigma_{xy}^{\rm dc}=\mathrm{Re}\,\sigma_{xy}(0).
$$

The output should also store \(\mathrm{Re}\,\sigma_{xy}(\omega)\) and \(\mathrm{Im}\,\sigma_{xy}(\omega)\). If later analysis needs the opposite sign convention, changing it should be isolated to the definition of \(J_y\) or to the documented tensor convention, not scattered through notebooks.

For notebook optical extrapolations, keep using the existing `spectra_opt_cond.csv` \(\mathrm{Re}\,\sigma_{xx}(\omega)\) data and fit \(\mathrm{Re}\,\sigma_{xy}(\omega)\) from `spectra_hall_cond.csv` on the same low-frequency window. The extrapolated optical resistivity is then

$$
\rho_{xx}^{\rm optical} =
\frac{\sigma_{xx}^{\rm fit}}{(\sigma_{xx}^{\rm fit})^2+(\sigma_{xy}^{\rm fit})^2}.
$$

This is an analysis-level tensor inversion using the available finite-\(\eta\) outputs. It should be labelled separately from the direct Kubo scalar `Longitudinal_Resistivity`.

## Architecture

### Current Operators

Extend the production current-operator builder in `src/Observables.jl` from x-only to direction-aware:

- `current_operator_matrix(cache, p; direction=:x, qx=0.0, qy=0.0)`
- `build_current_operator!(cache, p; direction=:x, qx=0.0, qy=0.0, store=:q0)`
- `probe_current_operator_matrix(cache, p; direction=:x, qx=0.0, qy=0.0)`
- `build_probe_H_BdG!(H, cache, p, state; direction=:x, λ, qx=0.0, qy=0.0)`

For \(J_x\), preserve the existing bonds:

- \(+\hat{x}\)
- \(+\hat{x}+\hat{y}\)
- \(+\hat{x}-\hat{y}\)

For \(J_y\), use:

- \(+\hat{y}\)
- \(+\hat{x}+\hat{y}\) with direction factor \(+1\)
- \(+\hat{x}-\hat{y}\) with direction factor \(-1\)

The sign comes from differentiating the Peierls phase by the bond displacement component \(d_\alpha\). This avoids duplicating bond-specific formulas for each direction.

The probe Hamiltonian used by derivative tests must use the same bond/displacement convention. A uniform \(A_y\) probe should phase the \(+\hat y\), \(+\hat x+\hat y\), and \(+\hat x-\hat y\) kinetic bonds by \(d_y A_y\), so the diagonal \(+\hat x-\hat y\) bond carries the opposite sign from \(+\hat x+\hat y\).

### Cache

Add one additional sparse cache slot to `ComputeCache`:

- `Jy_sparse_q0::SparseMatrixCSC{ComplexF64, Int}`

Keep the existing `J_mn` and `temp_JU` dense work buffers. Hall evaluation can compute \(J_x\) matrix elements, copy them to a local dense matrix or a reusable cache, then compute \(J_y\) matrix elements and combine them. If allocation cost is noticeable, add a second dense matrix cache such as `J_mn_aux`; prefer correctness and clarity first.

### Transport Results

Extend `TransportResult` and `SpectrumResult` with:

- `hall_conductivity::Float64`
- `longitudinal_resistivity::Float64`
- `hall_conductivity_eta::Vector{Float64}`
- `longitudinal_resistivity_eta::Vector{Float64}`
- `hall_optical_conductivity::Vector{ComplexF64}`
- `hall_optical_conductivity_eta::Matrix{ComplexF64}`

Existing fields must stay in place for compatibility. Existing code that reads `dc_conductivity` and `optical_conductivity` should continue to work.

### Simulation Output

Update `transport.csv` headers:

Without twist diagnostics:

```text
Sweep,Superfluid_Stiffness,DC_Conductivity,Hall_Conductivity,Longitudinal_Resistivity
```

With twist diagnostics, append Hall and resistivity before twist columns:

```text
Sweep,Superfluid_Stiffness,DC_Conductivity,Hall_Conductivity,Longitudinal_Resistivity,Twist_Qy,...
```

In `spectra_bins.jld2`, add per-bin keys:

- `hall_cond`
- `rho_xx`
- `hall_cond_eta`
- `rho_xx_eta`
- `hall_opt_cond`
- `hall_opt_cond_eta`

Keep `conductivity_convention` but update the value to describe the expanded tensor, for example `sigma_xx_regular_sigma_xy_kubo`.

`src/Simulation.jl` writes spectra bins through explicit accumulators, not by reflecting `SpectrumResult` fields. Add and maintain these accumulators:

- `accum_hall_cond_eta::Vector{Float64}`
- `accum_rho_xx_eta::Vector{Float64}`
- `accum_hall_opt_cond::Vector{ComplexF64}`
- `accum_hall_opt_eta::Matrix{ComplexF64}`

On the first bin, copy these fields from `spec_res`; on later bins, add them elementwise; at flush time divide by `bin_count` and write both single-eta and eta-first keys. `transport.csv` rows should use `spec_res.hall_conductivity` and `spec_res.longitudinal_resistivity`.

The final naming map is:

| Layer | DC Hall | DC longitudinal resistivity | Optical Hall |
| --- | --- | --- | --- |
| Julia field | `hall_conductivity` | `longitudinal_resistivity` | `hall_optical_conductivity` |
| Julia eta field | `hall_conductivity_eta` | `longitudinal_resistivity_eta` | `hall_optical_conductivity_eta` |
| `transport.csv` | `Hall_Conductivity` | `Longitudinal_Resistivity` | none |
| JLD2 single eta | `hall_cond` | `rho_xx` | `hall_opt_cond` |
| JLD2 multi eta | `hall_cond_eta` | `rho_xx_eta` | `hall_opt_cond_eta` |
| HPC summary | `Hall_Conductivity_mean/err` | `Longitudinal_Resistivity_mean/err` | none |
| spectra CSV | none | none | `spectra_hall_cond.csv` |

### Postprocessing

Update all top-level and HPC example processors:

- `scripts/process_spectra.jl`
- `scripts/batch_process_spectra.jl`
- `projectHPC/example/batch_process_spectra.jl`
- `scripts/spectra_postprocess_utils.jl`
- `projectHPC/example/spectra_postprocess_utils.jl`

Generate:

```text
spectra_hall_cond.csv
```

with columns:

```text
omega,Re_Sigma_xy,Re_Error,Im_Sigma_xy,Im_Error
```

Complex Hall arrays need componentwise statistics. Add a helper that computes complex means plus separate standard errors for the real and imaginary parts. Old JLD2 files without Hall keys should still process and should remove stale `processed_hall_cond.csv` / `spectra_hall_cond.csv` outputs.

If a selected eta factor is requested, also write selected scalar DC Hall/resistivity files if useful for analysis, or fold them into the existing selected DC output format. The minimum required scalar output is the T-summary columns from `transport.csv`; the minimum required frequency output is the Hall CSV above.

### Summary CSV

Update `projectHPC/example/batch_process_csv.jl` so one invalid column does not discard the whole `observables.csv` or `transport.csv`. The reader should compute means column by column, keep finite columns, and skip only columns whose selected values produce `NaN` or `Inf`. Error estimates should use the number of finite samples for each observable, not `real_n_conf`, when deciding whether a nonzero standard error can be computed.

### Notebooks

Update:

- `projectHPC/example/plot_stiffness.ipynb`
- `projectHPC/example/plot_conductivity.ipynb`

Notebook behavior:

- Prefer `Longitudinal_Resistivity_mean` when present.
- Use `Longitudinal_Resistivity_err` for error bars.
- Fall back to `1 / DC_Conductivity_mean` only for old data and label it as a proxy.
- In conductivity notebook, load and plot `spectra_hall_cond.csv` when present.
- Keep existing `spectra_opt_cond.csv` plots for \(\mathrm{Re}\,\sigma_{xx}(\omega)\).
- In `plot_conductivity.ipynb`, `build_dc_comparison_df` should read `Longitudinal_Resistivity_mean/err` explicitly. `R_dc_kubo` comes from those columns when present, not from `1 / sigma_dc_kubo`.
- For optical extrapolation, fit both \(\mathrm{Re}\,\sigma_{xx}(\omega)\) and \(\mathrm{Re}\,\sigma_{xy}(\omega)\) on the selected low-frequency window, then compute `R_dc_optical` from the tensor formula. Only if Hall spectra are absent should it fall back to the old `1 / sigma_xx` proxy, with the label showing that it is a proxy.

### Documentation

Update:

- `doc/observables.md`
- `doc/magnetic-field.md`

Documentation must include:

- \(J_x\) and \(J_y\) bond conventions.
- \( \sigma_{xy}(\omega) \) complex Kubo formula.
- \( \sigma_{xy}^{\rm dc}=\mathrm{Re}\,\sigma_{xy}(0) \).
- \( \rho_{xx}=\sigma_{xx}/(\sigma_{xx}^2+\sigma_{xy}^2) \).
- Clarification that `DC_Conductivity` remains regular \( \sigma_{xx} \) and excludes the superfluid delta peak.
- Removal or replacement of the old “Hall conductivity not implemented” warning.

## Testing Design

The tests should be stronger than “field exists and is finite”. They should isolate formula, sign, symmetry, and file IO.

### 1. Current-Operator Derivative Tests

Extend `test/test_magnetic_field.jl` with finite-difference derivative checks for both directions:

- Build probe Hamiltonians with a uniform \(A_x\) and compare \(-J_x\) or the documented sign relation to `current_operator_matrix(...; direction=:x)`.
- Build probe Hamiltonians with a uniform \(A_y\) and compare \(-J_y\) or the documented sign relation to `current_operator_matrix(...; direction=:y)`.
- Run for finite magnetic field with `boundary_condition=:magnetic_pbc`.
- Include a case where diagonal \(t'\) hopping is nonzero, so the \(+\hat{x}+\hat{y}\) and \(+\hat{x}-\hat{y}\) signs for \(J_y\) are exercised.

Expected failure mode before implementation: `direction=:y` is unsupported or the finite-difference comparison fails.

### 2. Zero-Field Symmetry Test

For a clean or weak-random zero-field lattice with a real pairing field:

- Compute `measure_transport_only`.
- Assert `abs(hall_conductivity) < tolerance`.
- Assert the Hall optical response is small over the frequency grid.
- Assert `longitudinal_resistivity ≈ 1 / dc_conductivity` when `dc_conductivity > 0`.

This checks time-reversal symmetry and the resistivity fallback relation. Do not use the default random complex `initialize_state` field for this test, because a complex pairing field can break time reversal by itself.

### 3. Magnetic-Field Reversal Test

For matched random states at \(+B\) and \(-B\):

- Use identical disorder fields.
- Use real pairing fields on both sides, or set \(\Delta(-B)=\Delta(+B)^*\).
- Diagonalize both.
- Assert `hall_conductivity(+B) ≈ -hall_conductivity(-B)` within a tolerance suitable for the small lattice.
- Assert `dc_conductivity(+B) ≈ dc_conductivity(-B)`.
- Assert `longitudinal_resistivity(+B) ≈ longitudinal_resistivity(-B)`.

This catches the most likely sign convention errors.

### 4. Direct Formula Regression Test

Add a small helper inside the test file that manually computes \(J_x\), \(J_y\), and \(\sigma_{xy}^{\rm dc}\) from the returned current matrices and eigenvectors, without using the production Hall accumulation helper.

Compare:

- `res.hall_conductivity`
- `res.hall_conductivity_eta[1]`
- `res.hall_optical_conductivity[1]` at the lowest frequency if the formula is evaluated on the same grid

This makes sure the transport result fields are not just populated but numerically tied to the documented formula.

### 5. Resistivity Formula Test

Directly verify

```text
rho_xx = sigma_xx / (sigma_xx^2 + sigma_xy^2)
```

for every eta value. Also verify that `sigma_xy = 0` gives `rho_xx = 1 / sigma_xx` when `sigma_xx > 0`.

### 6. Output Schema Test

Extend simulation output tests:

- `transport.csv` has `Hall_Conductivity` and `Longitudinal_Resistivity`.
- `spectra_bins.jld2` has `hall_cond`, `rho_xx`, `hall_opt_cond`, and eta variants.
- `conductivity_convention` metadata is updated.

Use tiny lattices and `n_measure=1` to keep runtime bounded.

### 7. Postprocessing Test

Extend `test/test_postprocess_spectra.jl` synthetic JLD2 fixtures:

- Include complex `hall_opt_cond_eta`.
- Verify `processed_hall_cond.csv` from `scripts/process_spectra.jl`.
- Verify `spectra_hall_cond.csv` from `scripts/batch_process_spectra.jl` and `projectHPC/example/batch_process_spectra.jl`.
- Verify all Hall CSV headers:

```text
omega,Re_Sigma_xy,Re_Error,Im_Sigma_xy,Im_Error
```

- Verify complex mean and standard error are computed componentwise.
- Verify old fixtures without Hall keys still process successfully and remove stale Hall CSV outputs in all three processors.
- Add a tiny fixture for `projectHPC/example/batch_process_csv.jl` confirming `Longitudinal_Resistivity_mean/err` reaches `summary_all.csv`, and confirming an invalid resistivity column does not discard finite `DC_Conductivity` and `Hall_Conductivity`.

### 8. Notebook Data Logic Test

Avoid brittle full notebook execution if possible. Instead, keep notebook changes simple and test the underlying CSV columns through existing batch processors. If a lightweight notebook smoke test is added, it should only execute the data-loading/resistivity cell against a tiny fixture and verify the selected plotted column is `Longitudinal_Resistivity_mean`.

## Error Handling and Compatibility

- Old `spectra_bins.jld2` files without Hall keys should still postprocess.
- Old `summary_all.csv` files should still open in notebooks; plots should label `1/sigma_xx` as proxy.
- If `dc_conductivity <= 0` or the tensor denominator is non-finite, write `NaN` for `Longitudinal_Resistivity` in raw outputs. Summary processing must be columnwise finite-value tolerant so this does not discard other valid transport columns.
- Multi-eta shape mismatches should raise the existing eta compatibility errors.
- `sigma_xy` arrays are complex; postprocessing must not discard the imaginary part.

## Verification Commands

Targeted commands after implementation:

```bash
julia --project test/test_magnetic_field.jl
julia --project test/test_postprocess_spectra.jl
julia --project test/test_simulation_tbc.jl
```

Broader verification:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

The full test suite may take time because of diagonalizations; targeted tests should be run first during iteration.
