# Hall Conductivity and Longitudinal Resistivity Design

## Goal

Add Hall conductivity output to finite-field transport measurements so the notebooks can plot the physical longitudinal resistivity

$$
\rho_{xx}^{\rm dc}=\frac{\sigma_{xx}^{\rm dc}}{(\sigma_{xx}^{\rm dc})^2+(\sigma_{xy}^{\rm dc})^2}
$$

instead of the finite-field proxy \(1/\sigma_{xx}^{\rm dc}\). The implementation should also output the frequency-dependent Hall response \(\sigma_{xy}(\omega)\).

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

Here \(f_n=f(E_n)\), \(J^\alpha_{nm}=\langle n|J^\alpha|m\rangle\), and strict \(n=m\) terms are skipped. This sign convention is chosen so that the \(\alpha=\beta=x\) real part reduces to the existing positive Lorentzian-broadened \(\mathrm{Re}\,\sigma_{xx}(\omega)\). For near-degenerate \(m\ne n\), use the same stable limit as the static stiffness path:

$$
\frac{f_n-f_m}{E_m-E_n}\rightarrow \beta f_n(1-f_n).
$$

The DC Hall conductivity is the zero-frequency value of the same broadened expression:

$$
\sigma_{xy}^{\rm dc}=\mathrm{Re}\,\sigma_{xy}(0).
$$

The output should also store \(\mathrm{Re}\,\sigma_{xy}(\omega)\) and \(\mathrm{Im}\,\sigma_{xy}(\omega)\). If later analysis needs the opposite sign convention, changing it should be isolated to the definition of \(J_y\) or to the documented tensor convention, not scattered through notebooks.

## Architecture

### Current Operators

Extend the production current-operator builder in `src/Observables.jl` from x-only to direction-aware:

- `current_operator_matrix(cache, p; direction=:x, qx=0.0, qy=0.0)`
- `build_current_operator!(cache, p; direction=:x, qx=0.0, qy=0.0, store=:q0)`

For \(J_x\), preserve the existing bonds:

- \(+\hat{x}\)
- \(+\hat{x}+\hat{y}\)
- \(+\hat{x}-\hat{y}\)

For \(J_y\), use:

- \(+\hat{y}\)
- \(+\hat{x}+\hat{y}\) with direction factor \(+1\)
- \(+\hat{x}-\hat{y}\) with direction factor \(-1\)

The sign comes from differentiating the Peierls phase by the bond displacement component \(d_\alpha\). This avoids duplicating bond-specific formulas for each direction.

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

### Postprocessing

Update both top-level and HPC example processors:

- `scripts/batch_process_spectra.jl`
- `projectHPC/example/batch_process_spectra.jl`
- shared helpers only if needed

Generate:

```text
spectra_hall_cond.csv
```

with columns:

```text
omega,Re_Sigma_xy,Re_Error,Im_Sigma_xy,Im_Error
```

If a selected eta factor is requested, also write selected scalar DC Hall/resistivity files if useful for analysis, or fold them into the existing selected DC output format. The minimum required output is the T-summary scalar columns from `transport.csv` and the frequency-dependent Hall CSV.

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

For a clean or weak-random zero-field lattice:

- Compute `measure_transport_only`.
- Assert `abs(hall_conductivity) < tolerance`.
- Assert the Hall optical response is small over the frequency grid.
- Assert `longitudinal_resistivity ≈ 1 / dc_conductivity` when `dc_conductivity > 0`.

This checks time-reversal symmetry and the resistivity fallback relation.

### 3. Magnetic-Field Reversal Test

For matched random states at \(+B\) and \(-B\):

- Use identical disorder and pairing fields.
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

### 5. Complex Formula Normalization Test

Using the same helper, set \(\alpha=\beta=x\) in the complex tensor formula and compare its real part against the existing `optical_conductivity` on the full frequency grid. This catches missing factors of \(\pi\), sign errors in the denominator, and mismatches between the complex Hall implementation and the existing diagonal Kubo-Greenwood path.

### 6. Output Schema Test

Extend simulation output tests:

- `transport.csv` has `Hall_Conductivity` and `Longitudinal_Resistivity`.
- `spectra_bins.jld2` has `hall_cond`, `rho_xx`, `hall_opt_cond`, and eta variants.
- `conductivity_convention` metadata is updated.

Use tiny lattices and `n_measure=1` to keep runtime bounded.

### 7. Postprocessing Test

Extend `test/test_postprocess_spectra.jl` synthetic JLD2 fixtures:

- Include complex `hall_opt_cond_eta`.
- Verify `processed_hall_cond.csv` and/or `spectra_hall_cond.csv` headers:

```text
omega,Re_Sigma_xy,Re_Error,Im_Sigma_xy,Im_Error
```

- Verify complex mean and standard error are computed componentwise.
- Verify old fixtures without Hall keys still process successfully and remove stale Hall CSV outputs.

### 8. Notebook Data Logic Test

Avoid brittle full notebook execution if possible. Instead, keep notebook changes simple and test the underlying CSV columns through existing batch processors. If a lightweight notebook smoke test is added, it should only execute the data-loading/resistivity cell against a tiny fixture and verify the selected plotted column is `Longitudinal_Resistivity_mean`.

## Error Handling and Compatibility

- Old `spectra_bins.jld2` files without Hall keys should still postprocess.
- Old `summary_all.csv` files should still open in notebooks; plots should label `1/sigma_xx` as proxy.
- If `dc_conductivity <= 0` or denominator is non-finite, write `NaN` for `Longitudinal_Resistivity`.
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
