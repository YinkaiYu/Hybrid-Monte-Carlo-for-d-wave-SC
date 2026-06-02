# LDOS Spectrum Output Design

## Goal

Add optional LDOS(omega) spectrum output while keeping current LDOS(omega=0) output and default file size unchanged.

## User-Facing Interface

`run_simulation` gets a new keyword:

```julia
write_ldos_spectrum::Bool=false
```

When false, no LDOS spectrum datasets are written. Existing `LDOS_0` and `LDOS_0_eta` behavior remains unchanged.

When true, each spectra bin writes:

```text
LDOS              shape (N, nω)
LDOS_eta          shape (nη, N, nω)
```

The frequency grid is the existing `dos_omega_grid`, built from the spectra parameters:

```text
-ω_max:Δω:ω_max
```

For TBC spectra this uses `spectra_delta_omega`; otherwise it uses `p.Δω`. The eta values use the existing spectra eta base and `spectra_eta_factors`.

## Data Flow

Untwisted spectra already loop over every eigenstate, site, eta, and omega for DOS and LDOS(0). The implementation will accumulate site-resolved Lorentzian weights over the full `dos_omega_grid` only when `write_ldos_spectrum=true`.

Twisted spectra will do the same inside the TBC sector loop, with the same final `Ltw^2` normalization used by `ldos_ω0_eta`.

Simulation binning will lazily allocate and average the new arrays only when the measurement result contains them. JLD2 metadata records `write_ldos_spectrum` and `ldos_spectrum_grid_key="dos_omega_grid"` for downstream checks.

## Post-Processing

Single-directory scripts write:

```text
processed_ldos.csv
```

HPC ensemble post-processing writes:

```text
spectra_ldos.csv
```

Both use long-table CSV format:

```text
x,y,site,omega,LDOS,Error
```

The selected `eta_factor` follows the existing multi-eta selection logic.

## HPC Files

`projectHPC/run_conf.jl` reads optional `write_ldos_spectrum` from `params.jl`, defaults to `false`, logs it, and forwards it to `run_simulation`.

`projectHPC/example/sweep_T.sh` includes:

```julia
write_ldos_spectrum = false
```

Users can set it true for runs where the larger output is desired.

Add `projectHPC/example/plot_ldos_spectrum.ipynb` to inspect `spectra_ldos.csv` or `processed_ldos.csv`.

## Tests

Regression tests cover:

- default simulations do not write `LDOS` or `LDOS_eta`;
- enabling `write_ldos_spectrum` writes arrays with expected shapes and `LDOS == LDOS_eta[1, :, :]`;
- post-processing exports long-table LDOS spectra when present;
- HPC scripts pass through the new option and keep the default false.
