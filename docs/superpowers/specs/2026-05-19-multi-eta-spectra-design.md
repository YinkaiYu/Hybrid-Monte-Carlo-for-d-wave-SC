# Multi-eta spectra and DC conductivity design

## Context

The current simulation writes `spectra_bins.jld2` after the heavy
transport/spectra measurement has already applied one Lorentzian broadening
parameter. Post-processing then averages those broadened arrays and extracts
path observables. This means a later change of `η` requires rerunning the
simulation, except for approximate one-way broadening of already broadened
curves.

The new workflow targets future data. During each heavy measurement the code
will compute a fixed list of Lorentzian broadenings from the same eigenvalues,
current matrix elements, FFT weights, and TBC sectors. This keeps the expensive
parts shared and stores exact results for each requested broadening.

## Goals

- Store exact results for `η`, `2η`, `4η`, `8η`, `16η`, `32η`, and `64η`.
- Make scalar DC conductivity and the antinodal `A_MX_path` spectra strictly
  selectable by `η` during post-processing.
- Cover the other spectra arrays with the same multi-`η` mechanism when they
  do not introduce disproportionate cost.
- Preserve the default CSV workflow: standard CSV files continue to represent
  `eta_factor = 1` unless a post-processing script explicitly selects another
  factor.
- Avoid high-GC implementations. Inner measurement loops should reuse buffers
  and write results in place.

## Non-goals

- Do not support arbitrary continuous `η` values in post-processing.
- Do not try to reconstruct new `η` values from old single-`η` data.
- Do not generate multiple `*_eta_x*.csv` files by default.
- Do not change the HMC sampling, forces, acceptance rule, or superfluid
  stiffness definition.

## User-facing Interface

`run_simulation` gains an optional keyword:

```julia
spectra_eta_factors = [1, 2, 4, 8, 16, 32, 64]
```

The actual broadening values are

```julia
eta_values = actual_spectra_eta .* spectra_eta_factors
```

where `actual_spectra_eta` is the existing single-`η` value after the current
TBC/non-TBC selection logic. The factors must be positive and non-empty.

Post-processing functions gain an optional selection keyword, defaulting to the
old behavior:

```julia
process_spectra_directory(target_dir; eta_factor=1)
process_single_directory(target_dir; eta_factor=1)
process_T_directory(t_dir; eta_factor=1)
```

The HPC scripts also accept:

```bash
DWAVEHMC_SPECTRA_ETA_FACTOR=8
```

When set, the post-processing scripts select the matching `eta_factor` slice
from JLD2 and export the standard CSV filenames for that one selected
broadening.

## Measurement Architecture

The expensive work remains shared:

- BdG diagonalization is performed once per untwisted heavy measurement.
- For TBC spectra, each twist sector is diagonalized once.
- Current matrix elements for transport are built once.
- FFTs and momentum weights are computed once per eigenstate and sector.

After these shared quantities are available, the measurement loops evaluate the
Lorentzian for each `eta_values[i]` and accumulate into multi-`η` result arrays.
Superfluid stiffness remains scalar because it does not depend on `η`.

Mandatory multi-`η` quantities:

- `dc_cond_eta`
- `A_MX_path_eta`

Additional multi-`η` quantities written by the same framework:

- `opt_cond_eta`
- `dos_eta`
- `dos_M_eta`
- `dos_M_patch_eta` for TBC spectra
- `LDOS_0_eta`
- `A_k0_eta`
- `A_XG_path_eta`
- `A_XG_node_patch_eta` for TBC spectra

The existing single-`η` fields are still written using the first factor. This
keeps old notebooks and scripts working.

## JLD2 Data Format

Top-level metadata added to `spectra_bins.jld2`:

```text
multi_eta_enabled      :: Bool
spectra_eta_factors    :: Vector{Float64}
eta_values             :: Vector{Float64}
spectra_eta_base       :: Float64
```

Each `sweep_*` group adds fields with `η` as the first dimension:

```text
dc_cond_eta             :: Vector{Float64}
opt_cond_eta            :: Matrix{Float64}       # n_eta x n_omega
dos_eta                 :: Matrix{Float64}       # n_eta x n_dos_omega
dos_M_eta               :: Matrix{Float64}
dos_M_patch_eta         :: Matrix{Float64}       # TBC only
LDOS_0_eta              :: Matrix{Float64}       # n_eta x N
A_k0_eta                :: Array{Float64,3}      # n_eta x Lx_eff x Ly_eff
A_MX_path_eta           :: Array{Float64,3}      # n_eta x n_k_mx x n_dos_omega
A_XG_path_eta           :: Array{Float64,3}
A_XG_node_patch_eta     :: Array{Float64,3}      # TBC only
```

Existing fields remain present and equal to the first `η` slice:

```text
opt_cond
dos
dos_M
dos_M_patch
LDOS_0
A_k0
A_MX_path
A_XG_path
A_XG_node_patch
```

`transport.csv` keeps the current columns and writes `DC_Conductivity` for
`eta_factor = 1`. Full multi-`η` DC information is stored in JLD2, not in the
default CSV output.

## Post-processing Behavior

Post-processing reads `eta_factor`, finds the corresponding index in
`spectra_eta_factors`, and uses that slice for all exported spectra CSV files.
The standard CSV filenames are reused:

- `processed_dos_AN.csv`
- `processed_MX_path.csv`
- `processed_dos.csv`
- `spectra_dos_AN.csv`
- `spectra_MX_path.csv`
- `spectra_dos.csv`

These files represent the selected broadening. With the default
`eta_factor = 1`, output is unchanged.

When post-processing summary data needs a non-default DC conductivity, it must
read the selected `dc_cond_eta` slice from JLD2. Existing CSV summary scripts
continue to read `transport.csv` and therefore report the default
`eta_factor = 1` unless explicitly extended to select from JLD2.

## Compatibility and Errors

- New files with `multi_eta_enabled = true` support any factor listed in
  `spectra_eta_factors`.
- Old files without multi-`η` fields support only `eta_factor = 1`.
- Selecting a factor not present in the file raises an error that includes the
  available factors.
- For floating-point factor matching, post-processing uses `isapprox` instead
  of exact equality.
- Invalid `spectra_eta_factors` values are rejected at simulation startup:
  empty lists, zero, negative, `NaN`, or `Inf`.
- If a config in an HPC ensemble has incompatible factors or malformed
  multi-`η` dimensions, the HPC post-processor skips that config with a warning,
  matching its existing incompatible-metadata strategy.
- Within one ensemble, selected `eta_factor`, grids, path metadata, and result
  shapes must match before averaging.

## Performance and Allocation Constraints

The implementation should avoid per-iteration allocation in the heavy loops:

- Store `eta_values`, result arrays, and Lorentzian work buffers in reusable
  measurement workspaces or `ComputeCache` where practical.
- Allocate multi-`η` result arrays once per measurement result shape, then
  fill and accumulate in place.
- In loops over eigenstates, matrix elements, frequencies, and `η`, do not
  create temporary arrays or comprehensions.
- Reuse a two-dimensional Lorentzian cache when helpful, with shape
  `n_eta x n_omega`, or a one-dimensional cache updated per `η` if that gives
  lower memory pressure.
- Simulation bin accumulators should be lazy-initialized once from the first
  result and then updated with `.+=` and normalized with `./=`.
- JLD2 writes should store already accumulated arrays directly and avoid
  building duplicate intermediate containers.

Expected cost:

- Diagonalization cost is unchanged.
- FFT cost is unchanged except for any small bookkeeping needed by multi-`η`
  arrays.
- Runtime increases mainly through Lorentzian evaluations and array
  accumulation over seven `η` values.
- JLD2 size increases roughly with the number of stored multi-`η` arrays. The
  default CSV output size does not increase.

## Testing Plan

- `run_simulation` smoke test writes `multi_eta_enabled`, `spectra_eta_factors`,
  `eta_values`, `dc_cond_eta`, and `A_MX_path_eta`.
- Multi-`η` field first dimensions equal `length(eta_values)`.
- For `eta_factor = 1`, the new multi-`η` first slices match old fields such as
  `A_MX_path`, `dos`, and `opt_cond`.
- Synthetic JLD2 post-processing with deliberately different slices proves that
  `eta_factor = 4` exports the selected slice to the standard CSV filenames.
- Old-format synthetic JLD2 still processes with `eta_factor = 1`.
- Old-format synthetic JLD2 rejects `eta_factor != 1` with a clear error.
- HPC post-processing averages compatible multi-`η` configs and skips configs
  with incompatible factors or malformed dimensions.
- Add a small allocation sanity check or benchmark around the core multi-`η`
  measurement path to catch obvious temporary allocations in inner loops.

## Migration

Existing data and notebooks continue to work because old fields and standard
CSV names remain available. New data carries richer multi-`η` JLD2 fields.
Users who want another broadening rerun only the post-processing script with a
selected `eta_factor`; they do not rerun HMC or TBC spectra measurements.
