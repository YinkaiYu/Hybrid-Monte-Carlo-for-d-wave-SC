# Twisted-Boundary Spectra Design

## Goal

Add an opt-in twisted-boundary-condition (TBC) measurement path for spectral observables, improving effective momentum resolution for DOS-derived and momentum-resolved spectra without changing HMC sampling dynamics or existing default outputs.

When TBC is disabled, `run_simulation` must preserve the current compute path, output shapes, and JLD2 keys. When TBC is enabled, only spectral arrays use TBC; transport, optical conductivity, current response, superfluid stiffness, finite-difference twist diagnostics, HMC proposals, and HMC forces continue to use the current untwisted BdG cache.

## Scope

Implement TBC for:

- `dos`
- exact-point `dos_AN`, preserving the current meaning
- new antinodal patch average `dos_AN_patch`
- `A_k_ω0`
- `A_kpath` on the existing antinodal path `(π,0) -> (π,π)`

Do not apply TBC to:

- `compute_forces!`
- `hmc_sweep!`
- `cache.H_base`, `cache.U`, or `cache.E_n`
- current operators
- optical conductivity
- DC conductivity
- superfluid stiffness
- existing finite-difference twist stiffness diagnostics

## Current-Code Constraints

The current repository already has twist-related stiffness code:

- `build_twisted_H_BdG!` in `src/Hamiltonian.jl`
- `measure_twist_stiffness` and `measure_twist_stiffness_qy` in `src/Observables.jl`
- `measure_twist`, `twist_Ax`, and `twist_qy` options in `run_simulation`

Those routines implement Peierls vector-potential perturbations for stiffness diagnostics. They are not the same object as spectral TBC boundary phases. The TBC implementation must avoid reusing or overloading those names for a different physical convention.

## Public Interface

Add a new source file:

```text
src/TwistedSpectra.jl
```

Include it from `src/DwaveHMC.jl` after `Observables.jl` and before `HMC.jl` or `Simulation.jl`. Export only:

```julia
measure_twisted_spectra
```

Add a result type, likely in `src/TwistedSpectra.jl`:

```julia
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
```

The primary measurement entry point should be:

```julia
measure_twisted_spectra(cache::ComputeCache,
                        p::ModelParameters,
                        state::SimulationState;
                        Ltw::Int=2,
                        antinode_patch_half_width::Float64=π / max(p.Lx, p.Ly),
                        reuse_buffers::Bool=false)
```

Validation:

- `Ltw > 0`
- `antinode_patch_half_width >= 0`
- if exact `(π,0)` path support is required, warn or error clearly when lattice sizes make the requested path ambiguous; existing code already uses nearest discrete indices, so the first implementation can follow that convention.

## Simulation Interface

Extend `run_simulation` with:

```julia
spectra_Ltw::Int = 1
use_twisted_spectra::Bool = spectra_Ltw > 1
antinode_patch_half_width::Float64 = π / max(p.Lx, p.Ly)
```

Behavior:

- `spectra_Ltw == 1` and `use_twisted_spectra == false` preserve existing behavior.
- `use_twisted_spectra == true` requires `spectra_Ltw > 0`.
- If `use_twisted_spectra == true` with `spectra_Ltw == 1`, run the TBC code path as a strict regression mode. This is useful for tests and debugging.
- Log whether TBC is enabled and the effective lattice size.

JLD2 metadata:

- Always write existing metadata keys.
- Add `spectra_Ltw`, `spectra_Lx_eff`, and `spectra_Ly_eff`.
- When TBC is enabled, write effective-grid `kpath_kx` and `kpath_ky`.
- When TBC is disabled, keep existing `kpath_kx_idx` and `kpath_ky_idx` keys.
- In sweep groups, write `dos_AN_patch` when TBC is enabled. The post-processing scripts must treat this key as optional for backward compatibility.

## Architecture

Use the recommended route: split existing heavy measurement internally while preserving the current public API.

1. Keep `measure_transport_and_spectra(cache, p; reuse_buffers=false)` public and behavior-compatible.
2. Extract transport and optical conductivity work into an internal helper returning the scalar and optical parts.
3. Extract current untwisted spectral work into an internal helper returning `dos`, `dos_AN`, `A_k_ω0`, and `A_kpath`.
4. Make `measure_transport_and_spectra` compose those two helpers exactly as before.
5. In `run_simulation`, when TBC is disabled, call `measure_transport_and_spectra`.
6. In `run_simulation`, when TBC is enabled, call the transport helper plus `measure_twisted_spectra`, then build a `SpectrumResult` with TBC spectral arrays.

This avoids duplicate old-spectrum work when TBC is enabled and keeps the non-TBC path unchanged.

## TBC Physics Convention

Use twist sectors:

```math
q_x = 2\pi n_x / L_{\mathrm{tw}},\quad q_y = 2\pi n_y / L_{\mathrm{tw}},
\quad n_x,n_y=0,\ldots,L_{\mathrm{tw}}-1.
```

For a directed bond crossing the original MC cell boundary, apply:

```math
\exp[-i(q_x w_x + q_y w_y)]
```

where `(w_x,w_y)` is the signed boundary winding of the directed bond. Internal bonds have zero winding.

The effective momentum grid is:

```math
L_x^{\mathrm{eff}} = L_x L_{\mathrm{tw}},\quad
L_y^{\mathrm{eff}} = L_y L_{\mathrm{tw}}.
```

For a small-cell FFT index `(m_x,m_y)` and twist index `(n_x,n_y)`, use:

```math
I_x = \operatorname{mod}(m_x L_{\mathrm{tw}} - n_x, L_x L_{\mathrm{tw}}),
\quad
I_y = \operatorname{mod}(m_y L_{\mathrm{tw}} - n_y, L_y L_{\mathrm{tw}}).
```

The corresponding momenta are:

```math
k_x = 2\pi I_x / L_x^{\mathrm{eff}},\quad
k_y = 2\pi I_y / L_y^{\mathrm{eff}}.
```

The implementation must not infer winding from neighbor index ordering. It must use coordinates and directed displacements.

## TBC Hamiltonian Builder

Add an internal builder named `build_tbc_H_BdG!` or equivalent, not `build_twisted_H_BdG!`.

Responsibilities:

- Allocate no persistent state in `ComputeCache`.
- Fill a caller-provided local `2N x 2N` matrix.
- Use onsite terms from `state.disorder_pot[i] - state.μ_eff`.
- Add nearest-neighbor hopping on unique `+x`, `+y` directed bonds with boundary phases.
- Add next-nearest-neighbor hopping on unique `+x+y`, `+x-y` directed bonds with boundary phases.
- Add pairing terms on stored `state.Δ[i,1]` and `state.Δ[i,2]`, including boundary phases on crossing bonds.
- Be compatible with `eigen!(Hermitian(work, :U))`.

Correctness detail: pairing boundary phases need explicit validation. The implementation should be checked against a direct supercell construction test before accepting the convention.

## Antinodal Patch

Keep `dos_AN` as the existing exact-point definition:

```math
A_{\mathrm{AN}}(\omega)=\frac12[A(\pi,0,\omega)+A(0,\pi,\omega)].
```

Add `dos_AN_patch`:

- Use the TBC effective momentum grid.
- Average spectral weights near both antinodes `(π,0)` and `(0,π)`.
- Include all effective grid points whose periodic distance from either antinode is less than or equal to `antinode_patch_half_width`.
- Normalize by the number of included effective grid points.
- If the patch is empty because the width is zero and exact points are unavailable, fall back to exact-point selection when possible; otherwise error with a clear message.

Default width:

```julia
π / max(p.Lx, p.Ly)
```

This keeps the default patch local in momentum space while allowing TBC to improve statistics/resolution around antinodes.

## Normalization

DOS:

```julia
dos ./= p.N * Ltw^2
```

`A_k_ω0`:

```julia
A_k_ω0 ./= p.N
```

Do not divide `A_k_ω0` by `Ltw^2`; each effective momentum point is supplied by one twist/FFT sector.

`A_kpath`:

- Use `abs2(FFT_value) / p.N` when accumulating.
- Do not divide again at the end.

`dos_AN`:

- Preserve exact-point semantics.
- Do not divide by `Ltw^2` if only the exact untwisted sector contributes.

`dos_AN_patch`:

- Accumulate effective-grid spectral weights with the same `abs2(FFT_value) / p.N` convention.
- Divide by the number of included antinodal patch momentum points, not by `Ltw^2`.

## Data Flow

For each measurement configuration:

1. HMC has already produced a current untwisted `cache.E_n` and `cache.U`.
2. If TBC is disabled:
   - call existing public `measure_transport_and_spectra`
   - write the current output format
3. If TBC is enabled:
   - compute transport and optical conductivity from current untwisted cache
   - allocate local TBC work arrays
   - for each twist sector, build and diagonalize the TBC BdG matrix
   - project eigenvectors to effective momentum grid
   - accumulate TBC spectral arrays
   - combine transport plus TBC spectra into `SpectrumResult`
   - write extra TBC metadata and optional `dos_AN_patch`

No TBC function may mutate `cache.H_base`, `cache.E_n`, or `cache.U`.

## Post-Processing

Update reusable post-processing scripts conservatively:

- Read `spectra_Ltw` when present; default to `1`.
- Use `spectra_Lx_eff` and `spectra_Ly_eff` for `A_k0` CSV grids when present; otherwise use `params.Lx` and `params.Ly`.
- Use stored `dos_omega_grid` when present.
- Use stored `kpath_kx` and `kpath_ky` for `A_kpath`.
- If bin groups contain `dos_AN_patch`, compute and output its mean/error. If absent, skip it without warning.

Do not change one-off notebooks unless needed later.

## Testing Plan

Add `test/test_twisted_spectra.jl` with deterministic, small-system tests:

1. **Hermiticity:** Build TBC Hamiltonians for several `(nx,ny)` sectors and check `Matrix(Hermitian(H,:U))` is Hermitian.
2. **Cache invariance:** Save copies of `cache.H_base`, `cache.E_n`, and `cache.U`; run `measure_twisted_spectra`; confirm exact equality afterward.
3. **`Ltw=1` regression:** Compare TBC result to existing untwisted spectra for `dos`, `dos_AN`, `A_k_ω0`, and `A_kpath` within tight tolerances.
4. **Effective dimensions:** For `Lx=Ly=4`, `Ltw=2`, assert `A_k_ω0` has size `(8,8)` and `A_kpath` has `5` momentum rows.
5. **Supercell equivalence:** For a small lattice, explicitly construct a repeated supercell BdG Hamiltonian and compare its full eigenvalue multiset to the union of all TBC twist-sector eigenvalues. This is the main phase-convention test.
6. **Normal-state dispersion:** With `Δ=0` and no disorder, verify strong spectral intensity appears near the tight-binding dispersion on the effective grid.
7. **`dos_AN_patch` behavior:** Check that patch point selection is nonempty, normalized, finite, and reduces to exact-point behavior in a controlled `Ltw=1` case.
8. **Simulation output:** Extend `test/test_simulation.jl` with a tiny TBC-enabled run. Confirm old runs do not write `dos_AN_patch`, while TBC runs write metadata and expected enlarged spectral array shapes.

Add an opt-in benchmark file:

```text
test/benchmark_twisted_spectra.jl
```

It should run small fixed configurations for `Ltw=1,2,4`, print timing and allocation summaries, and avoid being part of the default smoke path.

## Risks And Checks

Primary risks:

- Wrong sign convention between boundary phase and FFT projection.
- Missing boundary phases on pairing bonds.
- Accidentally using the existing stiffness twist builder for spectral TBC.
- Accidentally mutating HMC cache during measurement.
- Misnormalizing `A_k_ω0` or patch averages.
- Breaking existing post-processing scripts by assuming new keys always exist.

Required checks before accepting implementation:

- `julia --project test/test_twisted_spectra.jl`
- targeted `julia --project test/test_simulation.jl` with reduced environment settings if needed
- `julia --project -e 'using Pkg; Pkg.test()'` if runtime is acceptable
- manual opt-in benchmark for at least one representative small lattice

## Out Of Scope For First Implementation

- Parallelizing twist sectors.
- Adding persistent TBC work arrays to `ComputeCache`.
- Rewriting notebooks.
- Changing current operator, stiffness, conductivity, or HMC dynamics.
- Making TBC the default measurement mode.
