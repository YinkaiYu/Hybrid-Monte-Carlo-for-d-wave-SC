# Magnetic Field Design

## Goal

Add an opt-in finite orbital magnetic field path for the HMC BdG workflow using
magnetic periodic boundary conditions, Landau gauge, and Peierls phases on
normal hopping. The first production target is reliable finite-field
`Superfluid_Stiffness` and `DC_Conductivity` on the existing torus geometry,
with clear documentation and tests that prevent mixed gauge conventions.

The implementation follows the physics conventions of arXiv:2301.04175 where
the background field enters normal hopping through Peierls phases, the local
d-wave pairing used for vortex analysis is gauge-covariant, and Kubo stiffness
and currents include the same Peierls phases. Magnetic unit cells and magnetic
Bloch theorem are explicitly out of scope for this first version.

## Scope

Implement:

- uniform orbital magnetic field on a torus with magnetic periodic boundary
  conditions;
- integer superconducting flux quantum entry point `n_flux_sc`;
- Landau gauge link phases;
- Peierls phases for nearest-neighbor and next-nearest-neighbor normal hopping;
- Kubo `Superfluid_Stiffness` as the existing transverse `xx` Meissner
  estimator with finite `q_y`;
- Kubo-Greenwood `DC_Conductivity` as longitudinal `sigma_xx` at `q=0`;
- optional gauge-covariant bond pairing output for finite-field analysis;
- finite-field-safe spectra behavior that keeps DOS and LDOS but disables
  ordinary FFT momentum spectra by default.

Do not implement yet:

- magnetic unit cells or magnetic Bloch theorem;
- Hall conductivity, `sigma_xy`, or Onsager checks beyond allowing negative
  `n_flux_sc`;
- `yy` stiffness/conductivity or isotropic direction averages;
- cylinder/open-boundary production mode;
- finite-field twisted-boundary spectra;
- finite-field twist-stiffness finite-difference benchmark.

## Public Interface

`ModelParameters` gains a production magnetic-field entry point:

```julia
n_flux_sc::Int = 0
boundary_condition::Symbol = :periodic
```

`n_flux_sc` is the number of `hc/2e` superconducting flux quanta through the
full `Lx x Ly` simulation cell. It is also the expected vortex number. It may
be positive or negative; the sign is the magnetic-field orientation. Magnetic
periodic boundary conditions require `n_flux_sc` to be even, including negative
even values such as `-2`.

For finite-field torus production runs:

```julia
n_flux_sc != 0
boundary_condition == :magnetic_pbc
iseven(n_flux_sc)
```

The implementation derives:

```julia
flux_density_sc = n_flux_sc / (Lx * Ly)
plaquette_phase = pi * flux_density_sc
```

The name `flux_density_sc` is used instead of `phi` so it is not confused with
the Peierls link phase.

An advanced `flux_density_sc::Float64` route can exist internally or for later
cylinder/open-boundary tests, but torus magnetic PBC must derive the field from
integer `n_flux_sc`. Users should not be able to pass arbitrary continuous
magnetic flux for the magnetic torus production path.

## Magnetic Phase Layer

Add `src/MagneticField.jl` as a centralized phase layer. All physics code must
obtain Peierls factors from this layer; Landau gauge formulas should not be
hand-written inside Hamiltonian, current, diamagnetic, or pairing-observable
functions.

Core API:

```julia
validate_magnetic_field(p)
build_magnetic_cache(p)
link_phase(mag, i, dx, dy)
plaquette_phase(mag, x, y)
magnetic_metadata(mag)
```

This interface is intentionally close to a future oriented-bond-table design:
callers ask for a directed link phase using a site and displacement, rather
than knowing the gauge implementation.

Cache types:

- `NoFieldCache`: used when `n_flux_sc == 0`; `link_phase` returns
  `one(ComplexF64)` and should keep the zero-field path cheap.
- `LandauGaugeCache`: used for finite field; stores `n_flux_sc`,
  `flux_density_sc`, `plaquette_phase`, lattice sizes, and precomputed common
  directed phases.

`ComputeCache` will hold a magnetic cache built once in `initialize_cache(p)`.
Finite-field code should read precomputed phase arrays in hot loops and avoid
calling `cis` during HMC leapfrog or repeated Kubo accumulation.

## Landau Gauge Convention

Coordinates are interpreted as zero-based for phase formulas:

```text
x coordinate: x = 0, ..., Lx - 1
y coordinate: y = 0, ..., Ly - 1
```

Use the Landau gauge convention:

```text
U_y(x,y) = cis(plaquette_phase * x)
U_x(x,y) = 1                         for internal +x links
U_x(Lx-1,y) = cis(-plaquette_phase * Lx * y)  for +x boundary link
```

With this convention, each oriented plaquette has phase product:

```julia
cis(plaquette_phase)
```

The total torus flux is:

```julia
plaquette_phase * Lx * Ly = pi * n_flux_sc
```

Magnetic periodic closure therefore requires even `n_flux_sc`, so the total
phase is an integer multiple of `2pi`. For `n_flux_sc < 0`, all phases are
naturally complex conjugated relative to `abs(n_flux_sc)`.

Next-nearest-neighbor phases must come from the same `link_phase` convention.
For diagonal `t'` bonds, `link_phase` uses the straight-line Landau-gauge
Peierls integral, not an arbitrary product of two nearest-neighbor phases. For
internal diagonal links this gives the convention

```text
U_{dx,dy}(x,y) = cis(plaquette_phase * dy * (x + dx/2))
```

for `dx = +/-1` and `dy = +/-1`, before applying any magnetic boundary patch.
Boundary-crossing diagonal hops must use the same magnetic boundary patch as
nearest-neighbor hops, so `+x`, `+x+y`, and `+x-y` remain mutually consistent
at the torus edge.

For the first version the common directed bonds are:

```text
+x, +y, +x+y, +x-y
```

Negative or reverse-direction phases should be derived by the same API, usually
through conjugation of the reverse directed link, rather than by separate local
formula copies.

## Hamiltonian and HMC

The normal hopping block changes to:

```text
h_ij = -t_ij * U_ij
hole block = +t_ij * conj(U_ij)
```

where `U_ij = link_phase(cache.magnetic, i, dx, dy)`.

Apply Peierls phases to:

- nearest-neighbor normal hopping on `+x` and `+y` unique directed bonds;
- next-nearest-neighbor normal hopping on `+x+y` and `+x-y` unique directed
  bonds.

The pairing block remains the sampled auxiliary field:

```text
H_pair uses state.Delta[i,dir] directly
```

No Peierls factor is multiplied into the BdG pairing block. The gauge-covariant
pairing used for vortex or local pairing analysis is constructed at the
observable/output layer.

`init_static_H!` writes onsite and phased hopping terms. `update_H_BdG!`
continues to update only the pairing block. `compute_forces!` keeps the same
self-consistency target:

```text
F_ij = -beta/g_pair * (Delta_ij - g_pair * P_ij)
```

`P_ij` comes from the finite-field BdG eigenvectors. It should not be
multiplied by the link phase inside the HMC force because the sampled variable
in the Hamiltonian is the bare bond `Delta_ij`.

## Pairing Observables and Optional Bond Output

Existing CSV/JLD2 scalar pairing outputs keep their zero-field-compatible
meaning:

```text
Delta_Loc
Delta_Glob
Delta_Pair
Delta_LocalPair
d_local
```

They remain based on the old bare `Delta_x - Delta_y` or `P_x - P_y`
convention and are not overwritten by gauge-covariant definitions.

Add an optional run-simulation keyword:

```julia
write_gauge_pair_bonds::Bool = false
```

When enabled, `pairing_scatter.jld2` writes for each measured sweep:

```text
delta_bond_gauge :: N x 2 ComplexF64
pair_bond_gauge  :: N x 2 ComplexF64
```

Definitions:

```text
delta_bond_gauge[i,dir] = state.Delta[i,dir] * U_link(i,dir)
pair_bond_gauge[i,dir]  = g_pair * P[i,dir] * U_link(i,dir)
```

Finite-field vortex, local pairing, and gauge-covariant d-wave post-processing
should primarily use `pair_bond_gauge`, because it is the fermionic pair
expectation and therefore the more physical pairing observable. The auxiliary
field `delta_bond_gauge` is useful for HMC field inspection, self-consistency
debugging, and comparing sampled fields with fermionic expectation values.

Site-level gauge-covariant d-wave pairing is intentionally left to
post-processing:

```text
d_i = 1/4 * (Delta^g_{i,+x} + Delta^g_{i,-x}
             - Delta^g_{i,+y} - Delta^g_{i,-y})
```

The same formula can be applied to `pair_bond_gauge`.

## Kubo Stiffness and Conductivity

The first version keeps the existing `xx` channel only.

`Superfluid_Stiffness` remains the transverse Meissner estimator:

```text
rho_s = <-K_x> - Lambda_xx(qx=0, qy=2pi/Ly, omega=0)
```

The current/vector-potential direction is `x`; the probe momentum is transverse
to the current direction. Strict `m == n` terms are skipped as in the existing
finite-`q_y` stiffness path, while near-degenerate `m != n` terms use the
stable derivative limit.

`DC_Conductivity` remains the regular longitudinal Kubo-Greenwood
conductivity:

```text
sigma_xx(q=0, omega -> 0)
```

Both quantities must include Peierls phases consistently:

- `K_x` uses phased hopping on all bonds with nonzero `delta_x`, currently
  `+x`, `+x+y`, and `+x-y`;
- `J_x(q_y)` uses the same phased hopping for transverse stiffness;
- `J_x(q=0)` uses the same phased hopping for optical and DC conductivity.

`current_operator_matrix` or its replacement should obtain bond phases only
from `link_phase`. The diamagnetic loop should use the same bond list and the
same phases so Hamiltonian, current, and kinetic response cannot drift apart.

## Spectra Behavior at Finite Field

For `n_flux_sc != 0`, the following combinations are invalid and should throw
errors if explicitly enabled:

```julia
use_twisted_spectra == true
measure_twist == true
```

The code must fail fast rather than silently disabling these features.

Ordinary FFT momentum spectra are gauge-dependent in a finite orbital magnetic
field because ordinary lattice translation symmetry is broken by the background
vector potential. Without magnetic translation or magnetic Bloch basis, they
must not be treated as physical momentum-resolved spectra.

Add:

```julia
allow_gauge_dependent_spectra::Bool = false
```

No separate `measure_Ak` keyword is required in the first implementation.
Zero-field runs keep the current momentum-spectra behavior. Finite-field runs
skip ordinary FFT momentum spectra by default; setting
`allow_gauge_dependent_spectra=true` is the explicit opt-in for diagnostic
Landau-gauge FFT spectra.

Default finite-field behavior:

- compute and write DOS;
- compute and write LDOS;
- compute and write transport scalars and optical conductivity;
- do not write `A_k0`, `A_MX_path`, or `A_XG_path`;
- post-processing scripts treat momentum-resolved arrays as optional and skip
  the corresponding CSV outputs when absent.

If a future or script-level option explicitly requests ordinary FFT spectra at
`n_flux_sc != 0` without `allow_gauge_dependent_spectra=true`, throw an error
explaining that the result is gauge-dependent.

If `allow_gauge_dependent_spectra=true`, write diagnostic fields with warning
names, for example:

```text
A_k_omega0_landau_gauge_diagnostic
A_MX_path_landau_gauge_diagnostic
A_XG_path_landau_gauge_diagnostic
```

and metadata:

```text
gauge_dependent_spectra = true
spectra_gauge = "Landau gauge"
spectra_interpretation =
    "diagnostic only; not a gauge-invariant momentum-resolved spectral function"
```

## Simulation and HPC Integration

`run_simulation` gains:

```julia
write_gauge_pair_bonds::Bool = false
allow_gauge_dependent_spectra::Bool = false
```

Top-level JLD2 metadata should include:

```text
n_flux_sc
boundary_condition
flux_density_sc
plaquette_phase
magnetic_gauge
magnetic_pbc
gauge_dependent_spectra
spectra_gauge
spectra_interpretation
```

`projectHPC/run_conf.jl` should pass magnetic parameters from `params.jl` and
print them in `job.out` near the existing spectra/twist settings.

Defaults preserve existing zero-field behavior:

- `n_flux_sc = 0`;
- `boundary_condition = :periodic`;
- no magnetic metadata changes that break old post-processing;
- ordinary spectra behavior unchanged at zero field.

Finite-field runs should avoid large additional allocations in hot loops:

- build magnetic phase cache once in `initialize_cache(p)`;
- reuse existing dense/sparse buffers for Kubo work;
- keep gauge-covariant bond output disabled by default because it adds
  `N x 2` complex arrays per measured sweep;
- keep ordinary FFT momentum spectra disabled by default for finite field,
  reducing both runtime and JLD2 size.

## Testing Plan

Add `test/test_magnetic_field.jl` and extend existing targeted tests.

Magnetic phase tests:

- `n_flux_sc = 0` builds `NoFieldCache` and all `link_phase` calls return
  `1 + 0im`;
- positive and negative even `n_flux_sc` are valid;
- odd finite `n_flux_sc` errors for magnetic PBC;
- finite `n_flux_sc` requires `boundary_condition == :magnetic_pbc`;
- each plaquette phase product is `cis(plaquette_phase)`;
- torus closure gives total phase `cis(pi * n_flux_sc) == 1`;
- phases for `+B` and `-B` are complex conjugates.

Hamiltonian and HMC tests:

- zero-field Hamiltonian, eigenvalues, forces, and transport match the old path;
- finite-field Hamiltonian is Hermitian;
- selected boundary hoppings carry the expected Landau-gauge phases;
- for a clean real-Delta configuration, `+B` and `-B` spectra have the expected
  conjugation/symmetry behavior.

Kubo tests:

- current operator entries for `+x`, `+x+y`, and `+x-y` bonds match
  `link_phase`;
- diamagnetic term uses the same phases as the current operator;
- `Superfluid_Stiffness` and `DC_Conductivity` are finite and real for small
  `n_flux_sc = +/-2` smoke tests.

Output/fail-fast tests:

- finite field plus `use_twisted_spectra=true` errors;
- finite field plus `measure_twist=true` errors;
- finite field default output omits ordinary `A_k0`, `A_MX_path`, `A_XG_path`;
- diagnostic opt-in writes Landau-gauge warning field names and metadata;
- `write_gauge_pair_bonds=true` writes both `delta_bond_gauge` and
  `pair_bond_gauge`;
- post-processing handles missing momentum spectra and still writes DOS, LDOS,
  transport, and DC outputs.

## Documentation Plan

Update or add:

- `doc/magnetic-field.md`: central finite-field convention document;
- `doc/theory.md`: magnetic BdG Hamiltonian and magnetic PBC convention;
- `doc/observables.md`: gauge-covariant pairing, transverse stiffness,
  longitudinal conductivity, and finite-field spectra limitations;
- README or script comments only if needed for discoverability.

The documentation must clearly state:

- `n_flux_sc` means superconducting flux quanta through the full simulation
  cell and equals the expected vortex count;
- magnetic PBC on the torus requires even `n_flux_sc`;
- negative `n_flux_sc` reverses the field direction;
- `flux_density_sc` and `plaquette_phase` are derived quantities;
- exact Landau-gauge link convention;
- the BdG pairing block uses bare sampled `Delta_ij`;
- gauge-covariant bond pairing is constructed only for observables/output;
- finite-field ordinary FFT `A(k,omega)` is disabled by default and diagnostic
  only if explicitly enabled.

## Open Extension Points

The phase-layer API is designed so future work can add:

- a full oriented bond table;
- `yy` stiffness and conductivity;
- Hall conductivity and Onsager checks;
- cylinder/open boundary geometry with continuous `flux_density_sc`;
- magnetic unit cell and magnetic Bloch spectra.

Those extensions should build on `link_phase` or replace it with a bond-table
lookup without changing Hamiltonian, Kubo, and pairing-observable call sites.
