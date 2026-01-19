# Repository Guidelines

## Project Structure & Modules
- Core library lives in `src/` with `DwaveHMC.jl` exporting the public API and submodules `Types`, `Hamiltonian`, `HMC`, `Observables`, and `Simulation`.
- Tests and micro-benchmarks sit in `test/` (`test_hmc.jl`, `test_simulation.jl`, `benchmark_*`). Treat files prefixed with `benchmark_` as opt-in perf checks.
- Batch runners and post-processing tools are in `scripts/` (e.g., `batch_scan_beta.jl`, `batch_process_spectra.jl`); most write results to `data/`.
- Reference outputs and spectra/transport datasets are under `data/`; avoid committing new large artifacts.
- Background theory and algorithm notes live in `doc/`.

## Build, Test, and Development Commands
- Install deps once per machine: `julia --project -e 'using Pkg; Pkg.instantiate()'`.
- Run the full suite: `julia --project -e 'using Pkg; Pkg.test()'` (may take time due to diagonalizations).
- Targeted runs while iterating: `JULIA_NUM_THREADS=4 julia --project test/test_hmc.jl` or another single test file.
- Example batch sweep: `julia --project -t auto scripts/batch_scan_beta.jl` (writes to `data/beta_test_*`).
- For REPL-driven work, launch with `julia --project` and optionally `using Revise, DwaveHMC` to hot-reload changes.

## Coding Style & Naming Conventions
- Julia style with 4-space indentation and no tabs; favor explicit typing for structs and major arrays.
- Use `snake_case` for variables/fields, PascalCase for types, and append `!` to mutating functions (as in `compute_forces!`).
- Prefer in-place operations and preallocation for performance (see caches in `ComputeCache`); avoid allocations inside tight loops.
- Keep comments concise; physics explanations can remain bilingual but code should stay ASCII-friendly.

## Testing Guidelines
- Add regression tests in `test/` near related functionality; mirror naming like `test_hamiltonian.jl`.
- Randomized simulations should set seeds when feasible for reproducibility (`Random.seed!`).
- When adding new scripts, include a tiny sanity check (e.g., small lattice run) and point outputs to a dedicated subfolder in `data/`.

## Commit & Pull Request Guidelines
- Follow the existing short, action-first messages (often in Chinese), e.g., “完善结构体，去掉了dt” / “添加超流刚度等可观测量”.
- In PRs, describe physics parameters, lattice sizes, and any new output paths; attach small plots or CSV snippets if behavior changes.
- Note runtime impacts (threads, β grid sizes, FFT plan changes) and whether benchmarks in `test/benchmark_*` were exercised.
