using Random
using BenchmarkTools
using DwaveHMC

function parse_ltw_list(raw::AbstractString)
    values = Int[]
    for item in split(raw, ',')
        item = strip(item)
        isempty(item) && continue
        push!(values, parse(Int, item))
    end
    isempty(values) && error("DWAVEHMC_TBC_BENCH_LTW must contain at least one integer")
    return values
end

function setup_tbc_benchmark_fixture(L::Int)
    Random.seed!(13579)
    p = ModelParameters(L, L, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                        η=8.0 / (L * L), Δω=4.0 / (L * L), ω_max=3.0)
    state = initialize_state(p)
    cache = initialize_cache(p)

    init_static_H!(cache, p, state)
    update_H_BdG!(cache, p, state)
    diagonalize_H_BdG!(cache, p)

    return p, state, cache
end

function run_twisted_spectra_benchmark()
    L = parse(Int, get(ENV, "DWAVEHMC_TBC_BENCH_L", "6"))
    Ltw_values = parse_ltw_list(get(ENV, "DWAVEHMC_TBC_BENCH_LTW", "1,2,4"))

    p, state, cache = setup_tbc_benchmark_fixture(L)

    println("TBC spectra benchmark")
    println("System size: $(p.Lx)x$(p.Ly), N=$(p.N)")
    println("BdG dim: $(2 * p.N)")
    println("DOS grid length: $(length(cache.dos_omega_grid))")

    for Ltw in Ltw_values
        println()
        println("Ltw=$Ltw")
        println("Effective grid: $(p.Lx * Ltw)x$(p.Ly * Ltw)")
        println("TBC sectors: $(Ltw^2)")

        trial = @benchmark measure_twisted_spectra($cache, $p, $state; Ltw=$Ltw, reuse_buffers=false) samples=3 evals=1
        show(stdout, MIME("text/plain"), trial)
        println()
    end

    return nothing
end

run_twisted_spectra_benchmark()
