using Test

@testset "DwaveHMC default test suite" begin
    include("test_core_smoke.jl")
    include("test_twist_stiffness.jl")
    include("test_twisted_spectra.jl")
    include("test_simulation_tbc.jl")
    include("test_postprocess_spectra.jl")

    if get(ENV, "DWAVEHMC_RUN_SIMULATION_TESTS", "0") == "1"
        include("test_simulation.jl")
    end
end
