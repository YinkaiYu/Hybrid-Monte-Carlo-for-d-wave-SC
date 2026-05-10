using Test

@testset "DwaveHMC default test suite" begin
    include("test_core_smoke.jl")
    include("test_twist_stiffness.jl")

    if get(ENV, "DWAVEHMC_RUN_SIMULATION_TESTS", "0") == "1"
        include("test_simulation.jl")
    end
end
