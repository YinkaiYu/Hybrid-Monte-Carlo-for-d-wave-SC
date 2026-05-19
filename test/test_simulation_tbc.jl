using Test
using Random
using JLD2
using DwaveHMC

function tiny_simulation_parameters(Lx::Int=4, Ly::Int=4)
    return ModelParameters(Lx, Ly, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                           η=0.25, Δω=0.25, ω_max=2.0)
end

function run_tiny_spectra_simulation(p::ModelParameters, out_dir::String; kwargs...)
    Random.seed!(20260510)
    run_simulation(p, out_dir;
                   n_therm=0,
                   n_measure=1,
                   Nt_measure=1,
                   measure_transport_freq=1,
                   bin_size=1,
                   verbose=false,
                   kwargs...)
    return joinpath(out_dir, "spectra_bins.jld2")
end

@testset "spectra eta factor validation" begin
    @test DwaveHMC.validate_spectra_eta_factors([1, 2, 4]) == [1.0, 2.0, 4.0]
    @test DwaveHMC.validate_spectra_eta_factors((1, 2, 4, 8)) == [1.0, 2.0, 4.0, 8.0]
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors(Float64[])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([0, 1, 2])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([1, -2, 4])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([1, NaN, 4])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([1, Inf, 4])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([2, 4, 8])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([2, 1, 4])
    @test_throws ErrorException DwaveHMC.validate_spectra_eta_factors([1, 2, 2])

    factors = [1.0, 2.0, 4.0, 8.0]
    @test DwaveHMC.eta_factor_index(factors, 1) == 1
    @test DwaveHMC.eta_factor_index(factors, 4) == 3
    @test_throws ErrorException DwaveHMC.eta_factor_index(factors, 16)
end

@testset "Simulation spectra TBC integration" begin
    @testset "default simulation writes untwisted spectra metadata and bins" begin
        mktempdir() do out_dir
            spectra_path = run_tiny_spectra_simulation(tiny_simulation_parameters(), out_dir;
                                                       use_twisted_spectra=false)

            @test isfile(spectra_path)
            jldopen(spectra_path, "r") do file
                @test file["use_twisted_spectra"] == false
                @test file["spectra_Ltw"] == 1
                @test file["spectra_Lx_eff"] == 4
                @test file["spectra_Ly_eff"] == 4

                @test haskey(file, "omega_grid")
                @test haskey(file, "dos_omega_grid")
                    @test haskey(file, "mx_path_kx_idx")
                    @test haskey(file, "mx_path_ky_idx")
                    @test haskey(file, "mx_path_kx")
                    @test haskey(file, "mx_path_ky")
                    @test haskey(file, "xg_path_kx")
                    @test haskey(file, "xg_path_ky")

                @test haskey(file, "sweep_1")
                g = file["sweep_1"]
                for key in ("opt_cond", "dos", "dos_M", "LDOS_0", "A_k0", "A_MX_path", "A_XG_path", "count")
                    @test haskey(g, key)
                end
                @test length(g["LDOS_0"]) == 16
                @test size(g["A_k0"]) == (4, 4)
                @test !haskey(g, "dos_M_patch")
                @test !haskey(g, "dos_AN")
                @test !haskey(g, "dos_AN_patch")
            end
        end
    end

    @testset "enabled simulation writes TBC spectra metadata and patch bins" begin
        mktempdir() do out_dir
            spectra_path = run_tiny_spectra_simulation(tiny_simulation_parameters(), out_dir;
                                                       use_twisted_spectra=true,
                                                       spectra_Ltw=2)

            @test isfile(spectra_path)
            jldopen(spectra_path, "r") do file
                @test file["use_twisted_spectra"] == true
                @test file["spectra_Ltw"] == 2
                @test file["spectra_Lx_eff"] == 8
                @test file["spectra_Ly_eff"] == 8

                @test file["spectra_eta"] == 0.25
                @test file["spectra_delta_omega"] == 0.25
                @test length(file["dos_omega_grid"]) == length(collect(-2.0:0.25:2.0))

                for key in ("m_point_patch_half_width",
                            "dos_omega_grid",
                            "mx_path_kx",
                            "mx_path_ky",
                            "xg_path_kx",
                            "xg_path_ky",
                            "kx_grid",
                            "ky_grid")
                    @test haskey(file, key)
                end

                g = file["sweep_1"]
                @test size(g["A_k0"]) == (8, 8)
                @test size(g["A_MX_path"], 1) == 5
                @test size(g["A_XG_path"], 1) == 5
                @test size(g["A_XG_node_patch"], 1) == 5
                @test size(g["A_XG_node_patch"]) == size(g["A_XG_path"])
                @test haskey(g, "dos_M")
                @test haskey(g, "LDOS_0")
                @test haskey(g, "dos_M_patch")
                @test haskey(g, "A_XG_node_patch")
                @test !haskey(g, "dos_AN")
                @test !haskey(g, "dos_AN_patch")
                @test all(isfinite, g["dos_M_patch"])
                @test all(isfinite, g["LDOS_0"])
                @test length(g["LDOS_0"]) == 16
                @test length(g["dos_M_patch"]) == length(g["dos"])
                @test length(g["dos"]) == length(file["dos_omega_grid"])
            end
        end
    end

    @testset "TBC spectra reject odd effective dimensions before outputs" begin
        mktempdir() do out_dir
            p = tiny_simulation_parameters(3, 3)

            err = try
                run_simulation(p, out_dir;
                               n_therm=0,
                               n_measure=0,
                               use_twisted_spectra=true,
                               spectra_Ltw=1,
                               verbose=false)
                nothing
            catch e
                e
            end

            @test err isa ErrorException
            @test occursin("TBC spectra require even effective dimensions",
                           sprint(showerror, err))
            @test !isfile(joinpath(out_dir, "spectra_bins.jld2"))
        end
    end
end
