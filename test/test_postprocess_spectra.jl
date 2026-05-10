using Test
using JLD2
using DwaveHMC

const PROCESS_TARGET_REL = joinpath("data", "test_spectra_L24_J0.8_W1.0_imp0.0_T0.001_mu-1.4")
const BATCH_ROOT_REL = joinpath("data", "T_scan_L24_J0.8_W1.0_imp0.05_mu_-1.08")

module ProcessSpectraScript
include(normpath(joinpath(@__DIR__, "..", "scripts", "process_spectra.jl")))
end

module BatchProcessSpectraScript
include(normpath(joinpath(@__DIR__, "..", "scripts", "batch_process_spectra.jl")))
end

module HPCProcessSpectraScript
include(normpath(joinpath(@__DIR__, "..", "projectHPC", "example", "batch_process_spectra.jl")))
end

function tiny_params()
    return ModelParameters(2, 2, 1.0, -0.35, -0.5, 0.0, 0.0, 8.0, 1.0, 1.0;
                           η=0.5, Δω=1.0, ω_max=1.0)
end

function write_synthetic_spectra(dir; effective=(4, 4), metadata=true,
                                 patch_mode=:all, nsweeps=2, offset=0.0,
                                 dos_grid=[-2.0, 0.0, 2.0])
    mkpath(dir)
    p = tiny_params()
    jldopen(joinpath(dir, "spectra_bins.jld2"), "w") do file
        file["params"] = p
        file["omega_grid"] = [0.5, 1.0]
        if metadata
            file["dos_omega_grid"] = dos_grid
            file["spectra_Ltw"] = 2
            file["spectra_Lx_eff"] = effective[1]
            file["spectra_Ly_eff"] = effective[2]
        end

        for sweep in 1:nsweeps
            prefix = "sweep_$sweep"
            file["$prefix/opt_cond"] = [1.0, 2.0] .+ offset .+ sweep
            file["$prefix/dos"] = [3.0, 4.0, 5.0] .+ offset .+ sweep
            file["$prefix/dos_AN"] = [6.0, 7.0, 8.0] .+ offset .+ sweep
            file["$prefix/A_k0"] = reshape(collect(1.0:prod(effective)), effective) .+ offset .+ sweep

            has_patch = patch_mode == :all || (patch_mode == :mixed && sweep == 1)
            if has_patch
                file["$prefix/dos_AN_patch"] = [10.0, 20.0, 30.0] .+ offset .+ sweep
            end
        end
    end
end

csv_data_rows(path) = length(readlines(path)) - 1

function first_data_value(path, column)
    fields = split(readlines(path)[2], ",")
    return parse(Float64, fields[column])
end

function run_process_spectra_in_temp(; kwargs...)
    mktempdir() do root
        target_dir = joinpath(root, PROCESS_TARGET_REL)
        write_synthetic_spectra(target_dir; kwargs...)
        Base.invokelatest(ProcessSpectraScript.process_spectra_directory, target_dir)
        @test csv_data_rows(joinpath(target_dir, "processed_ak0.csv")) == 16
        @test isfile(joinpath(target_dir, "processed_dos_AN_patch.csv"))
        @test first_data_value(joinpath(target_dir, "processed_dos.csv"), 1) == -2.0
        @test first_data_value(joinpath(target_dir, "processed_dos_AN_patch.csv"), 2) == 11.5
    end
end

@testset "process_spectra.jl TBC metadata" begin
    run_process_spectra_in_temp()
end

@testset "process_spectra.jl legacy and mixed patch compatibility" begin
    mktempdir() do root
        target_dir = joinpath(root, PROCESS_TARGET_REL)
        write_synthetic_spectra(target_dir; effective=(2, 2), metadata=false, patch_mode=:none)
        write(joinpath(target_dir, "processed_dos_AN_patch.csv"), "stale\n")
        Base.invokelatest(ProcessSpectraScript.process_spectra_directory, target_dir)
        @test csv_data_rows(joinpath(target_dir, "processed_ak0.csv")) == 4
        @test !isfile(joinpath(target_dir, "processed_dos_AN_patch.csv"))
    end

    mktempdir() do root
        target_dir = joinpath(root, PROCESS_TARGET_REL)
        write_synthetic_spectra(target_dir; patch_mode=:mixed)
        write(joinpath(target_dir, "processed_dos_AN_patch.csv"), "stale\n")
        Base.invokelatest(ProcessSpectraScript.process_spectra_directory, target_dir)
        @test csv_data_rows(joinpath(target_dir, "processed_ak0.csv")) == 16
        @test !isfile(joinpath(target_dir, "processed_dos_AN_patch.csv"))
    end
end

@testset "batch_process_spectra.jl TBC metadata" begin
    mktempdir() do root
        root_dir = joinpath(root, BATCH_ROOT_REL)
        target_dir = joinpath(root_dir, "T_0.10")
        write_synthetic_spectra(target_dir)
        Base.invokelatest(BatchProcessSpectraScript.process_single_directory, target_dir)
        @test csv_data_rows(joinpath(target_dir, "processed_ak0.csv")) == 16
        @test isfile(joinpath(target_dir, "processed_dos_AN_patch.csv"))
        @test first_data_value(joinpath(target_dir, "processed_dos.csv"), 1) == -2.0
    end
end

@testset "batch_process_spectra.jl mixed patch compatibility" begin
    mktempdir() do root
        root_dir = joinpath(root, BATCH_ROOT_REL)
        target_dir = joinpath(root_dir, "T_0.10")
        write_synthetic_spectra(target_dir; patch_mode=:mixed)
        write(joinpath(target_dir, "processed_dos_AN_patch.csv"), "stale\n")
        Base.invokelatest(BatchProcessSpectraScript.process_single_directory, target_dir)
        @test csv_data_rows(joinpath(target_dir, "processed_ak0.csv")) == 16
        @test !isfile(joinpath(target_dir, "processed_dos_AN_patch.csv"))
    end
end

@testset "projectHPC batch processor TBC metadata" begin
    mktempdir() do root
        t_dir = joinpath(root, "T_0.10")
        write_synthetic_spectra(joinpath(t_dir, "conf_001"); offset=0.0)
        write_synthetic_spectra(joinpath(t_dir, "conf_002"); offset=10.0)
        Base.invokelatest(HPCProcessSpectraScript.process_T_directory, t_dir)
        @test csv_data_rows(joinpath(t_dir, "spectra_ak0.csv")) == 16
        @test isfile(joinpath(t_dir, "spectra_dos_AN_patch.csv"))
        @test first_data_value(joinpath(t_dir, "spectra_dos.csv"), 1) == -2.0
    end
end

@testset "projectHPC batch processor skips mismatched effective shapes" begin
    mktempdir() do root
        t_dir = joinpath(root, "T_0.10")
        write_synthetic_spectra(joinpath(t_dir, "conf_001"); effective=(4, 4), offset=0.0)
        write_synthetic_spectra(joinpath(t_dir, "conf_002"); effective=(2, 2), offset=10.0)
        write(joinpath(t_dir, "spectra_ak0.csv"), "stale\n")

        did_throw = false
        try
            Base.invokelatest(HPCProcessSpectraScript.process_T_directory, t_dir)
        catch
            did_throw = true
        end

        @test !did_throw
        @test csv_data_rows(joinpath(t_dir, "spectra_ak0.csv")) == 16
        @test first_data_value(joinpath(t_dir, "spectra_dos.csv"), 2) == 4.5
    end
end

@testset "projectHPC batch processor skips same-length mismatched DOS grids" begin
    mktempdir() do root
        t_dir = joinpath(root, "T_0.10")
        write_synthetic_spectra(joinpath(t_dir, "conf_001"); dos_grid=[-2.0, 0.0, 2.0], offset=0.0)
        write_synthetic_spectra(joinpath(t_dir, "conf_002"); dos_grid=[-3.0, 0.0, 3.0], offset=10.0)
        Base.invokelatest(HPCProcessSpectraScript.process_T_directory, t_dir)

        @test first_data_value(joinpath(t_dir, "spectra_dos.csv"), 1) == -2.0
        @test first_data_value(joinpath(t_dir, "spectra_dos.csv"), 2) == 4.5
    end
end

@testset "projectHPC batch processor skips mixed config patch" begin
    mktempdir() do root
        t_dir = joinpath(root, "T_0.10")
        write_synthetic_spectra(joinpath(t_dir, "conf_001"); patch_mode=:all)
        write_synthetic_spectra(joinpath(t_dir, "conf_002"); patch_mode=:none)
        write(joinpath(t_dir, "spectra_dos_AN_patch.csv"), "stale\n")
        Base.invokelatest(HPCProcessSpectraScript.process_T_directory, t_dir)
        @test csv_data_rows(joinpath(t_dir, "spectra_ak0.csv")) == 16
        @test !isfile(joinpath(t_dir, "spectra_dos_AN_patch.csv"))
    end
end
