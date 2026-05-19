using Test
using JLD2
using DwaveHMC

const PROCESS_TARGET_REL = joinpath("data", "test_spectra_L24_V0.8_W1.0_imp0.0_T0.001_mu-1.4")
const BATCH_ROOT_REL = joinpath("data", "T_scan_L24_V0.8_W1.0_imp0.05_mu_-1.08")

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

function path_fixture(kind::Symbol)
    if kind === :mx
        return [1.0 2.0 3.0;
                4.0 20.0 6.0;
                7.0 50.0 9.0;
                10.0 30.0 12.0;
                13.0 14.0 15.0]
    elseif kind === :xg
        return [1.0 3.0 5.0;
                8.0 80.0 10.0;
                6.0 30.0 9.0;
                4.0 20.0 8.0;
                2.0 1.0 0.0]
    end
    error("unknown path fixture")
end

function asymmetric_mx_path_fixture()
    return [100.0 2.0 100.0;
            4.0 20.0 6.0;
            7.0 50.0 9.0;
            10.0 30.0 12.0;
            500.0 14.0 500.0]
end

function write_synthetic_spectra(dir; effective=(4, 4), nsweeps=2, offset=0.0,
                                 dos_grid=[-2.0, 0.0, 2.0],
                                 mx_path=path_fixture(:mx),
                                 xg_path=path_fixture(:xg),
                                 xg_node_patch=path_fixture(:xg) .+ 100.0,
                                 mx_ky=[0.0, 0.25, 0.5, 0.75, 1.0],
                                 xg_k=[0.0, 0.25, 0.5, 0.75, 1.0],
                                 use_twisted_spectra=true,
                                 spectra_Ltw=2,
                                 multi_eta=true)
    mkpath(dir)
    p = tiny_params()
    jldopen(joinpath(dir, "spectra_bins.jld2"), "w") do file
        file["params"] = p
        file["use_twisted_spectra"] = use_twisted_spectra
        file["omega_grid"] = [0.5, 1.0]
        file["dos_omega_grid"] = dos_grid
        file["spectra_Ltw"] = spectra_Ltw
        file["spectra_Lx_eff"] = effective[1]
        file["spectra_Ly_eff"] = effective[2]
        file["spectra_eta"] = 0.125
        file["spectra_delta_omega"] = 0.25
        file["m_point_patch_half_width"] = 0.5
        file["mx_path_kx"] = pi
        file["mx_path_ky"] = mx_ky
        file["mx_path_kx_idx"] = effective[1] ÷ 2 + 1
        file["mx_path_ky_idx"] = collect(1:length(mx_ky))
        file["xg_path_kx"] = xg_k
        file["xg_path_ky"] = xg_k
        file["xg_path_kx_idx"] = collect(1:length(xg_k))
        file["xg_path_ky_idx"] = collect(1:length(xg_k))
        if multi_eta
            file["multi_eta_enabled"] = true
            file["spectra_eta_factors"] = [1.0, 2.0, 4.0]
            file["eta_values"] = [0.125, 0.25, 0.5]
            file["spectra_eta_base"] = 0.125
        end

        for sweep in 1:nsweeps
            prefix = "sweep_$sweep"
            file["$prefix/opt_cond"] = [1.0, 2.0] .+ offset .+ sweep
            file["$prefix/dos"] = [3.0, 4.0, 5.0] .+ offset .+ sweep
            file["$prefix/dos_M"] = [6.0, 7.0, 8.0] .+ offset .+ sweep
            if use_twisted_spectra
                file["$prefix/dos_M_patch"] = [10.0, 20.0, 30.0] .+ offset .+ sweep
            end
            file["$prefix/LDOS_0"] = collect(1.0:4.0) .+ offset .+ sweep
            file["$prefix/A_k0"] = reshape(collect(1.0:prod(effective)), effective) .+ offset .+ sweep
            file["$prefix/A_MX_path"] = mx_path .+ offset .+ sweep
            file["$prefix/A_XG_path"] = xg_path .+ offset .+ sweep
            if use_twisted_spectra
                file["$prefix/A_XG_node_patch"] = xg_node_patch .+ offset .+ sweep
            end
            if multi_eta
                stack_eta_vector(base) = vcat(reshape(base .* 1.0, 1, :),
                                              reshape(base .* 2.0, 1, :),
                                              reshape(base .* 4.0, 1, :))
                stack_eta_matrix(base) = permutedims(cat(base, base .* 2.0, base .* 4.0; dims=3),
                                                     (3, 1, 2))

                opt_base = [1.0, 2.0] .+ offset .+ sweep
                dos_base = [3.0, 4.0, 5.0] .+ offset .+ sweep
                dos_M_base = [6.0, 7.0, 8.0] .+ offset .+ sweep
                ldos_base = collect(1.0:4.0) .+ offset .+ sweep
                ak_base = reshape(collect(1.0:prod(effective)), effective) .+ offset .+ sweep
                mx_base = mx_path .+ offset .+ sweep
                xg_base = xg_path .+ offset .+ sweep
                node_patch_base = xg_node_patch .+ offset .+ sweep

                file["$prefix/dc_cond_eta"] = [100.0, 200.0, 400.0] .+ offset .+ sweep
                file["$prefix/opt_cond_eta"] = stack_eta_vector(opt_base)
                file["$prefix/dos_eta"] = stack_eta_vector(dos_base)
                file["$prefix/dos_M_eta"] = stack_eta_vector(dos_M_base)
                if use_twisted_spectra
                    file["$prefix/dos_M_patch_eta"] = stack_eta_vector([10.0, 20.0, 30.0] .+ offset .+ sweep)
                end
                file["$prefix/LDOS_0_eta"] = stack_eta_vector(ldos_base)
                file["$prefix/A_k0_eta"] = stack_eta_matrix(ak_base)
                file["$prefix/A_MX_path_eta"] = stack_eta_matrix(mx_base)
                file["$prefix/A_XG_path_eta"] = stack_eta_matrix(xg_base)
                if use_twisted_spectra
                    file["$prefix/A_XG_node_patch_eta"] = stack_eta_matrix(node_patch_base)
                end
            end
        end
    end
end

csv_data_rows(path) = length(readlines(path)) - 1

function first_data_value(path, column)
    fields = split(readlines(path)[2], ",")
    return parse(Float64, fields[column])
end

function header(path)
    return strip(readline(path))
end

@testset "process_spectra.jl M-point metadata and path spectra" begin
    mktempdir() do root
        target_dir = joinpath(root, PROCESS_TARGET_REL)
        write_synthetic_spectra(target_dir; nsweeps=1)
        Base.invokelatest(ProcessSpectraScript.process_spectra_directory, target_dir)

        @test csv_data_rows(joinpath(target_dir, "processed_ak0.csv")) == 16
        @test isfile(joinpath(target_dir, "processed_dos_M.csv"))
        @test isfile(joinpath(target_dir, "processed_dos_M_patch.csv"))
        @test isfile(joinpath(target_dir, "processed_ldos0.csv"))
        @test isfile(joinpath(target_dir, "processed_dos_AN.csv"))
        @test isfile(joinpath(target_dir, "processed_dos_node.csv"))
        @test isfile(joinpath(target_dir, "processed_MX_path.csv"))
        @test isfile(joinpath(target_dir, "processed_XG_path.csv"))
        @test header(joinpath(target_dir, "processed_dos_M.csv")) == "omega,DOS_M,Error"
        @test csv_data_rows(joinpath(target_dir, "processed_ldos0.csv")) == 4
        @test first_data_value(joinpath(target_dir, "processed_dos_AN.csv"), 2) == 8.0
        @test first_data_value(joinpath(target_dir, "processed_dos_node.csv"), 2) == 6.0
    end
end

@testset "batch_process_spectra.jl M-point metadata and path spectra" begin
    mktempdir() do root
        root_dir = joinpath(root, BATCH_ROOT_REL)
        target_dir = joinpath(root_dir, "T_0.10")
        write_synthetic_spectra(target_dir; nsweeps=1)
        Base.invokelatest(BatchProcessSpectraScript.process_single_directory, target_dir)

        @test csv_data_rows(joinpath(target_dir, "processed_ak0.csv")) == 16
        @test isfile(joinpath(target_dir, "processed_dos_M_patch.csv"))
        @test isfile(joinpath(target_dir, "processed_ldos0.csv"))
        @test isfile(joinpath(target_dir, "processed_dos_AN.csv"))
        @test isfile(joinpath(target_dir, "processed_dos_node.csv"))
        @test first_data_value(joinpath(target_dir, "processed_dos.csv"), 1) == -2.0
        @test first_data_value(joinpath(target_dir, "processed_dos_AN.csv"), 2) == 8.0
    end
end

@testset "batch_process_spectra.jl accepts untwisted Ltw1 spectra without patch fields" begin
    mktempdir() do root
        root_dir = joinpath(root, BATCH_ROOT_REL)
        target_dir = joinpath(root_dir, "T_0.10")
        write_synthetic_spectra(target_dir;
                                use_twisted_spectra=false,
                                spectra_Ltw=1,
                                nsweeps=1)
        Base.invokelatest(BatchProcessSpectraScript.process_single_directory, target_dir)

        @test isfile(joinpath(target_dir, "processed_dos.csv"))
        @test isfile(joinpath(target_dir, "processed_dos_node.csv"))
        @test !isfile(joinpath(target_dir, "processed_dos_M_patch.csv"))
        @test first_data_value(joinpath(target_dir, "processed_dos.csv"), 2) == 4.0
        @test first_data_value(joinpath(target_dir, "processed_dos_node.csv"), 2) == 6.0
    end
end

@testset "process_spectra.jl selects requested eta factor" begin
    mktempdir() do root
        target_dir = joinpath(root, PROCESS_TARGET_REL)
        write_synthetic_spectra(target_dir; nsweeps=1)
        Base.invokelatest(ProcessSpectraScript.process_spectra_directory,
                          target_dir;
                          eta_factor=4)

        @test first_data_value(joinpath(target_dir, "processed_dos.csv"), 2) == 16.0
        @test first_data_value(joinpath(target_dir, "processed_dos_AN.csv"), 2) == 32.0
    end
end

@testset "process_spectra.jl rejects old data for non-default eta" begin
    mktempdir() do root
        target_dir = joinpath(root, PROCESS_TARGET_REL)
        write_synthetic_spectra(target_dir; nsweeps=1, multi_eta=false)
        @test_throws ErrorException Base.invokelatest(ProcessSpectraScript.process_spectra_directory,
                                                      target_dir;
                                                      eta_factor=4)
    end
end

@testset "projectHPC batch processor M, antinode, and node outputs" begin
    mktempdir() do root
        t_dir = joinpath(root, "T_0.10")
        write_synthetic_spectra(joinpath(t_dir, "conf_001"); offset=0.0, nsweeps=1)
        write_synthetic_spectra(joinpath(t_dir, "conf_002"); offset=10.0, nsweeps=1)
        Base.invokelatest(HPCProcessSpectraScript.process_T_directory, t_dir)

        @test csv_data_rows(joinpath(t_dir, "spectra_ak0.csv")) == 16
        @test isfile(joinpath(t_dir, "spectra_dos_M_patch.csv"))
        @test isfile(joinpath(t_dir, "spectra_ldos0.csv"))
        @test isfile(joinpath(t_dir, "spectra_dos_AN.csv"))
        @test isfile(joinpath(t_dir, "spectra_dos_node.csv"))
        @test isfile(joinpath(t_dir, "spectra_MX_path.csv"))
        @test isfile(joinpath(t_dir, "spectra_XG_path.csv"))
        @test csv_data_rows(joinpath(t_dir, "spectra_ldos0.csv")) == 4
        @test isfile(joinpath(t_dir, "spectra_path_peaks.csv"))
        @test header(joinpath(t_dir, "spectra_dos.csv")) == "omega,DOS,DOS_Error,DOS_M,DOS_M_Error"
        @test first_data_value(joinpath(t_dir, "spectra_dos.csv"), 1) == -2.0
        @test first_data_value(joinpath(t_dir, "spectra_dos.csv"), 2) == 9.0
        @test first_data_value(joinpath(t_dir, "spectra_dos_AN.csv"), 2) == 13.0
        @test first_data_value(joinpath(t_dir, "spectra_dos_node.csv"), 2) == 114.0
    end
end

@testset "projectHPC batch processor selects requested eta factor" begin
    mktempdir() do root
        t_dir = joinpath(root, "T_0.10")
        write_synthetic_spectra(joinpath(t_dir, "conf_001"); offset=0.0, nsweeps=1)
        write_synthetic_spectra(joinpath(t_dir, "conf_002"); offset=10.0, nsweeps=1)
        Base.invokelatest(HPCProcessSpectraScript.process_T_directory,
                          t_dir;
                          eta_factor=4)

        @test first_data_value(joinpath(t_dir, "spectra_dos.csv"), 2) == 36.0
        @test first_data_value(joinpath(t_dir, "spectra_dos_AN.csv"), 2) == 52.0
    end
end

@testset "projectHPC batch processor uses wider AN window" begin
    mktempdir() do root
        t_dir = joinpath(root, "T_0.10")
        write_synthetic_spectra(joinpath(t_dir, "conf_001");
                                mx_path=asymmetric_mx_path_fixture(),
                                offset=0.0,
                                nsweeps=1)
        write_synthetic_spectra(joinpath(t_dir, "conf_002");
                                mx_path=asymmetric_mx_path_fixture(),
                                offset=10.0,
                                nsweeps=1)
        Base.invokelatest(HPCProcessSpectraScript.process_T_directory, t_dir)

        @test first_data_value(joinpath(t_dir, "spectra_dos_AN.csv"), 2) == 130.2
    end
end

@testset "projectHPC batch processor accepts untwisted Ltw1 spectra without patch fields" begin
    mktempdir() do root
        t_dir = joinpath(root, "T_0.10")
        write_synthetic_spectra(joinpath(t_dir, "conf_001");
                                use_twisted_spectra=false,
                                spectra_Ltw=1,
                                offset=0.0,
                                nsweeps=1)
        write_synthetic_spectra(joinpath(t_dir, "conf_002");
                                use_twisted_spectra=false,
                                spectra_Ltw=1,
                                offset=10.0,
                                nsweeps=1)
        Base.invokelatest(HPCProcessSpectraScript.process_T_directory, t_dir)

        @test isfile(joinpath(t_dir, "spectra_dos.csv"))
        @test isfile(joinpath(t_dir, "spectra_dos_node.csv"))
        @test !isfile(joinpath(t_dir, "spectra_dos_M_patch.csv"))
        @test first_data_value(joinpath(t_dir, "spectra_dos.csv"), 2) == 9.0
        @test first_data_value(joinpath(t_dir, "spectra_dos_node.csv"), 2) == 14.0
    end
end

@testset "projectHPC batch processor skips mismatched path metadata" begin
    mktempdir() do root
        t_dir = joinpath(root, "T_0.10")
        write_synthetic_spectra(joinpath(t_dir, "conf_001"); offset=0.0, nsweeps=1)
        write_synthetic_spectra(joinpath(t_dir, "conf_002");
                                mx_path=ones(3, 3),
                                mx_ky=[0.0, 0.5, 1.0],
                                offset=1000.0,
                                nsweeps=1)
        write_synthetic_spectra(joinpath(t_dir, "conf_003"); offset=4.0, nsweeps=1)

        did_throw = false
        try
            Base.invokelatest(HPCProcessSpectraScript.process_T_directory, t_dir)
        catch
            did_throw = true
        end

        @test !did_throw
        @test isfile(joinpath(t_dir, "spectra_MX_path.csv"))
        @test csv_data_rows(joinpath(t_dir, "spectra_MX_path.csv")) == 5 * 3
        @test first_data_value(joinpath(t_dir, "spectra_dos.csv"), 2) == 6.0
    end
end

@testset "projectHPC batch processor skips same-length mismatched DOS grids" begin
    mktempdir() do root
        t_dir = joinpath(root, "T_0.10")
        write_synthetic_spectra(joinpath(t_dir, "conf_001"); dos_grid=[-2.0, 0.0, 2.0], offset=0.0, nsweeps=1)
        write_synthetic_spectra(joinpath(t_dir, "conf_002"); dos_grid=[-3.0, 0.0, 3.0], offset=10.0, nsweeps=1)
        Base.invokelatest(HPCProcessSpectraScript.process_T_directory, t_dir)

        @test first_data_value(joinpath(t_dir, "spectra_dos.csv"), 1) == -2.0
        @test first_data_value(joinpath(t_dir, "spectra_dos.csv"), 2) == 4.0
    end
end
