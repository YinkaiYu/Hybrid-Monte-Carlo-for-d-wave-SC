using Test
using DelimitedFiles
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

function read_repo_file(parts...)
    return read(joinpath(REPO_ROOT, parts...), String)
end

function julia_cmd(args...; dir=pwd(), env=Pair{String,String}[])
    cmd = Cmd(`$(Base.julia_cmd()) $args`; dir=dir)
    return setenv(cmd, env)
end

function write_small_csv(path, content)
    mkpath(dirname(path))
    write(path, content)
end

function summary_value(path, column)
    data, header = readdlm(path, ',', header=true)
    names = string.(vec(header))
    idx = findfirst(==(column), names)
    idx === nothing && error("missing column $column")
    return Float64(data[1, idx])
end

function notebook_code_source(parts...)
    path = joinpath(REPO_ROOT, parts...)
    script = """
import ast
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    nb = json.load(f)

sources = []
for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    ast.parse(source, filename=f"{path}:cell{i}")
    sources.append(source)

sys.stdout.write("\\n".join(sources))
"""
    return read(`python -c $script $path`, String)
end

@testset "HPC CSV summary derives longitudinal resistivity from averaged conductivity tensor" begin
    mktempdir() do tmp
        t_dir = joinpath(tmp, "T_0.10")
        write_small_csv(joinpath(t_dir, "params.jl"), """
using DwaveHMC
T = 0.10
β = 10.0
Lx = 2
Ly = 2
""")
        for (conf, sx, sy, raw_rho, stiffness) in (("conf_001", 2.0, 1.0, 999.0, 0.5),
                                                   ("conf_002", 4.0, 3.0, 888.0, NaN))
            cdir = joinpath(t_dir, conf)
            write_small_csv(joinpath(cdir, "observables.csv"), """
Sweep,Energy,D2,D4,Avg_d2,Avg_d4
1,1.0,2.0,3.0,2.0,3.0
""")
            write_small_csv(joinpath(cdir, "transport.csv"), """
Sweep,Superfluid_Stiffness,DC_Conductivity,Hall_Conductivity,Longitudinal_Resistivity,Bad_Column
1,$stiffness,$sx,$sy,$raw_rho,NaN
""")
        end

        cmd = julia_cmd("--project=$(REPO_ROOT)",
                        joinpath(REPO_ROOT, "projectHPC", "example", "batch_process_csv.jl");
                        dir=tmp,
                        env=["DWAVEHMC_ANALYSIS_DIR" => tmp])
        @test success(cmd)

        summary = joinpath(tmp, "summary_all.csv")
        x = [2.0, 4.0]
        y = [1.0, 3.0]
        xbar = mean(x)
        ybar = mean(y)
        D = xbar^2 + ybar^2
        expected_rho = xbar / D
        gx = (ybar^2 - xbar^2) / D^2
        gy = -2.0 * xbar * ybar / D^2
        Cxx = var(x) / length(x)
        Cxy = cov(x, y) / length(x)
        Cyy = var(y) / length(y)
        expected_err = sqrt(max(gx^2 * Cxx + 2.0 * gx * gy * Cxy + gy^2 * Cyy, 0.0))

        @test summary_value(summary, "DC_Conductivity_mean") ≈ xbar
        @test summary_value(summary, "Hall_Conductivity_mean") ≈ ybar
        @test summary_value(summary, "Longitudinal_Resistivity_mean") ≈ expected_rho
        @test summary_value(summary, "Longitudinal_Resistivity_err") ≈ expected_err
        @test summary_value(summary, "Longitudinal_Resistivity_mean") != mean([999.0, 888.0])
        stiffness_err = summary_value(summary, "Superfluid_Stiffness_err")
        @test isfinite(stiffness_err)
        @test stiffness_err == 0.0

        summary_header = readline(summary)
        @test !occursin("Longitudinal_Resistivity_n_finite_conf", summary_header)
        @test !occursin("Bad_Column_mean", summary_header)
    end
end

@testset "HPC CSV summary ignores raw longitudinal resistivity without Hall tensor" begin
    mktempdir() do tmp
        t_dir = joinpath(tmp, "T_0.20")
        write_small_csv(joinpath(t_dir, "params.jl"), """
using DwaveHMC
T = 0.20
β = 5.0
Lx = 2
Ly = 2
""")
        for (conf, sx, raw_rho) in (("conf_001", 2.0, 999.0),
                                    ("conf_002", 4.0, 888.0))
            cdir = joinpath(t_dir, conf)
            write_small_csv(joinpath(cdir, "observables.csv"), """
Sweep,Energy,D2,D4,Avg_d2,Avg_d4
1,1.0,2.0,3.0,2.0,3.0
""")
            write_small_csv(joinpath(cdir, "transport.csv"), """
Sweep,Superfluid_Stiffness,DC_Conductivity,Longitudinal_Resistivity
1,0.5,$sx,$raw_rho
""")
        end

        cmd = julia_cmd("--project=$(REPO_ROOT)",
                        joinpath(REPO_ROOT, "projectHPC", "example", "batch_process_csv.jl");
                        dir=tmp,
                        env=["DWAVEHMC_ANALYSIS_DIR" => tmp])
        @test success(cmd)

        summary_header = readline(joinpath(tmp, "summary_all.csv"))
        @test !occursin("Longitudinal_Resistivity_mean", summary_header)
        @test !occursin("Longitudinal_Resistivity_err", summary_header)
    end
end

@testset "notebooks prefer tensor resistivity inputs" begin
    stiffness_nb = notebook_code_source("projectHPC", "example", "plot_stiffness.ipynb")
    conductivity_nb = notebook_code_source("projectHPC", "example", "plot_conductivity.ipynb")
    @test occursin("Longitudinal_Resistivity_mean", stiffness_nb)
    @test occursin("Hall_Conductivity_mean", stiffness_nb)
    @test occursin("old-data proxy", stiffness_nb)
    @test occursin("Longitudinal_Resistivity_mean", conductivity_nb)
    @test occursin("spectra_hall_cond.csv", conductivity_nb)
    @test occursin("sigma_xy", conductivity_nb)
end

@testset "HPC sweep_T production parameters" begin
    script = read_repo_file("projectHPC", "example", "sweep_T.sh")

    @test occursin("T_list=(0.001 0.002 0.003 0.005 0.008 0.010 0.015 0.020 0.030 0.040 0.050)", script)
    @test occursin("N_CORES=16", script)
    @test occursin("N_CONFS=16", script)
    @test occursin("USE_TARGET_N=1", script)
    @test occursin("target_n=7.316582713846e-01", script)
    @test occursin("W=2.0", script)
    @test occursin("n_therm=100", script)
    @test occursin("spectra_eta_factors=\"[1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0]\"", script)
    @test occursin("spectra_eta_factors = \$spectra_eta_factors", script)
    @test occursin("n_flux_sc=0", script)
    @test occursin("boundary_condition=:periodic", script)
    @test occursin("write_gauge_pair_bonds_freq=5", script)
    @test occursin("allow_gauge_dependent_spectra=false", script)
    @test occursin("write_ldos_spectrum=false", script)
    @test occursin("n_flux_sc = \$n_flux_sc", script)
    @test occursin("boundary_condition = \$boundary_condition", script)
    @test occursin("write_gauge_pair_bonds_freq = \$write_gauge_pair_bonds_freq", script)
    @test occursin("allow_gauge_dependent_spectra = \$allow_gauge_dependent_spectra", script)
    @test occursin("write_ldos_spectrum = \$write_ldos_spectrum", script)
    @test length(findall("n_flux_sc=n_flux_sc, boundary_condition=boundary_condition", script)) == 2
    @test occursin("JOB_TAG=\"n\${target_n}_Nv\${n_flux_sc}\"", script)
    @test occursin("JOB_TAG=\"m\${mu}_Nv\${n_flux_sc}\"", script)
    @test occursin("#SBATCH -J yyk/d-wave/L\${L}/imp\${n_imp}/W\${W}/T\${T}/Nv\${n_flux_sc}/\${JOB_TAG}", script)
end

@testset "run_conf forwards spectra and magnetic options" begin
    script = read_repo_file("projectHPC", "run_conf.jl")

    @test occursin("actual_spectra_eta_factors", script)
    @test occursin("actual_spectra_eta_factors,", script)
    @test occursin("spectra_eta_factors=spectra_eta_factors", script)
    @test occursin("write_gauge_pair_bonds_freq = isdefined(@__MODULE__, :write_gauge_pair_bonds_freq) ? getfield(@__MODULE__, :write_gauge_pair_bonds_freq) : 0", script)
    @test occursin("allow_gauge_dependent_spectra = isdefined(@__MODULE__, :allow_gauge_dependent_spectra) ? getfield(@__MODULE__, :allow_gauge_dependent_spectra) : false", script)
    @test occursin("write_ldos_spectrum = isdefined(@__MODULE__, :write_ldos_spectrum) ? getfield(@__MODULE__, :write_ldos_spectrum) : false", script)
    @test occursin(raw"Magnetic options: n_flux_sc=$(p.n_flux_sc), boundary_condition=$(p.boundary_condition)", script)
    @test occursin(raw"write_gauge_pair_bonds_freq=$(write_gauge_pair_bonds_freq)", script)
    @test occursin(raw"allow_gauge_dependent_spectra=$(allow_gauge_dependent_spectra)", script)
    @test occursin(raw"write_ldos_spectrum=$(write_ldos_spectrum)", script)
    @test occursin("worker_task(seed, p,", script)
    @test occursin("allow_gauge_dependent_spectra, write_gauge_pair_bonds_freq,\n                                 write_ldos_spectrum)", script)
    @test occursin("allow_gauge_dependent_spectra=allow_gauge_dependent_spectra", script)
    @test occursin("write_gauge_pair_bonds_freq=write_gauge_pair_bonds_freq", script)
    @test occursin("write_ldos_spectrum=write_ldos_spectrum", script)
    @test occursin("measure_twist, twist_Ax, twist_qy,\n                allow_gauge_dependent_spectra, write_gauge_pair_bonds_freq,\n                write_ldos_spectrum)", script)
end

@testset "run_conf executes with default magnetic optional params" begin
    mktempdir() do tmp
        write(joinpath(tmp, "params.jl"), """
using DwaveHMC
Lx, Ly = 2, 2
t, tp = 1.0, -0.3
mu = -0.5
W, n_imp = 0.0, 0.0
T = 1.0
beta = 1.0 / T
V = 1.0
mass = 1.0
spectra_Ltw = 1
use_twisted_spectra = false
spectra_eta = nothing
spectra_delta_omega = nothing
m_point_patch_half_width = pi / max(Lx, Ly)
measure_twist = false
twist_Ax = 0.001
twist_qy = 2pi / Ly
n_therm = 0
n_measure = 0
Nt_therm_init = 2
Nt_measure = 1
measure_transport_freq = 1
bin_size = 1
N_conf = 1
p = ModelParameters(Lx, Ly, t, tp, mu, W, n_imp, beta, V, mass)
""")

        cmd = julia_cmd("--project=$(REPO_ROOT)",
                        joinpath(REPO_ROOT, "projectHPC", "run_conf.jl");
                        dir=tmp,
                        env=["SLURM_NTASKS" => "0",
                             "DWAVEHMC_PROJECT_ROOT" => REPO_ROOT])
        @test success(cmd)
        @test isdir(joinpath(tmp, "conf_1"))
        @test isfile(joinpath(tmp, "conf_1", "spectra_bins.jld2"))
    end
end

@testset "sweep_T generated params are executable" begin
    mktempdir() do tmp
        fakebin = joinpath(tmp, "fakebin")
        mkdir(fakebin)
        fake_sbatch = joinpath(fakebin, "sbatch")
        write(fake_sbatch, "#!/bin/sh\nexit 0\n")
        chmod(fake_sbatch, 0o755)

        script = joinpath(REPO_ROOT, "projectHPC", "example", "sweep_T.sh")
        env = ["PATH" => string(fakebin, ":", get(ENV, "PATH", ""))]
        @test success(setenv(Cmd(`bash $script`; dir=tmp), env))

        generated_params = joinpath(tmp, "T_0.001", "params.jl")
        @test isfile(generated_params)

        check = """
include($(repr(generated_params)))
p.n_flux_sc == n_flux_sc || error("n_flux_sc mismatch")
p.boundary_condition == boundary_condition || error("boundary_condition mismatch")
isdefined(@__MODULE__, :write_gauge_pair_bonds_freq) || error("missing write_gauge_pair_bonds_freq")
isdefined(@__MODULE__, :allow_gauge_dependent_spectra) || error("missing allow_gauge_dependent_spectra")
isdefined(@__MODULE__, :write_ldos_spectrum) || error("missing write_ldos_spectrum")
write_gauge_pair_bonds_freq == 5 || error("unexpected write_gauge_pair_bonds_freq")
allow_gauge_dependent_spectra == false || error("unexpected allow_gauge_dependent_spectra")
write_ldos_spectrum == false || error("unexpected write_ldos_spectrum")
"""
        @test success(julia_cmd("--project=$(REPO_ROOT)", "-e", check; dir=tmp))
    end
end
