using Test

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

function read_repo_file(parts...)
    return read(joinpath(REPO_ROOT, parts...), String)
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
    @test occursin("write_gauge_pair_bonds_freq=0", script)
    @test occursin("allow_gauge_dependent_spectra=false", script)
    @test occursin("n_flux_sc = \$n_flux_sc", script)
    @test occursin("boundary_condition = \$boundary_condition", script)
    @test occursin("write_gauge_pair_bonds_freq = \$write_gauge_pair_bonds_freq", script)
    @test occursin("allow_gauge_dependent_spectra = \$allow_gauge_dependent_spectra", script)
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
    @test occursin(raw"Magnetic options: n_flux_sc=$(p.n_flux_sc), boundary_condition=$(p.boundary_condition)", script)
    @test occursin(raw"write_gauge_pair_bonds_freq=$(write_gauge_pair_bonds_freq)", script)
    @test occursin(raw"allow_gauge_dependent_spectra=$(allow_gauge_dependent_spectra)", script)
    @test occursin("worker_task(seed, p,", script)
    @test occursin("allow_gauge_dependent_spectra, write_gauge_pair_bonds_freq)", script)
    @test occursin("allow_gauge_dependent_spectra=allow_gauge_dependent_spectra", script)
    @test occursin("write_gauge_pair_bonds_freq=write_gauge_pair_bonds_freq", script)
    @test occursin("measure_twist, twist_Ax, twist_qy,\n                allow_gauge_dependent_spectra, write_gauge_pair_bonds_freq)", script)
end
