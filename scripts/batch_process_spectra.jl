using JLD2
using DwaveHMC

include(joinpath(@__DIR__, "spectra_postprocess_utils.jl"))

root_dir = "data/T_scan_L24_V0.8_W1.0_imp0.05_mu_-1.08"

function read_required_metadata(file)
    return (
        omega_grid=collect(file["omega_grid"]),
        dos_omega_grid=collect(file["dos_omega_grid"]),
        spectra_Ltw=Int(file["spectra_Ltw"]),
        spectra_Lx_eff=Int(file["spectra_Lx_eff"]),
        spectra_Ly_eff=Int(file["spectra_Ly_eff"]),
        mx_path_kx=Float64(file["mx_path_kx"]),
        mx_path_ky=collect(file["mx_path_ky"]),
        mx_path_kx_idx=Int(file["mx_path_kx_idx"]),
        mx_path_ky_idx=collect(file["mx_path_ky_idx"]),
        xg_path_kx=collect(file["xg_path_kx"]),
        xg_path_ky=collect(file["xg_path_ky"]),
        xg_path_kx_idx=collect(file["xg_path_kx_idx"]),
        xg_path_ky_idx=collect(file["xg_path_ky_idx"]),
    )
end

const DOS_M_KEY_PAIRS = [
    ("dos_M_eta", "dos_M"),
    ("dos_M_eta_landau_gauge_diagnostic", "dos_M_landau_gauge_diagnostic"),
]
const AK0_KEY_PAIRS = [
    ("A_k0_eta", "A_k0"),
    ("A_k_omega0_eta_landau_gauge_diagnostic", "A_k_omega0_landau_gauge_diagnostic"),
]
const MX_PATH_KEY_PAIRS = [
    ("A_MX_path_eta", "A_MX_path"),
    ("A_MX_path_eta_landau_gauge_diagnostic", "A_MX_path_landau_gauge_diagnostic"),
]
const XG_PATH_KEY_PAIRS = [
    ("A_XG_path_eta", "A_XG_path"),
    ("A_XG_path_eta_landau_gauge_diagnostic", "A_XG_path_landau_gauge_diagnostic"),
]

function collect_sweep_data(file; eta_factor=1)
    eta_idx = selected_eta_index(file, eta_factor)
    list_dc = Float64[]
    list_opt = Vector{Vector{Float64}}()
    list_dos = Vector{Vector{Float64}}()
    list_dos_M = Vector{Vector{Float64}}()
    list_dos_M_patch = Vector{Vector{Float64}}()
    list_ldos0 = Vector{Vector{Float64}}()
    list_ak = Vector{Matrix{Float64}}()
    list_mx_path = Vector{Matrix{Float64}}()
    list_xg_path = Vector{Matrix{Float64}}()

    for key in keys(file)
        if startswith(key, "sweep_")
            g = file[key]
            if haskey(g, "dc_cond_eta") || haskey(g, "dc_cond")
                push!(list_dc, selected_scalar(g, "dc_cond_eta", "dc_cond", eta_idx))
            end
            push!(list_opt, selected_vector(g, "opt_cond_eta", "opt_cond", eta_idx))
            push!(list_dos, selected_vector(g, "dos_eta", "dos", eta_idx))
            dos_M = selected_vector_any(g, DOS_M_KEY_PAIRS, eta_idx)
            if dos_M !== nothing
                push!(list_dos_M, dos_M)
            end
            if haskey(g, "dos_M_patch") || haskey(g, "dos_M_patch_eta")
                push!(list_dos_M_patch, selected_vector(g, "dos_M_patch_eta", "dos_M_patch", eta_idx))
            end
            if haskey(g, "LDOS_0") || haskey(g, "LDOS_0_eta")
                push!(list_ldos0, selected_vector(g, "LDOS_0_eta", "LDOS_0", eta_idx))
            end
            ak = selected_matrix_any(g, AK0_KEY_PAIRS, eta_idx)
            mx_path = selected_matrix_any(g, MX_PATH_KEY_PAIRS, eta_idx)
            xg_path = selected_matrix_any(g, XG_PATH_KEY_PAIRS, eta_idx)
            if ak !== nothing
                push!(list_ak, ak)
            end
            if mx_path !== nothing
                push!(list_mx_path, mx_path)
            end
            if xg_path !== nothing
                push!(list_xg_path, xg_path)
            end
        end
    end

    return (dc=list_dc,
            opt=list_opt,
            dos=list_dos,
            dos_M=list_dos_M,
            dos_M_patch=list_dos_M_patch,
            ldos0=list_ldos0,
            ak=list_ak,
            mx_path=list_mx_path,
            xg_path=list_xg_path,
            count=length(list_dos))
end

function process_single_directory(target_dir; eta_factor=1)
    jld_file = joinpath(target_dir, "spectra_bins.jld2")

    if !isfile(jld_file)
        @warn "File not found: $jld_file. Skipping this directory."
        return
    end

    println("========================================")
    println("Processing: $target_dir")
    println("Opening file: $jld_file")

    jldopen(jld_file, "r") do file
        params = file["params"]
        meta = read_required_metadata(file)
        println("  Params: L=$(params.Lx)x$(params.Ly), Beta=$(params.β), spectra_Ltw=$(meta.spectra_Ltw), effective=$(meta.spectra_Lx_eff)x$(meta.spectra_Ly_eff)")

        data = collect_sweep_data(file; eta_factor=eta_factor)
        if data.count == 0
            @warn "  No 'sweep_' data found in $jld_file."
            return
        end

        println("  Found $(data.count) bins. Calculating statistics...")

        mean_opt, err_opt = calc_stats(data.opt)
        mean_dos, err_dos = calc_stats(data.dos)

        if !isempty(data.dc) && !isapprox(Float64(eta_factor), 1.0; atol=DwaveHMC.ETA_FACTOR_ATOL, rtol=0.0)
            mean_dc, err_dc = calc_scalar_stats(data.dc)
            write_selected_dc_csv(joinpath(target_dir, "processed_dc_cond.csv"),
                                  eta_factor, mean_dc, err_dc)
        else
            rm(joinpath(target_dir, "processed_dc_cond.csv"); force=true)
        end

        write_series_csv(joinpath(target_dir, "processed_opt_cond.csv"),
                         "omega,Re_Sigma,Error", meta.omega_grid, mean_opt, err_opt)
        write_series_csv(joinpath(target_dir, "processed_dos.csv"),
                         "omega,DOS,Error", meta.dos_omega_grid, mean_dos, err_dos)
        if length(data.dos_M) == data.count
            mean_dos_M, err_dos_M = calc_stats(data.dos_M)
            write_series_csv(joinpath(target_dir, "processed_dos_M.csv"),
                             "omega,DOS_M,Error", meta.dos_omega_grid, mean_dos_M, err_dos_M)
        else
            rm(joinpath(target_dir, "processed_dos_M.csv"); force=true)
        end
        if !isempty(data.dos_M_patch)
            mean_dos_M_patch, err_dos_M_patch = calc_stats(data.dos_M_patch)
            write_series_csv(joinpath(target_dir, "processed_dos_M_patch.csv"),
                             "omega,DOS_M_patch,Error",
                             meta.dos_omega_grid, mean_dos_M_patch, err_dos_M_patch)
        else
            rm(joinpath(target_dir, "processed_dos_M_patch.csv"); force=true)
        end

        if length(data.ldos0) == data.count
            mean_ldos0, err_ldos0 = calc_stats(data.ldos0)
            write_ldos_csv(joinpath(target_dir, "processed_ldos0.csv"),
                           mean_ldos0, err_ldos0, params.Lx, params.Ly)
        else
            rm(joinpath(target_dir, "processed_ldos0.csv"); force=true)
        end

        if length(data.ak) == data.count
            mean_ak, err_ak = calc_stats(data.ak)
            write_ak_csv(joinpath(target_dir, "processed_ak0.csv"), mean_ak, err_ak)
        else
            rm(joinpath(target_dir, "processed_ak0.csv"); force=true)
        end

        mx_kx = fill(meta.mx_path_kx, length(meta.mx_path_ky))
        mx_kx_idx = fill(meta.mx_path_kx_idx, length(meta.mx_path_ky))
        if length(data.mx_path) == data.count && length(data.xg_path) == data.count
            mean_mx_path, err_mx_path = calc_stats(data.mx_path)
            mean_xg_path, err_xg_path = calc_stats(data.xg_path)
            write_path_csv(joinpath(target_dir, "processed_MX_path.csv"), mean_mx_path,
                           err_mx_path, meta.dos_omega_grid, mx_kx, meta.mx_path_ky,
                           mx_kx_idx, meta.mx_path_ky_idx)
            write_path_csv(joinpath(target_dir, "processed_XG_path.csv"), mean_xg_path,
                           err_xg_path, meta.dos_omega_grid, meta.xg_path_kx,
                           meta.xg_path_ky, meta.xg_path_kx_idx, meta.xg_path_ky_idx)

            dos_AN, err_AN, peak_AN = path_observable(mean_mx_path, meta.dos_omega_grid,
                                                      mx_kx, meta.mx_path_ky;
                                                      err_path=err_mx_path)
            dos_node, err_node, peak_node = path_observable(mean_xg_path, meta.dos_omega_grid,
                                                            meta.xg_path_kx, meta.xg_path_ky;
                                                            err_path=err_xg_path)
            write_series_csv(joinpath(target_dir, "processed_dos_AN.csv"),
                             "omega,DOS_AN,Error", meta.dos_omega_grid, dos_AN, err_AN)
            write_series_csv(joinpath(target_dir, "processed_dos_node.csv"),
                             "omega,DOS_node,Error", meta.dos_omega_grid, dos_node, err_node)
            write_peak_summary(joinpath(target_dir, "processed_path_peaks.csv"),
                               [(source="bins", kind="AN", peak_AN...),
                                (source="bins", kind="node", peak_node...)])
        else
            for name in ("processed_MX_path.csv", "processed_XG_path.csv",
                         "processed_dos_AN.csv", "processed_dos_node.csv",
                         "processed_path_peaks.csv")
                rm(joinpath(target_dir, name); force=true)
            end
        end

        println("  Done processing $target_dir")
    end
end

function process_batch_spectra_root(root_dir::AbstractString=root_dir; eta_factor=1)
    if !isdir(root_dir)
        error("Root directory does not exist: $root_dir")
    end

    println("Scanning directory: $root_dir")

    subdirs = filter(entry -> startswith(entry, "T_") && isdir(joinpath(root_dir, entry)),
                     readdir(root_dir))
    sort!(subdirs, by = x -> try
        parse(Float64, replace(x, "T_" => ""))
    catch
        x
    end)

    if isempty(subdirs)
        println("No directories starting with 'T_' found in $root_dir")
    else
        println("Found $(length(subdirs)) directories to process.")

        for subdir in subdirs
            full_path = joinpath(root_dir, subdir)
            try
                process_single_directory(full_path; eta_factor=eta_factor)
            catch e
                @error "Error processing $subdir: $e"
            end
        end
    end
end

function main()
    eta_factor = parse(Float64, get(ENV, "DWAVEHMC_SPECTRA_ETA_FACTOR", "1"))
    process_batch_spectra_root(root_dir; eta_factor=eta_factor)
    println("\nAll tasks completed.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
