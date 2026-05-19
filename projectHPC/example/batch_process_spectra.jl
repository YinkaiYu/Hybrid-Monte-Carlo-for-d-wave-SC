using JLD2
using Glob
using DwaveHMC

include(joinpath(@__DIR__, "spectra_postprocess_utils.jl"))

target_dir = get(ENV, "DWAVEHMC_ANALYSIS_DIR", @__DIR__)

const SPECTRA_OUTPUT_FILES = [
    "spectra_dc_cond.csv",
    "spectra_opt_cond.csv",
    "spectra_dos.csv",
    "spectra_dos_M.csv",
    "spectra_dos_M_patch.csv",
    "spectra_ldos0.csv",
    "spectra_ak0.csv",
    "spectra_MX_path.csv",
    "spectra_XG_path.csv",
    "spectra_dos_AN.csv",
    "spectra_dos_node.csv",
    "spectra_path_peaks.csv",
]

function remove_spectra_outputs!(dir_path)
    for filename in SPECTRA_OUTPUT_FILES
        rm(joinpath(dir_path, filename); force=true)
    end
end

function read_required_metadata(file)
    return Dict{String, Any}(
        "omega_grid" => collect(file["omega_grid"]),
        "dos_omega_grid" => collect(file["dos_omega_grid"]),
        "spectra_Ltw" => Int(file["spectra_Ltw"]),
        "spectra_Lx_eff" => Int(file["spectra_Lx_eff"]),
        "spectra_Ly_eff" => Int(file["spectra_Ly_eff"]),
        "mx_path_kx" => Float64(file["mx_path_kx"]),
        "mx_path_ky" => collect(file["mx_path_ky"]),
        "mx_path_kx_idx" => Int(file["mx_path_kx_idx"]),
        "mx_path_ky_idx" => collect(file["mx_path_ky_idx"]),
        "xg_path_kx" => collect(file["xg_path_kx"]),
        "xg_path_ky" => collect(file["xg_path_ky"]),
        "xg_path_kx_idx" => collect(file["xg_path_kx_idx"]),
        "xg_path_ky_idx" => collect(file["xg_path_ky_idx"]),
    )
end

function is_eta_selection_error(e)
    msg = sprint(showerror, e)
    return occursin("eta_factor", msg) ||
           occursin("multi-eta", msg) ||
           occursin("selected eta factor", msg) ||
           occursin("eta-first array", msg) ||
           occursin("eta dimension", msg) ||
           occursin("compatible with", msg)
end

function selected_eta_value(file, eta_idx::Int, eta_factor)
    if haskey(file, "eta_values")
        return Float64(file["eta_values"][eta_idx])
    end

    isapprox(Float64(eta_factor), 1.0; atol=DwaveHMC.ETA_FACTOR_ATOL, rtol=0.0) ||
        error("Missing eta_values metadata for eta_factor=$eta_factor")

    if haskey(file, "spectra_eta")
        return Float64(file["spectra_eta"])
    end
    return Float64(file["params"].η)
end

function selected_transport_eta_value(file, eta_idx::Int, eta_factor)
    if haskey(file, "transport_eta_values")
        return Float64(file["transport_eta_values"][eta_idx])
    elseif haskey(file, "eta_values")
        return Float64(file["eta_values"][eta_idx])
    end

    isapprox(Float64(eta_factor), 1.0; atol=DwaveHMC.ETA_FACTOR_ATOL, rtol=0.0) ||
        error("Missing transport eta metadata for eta_factor=$eta_factor")

    return Float64(file["params"].η)
end

function process_single_config(jld_path; eta_factor=1)
    if !isfile(jld_path) || filesize(jld_path) == 0
        return nothing
    end

    try
        jldopen(jld_path, "r") do file
            sweep_keys = filter(k -> startswith(k, "sweep_"), keys(file))
            isempty(sweep_keys) && return nothing
            haskey(file, "params") || return nothing
            sort!(sweep_keys)

            meta = read_required_metadata(file)
            g1 = file[sweep_keys[1]]
            eta_idx = selected_eta_index(file, eta_factor)
            selected_eta = selected_eta_value(file, eta_idx, eta_factor)
            selected_transport_eta = selected_transport_eta_value(file, eta_idx, eta_factor)
            has_dc = haskey(g1, "dc_cond_eta") || haskey(g1, "dc_cond")
            sum_dc = has_dc ? selected_scalar(g1, "dc_cond_eta", "dc_cond", eta_idx) : 0.0
            sum_opt = copy(selected_vector(g1, "opt_cond_eta", "opt_cond", eta_idx))
            sum_dos = copy(selected_vector(g1, "dos_eta", "dos", eta_idx))
            sum_dos_M = copy(selected_vector(g1, "dos_M_eta", "dos_M", eta_idx))
            sum_ak = copy(selected_matrix(g1, "A_k0_eta", "A_k0", eta_idx))
            sum_mx_path = copy(selected_matrix(g1, "A_MX_path_eta", "A_MX_path", eta_idx))
            sum_xg_path = copy(selected_matrix(g1, "A_XG_path_eta", "A_XG_path", eta_idx))
            has_ldos0 = haskey(g1, "LDOS_0") || haskey(g1, "LDOS_0_eta")
            sum_ldos0 = has_ldos0 ? copy(selected_vector(g1, "LDOS_0_eta", "LDOS_0", eta_idx)) : nothing
            has_node_patch = haskey(g1, "A_XG_node_patch") || haskey(g1, "A_XG_node_patch_eta")
            node_source_key = has_node_patch ? "A_XG_node_patch" : "A_XG_path"
            node_multi_key = has_node_patch ? "A_XG_node_patch_eta" : "A_XG_path_eta"
            sum_node_path = copy(selected_matrix(g1, node_multi_key, node_source_key, eta_idx))
            has_patch = haskey(g1, "dos_M_patch") || haskey(g1, "dos_M_patch_eta")
            sum_dos_M_patch = has_patch ? copy(selected_vector(g1, "dos_M_patch_eta", "dos_M_patch", eta_idx)) : nothing
            count = 1

            for i in 2:length(sweep_keys)
                g = file[sweep_keys[i]]
                if (haskey(g, "dos_M_patch") || haskey(g, "dos_M_patch_eta")) != has_patch
                    return nothing
                end
                if (haskey(g, "A_XG_node_patch") || haskey(g, "A_XG_node_patch_eta")) != has_node_patch
                    return nothing
                end
                if (haskey(g, "LDOS_0") || haskey(g, "LDOS_0_eta")) != has_ldos0
                    return nothing
                end
                if (haskey(g, "dc_cond_eta") || haskey(g, "dc_cond")) != has_dc
                    return nothing
                end
                if has_dc
                    sum_dc += selected_scalar(g, "dc_cond_eta", "dc_cond", eta_idx)
                end
                sum_opt .+= selected_vector(g, "opt_cond_eta", "opt_cond", eta_idx)
                sum_dos .+= selected_vector(g, "dos_eta", "dos", eta_idx)
                sum_dos_M .+= selected_vector(g, "dos_M_eta", "dos_M", eta_idx)
                sum_ak .+= selected_matrix(g, "A_k0_eta", "A_k0", eta_idx)
                sum_mx_path .+= selected_matrix(g, "A_MX_path_eta", "A_MX_path", eta_idx)
                sum_xg_path .+= selected_matrix(g, "A_XG_path_eta", "A_XG_path", eta_idx)
                if has_ldos0
                    sum_ldos0 .+= selected_vector(g, "LDOS_0_eta", "LDOS_0", eta_idx)
                end
                sum_node_path .+= selected_matrix(g, node_multi_key, node_source_key, eta_idx)
                if has_patch
                    sum_dos_M_patch .+= selected_vector(g, "dos_M_patch_eta", "dos_M_patch", eta_idx)
                end
                count += 1
            end

            res = (opt=sum_opt ./ count,
                   dc=has_dc ? (sum_dc / count) : nothing,
                   dos=sum_dos ./ count,
                   dos_M=sum_dos_M ./ count,
                   dos_M_patch=has_patch ? (sum_dos_M_patch ./ count) : nothing,
                   ak0=sum_ak ./ count,
                   ldos0=has_ldos0 ? (sum_ldos0 ./ count) : nothing,
                   mx_path=sum_mx_path ./ count,
                   xg_path=sum_xg_path ./ count,
                   node_path=sum_node_path ./ count,
                   node_from_patch=has_node_patch,
                   selected_eta=selected_eta,
                   selected_transport_eta=has_dc ? selected_transport_eta : nothing,
                   params=file["params"],
                   meta=meta)

            if any(isnan, res.opt) || any(isnan, res.dos) || any(isnan, res.dos_M) ||
               any(isnan, res.ak0) || (res.ldos0 !== nothing && any(isnan, res.ldos0)) ||
               any(isnan, res.mx_path) || any(isnan, res.xg_path) ||
               any(isnan, res.node_path) ||
               (res.dc !== nothing && isnan(res.dc)) ||
               (res.dos_M_patch !== nothing && any(isnan, res.dos_M_patch))
                return nothing
            end

            return res
        end
    catch e
        is_eta_selection_error(e) && rethrow()
        return nothing
    end
end

function same_metadata_value(a, b)
    if a === nothing || b === nothing
        return a === nothing && b === nothing
    elseif a isa AbstractArray || b isa AbstractArray
        return a isa AbstractArray && b isa AbstractArray && size(a) == size(b) && a == b
    else
        return isequal(a, b)
    end
end

function compatibility_signature(res)
    meta = res.meta
    return (
        omega_grid=meta["omega_grid"],
        dos_omega_grid=meta["dos_omega_grid"],
        spectra_Lx_eff=meta["spectra_Lx_eff"],
        spectra_Ly_eff=meta["spectra_Ly_eff"],
        opt_size=size(res.opt),
        dos_size=size(res.dos),
        dos_M_size=size(res.dos_M),
        dos_M_patch_size=res.dos_M_patch === nothing ? nothing : size(res.dos_M_patch),
        ak0_size=size(res.ak0),
        ldos0_size=res.ldos0 === nothing ? nothing : size(res.ldos0),
        mx_path_size=size(res.mx_path),
        xg_path_size=size(res.xg_path),
        node_path_size=size(res.node_path),
        node_from_patch=res.node_from_patch,
        selected_eta=res.selected_eta,
        selected_transport_eta=res.selected_transport_eta,
        mx_path_kx=meta["mx_path_kx"],
        mx_path_ky=meta["mx_path_ky"],
        mx_path_kx_idx=meta["mx_path_kx_idx"],
        mx_path_ky_idx=meta["mx_path_ky_idx"],
        xg_path_kx=meta["xg_path_kx"],
        xg_path_ky=meta["xg_path_ky"],
        xg_path_kx_idx=meta["xg_path_kx_idx"],
        xg_path_ky_idx=meta["xg_path_ky_idx"],
    )
end

function compatibility_mismatches(reference, candidate)
    mismatches = String[]
    for field in fieldnames(typeof(reference))
        if !same_metadata_value(getfield(reference, field), getfield(candidate, field))
            push!(mismatches, String(field))
        end
    end
    return mismatches
end

function process_T_directory(dir_path; eta_factor=1)
    conf_dirs = glob("conf_*", dir_path)
    if isempty(conf_dirs)
        remove_spectra_outputs!(dir_path)
        return
    end
    sort!(conf_dirs)

    println("Processing $(basename(dir_path))...")

    samples_opt = []
    samples_dos = []
    samples_dos_M = []
    samples_dos_M_patch = []
    samples_ak = []
    samples_ldos0 = []
    samples_mx_path = []
    samples_xg_path = []
    samples_node_path = []
    samples_dc = Float64[]
    reference_meta = nothing
    reference_params = nothing
    reference_signature = nothing

    for c_dir in conf_dirs
        jld_path = joinpath(c_dir, "spectra_bins.jld2")
        res = try
            process_single_config(jld_path; eta_factor=eta_factor)
        catch e
            if is_eta_selection_error(e)
                @warn "Skipping spectra config with incompatible eta selection." config=c_dir error=sprint(showerror, e)
                continue
            end
            rethrow()
        end
        if res === nothing
            continue
        end

        sig = compatibility_signature(res)
        if reference_signature === nothing
            reference_signature = sig
            reference_meta = res.meta
            reference_params = res.params
        else
            mismatches = compatibility_mismatches(reference_signature, sig)
            if !isempty(mismatches)
                @warn "Skipping incompatible spectra config." config=c_dir mismatches=join(mismatches, ", ")
                continue
            end
        end

        push!(samples_opt, res.opt)
        push!(samples_dos, res.dos)
        push!(samples_dos_M, res.dos_M)
        if res.dos_M_patch !== nothing
            push!(samples_dos_M_patch, res.dos_M_patch)
        end
        push!(samples_ak, res.ak0)
        if res.ldos0 !== nothing
            push!(samples_ldos0, res.ldos0)
        end
        push!(samples_mx_path, res.mx_path)
        push!(samples_xg_path, res.xg_path)
        push!(samples_node_path, res.node_path)
        if res.dc !== nothing
            push!(samples_dc, res.dc)
        end
    end

    real_n = length(samples_opt)
    if real_n == 0
        remove_spectra_outputs!(dir_path)
        println("  -> Skipped: No valid JLD2 data.")
        return
    end
    println("  -> Valid Samples: $real_n / $(length(conf_dirs))")

    final_opt, err_opt = calc_stats(samples_opt)
    final_dos, err_dos = calc_stats(samples_dos)
    final_dos_M, err_dos_M = calc_stats(samples_dos_M)
    final_ak, err_ak = calc_stats(samples_ak)
    final_mx_path, err_mx_path = calc_stats(samples_mx_path)
    final_xg_path, err_xg_path = calc_stats(samples_xg_path)
    final_node_path, err_node_path = calc_stats(samples_node_path)
    selected_dc_requested = !isapprox(Float64(eta_factor), 1.0; atol=DwaveHMC.ETA_FACTOR_ATOL, rtol=0.0)
    if selected_dc_requested && length(samples_dc) == real_n
        final_dc, err_dc = calc_scalar_stats(samples_dc)
        write_selected_dc_csv(joinpath(dir_path, "spectra_dc_cond.csv"),
                              eta_factor, final_dc, err_dc)
    else
        rm(joinpath(dir_path, "spectra_dc_cond.csv"); force=true)
    end

    meta = reference_meta
    omega_grid = meta["omega_grid"]
    dos_omega_grid = meta["dos_omega_grid"]

    write_series_csv(joinpath(dir_path, "spectra_opt_cond.csv"),
                     "omega,Re_Sigma,Error", omega_grid, final_opt, err_opt)

    open(joinpath(dir_path, "spectra_dos.csv"), "w") do io
        println(io, "omega,DOS,DOS_Error,DOS_M,DOS_M_Error")
        for i in eachindex(final_dos)
            @printf(io, "%.6f,%.6e,%.6e,%.6e,%.6e\n",
                    dos_omega_grid[i], final_dos[i], err_dos[i],
                    final_dos_M[i], err_dos_M[i])
        end
    end
    write_series_csv(joinpath(dir_path, "spectra_dos_M.csv"),
                     "omega,DOS_M,Error", dos_omega_grid,
                     final_dos_M, err_dos_M)

    if length(samples_dos_M_patch) == real_n
        final_dos_M_patch, err_dos_M_patch = calc_stats(samples_dos_M_patch)
        write_series_csv(joinpath(dir_path, "spectra_dos_M_patch.csv"),
                         "omega,DOS_M_patch,Error", dos_omega_grid,
                         final_dos_M_patch, err_dos_M_patch)
    else
        rm(joinpath(dir_path, "spectra_dos_M_patch.csv"); force=true)
    end

    write_ak_csv(joinpath(dir_path, "spectra_ak0.csv"), final_ak, err_ak)

    if length(samples_ldos0) == real_n
        final_ldos0, err_ldos0 = calc_stats(samples_ldos0)
        write_ldos_csv(joinpath(dir_path, "spectra_ldos0.csv"),
                       final_ldos0, err_ldos0,
                       reference_params.Lx, reference_params.Ly)
    else
        rm(joinpath(dir_path, "spectra_ldos0.csv"); force=true)
    end

    mx_kx = fill(meta["mx_path_kx"], length(meta["mx_path_ky"]))
    mx_kx_idx = fill(meta["mx_path_kx_idx"], length(meta["mx_path_ky"]))
    write_path_csv(joinpath(dir_path, "spectra_MX_path.csv"), final_mx_path,
                   err_mx_path, dos_omega_grid, mx_kx, meta["mx_path_ky"],
                   mx_kx_idx, meta["mx_path_ky_idx"])
    write_path_csv(joinpath(dir_path, "spectra_XG_path.csv"), final_xg_path,
                   err_xg_path, dos_omega_grid, meta["xg_path_kx"],
                   meta["xg_path_ky"], meta["xg_path_kx_idx"], meta["xg_path_ky_idx"])
    final_AN, err_AN, peak_AN = path_observable(final_mx_path, dos_omega_grid,
                                                mx_kx, meta["mx_path_ky"];
                                                err_path=err_mx_path,
                                                radius=DEFAULT_AN_PATH_WINDOW_RADIUS)
    final_node, err_node, peak_node = path_observable(final_node_path, dos_omega_grid,
                                                      meta["xg_path_kx"], meta["xg_path_ky"];
                                                      err_path=err_node_path,
                                                      radius=0)
    write_series_csv(joinpath(dir_path, "spectra_dos_AN.csv"),
                     "omega,DOS_AN,Error", dos_omega_grid, final_AN, err_AN)
    write_series_csv(joinpath(dir_path, "spectra_dos_node.csv"),
                     "omega,DOS_node,Error", dos_omega_grid, final_node, err_node)
    write_peak_summary(joinpath(dir_path, "spectra_path_peaks.csv"),
                       [(source="ensemble", kind="AN", peak_AN...),
                        (source="ensemble", kind="node", peak_node...)])
end

function main()
    println("Starting Robust T-scan Spectra Processing...")
    eta_factor = parse(Float64, get(ENV, "DWAVEHMC_SPECTRA_ETA_FACTOR", "1"))
    T_dirs = glob("T_*", target_dir)
    sort!(T_dirs, by = d -> try parse(Float64, split(basename(d), "_")[2]) catch; 0.0 end)

    for t_dir in T_dirs
        process_T_directory(t_dir; eta_factor=eta_factor)
    end

    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
