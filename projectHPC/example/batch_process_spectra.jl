using JLD2
using Glob
using DwaveHMC

include(normpath(joinpath(@__DIR__, "..", "..", "scripts", "spectra_postprocess_utils.jl")))

target_dir = get(ENV, "DWAVEHMC_ANALYSIS_DIR", @__DIR__)

const SPECTRA_OUTPUT_FILES = [
    "spectra_opt_cond.csv",
    "spectra_dos.csv",
    "spectra_dos_M.csv",
    "spectra_dos_M_patch.csv",
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

function process_single_config(jld_path)
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
            sum_opt = copy(g1["opt_cond"])
            sum_dos = copy(g1["dos"])
            sum_dos_M = copy(g1["dos_M"])
            sum_ak = copy(g1["A_k0"])
            sum_mx_path = copy(g1["A_MX_path"])
            sum_xg_path = copy(g1["A_XG_path"])
            has_patch = haskey(g1, "dos_M_patch")
            sum_dos_M_patch = has_patch ? copy(g1["dos_M_patch"]) : nothing
            count = 1

            for i in 2:length(sweep_keys)
                g = file[sweep_keys[i]]
                if haskey(g, "dos_M_patch") != has_patch
                    return nothing
                end
                sum_opt .+= g["opt_cond"]
                sum_dos .+= g["dos"]
                sum_dos_M .+= g["dos_M"]
                sum_ak .+= g["A_k0"]
                sum_mx_path .+= g["A_MX_path"]
                sum_xg_path .+= g["A_XG_path"]
                if has_patch
                    sum_dos_M_patch .+= g["dos_M_patch"]
                end
                count += 1
            end

            res = (opt=sum_opt ./ count,
                   dos=sum_dos ./ count,
                   dos_M=sum_dos_M ./ count,
                   dos_M_patch=has_patch ? (sum_dos_M_patch ./ count) : nothing,
                   ak0=sum_ak ./ count,
                   mx_path=sum_mx_path ./ count,
                   xg_path=sum_xg_path ./ count,
                   params=file["params"],
                   meta=meta)

            if any(isnan, res.opt) || any(isnan, res.dos) || any(isnan, res.dos_M) ||
               any(isnan, res.ak0) || any(isnan, res.mx_path) || any(isnan, res.xg_path) ||
               (res.dos_M_patch !== nothing && any(isnan, res.dos_M_patch))
                return nothing
            end

            return res
        end
    catch
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
        mx_path_size=size(res.mx_path),
        xg_path_size=size(res.xg_path),
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

function process_T_directory(dir_path)
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
    samples_mx_path = []
    samples_xg_path = []
    samples_AN = []
    samples_node = []
    peak_rows = []
    reference_meta = nothing
    reference_signature = nothing

    for c_dir in conf_dirs
        jld_path = joinpath(c_dir, "spectra_bins.jld2")
        res = process_single_config(jld_path)
        if res === nothing
            continue
        end

        sig = compatibility_signature(res)
        if reference_signature === nothing
            reference_signature = sig
            reference_meta = res.meta
        else
            mismatches = compatibility_mismatches(reference_signature, sig)
            if !isempty(mismatches)
                @warn "Skipping incompatible spectra config." config=c_dir mismatches=join(mismatches, ", ")
                continue
            end
        end

        mx_kx = fill(res.meta["mx_path_kx"], length(res.meta["mx_path_ky"]))
        AN_spectrum, _, peak_AN = path_observable(res.mx_path, res.meta["dos_omega_grid"],
                                                  mx_kx, res.meta["mx_path_ky"])
        node_spectrum, _, peak_node = path_observable(res.xg_path, res.meta["dos_omega_grid"],
                                                      res.meta["xg_path_kx"], res.meta["xg_path_ky"])

        push!(samples_opt, res.opt)
        push!(samples_dos, res.dos)
        push!(samples_dos_M, res.dos_M)
        if res.dos_M_patch !== nothing
            push!(samples_dos_M_patch, res.dos_M_patch)
        end
        push!(samples_ak, res.ak0)
        push!(samples_mx_path, res.mx_path)
        push!(samples_xg_path, res.xg_path)
        push!(samples_AN, AN_spectrum)
        push!(samples_node, node_spectrum)
        push!(peak_rows, (source=basename(c_dir), kind="AN", peak_AN...))
        push!(peak_rows, (source=basename(c_dir), kind="node", peak_node...))
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
    final_AN, err_AN = calc_stats(samples_AN)
    final_node, err_node = calc_stats(samples_node)

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

    mx_kx = fill(meta["mx_path_kx"], length(meta["mx_path_ky"]))
    mx_kx_idx = fill(meta["mx_path_kx_idx"], length(meta["mx_path_ky"]))
    write_path_csv(joinpath(dir_path, "spectra_MX_path.csv"), final_mx_path,
                   err_mx_path, dos_omega_grid, mx_kx, meta["mx_path_ky"],
                   mx_kx_idx, meta["mx_path_ky_idx"])
    write_path_csv(joinpath(dir_path, "spectra_XG_path.csv"), final_xg_path,
                   err_xg_path, dos_omega_grid, meta["xg_path_kx"],
                   meta["xg_path_ky"], meta["xg_path_kx_idx"], meta["xg_path_ky_idx"])
    write_series_csv(joinpath(dir_path, "spectra_dos_AN.csv"),
                     "omega,DOS_AN,Error", dos_omega_grid, final_AN, err_AN)
    write_series_csv(joinpath(dir_path, "spectra_dos_node.csv"),
                     "omega,DOS_node,Error", dos_omega_grid, final_node, err_node)
    write_peak_summary(joinpath(dir_path, "spectra_path_peaks.csv"), peak_rows)
end

function main()
    println("Starting Robust T-scan Spectra Processing...")
    T_dirs = glob("T_*", target_dir)
    sort!(T_dirs, by = d -> try parse(Float64, split(basename(d), "_")[2]) catch; 0.0 end)

    for t_dir in T_dirs
        process_T_directory(t_dir)
    end

    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
