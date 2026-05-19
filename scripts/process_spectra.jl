using JLD2
using DwaveHMC

include(joinpath(@__DIR__, "spectra_postprocess_utils.jl"))

target_dir = "data/test_spectra_L24_V0.8_W1.0_imp0.0_T0.001_mu-1.4"

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

function collect_sweep_data(file)
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
            push!(list_opt, g["opt_cond"])
            push!(list_dos, g["dos"])
            push!(list_dos_M, g["dos_M"])
            if haskey(g, "dos_M_patch")
                push!(list_dos_M_patch, g["dos_M_patch"])
            end
            if haskey(g, "LDOS_0")
                push!(list_ldos0, g["LDOS_0"])
            end
            push!(list_ak, g["A_k0"])
            push!(list_mx_path, g["A_MX_path"])
            push!(list_xg_path, g["A_XG_path"])
        end
    end

    return (opt=list_opt,
            dos=list_dos,
            dos_M=list_dos_M,
            dos_M_patch=list_dos_M_patch,
            ldos0=list_ldos0,
            ak=list_ak,
            mx_path=list_mx_path,
            xg_path=list_xg_path,
            count=length(list_dos))
end

function process_spectra_directory(target_dir::AbstractString=target_dir)
    jld_file = joinpath(target_dir, "spectra_bins.jld2")

    println("Opening file: $jld_file")

    jldopen(jld_file, "r") do file
        params = file["params"]
        meta = read_required_metadata(file)

        println("Params: L=$(params.Lx)x$(params.Ly), Beta=$(params.β), spectra_Ltw=$(meta.spectra_Ltw), effective=$(meta.spectra_Lx_eff)x$(meta.spectra_Ly_eff)")

        data = collect_sweep_data(file)
        if data.count == 0
            @warn "No 'sweep_' data found in $jld_file."
            return
        end

        println("Found $(data.count) bins. Calculating statistics...")

        mean_opt, err_opt = calc_stats(data.opt)
        mean_dos, err_dos = calc_stats(data.dos)
        mean_dos_M, err_dos_M = calc_stats(data.dos_M)
        mean_ak, err_ak = calc_stats(data.ak)
        mean_mx_path, err_mx_path = calc_stats(data.mx_path)
        mean_xg_path, err_xg_path = calc_stats(data.xg_path)

        write_series_csv(joinpath(target_dir, "processed_opt_cond.csv"),
                         "omega,Re_Sigma,Error", meta.omega_grid, mean_opt, err_opt)
        write_series_csv(joinpath(target_dir, "processed_dos.csv"),
                         "omega,DOS,Error", meta.dos_omega_grid, mean_dos, err_dos)
        write_series_csv(joinpath(target_dir, "processed_dos_M.csv"),
                         "omega,DOS_M,Error", meta.dos_omega_grid, mean_dos_M, err_dos_M)
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

        write_ak_csv(joinpath(target_dir, "processed_ak0.csv"), mean_ak, err_ak)

        mx_kx = fill(meta.mx_path_kx, length(meta.mx_path_ky))
        mx_kx_idx = fill(meta.mx_path_kx_idx, length(meta.mx_path_ky))
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
    end
end

function main()
    process_spectra_directory(target_dir)
    println("Processing Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
