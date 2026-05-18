#!/usr/bin/env julia

using JLD2
using Plots
using Printf
using Statistics

const SINGLE_REPORT_NAME = "pairing_phase_amplitude_temperature_report"
const MCAVG_REPORT_NAME = "pairing_mc_average_report"

struct Options
    root_dir::String
    mode::String
    single_report_name::String
    mcavg_report_name::String
    saturation::Float64
    lightness::Float64
    representatives_path::Union{Nothing, String}
end

struct SampleData
    T::Float64
    T_label::String
    conf_id::String
    file_path::String
    first_sweep::Int
    last_sweep::Int
    n_sweeps::Int
    Lx::Int
    Ly::Int
    single_d::Vector{ComplexF64}
    mcavg_d::Vector{ComplexF64}
end

function usage()
    return """
    Usage:
      julia --project scripts/generate_pairing_reports.jl ROOT_DIR [options]

    ROOT_DIR should contain T_*/conf_*/pairing_scatter.jld2 files.

    Options:
      --mode MODE              both, single, or mcavg. Default: both
      --single-report NAME     Output directory name for the single-sweep report.
                                Default: $(SINGLE_REPORT_NAME)
      --mcavg-report NAME      Output directory name for the MC-average report.
                                Default: $(MCAVG_REPORT_NAME)
      --saturation VALUE       HSL phase-map saturation. Default: 0.48
      --lightness VALUE        HSL phase-map lightness. Default: 0.56
      --representatives FILE   Optional CSV with columns T,conf_id to pin the
                                representative configuration for each T.
                                If omitted, ROOT_DIR/pairing_representatives.csv
                                is used when it exists; otherwise representatives
                                are selected by the single-sweep CV median rule.
      -h, --help               Show this help.

    Examples:
      julia --project scripts/generate_pairing_reports.jl data/HPC_disorder0.2_L40_Ltw1_V1.0
      julia --project scripts/generate_pairing_reports.jl data/new_scan --mode mcavg
    """
end

function parse_args(args::Vector{String})
    if isempty(args) || any(a -> a in ("-h", "--help"), args)
        println(usage())
        exit(0)
    end

    root_dir = args[1]
    mode = "both"
    single_report_name = SINGLE_REPORT_NAME
    mcavg_report_name = MCAVG_REPORT_NAME
    saturation = 0.48
    lightness = 0.56
    representatives_path = nothing

    i = 2
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--mode=")
            mode = split(arg, "=", limit=2)[2]
        elseif arg == "--mode"
            i += 1
            i <= length(args) || error("--mode requires a value")
            mode = args[i]
        elseif startswith(arg, "--single-report=")
            single_report_name = split(arg, "=", limit=2)[2]
        elseif arg == "--single-report"
            i += 1
            i <= length(args) || error("--single-report requires a value")
            single_report_name = args[i]
        elseif startswith(arg, "--mcavg-report=")
            mcavg_report_name = split(arg, "=", limit=2)[2]
        elseif arg == "--mcavg-report"
            i += 1
            i <= length(args) || error("--mcavg-report requires a value")
            mcavg_report_name = args[i]
        elseif startswith(arg, "--saturation=")
            saturation = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg == "--saturation"
            i += 1
            i <= length(args) || error("--saturation requires a value")
            saturation = parse(Float64, args[i])
        elseif startswith(arg, "--lightness=")
            lightness = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg == "--lightness"
            i += 1
            i <= length(args) || error("--lightness requires a value")
            lightness = parse(Float64, args[i])
        elseif startswith(arg, "--representatives=")
            representatives_path = split(arg, "=", limit=2)[2]
        elseif arg == "--representatives"
            i += 1
            i <= length(args) || error("--representatives requires a value")
            representatives_path = args[i]
        else
            error("Unknown option: $arg")
        end
        i += 1
    end

    mode in ("both", "single", "mcavg") || error("--mode must be one of: both, single, mcavg")
    0.0 <= saturation <= 1.0 || error("--saturation must be in [0, 1]")
    0.0 <= lightness <= 1.0 || error("--lightness must be in [0, 1]")

    return Options(root_dir, mode, single_report_name, mcavg_report_name,
                   saturation, lightness, representatives_path)
end

function sweep_number(key::AbstractString)
    m = match(r"^sweep_(\d+)$", key)
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

function parse_temperature_label(name::AbstractString)
    startswith(name, "T_") || return nothing
    return parse(Float64, name[3:end])
end

function infer_lattice_shape(n::Int)
    side = round(Int, sqrt(n))
    if side * side == n
        return side, side
    end
    error("Cannot infer a square lattice from vector length $n")
end

function read_pairing_file(file_path::AbstractString)
    jldopen(file_path, "r") do file
        sweep_ids = Int[]
        for key in keys(file)
            id = sweep_number(String(key))
            id === nothing || push!(sweep_ids, id)
        end
        isempty(sweep_ids) && error("No sweep_* groups found in $file_path")
        sort!(sweep_ids)

        first_sweep = first(sweep_ids)
        last_sweep = last(sweep_ids)
        acc = Vector{ComplexF64}()
        single_d = Vector{ComplexF64}()

        for (idx, sweep_id) in enumerate(sweep_ids)
            d = ComplexF64.(collect(file["sweep_$(sweep_id)"]["d_local"]))
            if idx == 1
                acc = zeros(ComplexF64, length(d))
            elseif length(d) != length(acc)
                error("Inconsistent d_local length in $file_path at sweep_$sweep_id")
            end
            acc .+= d
            if sweep_id == last_sweep
                single_d = d
            end
        end

        mcavg_d = acc ./ length(sweep_ids)
        Lx, Ly = infer_lattice_shape(length(single_d))
        return first_sweep, last_sweep, length(sweep_ids), Lx, Ly, single_d, mcavg_d
    end
end

function discover_samples(root_dir::AbstractString)
    isdir(root_dir) || error("Root directory does not exist: $root_dir")
    samples = SampleData[]

    for t_dir in sort(readdir(root_dir; join=true))
        isdir(t_dir) || continue
        t_value = parse_temperature_label(basename(t_dir))
        t_value === nothing && continue
        for conf_dir in sort(readdir(t_dir; join=true))
            isdir(conf_dir) || continue
            conf_id = basename(conf_dir)
            startswith(conf_id, "conf_") || continue
            file_path = joinpath(conf_dir, "pairing_scatter.jld2")
            isfile(file_path) || continue
            first_sweep, last_sweep, n_sweeps, Lx, Ly, single_d, mcavg_d =
                read_pairing_file(file_path)
            push!(samples, SampleData(t_value, @sprintf("%.3f", t_value), conf_id,
                                      file_path, first_sweep, last_sweep, n_sweeps,
                                      Lx, Ly, single_d, mcavg_d))
        end
    end

    isempty(samples) && error("No pairing_scatter.jld2 files found under $root_dir")
    sort!(samples, by=s -> (s.T, s.conf_id))
    return samples
end

safe_div(x, y) = abs(y) < eps(Float64) ? 0.0 : x / y

function gini_nonnegative(values::AbstractVector{<:Real})
    x = sort(Float64.(values))
    n = length(x)
    total = sum(x)
    if n == 0 || total <= 0
        return 0.0
    end
    weighted_sum = sum(i * x[i] for i in eachindex(x))
    return 2.0 * weighted_sum / (n * total) - (n + 1.0) / n
end

site_index(x, y, Lx) = (y - 1) * Lx + x

function nearest_neighbor_pairs(Lx::Int, Ly::Int)
    pairs = Tuple{Int, Int}[]
    sizehint!(pairs, 2 * Lx * Ly)
    for y in 1:Ly, x in 1:Lx
        i = site_index(x, y, Lx)
        right = site_index(x == Lx ? 1 : x + 1, y, Lx)
        up = site_index(x, y == Ly ? 1 : y + 1, Lx)
        push!(pairs, (i, right))
        push!(pairs, (i, up))
    end
    return pairs
end

function metrics_for_field(d::Vector{ComplexF64}, Lx::Int, Ly::Int)
    n = length(d)
    amp = abs.(d)
    mean_abs = mean(amp)
    std_abs = std(amp; corrected=false)
    p10_abs = quantile(amp, 0.10)
    p90_abs = quantile(amp, 0.90)
    sum_abs = sum(amp)
    sum_abs2 = sum(abs2, amp)
    sum_abs4 = sum(a -> a^4, amp)

    units = ComplexF64[]
    sizehint!(units, n)
    for z in d
        a = abs(z)
        push!(units, a > 0 ? z / a : 0.0 + 0.0im)
    end

    pairs = nearest_neighbor_pairs(Lx, Ly)
    nn_amp_delta = [abs(amp[i] - amp[j]) for (i, j) in pairs]
    nn_phase_cos = [real(conj(units[i]) * units[j]) for (i, j) in pairs
                    if abs(units[i]) > 0 && abs(units[j]) > 0]

    return (
        mean_abs=mean_abs,
        std_abs=std_abs,
        cv_abs=safe_div(std_abs, mean_abs),
        p10_abs=p10_abs,
        p90_abs=p90_abs,
        robust_contrast=safe_div(p90_abs - p10_abs, p90_abs + p10_abs),
        gini_abs=gini_nonnegative(amp),
        ipr_abs=sum_abs > 0 ? n * sum_abs2 / sum_abs^2 : 0.0,
        amp2_ipr=sum_abs2 > 0 ? n * sum_abs4 / sum_abs2^2 : 0.0,
        nn_amp_roughness=safe_div(mean(nn_amp_delta), mean_abs),
        global_abs=abs(mean(d)),
        weighted_phase_R=sum_abs > 0 ? abs(sum(d)) / sum_abs : 0.0,
        phase_R=abs(mean(units)),
        nn_phase_R=isempty(nn_phase_cos) ? 0.0 : mean(nn_phase_cos),
    )
end

function sample_metrics(samples::Vector{SampleData}; field::Symbol)
    rows = NamedTuple[]
    for s in samples
        d = field == :single ? s.single_d : s.mcavg_d
        m = metrics_for_field(d, s.Lx, s.Ly)
        push!(rows, merge((
            T=s.T,
            T_label=s.T_label,
            conf_id=s.conf_id,
            first_sweep=s.first_sweep,
            last_sweep=s.last_sweep,
            sweep=s.last_sweep,
            n_sweeps=s.n_sweeps,
            Lx=s.Lx,
            Ly=s.Ly,
        ), m))
    end
    return rows
end

function metric_value(row, key::Symbol)
    return getproperty(row, key)
end

function unique_temperatures(rows)
    temps = sort(unique(row.T for row in rows))
    return temps
end

function sem(values::Vector{Float64})
    length(values) <= 1 && return 0.0
    return std(values) / sqrt(length(values))
end

function summarize_by_temperature(rows, metric_keys::Vector{Symbol})
    summary = NamedTuple[]
    for T in unique_temperatures(rows)
        group = [row for row in rows if row.T == T]
        base = (T=T, T_label=@sprintf("%.3f", T), n=length(group))
        values = NamedTuple()
        for key in metric_keys
            xs = [Float64(metric_value(row, key)) for row in group]
            values = merge(values, NamedTuple{(Symbol("$(key)_mean"), Symbol("$(key)_sem"))}((mean(xs), sem(xs))))
        end
        push!(summary, merge(base, values))
    end
    return summary
end

function choose_representatives(single_rows)
    reps = Dict{Float64, NamedTuple}()
    for T in unique_temperatures(single_rows)
        group = sort([row for row in single_rows if row.T == T], by=row -> row.conf_id)
        cvs = sort([row.cv_abs for row in group])
        target = length(cvs) % 2 == 1 ? cvs[(length(cvs) + 1) ÷ 2] :
                 0.5 * (cvs[length(cvs) ÷ 2] + cvs[length(cvs) ÷ 2 + 1])
        best = group[argmin([abs(row.cv_abs - target) for row in group])]
        reps[T] = best
    end
    return reps
end

function load_representatives(path::AbstractString, single_rows)
    isfile(path) || error("Representative CSV does not exist: $path")
    lines = readlines(path)
    isempty(lines) && error("Representative CSV is empty: $path")

    header = split(strip(lines[1]), ",")
    t_col = findfirst(==("T"), header)
    conf_col = findfirst(==("conf_id"), header)
    t_col === nothing && error("Representative CSV must contain column T")
    conf_col === nothing && error("Representative CSV must contain column conf_id")

    reps = Dict{Float64, NamedTuple}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = split(strip(line), ",")
        length(fields) >= max(t_col, conf_col) ||
            error("Malformed representative row in $path: $line")
        T = parse(Float64, fields[t_col])
        conf_id = fields[conf_col]
        idx = findfirst(row -> isapprox(row.T, T; atol=1e-10) && row.conf_id == conf_id,
                        single_rows)
        idx === nothing &&
            error("Representative T=$T conf_id=$conf_id is not present in the data")
        reps[single_rows[idx].T] = single_rows[idx]
    end

    missing_T = setdiff(unique_temperatures(single_rows), collect(keys(reps)))
    isempty(missing_T) ||
        error("Representative CSV misses temperatures: $(join(fmt_fixed.(missing_T), ", "))")
    return reps
end

function representative_source(opts::Options)
    if opts.representatives_path !== nothing
        return opts.representatives_path
    end
    default_path = joinpath(opts.root_dir, "pairing_representatives.csv")
    return isfile(default_path) ? default_path : nothing
end

function clean_generated_assets(report_dir::AbstractString)
    mkpath(report_dir)
    for name in ("index.html", "sample_metrics.csv", "temperature_summary.csv",
                 "representative_samples.csv", "mc_average_metrics.csv",
                 "mc_average_temperature_summary.csv")
        path = joinpath(report_dir, name)
        isfile(path) && rm(path)
    end

    assets_dir = joinpath(report_dir, "assets")
    isdir(assets_dir) || return
    for path in readdir(assets_dir; join=true)
        if isfile(path) && (endswith(path, ".png") || endswith(path, ".csv"))
            rm(path)
        end
    end
end

function find_sample(samples::Vector{SampleData}, T::Float64, conf_id::AbstractString)
    idx = findfirst(s -> s.T == T && s.conf_id == conf_id, samples)
    idx === nothing && error("Cannot find sample T=$T conf=$conf_id")
    return samples[idx]
end

function csv_cell(x)
    if x isa AbstractFloat
        return @sprintf("%.12e", x)
    end
    return string(x)
end

function write_csv(path::AbstractString, header::Vector{String}, rows::Vector{Vector})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(csv_cell.(row), ","))
        end
    end
end

function write_field_csv(path::AbstractString, d::Vector{ComplexF64}, Lx::Int, Ly::Int)
    rows = Vector{Vector}()
    for y in 1:Ly, x in 1:Lx
        z = d[site_index(x, y, Lx)]
        push!(rows, Any[x, y, real(z), imag(z), abs(z), angle(z)])
    end
    write_csv(path, ["x", "y", "real_d", "imag_d", "abs_d", "phase"], rows)
end

function write_single_metrics(path::AbstractString, rows)
    table = Vector{Vector}()
    for row in rows
        push!(table, Any[row.T, row.conf_id, row.sweep, row.n_sweeps, row.Lx, row.Ly,
                         row.mean_abs, row.std_abs, row.cv_abs, row.p10_abs, row.p90_abs,
                         row.robust_contrast, row.gini_abs, row.ipr_abs, row.amp2_ipr,
                         row.nn_amp_roughness, row.global_abs, row.weighted_phase_R,
                         row.phase_R, row.nn_phase_R])
    end
    write_csv(path,
              ["T", "conf_id", "sweep", "n_sweeps", "Lx", "Ly", "mean_abs",
               "std_abs", "cv_abs", "p10_abs", "p90_abs", "robust_contrast",
               "gini_abs", "ipr_abs", "amp2_ipr", "nn_amp_roughness", "global_abs",
               "weighted_phase_R", "phase_R", "nn_phase_R"],
              table)
end

function write_mcavg_metrics(path::AbstractString, rows)
    table = Vector{Vector}()
    for row in rows
        csv_path = @sprintf("assets/mcavg_T_%.3f_%s.csv", row.T, row.conf_id)
        push!(table, Any[row.T, row.conf_id, row.first_sweep, row.last_sweep, row.n_sweeps,
                         row.Lx, row.Ly, row.mean_abs, row.std_abs, row.cv_abs,
                         row.p10_abs, row.p90_abs, row.robust_contrast, row.gini_abs,
                         row.ipr_abs, row.amp2_ipr, row.nn_amp_roughness, row.global_abs,
                         row.weighted_phase_R, row.phase_R, row.nn_phase_R, csv_path])
    end
    write_csv(path,
              ["T", "conf_id", "first_sweep", "last_sweep", "n_sweeps", "Lx", "Ly",
               "mean_abs", "std_abs", "cv_abs", "p10_abs", "p90_abs",
               "robust_contrast", "gini_abs", "ipr_abs", "amp2_ipr",
               "nn_amp_roughness", "global_abs", "weighted_phase_R", "phase_R",
               "nn_phase_R", "csv"],
              table)
end

function write_summary_csv(path::AbstractString, summary, metric_keys::Vector{Symbol})
    header = ["T", "n"]
    for key in metric_keys
        push!(header, "$(key)_mean")
        push!(header, "$(key)_sem")
    end
    table = Vector{Vector}()
    for row in summary
        values = Any[row.T, row.n]
        for key in metric_keys
            push!(values, getproperty(row, Symbol("$(key)_mean")))
            push!(values, getproperty(row, Symbol("$(key)_sem")))
        end
        push!(table, values)
    end
    write_csv(path, header, table)
end

function hsl_to_rgb(h::Float64, s::Float64, l::Float64)
    h = mod(h, 1.0)
    if s == 0
        return l, l, l
    end
    q = l < 0.5 ? l * (1 + s) : l + s - l * s
    p = 2 * l - q

    function hue_to_channel(t)
        t = mod(t, 1.0)
        if t < 1 / 6
            return p + (q - p) * 6 * t
        elseif t < 1 / 2
            return q
        elseif t < 2 / 3
            return p + (q - p) * (2 / 3 - t) * 6
        end
        return p
    end

    r = hue_to_channel(h + 1 / 3)
    g = hue_to_channel(h)
    b = hue_to_channel(h - 1 / 3)
    return r, g, b
end

function hex_color(r::Float64, g::Float64, b::Float64)
    ri = clamp(round(Int, 255 * r), 0, 255)
    gi = clamp(round(Int, 255 * g), 0, 255)
    bi = clamp(round(Int, 255 * b), 0, 255)
    return @sprintf("#%02x%02x%02x", ri, gi, bi)
end

function phase_colormap(saturation::Float64, lightness::Float64)
    colors = String[]
    for i in 0:256
        r, g, b = hsl_to_rgb(i / 256, saturation, lightness)
        push!(colors, hex_color(r, g, b))
    end
    return cgrad(colors)
end

function field_coordinates(Lx::Int, Ly::Int)
    xs = Int[]
    ys = Int[]
    sizehint!(xs, Lx * Ly)
    sizehint!(ys, Lx * Ly)
    for y in 1:Ly, x in 1:Lx
        push!(xs, x)
        push!(ys, y)
    end
    return xs, ys
end

function plot_spatial_field(path::AbstractString, d::Vector{ComplexF64}, Lx::Int, Ly::Int,
                            title::AbstractString, saturation::Float64,
                            lightness::Float64)
    xs, ys = field_coordinates(Lx, Ly)
    amp = abs.(d)
    phase = angle.(d)
    max_amp = maximum(amp)
    marker_sizes = max_amp > 0 ? 2.0 .+ 7.5 .* sqrt.(amp ./ max_amp) : fill(2.0, length(amp))
    p = scatter(xs, ys;
                marker_z=phase,
                color=phase_colormap(saturation, lightness),
                clims=(-pi, pi),
                markersize=marker_sizes,
                markerstrokewidth=0,
                markeralpha=0.95,
                xlabel="x",
                ylabel="y",
                xlim=(0.5, Lx + 0.5),
                ylim=(0.5, Ly + 0.5),
                yflip=true,
                aspect_ratio=:equal,
                title=title,
                legend=false,
                colorbar=true,
                colorbar_title="phase",
                size=(900, 780),
                dpi=160,
                grid=false,
                framestyle=:box,
                background_color=:white)
    savefig(p, path)
end

function plot_metric_series(path::AbstractString, summary, specs::Vector{Tuple{Symbol, String}},
                            title::AbstractString)
    temps = [row.T for row in summary]
    plots = Plots.Plot[]
    for (key, label) in specs
        ys = [getproperty(row, Symbol("$(key)_mean")) for row in summary]
        es = [getproperty(row, Symbol("$(key)_sem")) for row in summary]
        push!(plots, plot(temps, ys;
                          yerror=es,
                          marker=:circle,
                          linewidth=2,
                          label="mean ± SEM",
                          xlabel="T",
                          ylabel=label,
                          title=label,
                          framestyle=:box,
                          grid=true,
                          background_color=:white))
    end
    p = plot(plots...;
             layout=(length(plots), 1),
             size=(950, 300 * length(plots)),
             plot_title=title,
             margin=6Plots.mm)
    savefig(p, path)
end

function html_escape(x)
    s = string(x)
    s = replace(s, "&" => "&amp;")
    s = replace(s, "<" => "&lt;")
    s = replace(s, ">" => "&gt;")
    s = replace(s, "\"" => "&quot;")
    return s
end

fmt(x::Real; digits::Int=4) = @sprintf("%.*g", digits, x)
fmt_fixed(x::Real; digits::Int=3) = @sprintf("%.*f", digits, x)

function styles()
    return """
    <style>
      :root {
        color-scheme: light;
        --text: #202124;
        --muted: #62676d;
        --line: #d9dde2;
        --soft: #f6f7f8;
        --accent: #3d6f8f;
      }
      body {
        margin: 0;
        background: #ffffff;
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
        line-height: 1.65;
      }
      main {
        width: min(1120px, calc(100vw - 48px));
        margin: 0 auto;
        padding: 40px 0 64px;
      }
      h1 {
        margin: 0 0 12px;
        font-size: 30px;
        letter-spacing: 0;
      }
      h2 {
        margin: 46px 0 16px;
        padding-top: 28px;
        border-top: 1px solid var(--line);
        font-size: 22px;
      }
      h3 {
        margin: 24px 0 8px;
        font-size: 17px;
      }
      p {
        margin: 10px 0;
      }
      .lead {
        color: var(--muted);
        font-size: 16px;
        max-width: 920px;
      }
      .note {
        background: var(--soft);
        border-left: 4px solid var(--accent);
        padding: 14px 18px;
        margin: 24px 0;
      }
      .small {
        color: var(--muted);
        font-size: 14px;
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 18px;
      }
      .metric {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
      }
      figure {
        margin: 30px 0 42px;
      }
      figure img {
        width: 100%;
        max-width: 980px;
        display: block;
        border: 1px solid var(--line);
      }
      figcaption {
        margin-top: 10px;
        color: var(--muted);
        font-size: 14px;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        margin: 18px 0 28px;
        font-size: 14px;
      }
      th, td {
        border-bottom: 1px solid var(--line);
        padding: 9px 8px;
        text-align: right;
      }
      th:first-child, td:first-child,
      th:nth-child(3), td:nth-child(3) {
        text-align: left;
      }
      th {
        background: var(--soft);
        font-weight: 650;
      }
      code {
        background: var(--soft);
        padding: 1px 4px;
        border-radius: 4px;
      }
      .spatial {
        margin-top: 34px;
      }
      .spatial h3 {
        margin-top: 0;
      }
    </style>
    """
end

function mathjax()
    return raw"""
    <script>
      window.MathJax = { tex: { inlineMath: [['\\(', '\\)'], ['$', '$']] } };
    </script>
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
    """
end

function definitions_html()
    return replace(raw"""
    <h2>不均匀性指标的定义和意义</h2>
    <p>以下指标都从格点 pairing 场 \(d_i=A_i e^{i\\theta_i}\) 计算，其中 \(A_i=|d_i|\)，\(N=L_xL_y\)。幅度指标描述空间分布是否斑驳；相位指标描述不同区域的相位是否一致。</p>

    <div class="grid">
      <div class="metric">
        <h3>mean \(|d|\)</h3>
        <p>定义：\(\bar A=N^{-1}\\sum_i A_i\)。</p>
        <p class="small">意义：整体 pairing 幅度强弱。它本身不是不均匀性指标，因为两个样本可以有同样的平均幅度，但空间分布完全不同。</p>
      </div>
      <div class="metric">
        <h3>CV</h3>
        <p>定义：\(CV=\\sigma_A/\\bar A\)。</p>
        <p class="small">意义：相对起伏强度。\(CV=0\) 表示所有格点幅度相同；数值越大，说明相对于平均 pairing 幅度的空间起伏越强。</p>
      </div>
      <div class="metric">
        <h3>Gini 系数</h3>
        <p>定义：把 \(A_i\) 从小到大排序为 \(A_{(i)}\)，
        \[
        G={2\\sum_{i=1}^{N} i A_{(i)} \\over N\\sum_i A_i}-{N+1 \\over N}.
        \]</p>
        <p class="small">意义：衡量幅度分布的不平等程度。\(G=0\) 表示所有格点幅度完全一样；越接近 1，说明 pairing 幅度越被少数格点或少数区域主导。</p>
      </div>
      <div class="metric">
        <h3>IPR</h3>
        <p>定义：\(IPR_A=N\\sum_i A_i^2/(\\sum_i A_i)^2\)。</p>
        <p class="small">意义：衡量幅度权重的集中程度。均匀分布时为 1；如果 pairing 主要集中在少数格点，数值会升高。CSV 中还输出了 \(IPR_{A^2}=N\\sum_i A_i^4/(\\sum_i A_i^2)^2\)，它对大幅度点更敏感。</p>
      </div>
      <div class="metric">
        <h3>NN roughness</h3>
        <p>定义：\(\langle |A_i-A_j|\\rangle_{\langle ij\\rangle}/\\bar A\)。</p>
        <p class="small">意义：最近邻尺度上的幅度粗糙程度。CV/Gini 可以由大尺度起伏造成；NN roughness 更关注相邻格点之间是否有明显跳变。</p>
      </div>
      <div class="metric">
        <h3>weighted phase \(R_d\)</h3>
        <p>定义：\(R_d=|\\sum_i d_i|/\\sum_i |d_i|\)。</p>
        <p class="small">意义：幅度加权的全局相位 coherence。所有格点相位一致时 \(R_d=1\)；不同区域相位互相抵消时接近 0。</p>
      </div>
    </div>

    <h3>相位 coherence 图中的三条曲线</h3>
    <p>相位图同时画三个指标，是为了区分全局相位是否一致和局域邻近格点是否对齐。它们都越大表示相位越有序，但物理侧重点不同。</p>
    <div class="grid">
      <div class="metric">
        <h3>weighted \(|\\sum d|/\\sum |d|\)</h3>
        <p>\[
        R_d={\\left|\\sum_i d_i\\right|\\over\\sum_i |d_i|}
        ={\\left|\\sum_i A_i e^{i\\theta_i}\\right|\\over\\sum_i A_i}.
        \]</p>
        <p class="small">这是幅度加权的全局相位 coherence。幅度大的格点权重更大；如果大幅度区域相位一致，\(R_d\) 会较高。若不同区域相位相互抵消，即使局部区域内部有序，\(R_d\) 也会降低。</p>
      </div>
      <div class="metric">
        <h3>unweighted \(|\mathrm{mean}\,e^{i\theta}|\)</h3>
        <p>\[
        R_\\theta=\\left|{1\\over N}\\sum_i e^{i\\theta_i}\\right|.
        \]</p>
        <p class="small">这是不带幅度权重的全局相位 coherence。每个格点权重相同，因此对低幅度、相位噪声较大的格点更敏感。</p>
      </div>
      <div class="metric">
        <h3>NN mean \(\cos(\Delta\theta)\)</h3>
        <p>\[
        C_{\\mathrm{NN}}=\\left\\langle \\cos(\\theta_i-\\theta_j)\\right\\rangle_{\\langle ij\\rangle}.
        \]</p>
        <p class="small">这是最近邻相位相关，只看相邻格点之间是否对齐。若 \(C_{\\mathrm{NN}}\) 高但两个全局指标低，说明局域相位较平滑，但不同区域之间发生了全局抵消。</p>
      </div>
    </div>
    """, "\\\\" => "\\")
end

function summary_table_html(summary, reps)
    rows = IOBuffer()
    println(rows, "<table>")
    println(rows, "<thead><tr><th>T</th><th>样本数</th><th>代表构型</th><th>mean |d|</th><th>CV</th><th>Gini</th><th>IPR</th><th>NN roughness</th><th>phase \\(R_d\\)</th></tr></thead>")
    println(rows, "<tbody>")
    for row in summary
        rep = reps[row.T]
        println(rows, "<tr>")
        println(rows, "<td>$(fmt_fixed(row.T))</td>")
        println(rows, "<td>$(row.n)</td>")
        println(rows, "<td>$(html_escape(rep.conf_id))</td>")
        println(rows, "<td>$(fmt(getproperty(row, :mean_abs_mean))) ± $(fmt(getproperty(row, :mean_abs_sem)))</td>")
        println(rows, "<td>$(fmt(getproperty(row, :cv_abs_mean))) ± $(fmt(getproperty(row, :cv_abs_sem)))</td>")
        println(rows, "<td>$(fmt(getproperty(row, :gini_abs_mean))) ± $(fmt(getproperty(row, :gini_abs_sem)))</td>")
        println(rows, "<td>$(fmt(getproperty(row, :ipr_abs_mean))) ± $(fmt(getproperty(row, :ipr_abs_sem)))</td>")
        println(rows, "<td>$(fmt(getproperty(row, :nn_amp_roughness_mean))) ± $(fmt(getproperty(row, :nn_amp_roughness_sem)))</td>")
        println(rows, "<td>$(fmt(getproperty(row, :weighted_phase_R_mean))) ± $(fmt(getproperty(row, :weighted_phase_R_sem)))</td>")
        println(rows, "</tr>")
    end
    println(rows, "</tbody></table>")
    return String(take!(rows))
end

function spatial_sections_html(reps, image_prefix::AbstractString, report_kind::Symbol;
                               representative_note::AbstractString="")
    io = IOBuffer()
    for T in sort(collect(keys(reps)))
        rep = reps[T]
        image = @sprintf("assets/%s_T_%.3f.png", image_prefix, T)
        println(io, "<section class=\"spatial\">")
        println(io, "<h3>T = $(fmt_fixed(T))</h3>")
        if report_kind == :single
            println(io, "<p>$(html_escape(rep.conf_id)) / sweep $(rep.sweep)。$(html_escape(representative_note))</p>")
        else
            println(io, "<p>$(html_escape(rep.conf_id))，MC 平均 sweep $(rep.first_sweep)-$(rep.last_sweep)。代表构型沿用单 sweep 报告的选择。</p>")
        end
        println(io, "<figure>")
        println(io, "<img src=\"$(html_escape(image))\" alt=\"T $(fmt_fixed(T)) pairing phase amplitude map\">")
        if report_kind == :single
            println(io, "<figcaption>散点面积映射 \\(|d_i|\\)；颜色映射相位 \\(\\theta_i\\)，使用固定亮度、低饱和度的 HSL 循环色图。</figcaption>")
        else
            println(io, "<figcaption>散点面积映射 \\(|\\bar d_i|\\)；颜色映射相位 \\(\\arg \\bar d_i\\)，使用固定亮度、低饱和度的 HSL 循环色图。</figcaption>")
        end
        println(io, "</figure>")
        println(io, "</section>")
    end
    return String(take!(io))
end

function single_report_html(root_dir::AbstractString, summary, reps,
                            representative_note::AbstractString)
    template = raw"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Pairing 单 sweep 空间分布与不均匀性报告</title>
      __STYLES__
      __MATHJAX__
    </head>
    <body>
    <main>
      <h1>Pairing 单 sweep 空间分布与不均匀性报告</h1>
      <p class="lead">数据目录：<code>__ROOT_DIR__</code>。本报告对每个温度、每个 disorder 构型读取最后一个可用 sweep 的 \(d_i\)，用于观察未做 MC 平均的瞬时空间构型；趋势图把同一温度下的不同构型作为独立样本，显示均值和构型间标准误。</p>

      <div class="note">
        <p>空间图的读法：散点大小表示 \(A_i=|d_i|\)，颜色表示相位 \(\theta_i\)。颜色使用固定亮度、低饱和度 HSL 循环色图，因此 \(-\pi\) 和 \(\pi\) 首尾相接，亮度差异不会主导对散点大小的判断。</p>
      </div>

      __DEFINITIONS__

      <h2>温度趋势</h2>
      <figure>
        <img src="assets/inhomogeneity_vs_temperature.png" alt="幅度不均匀性指标随温度变化">
        <figcaption>幅度相关指标。每个温度下把不同 disorder 构型作为独立样本，误差棒为构型间标准误。</figcaption>
      </figure>
      <figure>
        <img src="assets/phase_coherence_vs_temperature.png" alt="相位 coherence 指标随温度变化">
        <figcaption>相位相关指标：weighted \(|\\sum d|/\\sum |d|\) 是幅度加权的全局相位一致性；unweighted \(|\mathrm{mean}\,e^{i\\theta}|\) 只看相位角本身；NN mean \(\cos(\Delta\theta)\) 看最近邻局域相位对齐。</figcaption>
      </figure>

      <h2>数值摘要</h2>
      __SUMMARY_TABLE__

      <h2>代表构型空间图</h2>
      <p>每个温度只展示一个代表构型。__REPRESENTATIVE_NOTE__</p>
      __SPATIAL_SECTIONS__
    </main>
    </body>
    </html>
    """
    template = replace(template, "\\\\" => "\\")
    return replace(template,
                   "__STYLES__" => styles(),
                   "__MATHJAX__" => mathjax(),
                   "__ROOT_DIR__" => html_escape(root_dir),
                   "__DEFINITIONS__" => definitions_html(),
                   "__SUMMARY_TABLE__" => summary_table_html(summary, reps),
                   "__REPRESENTATIVE_NOTE__" => html_escape(representative_note),
                   "__SPATIAL_SECTIONS__" => spatial_sections_html(reps, "phase_amp_scatter", :single;
                                                                     representative_note=representative_note))
end

function mcavg_report_html(root_dir::AbstractString, summary, reps)
    template = raw"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Pairing MC 平均后空间分布与不均匀性报告</title>
      __STYLES__
      __MATHJAX__
    </head>
    <body>
    <main>
      <h1>Pairing MC 平均后空间分布与不均匀性报告</h1>
      <p class="lead">数据目录：<code>__ROOT_DIR__</code>。本报告先在每个 disorder 构型内部做复数 MC 平均
      \[
      \\bar d_i^{(c)}={1\\over N_{MC}^{(c)}}\\sum_s d_i^{(c)}(s),
      \]
      然后分析每个构型的 \(\bar d_i^{(c)}\)。报告没有把不同 disorder 构型的空间场再平均成一张图；趋势图把不同构型一起作为独立样本统计，点为构型均值，误差棒为构型间标准误。</p>

      <div class="note">
        <p>空间图的读法：散点大小表示 \(|\\bar d_i^{(c)}|\)，颜色表示相位 \(\arg \\bar d_i^{(c)}\)。代表构型沿用单 sweep 报告的选择，便于直接比较“瞬时构型”和“MC 平均后构型”。</p>
      </div>

      __DEFINITIONS__

      <h2>温度趋势</h2>
      <figure>
        <img src="assets/mcavg_inhomogeneity_vs_temperature.png" alt="MC 平均后幅度不均匀性指标随温度变化">
        <figcaption>幅度相关指标。每个温度下把不同 disorder 构型作为独立样本，误差棒为构型间标准误；未对构型空间场做平均。</figcaption>
      </figure>
      <figure>
        <img src="assets/mcavg_phase_coherence_vs_temperature.png" alt="MC 平均后相位 coherence 指标随温度变化">
        <figcaption>相位相关指标：weighted \(|\\sum \\bar d|/\\sum |\\bar d|\) 是幅度加权的全局相位一致性；unweighted \(|\mathrm{mean}\,e^{i\\theta}|\) 只看相位角本身；NN mean \(\cos(\Delta\theta)\) 看最近邻局域相位对齐。</figcaption>
      </figure>

      <h2>数值摘要</h2>
      __SUMMARY_TABLE__

      <h2>代表构型空间图</h2>
      <p>每个温度只展示一个代表构型。代表构型沿用单 sweep 报告中按 CV 中位数选出的构型，但这里展示的是该构型内部做 MC 平均后的 \(\bar d_i\)。</p>
      __SPATIAL_SECTIONS__
    </main>
    </body>
    </html>
    """
    template = replace(template, "\\\\" => "\\")
    return replace(template,
                   "__STYLES__" => styles(),
                   "__MATHJAX__" => mathjax(),
                   "__ROOT_DIR__" => html_escape(root_dir),
                   "__DEFINITIONS__" => definitions_html(),
                   "__SUMMARY_TABLE__" => summary_table_html(summary, reps),
                   "__SPATIAL_SECTIONS__" => spatial_sections_html(reps, "mcavg_representative", :mcavg))
end

function single_representative_rows(reps)
    rows = Vector{Vector}()
    for T in sort(collect(keys(reps)))
        row = reps[T]
        csv_path = @sprintf("assets/representative_T_%.3f_%s_sweep_%d.csv",
                            row.T, row.conf_id, row.sweep)
        push!(rows, Any[row.T, row.conf_id, row.sweep, row.Lx, row.Ly,
                       row.cv_abs, row.mean_abs, row.weighted_phase_R, csv_path])
    end
    return rows
end

function make_single_report(root_dir::AbstractString, report_dir::AbstractString,
                            samples::Vector{SampleData}, single_rows, reps,
                            saturation::Float64, lightness::Float64,
                            representative_note::AbstractString)
    println("Writing single-sweep report to $report_dir")
    clean_generated_assets(report_dir)
    assets_dir = joinpath(report_dir, "assets")
    mkpath(assets_dir)

    metric_keys = [:mean_abs, :cv_abs, :gini_abs, :ipr_abs, :nn_amp_roughness,
                   :weighted_phase_R, :phase_R, :nn_phase_R]
    summary = summarize_by_temperature(single_rows, metric_keys)

    write_single_metrics(joinpath(report_dir, "sample_metrics.csv"), single_rows)
    write_summary_csv(joinpath(report_dir, "temperature_summary.csv"), summary, metric_keys)
    write_csv(joinpath(report_dir, "representative_samples.csv"),
              ["T", "conf_id", "sweep", "Lx", "Ly", "cv_abs", "mean_abs",
               "weighted_phase_R", "csv"],
              single_representative_rows(reps))

    for T in sort(collect(keys(reps)))
        rep = reps[T]
        sample = find_sample(samples, T, rep.conf_id)
        csv_path = joinpath(assets_dir, @sprintf("representative_T_%.3f_%s_sweep_%d.csv",
                                                 T, rep.conf_id, rep.sweep))
        write_field_csv(csv_path, sample.single_d, sample.Lx, sample.Ly)
        png_path = joinpath(assets_dir, @sprintf("phase_amp_scatter_T_%.3f.png", T))
        plot_spatial_field(png_path, sample.single_d, sample.Lx, sample.Ly,
                           @sprintf("T = %.3f, %s, sweep %d", T, rep.conf_id, rep.sweep),
                           saturation, lightness)
    end

    plot_metric_series(joinpath(assets_dir, "inhomogeneity_vs_temperature.png"),
                       summary,
                       [(:mean_abs, "mean |d|"),
                        (:cv_abs, "CV"),
                        (:gini_abs, "Gini"),
                        (:nn_amp_roughness, "NN roughness")],
                       "single sweep amplitude metrics")
    plot_metric_series(joinpath(assets_dir, "phase_coherence_vs_temperature.png"),
                       summary,
                       [(:weighted_phase_R, "weighted |sum d| / sum |d|"),
                        (:phase_R, "unweighted |mean exp(iθ)|"),
                        (:nn_phase_R, "NN mean cos(Δθ)")],
                       "single sweep phase coherence")

    open(joinpath(report_dir, "index.html"), "w") do io
        write(io, single_report_html(root_dir, summary, reps, representative_note))
    end
end

function make_mcavg_report(root_dir::AbstractString, report_dir::AbstractString,
                           samples::Vector{SampleData}, mc_rows, single_reps,
                           saturation::Float64, lightness::Float64)
    println("Writing MC-average report to $report_dir")
    clean_generated_assets(report_dir)
    assets_dir = joinpath(report_dir, "assets")
    mkpath(assets_dir)

    metric_keys = [:mean_abs, :cv_abs, :gini_abs, :ipr_abs, :nn_amp_roughness,
                   :weighted_phase_R, :phase_R, :nn_phase_R]
    summary = summarize_by_temperature(mc_rows, metric_keys)

    mc_reps = Dict{Float64, NamedTuple}()
    for T in sort(collect(keys(single_reps)))
        single_rep = single_reps[T]
        idx = findfirst(row -> row.T == T && row.conf_id == single_rep.conf_id, mc_rows)
        idx === nothing && error("Cannot find MC-averaged row for T=$T conf=$(single_rep.conf_id)")
        mc_reps[T] = mc_rows[idx]
    end

    write_mcavg_metrics(joinpath(report_dir, "mc_average_metrics.csv"), mc_rows)
    write_summary_csv(joinpath(report_dir, "mc_average_temperature_summary.csv"),
                      summary, metric_keys)

    for row in mc_rows
        sample = find_sample(samples, row.T, row.conf_id)
        csv_path = joinpath(assets_dir, @sprintf("mcavg_T_%.3f_%s.csv", row.T, row.conf_id))
        write_field_csv(csv_path, sample.mcavg_d, sample.Lx, sample.Ly)
    end

    for T in sort(collect(keys(mc_reps)))
        rep = mc_reps[T]
        sample = find_sample(samples, T, rep.conf_id)
        png_path = joinpath(assets_dir, @sprintf("mcavg_representative_T_%.3f.png", T))
        plot_spatial_field(png_path, sample.mcavg_d, sample.Lx, sample.Ly,
                           @sprintf("T = %.3f, %s, MC average", T, rep.conf_id),
                           saturation, lightness)
    end

    plot_metric_series(joinpath(assets_dir, "mcavg_inhomogeneity_vs_temperature.png"),
                       summary,
                       [(:mean_abs, "mean |d|"),
                        (:cv_abs, "CV"),
                        (:gini_abs, "Gini"),
                        (:nn_amp_roughness, "NN roughness")],
                       "MC-averaged amplitude metrics")
    plot_metric_series(joinpath(assets_dir, "mcavg_phase_coherence_vs_temperature.png"),
                       summary,
                       [(:weighted_phase_R, "weighted |sum d| / sum |d|"),
                        (:phase_R, "unweighted |mean exp(iθ)|"),
                        (:nn_phase_R, "NN mean cos(Δθ)")],
                       "MC-averaged phase coherence")

    open(joinpath(report_dir, "index.html"), "w") do io
        write(io, mcavg_report_html(root_dir, summary, mc_reps))
    end
end

function main(args=ARGS)
    opts = parse_args(args)
    samples = discover_samples(opts.root_dir)
    println("Found $(length(samples)) pairing files under $(opts.root_dir)")

    single_rows = sample_metrics(samples; field=:single)
    rep_source = representative_source(opts)
    representatives_pinned = rep_source !== nothing
    reps = if !representatives_pinned
        println("Selecting representatives by single-sweep CV median.")
        choose_representatives(single_rows)
    else
        println("Using pinned representatives from $rep_source")
        load_representatives(rep_source, single_rows)
    end
    representative_note = representatives_pinned ?
        "代表样本由固定名单指定，用于沿用上一份报告的代表构型。" :
        "代表样本按该温度内单 sweep 的 CV 中位数自动选取，避免专门挑最均匀或最不均匀的构型。"

    if opts.mode in ("both", "single")
        make_single_report(opts.root_dir,
                           joinpath(opts.root_dir, opts.single_report_name),
                           samples, single_rows, reps, opts.saturation, opts.lightness,
                           representative_note)
    end

    if opts.mode in ("both", "mcavg")
        mc_rows = sample_metrics(samples; field=:mcavg)
        make_mcavg_report(opts.root_dir,
                          joinpath(opts.root_dir, opts.mcavg_report_name),
                          samples, mc_rows, reps, opts.saturation, opts.lightness)
    end

    println("Pairing reports generated.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
