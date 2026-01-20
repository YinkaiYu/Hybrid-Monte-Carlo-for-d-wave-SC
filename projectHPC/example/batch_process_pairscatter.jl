using JLD2
using Glob
using Printf

# ================= 设置区域 =================
target_dir = @__DIR__
output_filename = "pairscatter_points.csv"
# ===========================================

function append_scatter!(io, jld_path, conf_id)
    if !isfile(jld_path) || filesize(jld_path) == 0
        return 0
    end

    total = 0
    try
        jldopen(jld_path, "r") do file
            for key in keys(file)
                if !startswith(key, "sweep_")
                    continue
                end
                sweep = try
                    parse(Int, split(key, "_")[2])
                catch
                    -1
                end
                d_local = file[key]["d_local"]
                @inbounds for (idx, val) in enumerate(d_local)
                    @printf(io, "%s,%d,%d,%.8e,%.8e\n",
                            conf_id, sweep, idx, real(val), imag(val))
                end
                total += length(d_local)
            end
        end
    catch
        return 0
    end

    return total
end

function process_T_directory(dir_path)
    conf_dirs = glob("conf_*", dir_path)
    if isempty(conf_dirs)
        return
    end

    println("Processing $(basename(dir_path))...")

    output_path = joinpath(dir_path, output_filename)
    total = 0

    open(output_path, "w") do io
        println(io, "conf_id,sweep,site_idx,re,im")
        for c_dir in conf_dirs
            conf_id = basename(c_dir)
            jld_path = joinpath(c_dir, "pairing_scatter.jld2")
            total += append_scatter!(io, jld_path, conf_id)
        end
    end

    if total == 0
        println("  -> Skipped: No valid JLD2 data.")
        return
    end

    println("  -> Saved: $output_path (samples=$total)")
end

println("Starting Pairing Scatter Processing...")

T_dirs = glob("T_*", target_dir)
sort!(T_dirs, by = d -> try parse(Float64, split(basename(d), "_")[2]) catch; 0.0 end)

for t_dir in T_dirs
    process_T_directory(t_dir)
end

println("Done.")
