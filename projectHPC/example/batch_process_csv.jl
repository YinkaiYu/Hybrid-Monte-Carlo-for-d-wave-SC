using DelimitedFiles
using Statistics
using Printf
using Glob
using DwaveHMC

# ================= 设置区域 =================
target_dir = @__DIR__
output_filename = "summary_all.csv"
# ===========================================

# --- 1. 鲁棒的读取函数 (核心) ---
function read_conf_robust(filepath)
    # 1. 文件存在且不为空
    if !isfile(filepath) || filesize(filepath) == 0
        return nothing, nothing
    end

    try
        # 2. 读取 CSV
        data, header = readdlm(filepath, ',', header=true)
        
        # 3. 检查是否只有表头没有数据
        if size(data, 1) == 0
            return nothing, nothing
        end

        col_names = vec(header)
        # 排除非物理量列
        exclude = ["Sweep", "Accepted", "dH", "Step", "Iter"]
        indices = findall(x -> !(string(x) in exclude), col_names)

        if isempty(indices) return nothing, nothing end

        # 4. 计算热平均
        means = mean(data[:, indices], dims=1) |> vec

        # 5. 检查计算结果是否含 NaN / Inf
        if any(isnan, means) || any(isinf, means)
            return nothing, nothing
        end

        return col_names[indices], means
    catch
        return nothing, nothing
    end
end

# --- 2. 解析 params.jl ---
function parse_params(param_file)
    if !isfile(param_file) return Dict{String, Any}() end
    m = Module()
    Base.eval(m, :(using DwaveHMC))
    try
        Base.include(m, param_file)
    catch
        return Dict{String, Any}()
    end
    data = Dict{String, Any}()
    for n in names(m; all=true)
        s = string(n)
        if startswith(s, "#") || s=="eval" || s=="include" || s==string(nameof(m)) continue end
        try
            val = getfield(m, n)
            if isa(val, Number) || isa(val, String) || isa(val, Bool)
                data[s] = val
            end
        catch; end
    end
    return data
end

# --- 3. 主程序 ---
println("Starting Robust T-scan CSV processing...")

# 1. 找到所有 T_* 目录
T_dirs = glob("T_*", target_dir)
# 按 T 的数值排序 (防止 T_0.1 排在 T_0.02 前面)
sort!(T_dirs, by = d -> try parse(Float64, split(basename(d), "_")[2]) catch; 0.0 end)

all_records = []
all_keys = Set{String}()

for t_dir in T_dirs
    dir_name = basename(t_dir)
    println("Processing $dir_name ...")
    
    # 解析参数
    p_dict = parse_params(joinpath(t_dir, "params.jl"))
    if isempty(p_dict)
        println("  -> Skipped: No params.jl found.")
        continue
    end
    union!(all_keys, keys(p_dict))
    
    # 寻找该温度下的所有 conf；若没有 conf_*，则直接用 T 目录
    conf_dirs = glob("conf_*", t_dir)
    if isempty(conf_dirs)
        obs_path = joinpath(t_dir, "observables.csv")
        if isfile(obs_path)
            conf_dirs = [t_dir]
        end
    end
    obs_dict = Dict{String, Vector{Float64}}()
    
    real_n_conf = 0 # 有效构型计数器
    
    for c_dir in conf_dirs
        # 读取 observables
        names, vals = read_conf_robust(joinpath(c_dir, "observables.csv"))
        
        if names !== nothing
            real_n_conf += 1
            conf_map = Dict{String, Float64}()
            for (k, v) in zip(names, vals)
                key = string(k)
                conf_map[key] = v
                if !haskey(obs_dict, key) obs_dict[key] = Float64[] end
                push!(obs_dict[key], v)
            end

            # Binder ratios from per-conf means
            if haskey(conf_map, "D2") && haskey(conf_map, "D4")
                d2 = conf_map["D2"]
                if d2 > 0
                    b_global = 1.0 - conf_map["D4"] / (2.0 * d2 * d2)
                    if !haskey(obs_dict, "B_global") obs_dict["B_global"] = Float64[] end
                    push!(obs_dict["B_global"], b_global)
                end
            end
            if haskey(conf_map, "Avg_d2") && haskey(conf_map, "Avg_d4")
                sd2 = conf_map["Avg_d2"]
                if sd2 > 0
                    b_local = 1.0 - conf_map["Avg_d4"] / (2.0 * sd2 * sd2)
                    if !haskey(obs_dict, "B_local") obs_dict["B_local"] = Float64[] end
                    push!(obs_dict["B_local"], b_local)
                end
            end
            
            # 尝试读取 transport (可选)
            t_names, t_vals = read_conf_robust(joinpath(c_dir, "transport.csv"))
            if t_names !== nothing
                for (k, v) in zip(t_names, t_vals)
                    key = string(k)
                    if !haskey(obs_dict, key) obs_dict[key] = Float64[] end
                    push!(obs_dict[key], v)
                end
            end
        end
    end
    
    # 只有当至少有一个构型跑出数据时，才记录该温度点
    if real_n_conf > 0
        row = copy(p_dict)
        row["real_n_conf"] = real_n_conf
        push!(all_keys, "real_n_conf")
        
        for (k, vals) in obs_dict
            if !isempty(vals)
                k_mean = "$(k)_mean"
                k_err = "$(k)_err"
                row[k_mean] = mean(vals)
                row[k_err] = real_n_conf > 1 ? std(vals) / sqrt(length(vals)) : 0.0
                push!(all_keys, k_mean)
                push!(all_keys, k_err)
            end
        end
        push!(all_records, row)
        println("  -> OK. Valid Samples: $real_n_conf / $(length(conf_dirs))")
    else
        println("  -> Warning: No complete data found in $dir_name. Skipped.")
    end
end

# --- 4. 写入 ---
if isempty(all_records)
    println("No valid data found in any directory.")
    exit()
end

# 确保按 T 排序
sort!(all_records, by = r -> get(r, "T", 0.0))

param_keys = sort(collect(filter(k -> !endswith(k, "_mean") && !endswith(k, "_err"), all_keys)))
data_keys = sort(collect(filter(k -> endswith(k, "_mean") || endswith(k, "_err"), all_keys)))

# 调整列顺序
priority = ["T", "β", "beta", "n_imp", "real_n_conf"]
final_params = filter(k -> k in priority, param_keys)
append!(final_params, filter(k -> !(k in priority), param_keys))
final_header = vcat(final_params, data_keys)

open(joinpath(target_dir, output_filename), "w") do io
    println(io, join(final_header, ","))
    for row in all_records
        vals = [get(row, k, "") for k in final_header]
        println(io, join(vals, ","))
    end
end
println("Summary saved to $output_filename")
