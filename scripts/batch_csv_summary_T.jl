using DelimitedFiles
using Statistics
using Printf

# ==========================================
# 1. 设置
# ==========================================
# 修改：匹配之前的 T_scan 目录命名格式
target_dir = "data/T_scan_L24_J0.8_W1.0_imp0.05_mu_-1.08" 

output_filename = "summary_all.csv"

# ==========================================
# 2. 辅助函数 (保持不变)
# ==========================================

"""
    process_csv(filepath)

读取 CSV，忽略 'Sweep' 列，计算剩余列的均值和误差。
返回: (列名数组, 均值数组, 误差数组)
"""
function process_csv(filepath)
    if !isfile(filepath)
        return nothing, nothing, nothing
    end

    try
        data, header = readdlm(filepath, ',', header=true)
        col_names = vec(header)
        
        # 排除 Sweep 列
        exclude_indices = findall(x -> lowercase(string(x)) == "sweep", col_names)
        keep_indices = setdiff(1:length(col_names), exclude_indices)
        
        if isempty(keep_indices)
            return nothing, nothing, nothing
        end
        
        final_names = col_names[keep_indices]
        final_data = data[:, keep_indices]
        
        n_steps = size(final_data, 1)
        
        if n_steps < 2
            # 只有1行数据没法算标准差，但在 summary 中可以仅输出均值，
            # 这里为了安全还是返回 nothing 或者你可以改为接受
            # 也可以简单处理：
            if n_steps == 1
                 means = final_data[1, :]
                 errs = zeros(length(means))
                 return final_names, means, errs
            end
            println("  Warning: Not enough data in $filepath (rows=$n_steps)")
            return nothing, nothing, nothing
        end
        
        means = mean(final_data, dims=1) |> vec
        stds = std(final_data, dims=1) |> vec
        errs = stds ./ sqrt(n_steps)
        
        return final_names, means, errs
        
    catch e
        println("  Error processing $filepath: $e")
        return nothing, nothing, nothing
    end
end

# ==========================================
# 3. 主逻辑
# ==========================================

println("Starting Post-Processing in: $target_dir")

if !isdir(target_dir)
    println("Error: Directory '$target_dir' does not exist.")
    exit()
end

# 修改：1. 扫描所有以 "T_" 开头的子目录
subdirs = filter(d -> isdir(joinpath(target_dir, d)) && startswith(d, "T_"), readdir(target_dir))

if isempty(subdirs)
    println("No 'T_*' directories found!")
    exit()
end

println("Found $(length(subdirs)) directories.")

all_results = []
all_keys = Set{String}(["T", "Beta"])

for subdir in subdirs
    full_path = joinpath(target_dir, subdir)
    
    # 修改：2. 解析 T 值
    # 假设目录名格式 T_0.00123 (基于 sigdigits=3)
    try
        T_str = replace(subdir, "T_" => "")
        T_val = parse(Float64, T_str)
        
        # 计算对应的 Beta
        beta_val = 1.0 / T_val
        
        println("Processing T = $T_val (Beta ≈ $(round(beta_val, digits=2))) ...")
        
        row_dict = Dict{String, Any}()
        row_dict["Beta"] = beta_val
        row_dict["T"] = T_val
        
        # --- 处理 observables.csv ---
        obs_names, obs_means, obs_errs = process_csv(joinpath(full_path, "observables.csv"))
        if obs_names !== nothing
            for i in 1:length(obs_names)
                name = string(obs_names[i])
                row_dict["$(name)_mean"] = obs_means[i]
                row_dict["$(name)_err"] = obs_errs[i]
                push!(all_keys, "$(name)_mean")
                push!(all_keys, "$(name)_err")
            end
        end
        
        # --- 处理 transport.csv ---
        trans_names, trans_means, trans_errs = process_csv(joinpath(full_path, "transport.csv"))
        if trans_names !== nothing
            for i in 1:length(trans_names)
                name = string(trans_names[i])
                row_dict["$(name)_mean"] = trans_means[i]
                row_dict["$(name)_err"] = trans_errs[i]
                push!(all_keys, "$(name)_mean")
                push!(all_keys, "$(name)_err")
            end
        end
        
        push!(all_results, row_dict)
        
    catch e
        println("  Skipping $subdir: $e")
    end
end

# ==========================================
# 4. 排序与写入
# ==========================================

# 按温度 T 从低到高排序
sort!(all_results, by = x -> x["T"])

sorted_keys = sort(collect(all_keys))
filter!(x -> x != "T" && x != "Beta", sorted_keys)
final_header = vcat(["T", "Beta"], sorted_keys)

output_path = joinpath(target_dir, output_filename)
open(output_path, "w") do io
    println(io, join(final_header, ","))
    
    for row in all_results
        vals = []
        for key in final_header
            val = get(row, key, NaN)
            push!(vals, val)
        end
        println(io, join(vals, ","))
    end
end

println("--------------------------------------------------")
println("Done! Summary saved to: $output_path")
println("Columns: $(join(final_header, ", "))")