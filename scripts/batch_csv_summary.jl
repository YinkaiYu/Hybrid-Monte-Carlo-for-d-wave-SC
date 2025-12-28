using DelimitedFiles
using Statistics
using Printf

# ==========================================
# 1. 设置
# ==========================================
# 请修改为你存放数据的目录
target_dir = "data/beta_test_L12_J0.8_imp0.0" 

output_filename = "summary_all.csv"

# ==========================================
# 2. 辅助函数
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

    # 读取数据和表头
    try
        data, header = readdlm(filepath, ',', header=true)
        # header 是 1xN 矩阵，转为 Vector
        col_names = vec(header)
        
        # 找到需要排除的列 (Sweep, sweep, possibly Accepted if needed)
        # 这里我们只排除 Sweep，保留 Accepted 因为它的均值就是接受率
        exclude_indices = findall(x -> lowercase(string(x)) == "sweep", col_names)
        
        # 保留的列索引
        keep_indices = setdiff(1:length(col_names), exclude_indices)
        
        if isempty(keep_indices)
            return nothing, nothing, nothing
        end
        
        final_names = col_names[keep_indices]
        final_data = data[:, keep_indices]
        
        # 计算统计量
        # rows are steps, cols are observables
        n_steps = size(final_data, 1)
        
        # 如果数据行数太少，给警告
        if n_steps < 2
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

# 1. 扫描所有子目录
subdirs = filter(d -> isdir(joinpath(target_dir, d)) && startswith(d, "beta_"), readdir(target_dir))

if isempty(subdirs)
    println("No 'beta_*' directories found!")
    exit()
end

println("Found $(length(subdirs)) directories.")

# 用于存储所有行的结果 (Dictionary list)
# 结构: [ Dict("T"=>..., "Beta"=>..., "Energy_mean"=>..., ...), ... ]
all_results = []
# 用于收集所有出现过的列名 (保证最后的表头完整)
all_keys = Set{String}(["T", "Beta"])

for subdir in subdirs
    full_path = joinpath(target_dir, subdir)
    
    # 解析 Beta
    # 假设目录名格式 beta_10.0 或 beta_10.000
    try
        beta_str = replace(subdir, "beta_" => "")
        beta_val = parse(Float64, beta_str)
        T_val = 1.0 / beta_val
        
        println("Processing Beta = $beta_val ...")
        
        # 当前行的记录
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
sort!(all_results, by = x -> x["Beta"])

# 构建最终的列顺序
# 强制 T 和 Beta 在最前面
sorted_keys = sort(collect(all_keys))
# 移除 T 和 Beta 以便重新插入
filter!(x -> x != "T" && x != "Beta", sorted_keys)
final_header = vcat(["T", "Beta"], sorted_keys)

output_path = joinpath(target_dir, output_filename)
open(output_path, "w") do io
    # 1. 写入表头
    println(io, join(final_header, ","))
    
    # 2. 写入数据行
    for row in all_results
        vals = []
        for key in final_header
            # 如果某一行缺少某个key (比如某个beta下没跑transport), 填 NaN
            val = get(row, key, NaN)
            push!(vals, val)
        end
        println(io, join(vals, ","))
    end
end

println("--------------------------------------------------")
println("Done! Summary saved to: $output_path")
println("Contains data for: $(join(final_header, ", "))")