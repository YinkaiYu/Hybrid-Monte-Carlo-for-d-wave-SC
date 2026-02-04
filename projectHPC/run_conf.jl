using Printf
using Random

# ---------------------------------------------------------
# 1. 环境加载
# ---------------------------------------------------------
const PROJECT_ROOT = @__DIR__
using Pkg
Pkg.activate(PROJECT_ROOT)
using DwaveHMC

# ---------------------------------------------------------
# 2. 读取参数
# ---------------------------------------------------------
# 获取 params.jl 在当前工作目录下的绝对路径
params_path = joinpath(pwd(), "params.jl")
if !isfile(params_path)
    error("params.jl not found at $params_path")
end
# 使用绝对路径 include
include(params_path)

# 如果 params.jl 未指定 max_iter，则使用模型参数默认值
if !@isdefined(max_iter)
    max_iter = p.Δ_MF_max_iter
end

println("Parameters loaded. T = $(T), max_iter = $(max_iter)")
flush(stdout)

# ---------------------------------------------------------
# 3. 单次运行 (平均场无无序)
# ---------------------------------------------------------
try
    println("Running mean-field simulation...")
    flush(stdout)
    run_simulation(p, "conf_1"; max_iter=max_iter, verbose=false)
    println("Done.")
    flush(stdout)
catch e
    println("ERROR: $e")
    exit(1)
end
flush(stdout)

if success_count != N_conf
    exit(1) # 如果有任务失败，让作业状态显示为 Failed
end
