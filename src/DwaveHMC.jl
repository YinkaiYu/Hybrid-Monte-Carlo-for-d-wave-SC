module DwaveHMC

export ModelParameters, SimulationState, ComputeCache
export initialize_state, initialize_cache
export init_static_H!, update_H_BdG!, diagonalize_H_BdG!
export compute_forces!
export hmc_sweep!, compute_total_energy # 新增

include("Types.jl")
include("Hamiltonian.jl")
include("Observables.jl")
include("HMC.jl") # 新增

end