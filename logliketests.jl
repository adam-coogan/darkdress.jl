using Distributions
using Random
using PyPlot

# %%
include("utils.jl")
include("binary.jl")
include("analysis.jl")  # needs types from binary.jl to be loaded

"""
Get the benchmark Eda et al system `s` at some distance.
"""
function get_system_s(dₗ_ref=1e8 * pc)
    m₁_ref = 1e3 * MSun
    m₂_ref = 1 * MSun
    ρₛ_ref = 226 * MSun / pc^3
    γₛ_ref = 7 / 3.
    ι_ref = 0.
    Φ_c_ref = 0.
    t_c_ref = -dₗ_ref / c
    return make_dress(
        DynamicDress, m₁_ref, m₂_ref, ρₛ_ref, γₛ_ref, dₗ_ref, ι_ref, Φ_c_ref, t_c_ref
    )
end

const system_s = get_system_s()

# %%
"""
Samples a dark dress `h` with parameters near the given one `s`.
"""
function sample_dd(system_s, method="uniform")
    if method == "uniform"
        # Estimated from Bradley's MCMC run
        r_γₛ = 0.02
        r_c_f = 0.5
        r_ℳ = 0.0002
        r_q = 0.2
        Δt̃_c = 130.

        γₛ = rand(Uniform(system_s.γₛ * (1 - r_γₛ), system_s.γₛ * (1 + r_γₛ)))
        c_f = rand(Uniform(system_s.c_f * (1 - r_c_f), system_s.c_f * (1 + r_c_f)))
        ℳ = rand(Uniform(system_s.ℳ * (1 - r_ℳ), system_s.ℳ * (1 + r_ℳ)))
        q = rand(Uniform(system_s.q * (1 - r_q), system_s.q * (1 + r_q)))
        Φ_c = system_s.Φ_c
        t̃_c = rand(Uniform(system_s.t̃_c - Δt̃_c, system_s.t̃_c + Δt̃_c))
        dₗ_ι = system_s.dₗ_ι
    elseif method == "fim"
        idxs = [1, 2, 3, 4, 6]
        μ = convert(Array{Float64,1}, system_s)[idxs]
        
        # 5D covariance matrix
        fc_s = f_isco(m₁(system_s.ℳ, system_s.q))
        fₗ = f_of_t_to_c(5 * yr, fc_s, system_s)
        Σ = inv(fim(fₗ, fc_s, fc_s, system_s)[idxs, idxs])
        Σ = 1/2 * (Σ + Σ') * 0.01

        # Enforce positivity
        while true
            γₛ, c_f, ℳ, q, t̃_c = rand(MultivariateNormal(μ, Σ))
            if γₛ > 0 && c_f > 0 && ℳ > 0 && q > 0
                break
            end
        end

        Φ_c = system_s.Φ_c
        dₗ_ι = system_s.dₗ_ι
    end

    return DynamicDress(γₛ, c_f, ℳ, q, Φ_c, t̃_c, dₗ_ι)
end

"""
Samples a dark dress `h` near the given one `s` and computes `log L(h|s)`.
"""
function sample_loglike(system_s, N_nodess, method)
    system_h = sample_dd(system_s, method)
    
    fc_s = f_isco(m₁(system_s.ℳ, system_s.q))
    fc_h = f_isco(m₁(system_h.ℳ, system_h.q))
    fₗ = f_of_t_to_c(5 * yr, fc_s, system_s)

    loglikes = zeros(size(N_nodess))
    for (i, N_nodes) in enumerate(N_nodess)
        loglikes[i] = calculate_loglike(
            system_h, system_s, fₗ, fc_s, fc_h, fc_s; N_nodes=N_nodes
        )
    end

    return loglikes, system_h
end

# %%
"""
Check how log likelihood varies with number of points for a randomly sampled
system.
"""
function plot_sample_loglikes()
    # Check 1_000 - 100_000 points
    N_nodess = Array{Int64,1}(round.(10 .^ range(3, 5, length=15)))
    loglikes, system_h = sample_loglike(system_s, N_nodess, "fim")
    println(loglikes)

    PyPlot.close("all")
    PyPlot.loglog(N_nodess, loglikes)
    PyPlot.xlabel("Number of nodes")
    PyPlot.ylabel("Log L(h|s)")
    PyPlot.title("s: Eda+ system. h: samples from FIM.")
    PyPlot.display(PyPlot.gcf())
end

plot_sample_loglikes()