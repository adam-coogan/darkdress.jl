# %%
include("utils.jl")
include("binary.jl")
include("analysis.jl")  # needs types from binary.jl to be loaded
# Needs above files to be included
include("tests.jl")
include("plotting.jl")

# %%
test_calcloglike()
benchmark_calcloglike()

# %%
test_fim_SNR()
benchmark_fim()

# %%
"""
Regenerate all plots (takes ~3 minutes)
"""
function main()
    # Configure masses and parameter grids
    m₁_ref = 1.4e3 * MSun
    m₂_ref = 1.4 * MSun
    ρₛs_ref = 10.0.^range(-2, log10(3e4), length=15) * MSun/pc^3
    γₛs_ref = collect(range(2.2, 2.5, length=14))
    t_to_merger = 5 * yr  # observing time
    f_c = f_isco(m₁_ref)  # frequency at coalescence
    fₕ = f_c  # for setting frequency observation window

    make_all_plots(m₁_ref, m₂_ref, ρₛs_ref, γₛs_ref, fₕ, f_c, t_to_merger)
end