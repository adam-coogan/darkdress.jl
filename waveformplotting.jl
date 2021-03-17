# %%
import PyPlot
using LaTeXStrings
PyPlot.rc("text", usetex="True")

include("utils.jl")
include("binary.jl")
include("analysis.jl")  # needs types from binary.jl to be loaded

"""
This file contains code for plotting and checking waveforms.
"""

# %%
function plot_Φ()
    # Benchmark system
    m₁_ref = 1e3 * MSun
    m₂_ref = 1 * MSun
    ρₛ_ref = 226 * MSun / pc^3
    γₛ_ref = 7/3.
    dₗ_ref = 1e8 * pc
    ι_ref = 0.
    Φ_c_ref = 0.
    t_c_ref = -dₗ_ref / c
    dd = make_dress(
        DynamicDress, m₁_ref, m₂_ref, ρₛ_ref, γₛ_ref, dₗ_ref, ι_ref, Φ_c_ref, t_c_ref
    )
    sd = make_dress(
        StaticDress, m₁_ref, m₂_ref, ρₛ_ref, γₛ_ref, dₗ_ref, ι_ref, Φ_c_ref, t_c_ref
    )
    vb = make_vacuum_binary(m₁_ref, m₂_ref, dₗ_ref, ι_ref, Φ_c_ref, t_c_ref)

    dd_alt = DynamicDress(
        2.27843666, 7.38694332e-5, 3.1518396596159997e31, 0.000645246955, 0.0, -1.98186232e+02, -56.3888135025341
    )
    sd_alt = convert(StaticDress{Float64}, dd_alt)
    vb_alt = convert(VacuumBinary{Float64}, dd_alt)

    println("$(m₁(dd.ℳ, dd.q) / MSun), $(m₂(dd.ℳ, dd.q) / MSun)")
    println("$(m₁(dd_alt.ℳ, dd_alt.q) / MSun), $(m₂(dd_alt.ℳ, dd_alt.q) / MSun)")

    f_c = f_isco(m₁(dd.ℳ, dd.q))
    f_c_alt = f_isco(m₁(dd_alt.ℳ, dd_alt.q))
    fₗ = f_of_t_to_c(5 * yr, f_c, dd)
    fs = geomspace(fₗ, max(f_c, f_c_alt), 500)

    PyPlot.close("all")
    
    fig, axes = PyPlot.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[1]
    ax.loglog(fs, Φ_to_c.(fs, [f_c], [dd]), label="D")
    ax.loglog(fs, Φ_to_c.(fs, [f_c_alt], [dd_alt]), label="D (alt)")
    ax.loglog(fs, Φ_to_c.(fs, [f_c], [sd]), "--", label="S")
    ax.loglog(fs, Φ_to_c.(fs, [f_c_alt], [sd_alt]), "--", label="S (alt)")
    ax.loglog(fs, Φ_to_c.(fs, [f_c], [vb]), ":", label="V")
    ax.loglog(fs, Φ_to_c.(fs, [f_c], [vb_alt]), ":", label="V (alt)")
    ax.set_ylabel(L"\Phi~[\mathrm{rad}]")
    ax.set_title("Phase")

    ax = axes[2]
    ax.loglog(fs, Φ_to_c.(fs, [f_c], [vb]) .- Φ_to_c.(fs, [f_c], [dd]), label="V - D")
    ax.loglog(fs, Φ_to_c.(fs, [f_c_alt], [vb_alt]) .- Φ_to_c.(fs, [f_c_alt], [dd_alt]), label="V - D (alt)")
    ax.loglog(fs, Φ_to_c.(fs, [f_c], [vb]) .- Φ_to_c.(fs, [f_c], [sd]), "--", label="V - S")
    ax.loglog(fs, Φ_to_c.(fs, [f_c_alt], [vb_alt]) .- Φ_to_c.(fs, [f_c_alt], [sd_alt]), "--", label="V - S (alt)")
    ΔΦ_ref_alt = Φ_to_c.(fs, [f_c], [dd]) .- Φ_to_c.(fs, [f_c_alt], [dd_alt])
    ax.loglog(fs, ΔΦ_ref_alt .- ΔΦ_ref_alt[end], label="D - D (alt)")
    # ax.loglog(fs, Φ_to_c.(fs, [f_c], [vb]), ":", label="V")
    ax.set_ylabel(L"\Delta\Phi~[\mathrm{rad}]")
    ax.set_title("Dephasing")

    for ax in axes
        ax.axvline(convert(HypParams{Float64}, dd).fₜ, color="C0")
        ax.axvline(convert(HypParams{Float64}, dd_alt).fₜ, color="C1")
        ax.legend(frameon=false)
        ax.set_xlabel(L"f~[\mathrm{Hz}]")
    end

    fig.tight_layout()
    PyPlot.display(fig)

    return fig, axes
end

fig = plot_Φ()[1]