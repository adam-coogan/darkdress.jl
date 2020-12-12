include("utils.jl")
include("binary.jl")
include("analysis.jl")  # needs types from binary.jl to be loaded

import ProgressMeter
import PyPlot
using LaTeXStrings
PyPlot.rc("text", usetex="True")

function latex_name(p::Symbol)
    if p == :ℳ
        return "{\\mathcal{M}}"
    elseif p == :q
        return "{q}"
    elseif p == :γₛ
        return "{\\gamma_\\mathrm{sp}}"
    elseif p == :c_f
        return "{c_f}"
    else
        throw(ArgumentError("invalid parameter"))
    end
end

# Configure masses and parameter grids
const m₁_ref = 1e3 * MSun
const m₂_ref = 1 * MSun
get_ρₛs(n) = 10.0.^range(-2, 3, length=n) * MSun/pc^3
get_γₛs(n) = collect(range(2.25, 2.5, length=n))
const t_to_merger = 5 * yr  # observing time
const f_c = f_isco(m₁_ref)  # frequency at coalescence
const fₕ = f_c  # for setting frequency observation window

"""
Verifies the code runs.
"""
function basic_checks()
    println("Static checks")
    sd = make_dress(StaticDress, 1e3 * MSun, 1. * MSun, 226. * MSun/pc^3, 7/3)
    println(sd)
    fₕ = f_c = f_isco(1e3 * MSun)
    fₗ = f_of_t_to_c(5 * yr, f_c, sd)
    snr(fₗ, fₕ, sd)  # agrees with David's notebook
    println(log10(fim_errs(fₗ, fₕ, f_c, sd)[1]))
    println(log10(fim_errs(fₗ, fₕ, f_c, sd)[2]))
    println(log10(fim_errs(fₗ, fₕ, f_c, sd)[3]))
    println()
    
    println("Dynamic checks")
    dd = make_dress(DynamicDress, 1e3 * MSun, 1. * MSun, 226. * MSun/pc^3, 7/3)
    println(dd)
    fₕ = f_c = f_isco(1e3 * MSun)
    fₗ = f_of_t_to_c(5 * yr, f_c, dd)
    snr(fₗ, fₕ, dd)  # agrees with David's notebook
    println(log10(fim_errs(fₗ, fₕ, f_c, dd)[1]))
    println(log10(fim_errs(fₗ, fₕ, f_c, dd)[2]))
    println(log10(fim_errs(fₗ, fₕ, f_c, dd)[3]))
    println(log10(fim_errs(fₗ, fₕ, f_c, dd)[4])) 
end

# Plotting
function plot_snr(T::Type{DT}, n_ρₛ=15, n_γₛ=10) where DT <: Binary
    ρₛs = get_ρₛs(n_ρₛ)
    γₛs = get_γₛs(n_γₛ)
    snrs = zeros(n_γₛ, n_ρₛ)

    for i in 1:n_ρₛ, j in 1:n_γₛ
        sd = make_dress(T, m₁_ref, m₂_ref, ρₛs[i], γₛs[j])
        fₗ = find_zero(f -> t_to_c(f, f_c, sd) - t_to_merger, (0.0001 * f_c, f_c))
        snrs[j, i] = snr(fₗ, fₕ, sd)  # remember to transpose!
    end

    fig, ax = PyPlot.subplots(1, 1, figsize=(4, 3.2))

    cs_f = ax.contourf(ρₛs / (MSun/pc^3), γₛs, snrs, cmap=T == StaticDress ? "plasma" : "viridis")
    PyPlot.colorbar(cs_f, ax=ax)

    ax.set_xscale("log")
    ax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])  # tick locators not supported...
    ax.set_xlabel(L"\log_{10} \, \frac{\rho_s}{\mathrm{M}_\odot / \mathrm{pc}^3}")
    ax.set_ylabel(L"\gamma_s")
    ax.set_title(T == StaticDress ? "Static dress" : "Dynamic dress")
    fig.tight_layout()

    return fig
end

function plot_ΔN_cycles(T::Type{DT}, n_ρₛ=15, n_γₛ=10) where DT <: Binary
    ρₛs = get_ρₛs(n_ρₛ)
    γₛs = get_γₛs(n_γₛ)
    ΔN_cycless = zeros(n_γₛ, n_ρₛ)

    vb = make_vacuum_binary(m₁_ref, m₂_ref)

    for i in 1:n_ρₛ, j in 1:n_γₛ
        sd = make_dress(T, m₁_ref, m₂_ref, ρₛs[i] * MSun/pc^3, γₛs[j])
        fₗ = find_zero(f -> t_to_c(f, f_c, sd) - t_to_merger, (0.0001 * f_c, f_c))
        # Remember to transpose!
        ΔN_cycless[j, i] = (Φ_to_c(fₗ, f_c, vb) - Φ_to_c(fₗ, f_c, sd)) / (2 * π)
    end

    fig, ax = PyPlot.subplots(1, 1, figsize=(4, 3.2))

    # PyPlot.jl doesn't have locators...
    if T == DynamicDress
        levels = log10.([3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3])
        cbar_ticklabels = [L"10^{-2}", L"10^{-1}", L"10^0", L"10^1", L"10^2", L"10^3"]
        cmap = "viridis"
        title = "Dynamic dress"
    elseif T == StaticDress
        levels = log10.([3e-1, 1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4])
        cbar_ticklabels = [L"10^0", L"10^1", L"10^2", L"10^3", L"10^4"]
        cmap = "plasma"
        title = "Static dress"
    end

    cs_f = ax.contourf(ρₛs / (MSun/pc^3), γₛs, log10.(ΔN_cycless), cmap=cmap, levels=levels)
    cbar = PyPlot.colorbar(cs_f, ax=ax)
    cbar.ax.set_yticklabels(cbar_ticklabels)
    
    ax.set_xscale("log")
    ax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])  # tick locators not supported...
    ax.set_xlabel(L"\log_{10} \, \frac{\rho_s}{\mathrm{M}_\odot / \mathrm{pc}^3}")
    ax.set_ylabel(L"\gamma_s")
    ax.set_title(title)
    fig.tight_layout()

    return fig
end

function plot_errs(T::Type{DT}, n_ρₛ=15, n_γₛ=10) where DT <: Binary
    ρₛs = get_ρₛs(n_ρₛ)
    γₛs = get_γₛs(n_γₛ)
    n_intrinsic = length(intrinsics(T))
    errs = zeros(n_γₛ, n_ρₛ, n_intrinsic)

    ProgressMeter.@showprogress  for i in 1:n_ρₛ, j in 1:n_γₛ
        sd = make_dress(T, m₁_ref, m₂_ref, ρₛs[i], γₛs[j])
        fₗ = find_zero(f -> t_to_c(f, f_c, sd) - t_to_merger, (0.0001 * f_c, f_c))
        errs[j, i, :] = fim_errs(fₗ, fₕ, f_c, sd)[1:n_intrinsic]
    end

    fig, axes = PyPlot.subplots(2, 2, figsize=(2 * 4, 2 * 3.4))
    axes = vec(axes)
    cmap = T == StaticDress ? "plasma" : "viridis"

    for i in 1:size(errs, 3)
        ax = axes[i]

        # Plot
        cs_f = ax.contourf(ρₛs / (MSun/pc^3), γₛs, log10.(errs[:, :, i]), cmap=cmap)
        PyPlot.colorbar(cs_f, ax=ax)

        # Format
        ax.set_xscale("log")
        ax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])  # tick locators not supported...
        ax.set_xlabel(L"\log_{10} \, \frac{\rho_s}{\mathrm{M}_\odot / \mathrm{pc}^3}")
        ax.set_ylabel(L"\gamma_s")
        lp = latex_name(intrinsics(T)[i])
        ax.set_title(latexstring("\\log_{10} \\, \\Delta", lp, " / ", lp))
    end

    # For static case
    for i in size(errs, 3)+1:length(axes)
        axes[i].set_axis_off()
    end

    fig.suptitle(T == StaticDress ? "Static dress" : "Dynamic dress")
    fig.tight_layout()
    PyPlot.subplots_adjust(top=0.91)

    return fig
end

"""
Call to remake and save figures for the paper.
"""
function make_paper_plots()
    fig = plot_snr(StaticDress, 50, 45)
    fig.savefig("figures/sd_snr.pdf")

    fig = plot_snr(DynamicDress, 50, 45)
    fig.savefig("figures/dd_snr.pdf")

    fig = plot_ΔN_cycles(StaticDress, 50, 45)
    fig.savefig("figures/sd_dN_cycles.pdf")

    fig = plot_ΔN_cycles(DynamicDress, 50, 45)
    fig.savefig("figures/dd_dN_cycles.pdf")

    fig = plot_errs(StaticDress, 50, 45)
    fig.savefig("figures/sd_post.pdf")

    fig = plot_errs(DynamicDress, 50, 45)
    fig.savefig("figures/dd_post.pdf")

    PyPlot.close("all")
end

make_paper_plots()

#####################
# Waveform plotting #
#####################
function plot_amp₊()
    m₁_ = 1e3 * MSun
    m₂_ = 1.0 * MSun
    γₛ_ = 7/3
    ρₛ_ = 500. * MSun / pc^3
    vb = make_vacuum_binary(m₁_, m₂_)
    sd = make_static_dress(m₁_, m₂_, ρₛ_, γₛ_)
    dd = make_dynamic_dress(m₁_, m₂_, ρₛ_, γₛ_)

    f_c = f_isco(m₁_)
    fs = geomspace(3e-3, f_c * 0.999, 1000)
    
    plot(fs, (f -> f * amp₊(f, vb)).(fs), legend=:topright, label=L"f\, A_{+,V}(f)")
    plot!(fs, (f -> f * amp₊(f, sd)).(fs), label=L"f\, A_{+,S}(f)")
    plot!(fs, (f -> f * amp₊(f, dd)).(fs), label=L"f\, A_{+,D}(f)")
    plot!(fs, sqrt.(Sₙ_LISA.(fs)), label=L"\sqrt{S_n^\mathrm{LISA}}")
    xaxis!(L"f~\mathrm{[Hz]}", :log)
    yaxis!(:log)
end

function plot_Φ()
    m₁_ = 1e4 * MSun
    m₂_ = 1.0 * MSun
    γₛ_ = 7/3
    ρₛ_ = 200. * MSun / pc^3
    vb = make_vacuum_binary(m₁_, m₂_)
    sd = make_static_dress(m₁_, m₂_, ρₛ_, γₛ_)
    dd = make_dynamic_dress(m₁_, m₂_, ρₛ_, γₛ_)

    f_c = f_isco(m₁_)
    fs = geomspace(1e-5, f_c * 0.999, 1000)
    
    plot(fs, (f -> Φ_to_c(f, f_c, vb)).(fs), label="V")
    plot!(fs, (f -> Φ_to_c(f, f_c, vb) - Φ_to_c(f, f_c, sd)).(fs), legend=:bottomleft, label="V - S")
    plot!(fs, (f -> Φ_to_c(f, f_c, vb) - Φ_to_c(f, f_c, dd)).(fs), legend=:bottomleft, label="V - D")

    xaxis!(L"f~\mathrm{[Hz]}", :log)
    yaxis!(L"\mathrm{Phase~[rad]}", :log, (1e-4, 1e12))
end

function plot_t_to_c()
    m₁_ = 1e4 * MSun
    m₂_ = 1.0 * MSun
    γₛ_ = 7/3
    ρₛ_ = 200. * MSun / pc^3
    vb = make_vacuum_binary(m₁_, m₂_)
    sd = make_static_dress(m₁_, m₂_, ρₛ_, γₛ_)
    dd = make_dynamic_dress(m₁_, m₂_, ρₛ_, γₛ_)

    f_c = f_isco(m₁_)
    fs = geomspace(1e-5, f_c * 0.999, 1000)
    
    plot(fs, (f -> t_to_c(f, f_c, vb)).(fs), label="V")
    plot!(fs, (f -> t_to_c(f, f_c, vb) - t_to_c(f, f_c, sd)).(fs), legend=:bottomleft, label="V - S")
    plot!(fs, (f -> t_to_c(f, f_c, vb) - t_to_c(f, f_c, dd)).(fs), legend=:bottomleft, label="V - D")
    xaxis!(L"f~\mathrm{[Hz]}", :log)
    yaxis!(L"t~[\mathrm{s}]", :log)
end