"""
Plotting functions.
"""

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
    elseif p == :Φ_c
        return "{\\Phi_c}"
    elseif p == :t̃_c
        return "{\\tilde{t}_c}"
    elseif p == :dₗ_ι
        return "{d\\iota}"
    else
        throw(ArgumentError("unrecognized parameter name"))
    end
end

"""
Indicates position of system analyzed by Eda+ 2014 and the PBH dress from Adamek+ 2019
in the (ρₛ, γₛ) plane.
"""
function mark_benchmarks(ax, m₁, m₂)
    if m₁ != 1e3 * MSun || m₂ != 1 * MSun
        println("BH masses differ from Eda+ 2014...")
    end

    ax.scatter(226., 7/3, marker="*", color="b", edgecolor="k", lw=0.5, s=100)

    ρₛ_pbh = 17984.2  # this is `(ρ_eq / 2) * t_eq^(3/2) * (2 * Gₙ)^(3/4)`
    ax.scatter(ρₛ_pbh, 9/4, marker="*", color="r", edgecolor="k", lw=0.5, s=100)
end

function plot_snr(T::Type{DT}, m₁, m₂, ρₛs, γₛs, fₕ, f_c, t_to_merger) where DT <: Binary
    n_ρₛ = length(ρₛs)
    n_γₛ = length(γₛs)
    snrs = zeros(n_γₛ, n_ρₛ)

    for i in 1:n_ρₛ, j in 1:n_γₛ
        system = make_dress(T, m₁, m₂, ρₛs[i], γₛs[j])
        fₗ = find_zero(f -> t_to_c(f, f_c, system) - t_to_merger, (0.00001 * f_c, f_c))
        snrs[j, i] = calculate_SNR(system, fₗ, fₕ, f_c; N_nodes=1000)  # remember to transpose!
    end

    fig, ax = PyPlot.subplots(1, 1, figsize=(4, 3.2))

    cs_f = ax.contourf(ρₛs / (MSun/pc^3), γₛs, snrs, cmap=T == StaticDress ? "plasma" : "viridis")
    PyPlot.colorbar(cs_f, ax=ax)

    mark_benchmarks(ax, m₁, m₂)

    ax.set_xscale("log")
    ax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])  # tick locators not supported...
    ax.set_xlabel(L"\rho_s~[\mathrm{M}_\odot / \mathrm{pc}^3]")
    ax.set_ylabel(L"\gamma_s")
    ax.set_title(T == StaticDress ? "Static dress" : "Dynamic dress")
    fig.tight_layout()

    return fig
end

function plot_ΔN_cycles(T::Type{DT}, m₁, m₂, ρₛs, γₛs, f_c, t_to_merger) where DT <: Binary
    n_ρₛ = length(ρₛs)
    n_γₛ = length(γₛs)
    ΔN_cycless = zeros(n_γₛ, n_ρₛ)

    vb = make_vacuum_binary(m₁, m₂)

    for i in 1:n_ρₛ, j in 1:n_γₛ
        system = make_dress(T, m₁, m₂, ρₛs[i], γₛs[j])
        fₗ = find_zero(f -> t_to_c(f, f_c, system) - t_to_merger, (0.0001 * f_c, f_c))
        # Remember to transpose!
        ΔN_cycless[j, i] = (Φ_to_c(fₗ, f_c, vb) - Φ_to_c(fₗ, f_c, system)) / (2 * π)
    end

    fig, ax = PyPlot.subplots(1, 1, figsize=(4, 3.2))

    # PyPlot.jl doesn't have locators, so unfortunately we must manually set
    # tick positions and labels...
    if T == DynamicDress
        levels = log10.(geomspace(1e1, 1e6, 11))
        cmap = "viridis"
        title = "Dynamic dress"
    elseif T == StaticDress
        levels = log10.(geomspace(1e3, 1e9, 13))
        cmap = "plasma"
        title = "Static dress"
    end

    cs_f = ax.contourf(ρₛs / (MSun/pc^3), γₛs, log10.(ΔN_cycless), cmap=cmap, levels=levels)
    cbar = PyPlot.colorbar(cs_f, ax=ax)
    cbar.ax.set_yticklabels([latexstring("10^{$(convert(Int, l))}") for l in levels[1:2:end]])

    mark_benchmarks(ax, m₁, m₂)
    
    ax.set_xscale("log")
    ax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])  # tick locators not supported...
    ax.set_xlabel(L"\rho_s~[\mathrm{M}_\odot / \mathrm{pc}^3]")
    ax.set_ylabel(L"\gamma_s")
    ax.set_title(title)
    fig.tight_layout()

    return fig
end

function plot_errs(T::Type{DT}, m₁, m₂, ρₛs, γₛs, fₕ, f_c, t_to_merger) where DT <: Binary
    n_ρₛ = length(ρₛs)
    n_γₛ = length(γₛs)
    n_intrinsic = length(intrinsics(T))
    errs = zeros(n_γₛ, n_ρₛ, n_intrinsic)

    ProgressMeter.@showprogress for i in 1:n_ρₛ, j in 1:n_γₛ
        system = make_dress(T, m₁, m₂, ρₛs[i], γₛs[j])
        fₗ = find_zero(f -> t_to_c(f, f_c, system) - t_to_merger, (0.0001 * f_c, f_c))
        errs[j, i, :] = fim_errs(fₗ, fₕ, f_c, system)[1:n_intrinsic]
    end

    fig, axes = PyPlot.subplots(2, 2, figsize=(2 * 4, 2 * 3.4))
    axes = vec(axes)
    cmap = T == StaticDress ? "plasma" : "viridis"

    for i in 1:size(errs, 3)
        ax = axes[i]

        # Plot
        cs_f = ax.contourf(ρₛs / (MSun/pc^3), γₛs, log10.(errs[:, :, i]), cmap=cmap)
        PyPlot.colorbar(cs_f, ax=ax)

        mark_benchmarks(ax, m₁, m₂)

        # Format
        ax.set_xscale("log")
        ax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])  # tick locators not supported...
        ax.set_xlabel(L"\rho_s~[\mathrm{M}_\odot / \mathrm{pc}^3]", fontsize=16)
        ax.set_ylabel(L"\gamma_s", fontsize=16)
        lp = latex_name(intrinsics(T)[i])
        ax.set_title(latexstring("\\log_{10} \\, \\sigma_{", lp, "} / ", lp), fontsize=16)
    end

    # For static case
    for i in size(errs, 3)+1:length(axes)
        axes[i].set_axis_off()
    end

    fig.suptitle(T == StaticDress ? "Static dress" : "Dynamic dress", fontsize=18)
    fig.tight_layout()
    PyPlot.subplots_adjust(top=0.91)

    return fig
end

"""
Remake and save SNR plots.
"""
function make_snr_plots(m₁, m₂, ρₛs, γₛs, fₕ, f_c, t_to_merger; suffix="")
    println("Plotting SNRs...")
    sd_fig = plot_snr(StaticDress, m₁, m₂, ρₛs, γₛs, fₕ, f_c, t_to_merger)
    sd_fig.savefig("figures/sd_snr$(suffix).pdf")

    dd_fig = plot_snr(DynamicDress, m₁, m₂, ρₛs, γₛs, fₕ, f_c, t_to_merger)
    dd_fig.savefig("figures/dd_snr$(suffix).pdf")
    
    return sd_fig, dd_fig
end

"""
Remake and save ΔN_cycles plots.
"""
function make_ΔN_cycles_plots(m₁, m₂, ρₛs, γₛs, f_c, t_to_merger; suffix="")
    println("\nPlotting ΔN_cycles...")
    sd_fig = plot_ΔN_cycles(StaticDress, m₁, m₂, ρₛs, γₛs, f_c, t_to_merger)
    sd_fig.savefig("figures/sd_dN_cycles$(suffix).pdf")

    dd_fig = plot_ΔN_cycles(DynamicDress, m₁, m₂, ρₛs, γₛs, f_c, t_to_merger)
    dd_fig.savefig("figures/dd_dN_cycles$(suffix).pdf")
    
    return sd_fig, dd_fig
end

"""
Remake and save Fisher error plots.
"""
function make_err_plots(m₁, m₂, ρₛs, γₛs, fₕ, f_c, t_to_merger; suffix="")
    println("\nPlotting errors...")
    sd_fig = plot_errs(StaticDress, m₁, m₂, ρₛs, γₛs, fₕ, f_c, t_to_merger)
    sd_fig.savefig("figures/sd_post$(suffix).pdf")

    dd_fig = plot_errs(DynamicDress, m₁, m₂, ρₛs, γₛs, fₕ, f_c, t_to_merger)
    dd_fig.savefig("figures/dd_post$(suffix).pdf")
    return dd_fig
    
    return sd_fig, dd_fig
end

"""
Call to remake and save figures for the paper.
"""
function make_all_plots(m₁, m₂, ρₛs, γₛs, fₕ, f_c, t_to_merger; suffix="")
    make_snr_plots(m₁, m₂, ρₛs, γₛs, fₕ, f_c, t_to_merger; suffix)
    make_ΔN_cycles_plots(m₁, m₂, ρₛs, γₛs, f_c, t_to_merger; suffix)
    make_err_plots(m₁, m₂, ρₛs, γₛs, fₕ, f_c, t_to_merger; suffix)

    PyPlot.close("all")
end