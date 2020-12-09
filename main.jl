include("utils.jl")
include("binary.jl")
include("analysis.jl")  # needs types from binary.jl to be loaded

#################
# Static checks #
#################
sd = make_static_dress(1e3 * MSun, 1. * MSun, 226. * MSun/pc^3, 7/3)
fₕ = f_c = f_isco(1e3 * MSun)
fₗ = f_of_t_to_c(5 * yr, f_c, sd)
snr(fₗ, fₕ, sd, 1000)  # agrees with David's notebook
println(log10(fim_errs(fₗ, fₕ, f_c, sd, 100)[1]))
println(log10(fim_errs(fₗ, fₕ, f_c, sd, 100)[2]))
println(log10(fim_errs(fₗ, fₕ, f_c, sd, 100)[3]))

##################
# Dynamic checks #
##################
dd = make_dynamic_dress(1e3 * MSun, 1. * MSun, 226. * MSun/pc^3, 7/3)
fₕ = f_c = f_isco(1e3 * MSun)
fₗ = f_of_t_to_c(5 * yr, f_c, dd)
snr(fₗ, fₕ, dd, 1000)  # agrees with David's notebook
println(log10(fim_errs(fₗ, fₕ, f_c, dd, 100)[1]))
println(log10(fim_errs(fₗ, fₕ, f_c, dd, 100)[2]))
println(log10(fim_errs(fₗ, fₕ, f_c, dd, 100)[3]))
println(log10(fim_errs(fₗ, fₕ, f_c, dd, 100)[4]))


############
# Plotting #
############
using Plots
using LaTeXStrings

function plot_static_errs(m₁, m₂)
    fₕ = f_c = f_isco(m₁)
    t_c = 5.0 * yr
    n_intrinsic = length(intrinsics(StaticDress))

    function fn(log10_ρₛ, γₛ)
        sd = make_static_dress(m₁, m₂, 10^log10_ρₛ * MSun/pc^3, γₛ)
        fₗ = f_of_t_to_c(t_c, f_c, sd)
        return log10.(fim_errs(fₗ, fₕ, fₕ, sd, 80)[1:n_intrinsic])
    end

    # Apply to grid
    log10_ρₛs = -2:0.4:3
    γₛs = 2.25:0.01:2.4
    uncertainties = zeros(length(γₛs), length(log10_ρₛs), n_intrinsic)
    print(size(uncertainties))
    for (i, log10_ρₛ) in enumerate(log10_ρₛs)
        for (j, γₛ) in enumerate(γₛs)
            uncertainties[j, i, :] = fn(log10_ρₛ, γₛ)
        end
    end
    
    plots = []
    for i in 1:size(uncertainties, 3)
        p = contour(log10_ρₛs, γₛs, uncertainties[:, :, i], fill=true, fillcolor=cgrad(:inferno, rev=true), size=(550, 500))
        xaxis!(L"\log_{10} \rho_s", (minimum(log10_ρₛs), maximum(log10_ρₛs)))
        yaxis!(L"\gamma_s", (minimum(γₛs), maximum(γₛs)))
        title!("$(fieldnames(StaticDress)[i])")
        push!(plots, p)
    end
    plot(plots...)
end

function plot_dynamic_errs(m₁, m₂)
    fₕ = f_c = f_isco(m₁)
    t_c = 5.0 * yr
    n_intrinsic = length(intrinsics(DynamicDress))

    function fn(log10_ρₛ, γₛ)
        sd = make_dynamic_dress(m₁, m₂, 10^log10_ρₛ * MSun/pc^3, γₛ)
        fₗ = find_zero(f -> t_to_c(f, f_c, sd) - t_c, (0.001 * f_c, f_c))
        return log10.(fim_errs(fₗ, fₕ, fₕ, sd, 2000)[1:n_intrinsic])
    end

    # Apply to grid
    log10_ρₛs = -2:0.8:3
    γₛs = 2.25001:0.035:2.5
    uncertainties = zeros(length(γₛs), length(log10_ρₛs), n_intrinsic)
    for (i, log10_ρₛ) in enumerate(log10_ρₛs)
        for (j, γₛ) in enumerate(γₛs)
            uncertainties[j, i, :] = fn(log10_ρₛ, γₛ)
        end
    end
    
    plots = []
    for i in 1:size(uncertainties, 3)
        p = contour(log10_ρₛs, γₛs, uncertainties[:, :, i], fill=true, fillcolor=cgrad(:inferno, rev=true), size=(550, 500))
        xaxis!(L"\log_{10} \rho_s", (minimum(log10_ρₛs), maximum(log10_ρₛs)))
        yaxis!(L"\gamma_s", (minimum(γₛs), maximum(γₛs)))
        title!("$(fieldnames(DynamicDress)[i])")
        push!(plots, p)
    end
    plot(plots...)

    return (log10_ρₛs, γₛs, uncertainties)
end

function plot_snr(m₁, m₂)
    log10_ρₛs = -2:0.2:3
    γₛs = 2.25:0.035:2.5

    fₕ = f_c = f_isco(m₁ * MSun)
    t_c = 5.0 * yr

    function fn(log10_ρₛ, γₛ)
        sd = make_static_dress(m₁, m₂, 10^log10_ρₛ * MSun/pc^3, γₛ)
        fₗ = find_zero(f -> t_to_c(f, f_c, sd) - t_c, (0.001 * f_c, f_c))
        return snr(fₗ, fₕ, sd)
    end

    heatmap(
        log10_ρₛs,
        γₛs,
        fn,
        size=(550, 500)
    )
    xaxis!(L"\log_{10} \rho_s")
    yaxis!(L"\gamma_s")
end

function plot_static_ΔN_cycles(m₁, m₂)
    log10_ρₛs = -2:0.2:3
    γₛs = 2.25:0.035:2.5

    f_c = f_isco(m₁)
    t_c = 5.0 * yr

    function fn(log10_ρₛ, γₛ)
        vb = make_vacuum_binary(m₁, m₂)
        sd = make_static_dress(m₁, m₂, 10^log10_ρₛ * MSun/pc^3, γₛ)
        fₗ = find_zero(f -> t_to_c(f, f_c, sd) - t_c, (0.0001 * f_c, f_c))
        return log10((Φ_to_c(fₗ, f_c, vb) - Φ_to_c(fₗ, f_c, sd)) / (2 * π))
    end

    contour(log10_ρₛs, γₛs, fn, fill=true, fillcolor=cgrad(:inferno, rev=true), size=(550, 500))
    xaxis!(L"\log_{10} \rho_s")
    yaxis!(L"\gamma_s")
    title!(L"\log_{10} \Delta N_\mathrm{cycles}")
end

function plot_dynamic_ΔN_cycles(m₁, m₂)
    log10_ρₛs = -2:0.2:3
    γₛs = 2.25:0.035:2.5

    f_c = f_isco(m₁)
    t_c = 5.0 * yr

    function fn(log10_ρₛ, γₛ)
        vb = make_vacuum_binary(m₁, m₂)
        sd = make_dynamic_dress(m₁, m₂, 10^log10_ρₛ * MSun/pc^3, γₛ)
        fₗ = find_zero(f -> t_to_c(f, f_c, sd) - t_c, (0.00001 * f_c, f_c))
        return log10((Φ_to_c(fₗ, f_c, vb) - Φ_to_c(fₗ, f_c, sd)) / (2 * π))
    end

    contour(log10_ρₛs, γₛs, fn, fill=true, fillcolor=cgrad(:inferno, rev=true), size=(550, 500))
    xaxis!(L"\log_{10} \rho_s")
    yaxis!(L"\gamma_s")
    title!(L"\log_{10} \Delta N_\mathrm{cycles}")
end

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