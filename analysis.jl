# using Zygote  # slow
using ForwardDiff
using LinearAlgebra: eigvals, Diagonal
using QuadGK
using Base
using StaticArrays

include("utils.jl")

# Factors for rescaling some parameter derivatives to relative ones (d/dx d/dlog(x))
# TODO: automate with `intrinsics`?
rescalings(sd::StaticDress) = [sd.γₛ, sd.c_f, sd.ℳ, 1., 1., 1.]
rescalings(dd::DynamicDress) = [dd.γₛ, dd.c_f, dd.ℳ, dd.q, 1., 1., 1.]
rescalings(system::Binary) = ones(length(system))  # fallback: don't use relative errors

"""
Numerator of integrand in Fisher information matrix.
"""
function fim_integrand_num(f, f_c, system::B) where B <: Binary
    # Get higher-order constructor to make binaries where the type parameter is
    # a dual number
    Constructor = Base.typename(B).wrapper

    # Zygote returns `nothing` for parameters that don't have gradients. These must be
    # replaced with zeros. This is not the case for ForwardDiff.
    # ∂amp₊ = collect(values(gradient(s -> amp₊(f, s), system)[1]))  # zygote
    ∂amp₊ = ForwardDiff.gradient(
        vals -> amp₊(f, Constructor(vals...)), convert(Array{Float64,1}, system)
    )
    ∂amp₊[∂amp₊ .=== nothing] .= 0.

    # ∂Ψ = collect(values(gradient(s -> Ψ(f, f_c, s), system)[1]))  # zygote
    ∂Ψ = ForwardDiff.gradient(
        vals -> Ψ(f, f_c, Constructor(vals...)), convert(Array{Float64,1}, system)
    )
    ∂Ψ[∂Ψ .=== nothing] .= 0.

    # Convert to log derivatives for intrinsic parameters
    scales = rescalings(system)
    ∂amp₊ .*= scales
    ∂Ψ .*= scales
    
    return 4 * (∂amp₊ * ∂amp₊' + amp₊(f, system)^2 * ∂Ψ * ∂Ψ')
end

"""
Compute Fisher information matrix.
"""
function fim(fₗ, fₕ, f_c, system)
    integrand(f) = SMatrix{length(system), length(system), Float64}(
        fim_integrand_num(f, f_c, system) / Sₙ_LISA(f)
    )
    Γ_raw = quadgk(integrand, fₗ, fₕ, rtol=1e-10)[1]
    # Conversion to `Array` required to work with `eigenvals`
    return Array{Float64}(1/2 * (Γ_raw + Γ_raw'))  # symmetrize, just in case
end

"""
Estimate covariance matrix from Fisher information matrix.
"""
function fim_cov(fₗ, fₕ, f_c, system)  # where T <: Binary
    Γ = fim(fₗ, fₕ, f_c, system)

    # Improve stability of inversion
    scales = sqrt(inv(Diagonal(Γ)))
    Γr = scales * Γ * scales

    @assert all(eigvals(Γr) .> 0) eigvals(Γr)

    return scales * inv(Γr) * scales
end

"""
Estimate 1D uncertainties from Fisher information matrix.
"""
function fim_errs(fₗ, fₕ, f_c, system)  # where T <: Binary
    Σ = fim_cov(fₗ, fₕ, f_c, system)
    return sqrt.([Σ[i, i] for i in 1:size(Σ)[1]])
end

using FastGaussQuadrature
using Trapz
using LinearAlgebra

# These take fc so they can work with VacuumBinary and StaticDress as well as
# DynamicDress.

function calculate_SNR(system, fₗ, fₕ, fc; N_nodes=1000, USE_GAUSS=true, LOG_SPACE=true)
    fh = min(fₕ, fc)

    modh_integrand(f) = 4 * amp₊(f, system)^2 / Sₙ_LISA(f)

    #Use fixed Gauss quadrature, or Trapz
    if USE_GAUSS
        nodes, weights = gausslegendre(N_nodes)

        if LOG_SPACE
            log_fₗ = log(fₗ)
            log_fh = log(fh)
            log_f̄ = (log_fh + log_fₗ)/2
            log_Δf = log_fh - log_fₗ
            log_f_vals = log_f̄ .+ nodes * log_Δf / 2
            f_vals = exp.(log_f_vals)
            modh = log_Δf/2 * dot(weights, modh_integrand.(f_vals) .* f_vals)
        else
            f̄ = (fh + fₗ)/2
            Δf = fh - fₗ
            f_vals = f̄ .+ nodes * Δf / 2
            modh = (Δf / 2 * dot(weights, modh_integrand.(f_vals)))
        end
    else  # trapz
        f_vals = LOG_SPACE ? geomspace(fₗ, fh, N_nodes) : range(fₗ, fh, length=N_nodes)
        modh = trapz(f_vals, modh_integrand.(f_vals))
    end

    return sqrt(modh)
end

function calculate_match_unnorm(
    system_1, system_2, fₗ, fₕ, fc_1, fc_2; N_nodes=15000, USE_GAUSS=true, LOG_SPACE=true
)
    # Cut integral off at ISCO of first system to merge
    fh = min(fₕ, fc_1, fc_2)

    # TODO: should it be fc_1 or fc_2 here?
    calcΔΨ(f) = Ψ(f, fc_1, system_1) - Ψ(f, fc_2, system_2)
    calcAmp(f) = 4 * amp₊(f, system_1) * amp₊(f, system_2) / Sₙ_LISA(f)

    if USE_GAUSS
        nodes, weights = gausslegendre(N_nodes)

        if LOG_SPACE
            log_fₗ = log(fₗ)
            log_fh = log(fh)
            log_f̄ = (log_fh + log_fₗ)/2
            log_Δf = log_fh - log_fₗ
            log_f_vals = log_f̄ .+ nodes * log_Δf / 2
            f_vals = exp.(log_f_vals)
        else
            f̄ = (fh + fₗ)/2
            Δf = fh - fₗ
            f_vals = f̄ .+ nodes * Δf / 2
        end
    else
        f_vals = LOG_SPACE ? geomspace(fₗ, fh, N_nodes) : range(fₗ, fh, length=N_nodes)
    end

    ΔΨ = calcΔΨ.(f_vals)
    Amp = calcAmp.(f_vals)
    integs_re = cos.(ΔΨ) .* Amp
    integs_im = sin.(ΔΨ) .* Amp

    if USE_GAUSS
        if LOG_SPACE
            x_re = log_Δf/2 * dot(weights, integs_re .* f_vals)
            x_im = log_Δf/2 * dot(weights, integs_im .* f_vals)
        else
            x_re = Δf/2 * dot(weights, integs_re)
            x_im = Δf/2 * dot(weights, integs_im)
        end
    else
        x_re = trapz(f_vals, integs_re)
        x_im = trapz(f_vals, integs_im)
    end

    return x_re, x_im
end

function calculate_loglike(
    system_h, system_d, fₗ, fₕ, fc_h, fc_d;
    optimize_dₗ_ι=true, optimize_Φ_c=true, N_nodes=1000, USE_GAUSS=true, LOG_SPACE=true
)
    # Calculate real and imaginary parts of the overlap <d|h>
    dh_re, dh_im = calculate_match_unnorm(
        system_d, system_h, fₗ, fₕ, fc_d, fc_h; N_nodes, USE_GAUSS, LOG_SPACE
    )

    # Calculate the overlap <h|h>
    hh = calculate_SNR(system_h, fₗ, fₕ, fc_h; N_nodes, USE_GAUSS, LOG_SPACE)^2
    

    # Optimise over phase at coalescence and dₗ_ι
    dh = optimize_Φ_c ? sqrt(dh_re^2 + dh_im^2) : dh_re
    r = optimize_dₗ_ι ? dh / hh : 1.

    return r * dh - 1/2 * r^2 * hh    
end

function like_wrapper(x, dd_ref::B, fₗ, fₕ; N_nodes=3000, USE_GAUSS=true, LOG_SPACE=true) where B <: Binary

    α0 = -56.3888135025341

    γₛ  = exp(x[1])
    c_f = exp(x[2])
    ℳ   = exp(x[3])*MSun 
    q   = exp(x[4]) 
    t̃_c = x[5]

    Φ_c = 0.0

    #Set some parameter boundaries
    if (γₛ < 0)
        return -1e30
    end
    #if (γₛ > 3)
    #    return -1e30
    #end
    if (c_f < 0)
        return -1e30
    end
    if (q < 0)
        return -1e30
    end
    if (q > 1)
        return -1e30
    end

    # Get higher-order constructor to make binaries where the type parameter is
    # a dual number
    Constructor = Base.typename(B).wrapper

    #Construct alternative system
    vals = [γₛ, c_f, ℳ, q, Φ_c, t̃_c, α0]
    dd_alt = Constructor(vals...)

    fc_ref = f_isco(m₁(dd_ref.ℳ, dd_ref.q))
    fc_alt = f_isco(m₁(ℳ, q))

    return calculate_loglike(dd_alt, dd_ref, fₗ, fₕ, fc_alt, fc_ref, N_nodes=N_nodes, USE_GAUSS=USE_GAUSS, LOG_SPACE=LOG_SPACE)
end