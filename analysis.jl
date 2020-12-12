using Zygote  # slow
using ForwardDiff
using LinearAlgebra: eigvals, Diagonal
using QuadGK
using Base
using StaticArrays

# Factors for rescaling some parameter derivatives to relative ones (d/dx d/dlog(x))
# TODO: automate with `intrinsics`?
rescalings(sd::StaticDress) = [sd.γₛ, sd.c_f, sd.ℳ, 1., 1., 1.]
rescalings(dd::DynamicDress) = [dd.γₛ, dd.c_f, dd.ℳ, dd.q, 1., 1., 1.]
rescalings(system::Binary) = ones(length(system))  # fallback: don't use relative errors

"""
Numerator of integrand in Fisher information matrix.
"""
function fim_integrand_num(f, f_c, system::T) where T <: Binary
    # Zygote returns `nothing` for parameters that don't have gradients. These must be
    # replaced with zeros. This is not the case for ForwardDiff.
    # ∂amp₊ = collect(values(gradient(s -> amp₊(f, s), system)[1]))  # zygote
    ∂amp₊ = ForwardDiff.gradient(vals -> amp₊(f, T(vals...)), collect(values(system)))
    ∂amp₊[∂amp₊ .=== nothing] .= 0.
    ∂amp₊ = convert(Array{Float64}, ∂amp₊)

    # ∂Ψ = collect(values(gradient(s -> Ψ(f, f_c, s), system)[1]))  # zygote
    ∂Ψ = ForwardDiff.gradient(vals -> Ψ(f, f_c, T(vals...)), collect(values(system)))
    ∂Ψ[∂Ψ .=== nothing] .= 0.
    ∂Ψ = convert(Array{Float64}, ∂Ψ)

    # Convert to log derivatives for intrinsic parameters
    scales = rescalings(system)
    ∂amp₊ .*= scales
    ∂Ψ .*= scales
    
    return 4 * (∂amp₊ * ∂amp₊' + amp₊(f, system)^2 * ∂Ψ * ∂Ψ')
end

"""
Compute Fisher information matrix.
"""
function fim(fₗ, fₕ, f_c, system::T) where T <: Binary
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
function fim_cov(fₗ, fₕ, f_c, system::T) where T <: Binary
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
function fim_errs(fₗ, fₕ, f_c, system::T) where T <: Binary
    Σ = fim_cov(fₗ, fₕ, f_c, system)
    return sqrt.([Σ[i, i] for i in 1:size(Σ)[1]])
end

"""
Signal-to-noise ratio.
"""
function snr(fₗ, fₕ, system::T) where T <: Binary
    integrand(f) = amp₊(f, system)^2 / Sₙ_LISA(f)
    return √(4 * quadgk(integrand, fₗ, fₕ, rtol=1e-7)[1])
end