include("Utils.jl")
include("Systems.jl")
include("Detectors.jl")

using Zygote
using Revise
using Trapz
using LinearAlgebra: eigvals, cholesky
using .Utils: geomspace
using .Detectors: Detector, LISA, snr, Sₙ, F₊, Fₓ
using .Systems: Binary, VacuumBinary, DynamicDress, amp₊, Ψ, length, t_to_c, ℳ, Φ_to_c

"""
Trapezoidal integration where `ys` is an array of matrices (2D arrays).
"""
function trapz_1d_mat(xs, ys)
    @assert length(xs) == length(ys)
    result = zeros(size(ys[1]))
    for i in range(1, stop=length(xs) - 1)
        result .+= (xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) ./ 2
    end
    return result
end

"""
Computes the numerator of the integrand of the Fisher information matrix:

    ∂ᵢa ∂ⱼa + a² ∂ᵢΨ ∂ⱼΨ

where i and j index the extrinsic and intrinsic parameters of the system:

    d, ι, Φ_c, t_c, system...
"""
function fim_integrand_num(f, d, ι, Φ_c, t_c, system::T) where T <: Binary
    # amp_fn(d, ι, Φ_c, t_c, system) = amp₊(f, d, ι, system)
    # ∂amp₊ = collect(gradient(amp_fn, d, ι, Φ_c, t_c, system))
    # Ψ_fn(d, ι, Φ_c, t_c, system) = Ψ(f, d, Φ_c, t_c, system)
    # ∂Ψ = collect(gradient(Ψ_fn, d, ι, Φ_c, t_c, system))

    amp_fn(d, ι, Φ_c, system) = amp₊(f, d, ι, system)
    ∂amp₊ = collect(gradient(amp_fn, d, ι, Φ_c, system))
    Ψ_fn(d, ι, Φ_c, system) = Ψ(f, d, Φ_c, t_c, system)
    ∂Ψ = collect(gradient(Ψ_fn, d, ι, Φ_c, system))

    n_params = length(∂amp₊) - 1  # ignoring t_c
    for i in range(1, stop=n_params)
        if ∂amp₊[i] === nothing
            ∂amp₊[i] = 0.
        end
        if ∂Ψ[i] === nothing
            ∂Ψ[i] = 0.
        end
    end

    ∂amp₊ = collect(Iterators.flatten(values.(∂amp₊)))
    ∂Ψ = collect(Iterators.flatten(values.(∂Ψ)))
    # ∂ᵢ → ∂/∂log(θᵢ)
    # param_vals = [system.m₁, system.m₂]
    # ∂amp₊ .*= param_vals
    # ∂Ψ .*= param_vals
    return 4 * (∂amp₊ * ∂amp₊' + amp₊(f, d, ι, system)^2 * ∂Ψ * ∂Ψ')
end

"""
Compute the Fisher information matrix.
"""
function fim(fₗ, fₕ, d, ι, Φ_c, t_c, detector::Detector, system::Binary, n=5000)
    fs = geomspace(fₗ, fₕ, n)
    nums = map(f -> fim_integrand_num(f, d, ι, Φ_c, t_c, system), fs)
    Sₙs = map(f -> Sₙ(f, detector), fs)
    return trapz_1d_mat(fs, nums ./ Sₙs)
end

"""
Get the Fisher information matrix estimates for parameter measurement uncertainties.
"""
function fim_uncertainties(fₗ, fₕ, d, ι, Φ_c, t_c, detector::Detector, system::Binary, n=1000)
    Γ = fim(fₗ, fₕ, d, ι, Φ_c, t_c, detector, system, n)
    Γ = (Γ .+ Γ') ./ 2

    @assert all(eigvals(Γ) .> 0) eigvals(Γ)
    # println("evs: ", eigvals(Γ))
    # println("")
    
    # L = cholesky(Γ).L
    # L⁻¹ = inv(L)
    # Σ = L⁻¹' * L⁻¹
    # println("Σ * Γ")
    # for i in 1:size(Σ * Γ)[1]
    #     println((L⁻¹' * ((L⁻¹ * L) * L'))[i, :])
    # end

    Σ = inv(Γ)
    # println("Σ * Γ")
    # for i in 1:size(Σ * Γ)[1]
    #     println((Γ * Σ)[i, :])
    # end
    # println("")

    return sqrt.([Σ[i, i] for i in 1:size(Σ)[1]])
end

# Playground
system = VacuumBinary(1e3, 1.)
lisa = LISA(false)
snr(1e-2, 1., 500e6, 0., lisa, system)
fim_uncertainties(1e-2, 1, 500e6, π/3, 0., 0., lisa, system, 2000)

dd = DynamicDress(1e3, 1., 200., 7/3.)
snr(1e-2, 1., 500e6, 0., lisa, dd)
fim_uncertainties(1e-2, 1, 500e6, π/3, 0., 0., lisa, dd, 2000)



# # "Bad terms"
# using HypergeometricFunctions: _₂F₁
# using SpecialFunctions: gamma
# B(b, x) = gamma(1 - b) * gamma(1 + b) - _₂F₁(1, -b, 1 - b, -x) / x^b

# using Plots

# function plot_stuff()
#     bs = range(1.6, 3.833, step=0.5)
#     log10_xs = range(-5, 5, length=100)

#     Plots.plot(log10_xs, -B.(bs[1], 10 .^ log10_xs), label="b = $(bs[1])")
#     for b in bs[2:end]
#         Plots.plot!(log10_xs, -B.(b, 10 .^ log10_xs), label="b = $b")
#     end
#     xaxis!("log_10(x)", [-5, 5])
#     yaxis!("-B(b, x)", :log10, [1e-20, 1e20])
# end