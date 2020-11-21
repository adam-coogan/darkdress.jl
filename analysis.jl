include("Utils.jl")
include("Systems.jl")
include("Detectors.jl")

using Zygote
using Revise
using Trapz
using .Utils: geomspace
using .Detectors: Detector, LISA, snr, Sₙ
using .Systems: Binary, VacuumBinary, amp₊, Ψ, length, t_to_c

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
    amp_fn(d, ι, Φ_c, t_c, system) = amp₊(f, d, ι, system)
    ∂amp₊ = collect(gradient(amp_fn, d, ι, Φ_c, t_c, system))
    Ψ_fn(d, ι, Φ_c, t_c, system) = Ψ(f, d, Φ_c, t_c, system)
    ∂Ψ = collect(gradient(Ψ_fn, d, ι, Φ_c, t_c, system))

    for i in range(1, stop=4 + 1)
        if ∂amp₊[i] === nothing
            ∂amp₊[i] = 0.
        end
        if ∂Ψ[i] === nothing
            ∂Ψ[i] = 0.
        end
    end

    ∂amp₊ = collect(Iterators.flatten(values.(∂amp₊)))
    ∂Ψ = collect(Iterators.flatten(values.(∂Ψ)))
    4 * (∂amp₊ * ∂amp₊' + amp₊(f, d, ι, system)^2 * ∂Ψ * ∂Ψ')
end

"""
Compute the Fisher information matrix.
"""
function fim(fₗ, fₕ, d, ι, Φ_c, t_c, detector::Detector, system::Binary, n=1000)
    fs = geomspace(fₗ, fₕ, n)
    nums = map(f -> fim_integrand_num(f, d, ι, Φ_c, t_c, system), fs)
    Sₙs = map(f -> Sₙ(f, detector), fs)
    trapz_1d_mat(fs, nums ./ Sₙs)
end

"""
Get the Fisher information matrix estimates for parameter measurement uncertainties.
"""
function fim_uncertainties(fₗ, fₕ, d, ι, Φ_c, t_c, detector::Detector, system::Binary, n=1000)
    Γ = fim(fₗ, fₕ, d, ι, Φ_c, t_c, detector, system, n)
    Γ = (Γ .+ Γ') ./ 2

    Σ = inv(Γ)
    # println(Γ * Σ)  # should be the identity...
    # println(eigvals(Γ))  # should all be positive...
    # println([Σ[i, i] for i in 1:size(Σ)[1]])
    sqrt.([Σ[i, i] for i in 1:size(Σ)[1]])
end


# Playground
system = VacuumBinary(1e3, 1.)
lisa = LISA(false)
snr(1e-2, 1., 500e6, 0., lisa, system)
fim_uncertainties(1e-2, 1, 500e6, π/3, 0., 0., lisa, system, 2000)
