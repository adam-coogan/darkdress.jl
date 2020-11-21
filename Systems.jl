module Systems

using Zygote
using HypergeometricFunctions
using ..Utils: Gₙ, c, pc_to_km, km_to_pc

# FloatOrAbsArr = Union{AbstractFloat, AbstractArray{AbstractFloat, 1}}

import Base.length
export Binary, VacuumBinary, amp₊, ampₓ, length

abstract type Binary end

struct VacuumBinary <: Binary
    m₁::AbstractFloat
    m₂::AbstractFloat
end

struct DarkDress <: Binary
    m₁::AbstractFloat
    m₂::AbstractFloat
    ρₛ::AbstractFloat
    γₛ::AbstractFloat
end

length(system::VacuumBinary) = 2
length(system::DarkDress) = 4

"""
Chirp mass [Mₒ]
"""
ℳ(system::Binary) = (system.m₁ * system.m₂)^(3/5) / (system.m₁ + system.m₂)^(1/5)

"""
Factor for computing vacuum phase
"""
ψᵥ(system::Binary) = 1 / 16 * (c^3 / (π * Gₙ * pc_to_km * ℳ(system)))^(5/3)

"""
'Intrinsic' part of the strain amplitude [pc].
"""
function h₀(f, system::Binary)
    M = system.m₁ + system.m₂
    ωₛ = (f / 2) * 2 * π  # orbital angular frequency, Hz
    r = (Gₙ * km_to_pc^2 * M / (π * f)^2)^(1 / 3)  # separation, pc
    # Time domain
    amp_td = 4 * Gₙ * system.m₂ * ωₛ^2 * r^2 / c^4 / km_to_pc^2  # 1
    # Fourier domain
    1 / 2 * amp_td * √(2 * π / d²Φ_dt²(f, system))  # s
end

"""
Amplitude of h₊ [1].
"""
amp₊(f, d, ι, system::Binary) = h₀(f, system) * 1 / d * (1 + cos(ι)^2) / 2

"""
Amplitude of hₓ [1].
"""
ampₓ(f, d, ι, system::Binary) = h₀(f, system) * 1 / d * cos(ι)

"""
Phase of waveform in Fourier domain [1].
"""
function Ψ(f, d, Φ_c, t_c, system::Binary)
    t = t_c - t_to_c(f, system)
    Φ = Φ_c - Φ_to_c(f, system)
    2 * π * f * (t + d * pc_to_km / c) - Φ - π / 4
end

h₊(f, d, ι, Φ_c, t_c, system::Binary) = exp(1im * Ψ(f, d, Φ_c, t_c, system)) * amp₊(f, d, ι, system)

hₓ(f, d, ι, Φ_c, t_c, system::Binary) = exp(1im * Ψ(f, d, Φ_c, t_c, system)) * ampₓ(f, d, ι, system)

"""
ISCO radius [pc].
"""
r_isco(system::Binary) = 6 * Gₙ * system.m₁ / c^2

"""
ISCO frequency [Hz].
"""
f_isco(system::Binary) = √(Gₙ * km_to_pc^2 * system.m₁ / r_isco(system)^3) / π

### Subtypes should implement the following functions
"""
Frequency [Hz] at which the system will merge in time t [s].
"""
f_to_c(t, system::VacuumBinary) = (5 * ψᵥ(system) / (π * t))^(3/8) / (2 * √(2))

"""
Phase remaining to coalescence [1].
"""
Φ_to_c(f, system::VacuumBinary) = ψᵥ(system) / f^(5/3)

"""
Time remaining until coalescence [s].
"""
t_to_c(f, system::VacuumBinary) = 5 * ψᵥ(system) / (16 * π * f^(8/3))

"""
Second derivative of phase [Hz^2].

TODO: can probably compute with autograd!
"""
d²Φ_dt²(f, system::VacuumBinary) = 12 * π^2 * f^(11/3) / (5 * ψᵥ(system))

end  # module Systems
