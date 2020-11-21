module Detectors

using Trapz
# include("systems.jl")
using ..Systems: Binary, amp₊, ampₓ
# include("utils.jl")
using ..Utils: geomspace

export LISA, snr, Sₙ

abstract type Detector end

function snr(fₗ, fₕ, d, ι, detector::Detector, system::Binary, n::Int=5000)
    fs = geomspace(fₗ, fₕ, n)
    amp²(f) = (F₊(detector) * amp₊(f, d, ι, system))^2 + (Fₓ(detector) * ampₓ(f, d, ι, system))^2
    amp²s = map(amp², fs)
    Sₙs = map(f -> Sₙ(f, detector), fs)
    sqrt.(4 * trapz(fs, amp²s ./ Sₙs))
end

struct LISA <: Detector
    wd::Bool
end

F₊(detector::LISA) = 1.
Fₓ(detector::LISA) = 0.

function Sₙ(f, detector::LISA)
    cSI = 299792458.0

    Larm = 2.5e9
    fstar = 0.5 * cSI / Larm / π

    POMS = 2.25e-22 * (1.0 + 1.6e-11 / f^4)
    Pacc = 9e-30 * (1.0 + 1.6e-7 / f / f) * (1.0 + f^4 / 4.096e-9)
    CosTerm = cos(f / fstar)
    Pnf = (
        POMS + (1.0 + CosTerm * CosTerm) * Pacc / (8.0 * π^4 * f^4)
    ) / Larm^2

    Rinvf = 10.0 * (1.0 + 0.6 * f^2 / fstar^2) / 3.0

    Scf = (
        9e-45
        * f^(-7.0 / 3.0)
        * exp(-(f^0.138) - 221.0 * f * sin(521.0 * f))
        * (1 + tanh(1680.0 * (1.13e-3 - f)))
    )

    if detector.wd
        return Pnf * Rinvf + Scf
    else
        return Pnf * Rinvf
    end
end

end  # module Detectors
