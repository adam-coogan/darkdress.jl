module Utils

export Gₙ, c, s_to_yr, yr_to_s, tᵤ, ħ, pc_to_km, km_to_pc, GeV_to_MSun, MSun_to_GeV, GeV_to_g, g_to_GeV, MSun_to_g

const Gₙ = 4.3021937e-3  # (km/s)^2 pc/M_sun
const c = 2.9979e5  # km/s
const s_to_yr = 1. / (365. * 24 * 60 * 60)
const yr_to_s = 1 / s_to_yr
const tᵤ = 13.8e9 * yr_to_s  # age of universe (s)
const ħ = 6.582e-16 * 1e-9  # GeV * s
const pc_to_km = 3.085677581e13
const km_to_pc = 1 / pc_to_km
const GeV_to_MSun = 8.97e-58
const MSun_to_GeV = 1 / GeV_to_MSun
const GeV_to_g = 1.783e-24
const g_to_GeV = 1 / GeV_to_g
const MSun_to_g = MSun_to_GeV * GeV_to_g

export geomspace, logspace, heaviside

geomspace(x_start, x_end, n::Int=20) = [10^y for y in range(log10(x_start), log10(x_end), length=n)]
logspace(l_start, l_end, n::Int=20) = [10^y for y in range(l_start, l_end, length=n)]

end  # module Utils
