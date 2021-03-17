# Constants in SI units
# For some reason the code gets EXTREMELY slow if these are const, which is confusing...
const Gₙ = 6.67408e-11  # m^3 s^-2 kg^-1
const c = 299792458.  # m/s
const MSun = 1.98855e30  # kg
const pc = 3.08567758149137e16 # m
const yr = 365.25 * 24 * 3600  # s

geomspace(x_start, x_end, n::Int=20) = [10^y for y in range(log10(x_start), log10(x_end), length=n)]

# LISA noise curve
Sₙ_LISA(f) = 1 / f^14 * 1.80654e-17 * (0.000606151 + f^2) * (3.6864e-76 + 3.6e-37 * f^8 + 2.25e-30 * f^10 + 8.66941e-21 * f^14)

"""
1D trapz for matrix-values functions
"""
function trapz_1d_mat(xs, ys)
    # @assert length(xs) == length(ys)
    # result = zeros(size(ys[1]))
    # for i in range(1, stop=length(xs) - 1)
    #     result .+= (xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) ./ 2
    # end
    # return result

    @assert length(xs) == length(ys)
    return sum((xs[2:end] .- xs[1:end - 1]) .* (ys[2:end] .+ ys[1:end - 1]) ./ 2)
end