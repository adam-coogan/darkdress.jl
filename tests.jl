# %%
"""
Verifies the code runs.
"""
function test_fim_SNR()
    println("Making sure FIM and SNR functions run.")
    println("Static checks")
    sd = make_dress(StaticDress{Float64}, 1e3 * MSun, 1. * MSun, 226. * MSun / pc^3, 7 / 3)
    println(sd)
    fₕ = f_c = f_isco(1e3 * MSun)
    fₗ = find_zero(f -> t_to_c(f, f_c, sd) - 5 * yr, (0.0001 * f_c, f_c))
    calculate_SNR(sd, fₗ, fₕ, f_c)  # agrees with David's notebook
    println(log10(fim_errs(fₗ, fₕ, f_c, sd)[1]))
    println(log10(fim_errs(fₗ, fₕ, f_c, sd)[2]))
    println(log10(fim_errs(fₗ, fₕ, f_c, sd)[3]))
    println()
    
    println("Dynamic checks")
    dd = make_dress(DynamicDress{Float64}, 1e3 * MSun, 1. * MSun, 226. * MSun / pc^3, 7 / 3)
    println(dd)
    fₕ = f_c = f_isco(1e3 * MSun)
    fₗ = find_zero(f -> t_to_c(f, f_c, dd) - 5 * yr, (0.0001 * f_c, f_c))
    calculate_SNR(dd, fₗ, fₕ, f_c)  # agrees with David's notebook
    println(log10(fim_errs(fₗ, fₕ, f_c, dd)[1]))
    println(log10(fim_errs(fₗ, fₕ, f_c, dd)[2]))
    println(log10(fim_errs(fₗ, fₕ, f_c, dd)[3]))
    println(log10(fim_errs(fₗ, fₕ, f_c, dd)[4])) 
    println("Done.")
end

# %%
function test_calcloglike()
    dd_s = DynamicDress(
        2.3333333333333335, 0.00018806659428775589, 3.151009407916561e31, 0.001, 0.0, 0.0, -56.3888135025341
    )
    dd_h = DynamicDress(
        2.27843666, 7.38694332e-5, 3.1518396596159997e31, 0.000645246955, 0.0, -1.98186232e+02, -56.3888135025341
    )
    fₗ = 0.022607529999065474
    fc_s = f_isco(m₁(dd_s.ℳ, dd_s.q))
    fc_h = f_isco(m₁(dd_h.ℳ, dd_h.q))
    
    snr_s = calculate_SNR(dd_s, fₗ, fc_s, fc_s)
    snr_h = calculate_SNR(dd_s, fₗ, fc_h, fc_h)
    ll_ss = calculate_loglike(dd_s, dd_s, fₗ, fc_s, fc_s, fc_s)
    ll_hh = calculate_loglike(dd_h, dd_h, fₗ, fc_h, fc_h, fc_h)
    ll_hs = calculate_loglike(dd_h, dd_s, fₗ, fc_s, fc_h, fc_s)

    @assert isapprox(ll_ss, 1/2 * snr_s^2, rtol=1e-3) "log L(s|s) ≉ 1/2 SNR(s)^2"
    @assert isapprox(ll_hh, 1/2 * snr_h^2, rtol=1e-3) "log L(h|h) ≉ 1/2 SNR(h)^2"

    println("SNR(s) = $(snr_s)")
    println("SNR(h) = $(snr_h)")
    println("log L(s|s) = $(ll_ss)")
    println("log L(h|h) = $(ll_hh)")
    println("log L(h|s) = $(ll_hs)")
end

# %%
"""
No type annotations, non-constant globals: ~??? s/it
Type annotations, constant globals: ~0.02 s/it
"""
function benchmark_fim()
    sd = StaticDress{Float64}(
        2.3333333333333335, 0.00018806659428775589, 3.151009407916561e31, 0.0, 0.0, -56.3888135025341
    )
    dd = DynamicDress{Float64}(
        2.3333333333333335, 0.00018806659428775589, 3.151009407916561e31, 0.001, 0.0, 0.0, -56.3888135025341
    )
    fₗ = 0.022607529999065474
    fc = f_isco(m₁(dd.ℳ, dd.q))

    for i in 1:10
        println(
            @time(
                begin
                    fim_errs(fₗ, fc, fc, sd)
                    fim_errs(fₗ, fc, fc, dd)
                    ""
                end
            )
        )
    end
end

# %%
"""
No type annotations, non-constant globals: ~0.1 s/it
Type annotations, constant globals: ~0.02 s/it
"""
function benchmark_calcloglike()
    dd_s = DynamicDress{Float64}(
        2.3333333333333335, 0.00018806659428775589, 3.151009407916561e31, 0.001, 0.0, 0.0, -56.3888135025341
    )
    dd_h = DynamicDress{Float64}(
        2.27843666, 7.38694332e-5, 3.1518396596159997e31, 0.000645246955, 0.0, -1.98186232e+02, -56.3888135025341
    )
    fₗ = 0.022607529999065474
    fc_s = f_isco(m₁(dd_s.ℳ, dd_s.q))
    fc_h = f_isco(m₁(dd_h.ℳ, dd_h.q))

    for i in 1:5
        println(
            @time(
                begin
                    calculate_loglike(dd_s, dd_s, fₗ, fc_s, fc_s, fc_s)
                    calculate_loglike(dd_h, dd_h, fₗ, fc_h, fc_h, fc_h)
                    calculate_loglike(dd_h, dd_s, fₗ, fc_s, fc_h, fc_s)
                    ""
                end
            )
        )
    end
end