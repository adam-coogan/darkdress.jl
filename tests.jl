# %%
"""
Verifies the code runs.
"""
function test_fim_SNR()
    println("Making sure FIM and SNR functions run.")
    println("Static checks")
    sd = make_dress(StaticDress, 1e3 * MSun, 1. * MSun, 226. * MSun / pc^3, 7 / 3)
    println(sd)
    fₕ = f_c = f_isco(1e3 * MSun)
    fₗ = find_zero(f -> t_to_c(f, f_c, sd) - 5 * yr, (0.0001 * f_c, f_c))
    calculate_SNR(sd, fₗ, fₕ, f_c)  # agrees with David's notebook
    println(log10(fim_errs(fₗ, fₕ, f_c, sd)[1]))
    println(log10(fim_errs(fₗ, fₕ, f_c, sd)[2]))
    println(log10(fim_errs(fₗ, fₕ, f_c, sd)[3]))
    println()
    
    println("Dynamic checks")
    dd = make_dress(DynamicDress, 1e3 * MSun, 1. * MSun, 226. * MSun / pc^3, 7 / 3)
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
"""
Expected output:

    SNR(s) = 9.200834846576726
    SNR(h) = 9.200834590962279
    log L(s|s) = 42.327680936990355
    log L(h|h) = 42.35282361247973
    log L(h|s) = 5.606088315541141e-7

Accurately evaluating the last likelihood requires ~100000 points.
"""
function test_calcloglike(N_nodes=100000, USE_GAUSS=true, LOG_SPACE=true)
    dd_s = DynamicDress(
        2.3333333333333335, 0.00018806659428775589, 3.151009407916561e31, 0.001, 0.0, 0.0, -56.3888135025341
    )
    dd_h = DynamicDress(
        2.27843666, 7.38694332e-5, 3.1518396596159997e31, 0.000645246955, 0.0, -1.98186232e+02, -56.3888135025341
    )
    fₗ = 0.022607529999065474
    fc_s = f_isco(m₁(dd_s.ℳ, dd_s.q))
    fc_h = f_isco(m₁(dd_h.ℳ, dd_h.q))
    fₕ = max(fc_s, fc_h)
    
    snr_s = calculate_SNR(dd_s, fₗ, fc_s, fc_s)
    snr_h = calculate_SNR(dd_s, fₗ, fc_h, fc_h)
    ll_ss = calculate_loglike(dd_s, dd_s, fₗ, fc_s, fc_s, fc_s; N_nodes, USE_GAUSS, LOG_SPACE)
    ll_hh = calculate_loglike(dd_h, dd_h, fₗ, fc_h, fc_h, fc_h; N_nodes, USE_GAUSS, LOG_SPACE)
    ll_hs = calculate_loglike(dd_h, dd_s, fₗ, fₕ, fc_h, fc_s; N_nodes, USE_GAUSS, LOG_SPACE)

    @assert isapprox(ll_ss, 1/2 * snr_s^2, rtol=1e-3) "log L(s|s) ≉ 1/2 SNR(s)^2"
    @assert isapprox(ll_hh, 1/2 * snr_h^2, rtol=1e-3) "log L(h|h) ≉ 1/2 SNR(h)^2"

    println("SNR(s) = $(snr_s)")
    println("SNR(h) = $(snr_h)")
    println("log L(s|s) = $(ll_ss)")
    println("log L(h|h) = $(ll_hh)")
    println("log L(h|s) = $(ll_hs)")
end

# %%
function benchmark_fim()
    sd = StaticDress(
        2.3333333333333335, 0.00018806659428775589, 3.151009407916561e31, 0.0, 0.0, -56.3888135025341
    )
    dd = DynamicDress(
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
function benchmark_calcloglike(N_nodes=10000, USE_GAUSS=true, LOG_SPACE=true)
    dd_s = DynamicDress(
        2.3333333333333335, 0.00018806659428775589, 3.151009407916561e31, 0.001, 0.0, 0.0, -56.3888135025341
    )
    dd_h = DynamicDress(
        2.27843666, 7.38694332e-5, 3.1518396596159997e31, 0.000645246955, 0.0, -1.98186232e+02, -56.3888135025341
    )
    fₗ = 0.022607529999065474
    fc_s = f_isco(m₁(dd_s.ℳ, dd_s.q))
    fc_h = f_isco(m₁(dd_h.ℳ, dd_h.q))

    for i in 1:5
        println(
            @time(
                begin
                    calculate_loglike(dd_s, dd_s, fₗ, fc_s, fc_s, fc_s; N_nodes, USE_GAUSS, LOG_SPACE)
                    calculate_loglike(dd_h, dd_h, fₗ, fc_h, fc_h, fc_h; N_nodes, USE_GAUSS, LOG_SPACE)
                    calculate_loglike(dd_h, dd_s, fₗ, fc_s, fc_h, fc_s; N_nodes, USE_GAUSS, LOG_SPACE)
                    ""
                end
            )
        )
    end
end

# %%
function test_N_nodes(;N_samples=1000, USE_GAUSS=true)

    println("> Testing number of nodes...")

    dd = make_dress(DynamicDress, 1000.0 * MSun, 1.0*MSun, 226 * MSun/pc^3, 7.0/3.0, 1e8*pc, 0.0, 0.0, -1e8*pc/c)
    f_c = f_isco(m₁(dd))
    fₗ = f_of_t_to_c(5 * yr, f_c, dd)
    fₕ = f_c

    x_true = [(dd.γₛ), dd.c_f, (dd.ℳ)/MSun, dd.q, dd.t̃_c]
    x_true[1:4] = log.(x_true[1:4])

    loglike(x) = like_wrapper(x, dd, fₗ, fₕ)
    FIM = -1.0*ForwardDiff.hessian(loglike, x_true)
    covmat = inv(FIM)
        
    N_nodes_list = [300, 1000, 3000, 10000, 300000, 100000]
    likes = zeros(length(N_nodes_list), N_samples)

    for i in 1:length(N_nodes_list)
        println("> Sampling for N_nodes = $(N_nodes_list[i])...")
        loglike_test(x) = like_wrapper(x, dd, fₗ, fₕ, N_nodes=N_nodes_list[i], USE_GAUSS=USE_GAUSS)
        if (i == 1)
            full_chain = RandomSample(loglike_test, x_true, N_samples, covmat)
            global samples = full_chain[:,3:end]
            likes[1,:] = full_chain[:,2]
        else
            for j in 1:N_samples           
                likes[i,j] = loglike_test(samples[j,:])
            end
        end
    end

    l300, l1000, l3000, l10000, l30000, l100000 = [likes[i, :] for i in 1:length(N_nodes_list)]

    Δ300 = abs.(l300 - l100000)./(l100000)
    Δ3000 = abs.(l3000 - l100000)./(l100000)
    Δ10000 = abs.(l10000 - l100000)./(l100000)
    Δ30000 = abs.(l30000 - l100000)./(l100000)

    println("Maximum error (N_nodes = 300): ", maximum(Δ300))
    println("Maximum error (N_nodes = 3000): ", maximum(Δ3000))
    println("Maximum error (N_nodes = 10000): ",maximum(Δ10000))
    println("Maximum error (N_nodes = 30000): ",maximum(Δ30000))

    return samples, likes

end