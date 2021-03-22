include("utils.jl")
include("binary.jl")
include("analysis.jl")  # needs types from binary.jl to be loaded
include("MCMC.jl")

using FastGaussQuadrature, Trapz, LinearAlgebra, StatsBase, Statistics
using Optim, DelimitedFiles, ProgressMeter, LaTeXStrings, PyPlot

#font = Plots.font("Helvetica", 14)
#PyPlot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font)

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 18

#import PyPlot
#PyPlot.rc("text", usetex="True")

# Information about the match can be found in https://arxiv.org/abs/1807.07163


function latex_name(p::Symbol)
    if p == :ℳ
        return "{\\mathcal{M}}"
    elseif p == :q
        return "{q}"
    elseif p == :γₛ
        return "{\\gamma_\\mathrm{sp}}"
    elseif p == :c_f
        return "{c_f}"
    else
        throw(ArgumentError("invalid parameter"))
    end
end

# Configure masses and parameter grids
const m₁_ref = 1e3 * MSun
const m₂_ref = 1 * MSun
const t_to_merger = 5 * yr  # observing time
const f_c = f_isco(m₁_ref)  # frequency at coalescence
const fₕ = f_c  # for setting frequency observation window


function calculate_SNR_r(T, fₗ, fₕ, N_nodes=10000)


    fc_1 = f_isco(m₁(T.ℳ, T.q))
    #fc_2 = f_isco(m₁(T2.ℳ, T2.q))
    #fc_2 = f_isco(m₁(T1.ℳ, T1.q))


    #match_integrand(f) = (f -> amp₊(f, vb)*amp₊(f, dd)/Sₙ_LISA(f))*exp( im*(Ψ(f, fc_test, vb) - Ψ(f, fc_true, dd)) )
    #match_integrand(f) = 4*(amp₊(f, T1)*amp₊(f, T2)/Sₙ_LISA(f))*exp( im*(Ψ(f, fc_1, T1) - Ψ(f, fc_2, T2)) )
    #match_integrand(f) = 4*(amp₊(f, vb)*amp₊(f, dd)/Sₙ_LISA(f))*exp( im*Ψ(f, fc_test, vb))* exp(-im*Ψ(f, fc_true, dd)) 
    modh_integrand(f) = 4*(amp₊(f, T)^2/Sₙ_LISA(f))
    #println("Here (a)")
    #fₗ = f_of_t_to_c(5 * yr, fc_1, T)
    #println("Here (b)")
    #fₗ = f_of_t_to_c(5 * yr, fc_1, T)
    #fₕ = fc_1

    fh_temp = fₕ
    if (fc_1 < fh_temp)
        fh_temp = fc_1
    end

    f̄ = (fh_temp + fₗ)/2
    Δf = (fh_temp - fₗ)

    #Use fixed Gauss quadrature, or Trapz
    USE_GAUSS = true

    if (USE_GAUSS)
        nodes, weights = gausslegendre( N_nodes )
        f_vals = f̄ .+ nodes*Δf/2
        modh = (Δf/2 * dot( weights, modh_integrand.(f_vals) ))
    else
        f_vals = range(fₗ, stop=fh_temp, length=N_nodes)
        modh = trapz(f_vals, modh_integrand.(f_vals))
    end

    return modh

    #println("quadgk:")
    #x = quadgk(match_integrand, fₗ, fₕ, rtol=1e-3)[1]
    #println(x);

    #Don't forget the factor of 4
end

function calculate_match_unnorm_r(T1, T2, fₗ, fₕ, N_nodes=10000)


    fc_1 = f_isco(m₁(T1.ℳ, T1.q))
    #fc_2 = f_isco(m₁(T2.ℳ, T2.q))
    fc_2 = f_isco(m₁(T2.ℳ, T2.q))


    fh_temp = fₕ
    if (fc_1 < fh_temp)
        fh_temp = fc_1
    end
    if (fc_2 < fh_temp)
        fh_temp = fc_2
    end

    f̄ = (fh_temp + fₗ)/2
    Δf = (fh_temp - fₗ)



    #Should it be fc_1 or fc_2 here?
    calcΔΨ(f) = Ψ(f, fc_1, T1) - Ψ(f, fc_2, T2)
    calcAmp(f) = 4*(amp₊(f, T1)*amp₊(f, T2)/Sₙ_LISA(f))

    #Use fixed Gauss quadrature, or Trapz
    USE_GAUSS = true

    if (USE_GAUSS) 
        nodes, weights = gausslegendre( N_nodes )
        f_vals = f̄ .+ nodes*Δf/2
    else
        f_vals = range(fₗ, stop=fh_temp, length=N_nodes)
    end

    ΔΨ = calcΔΨ.(f_vals)
    Amp = calcAmp.(f_vals)
    integs_re = cos.(ΔΨ).*Amp
    integs_im = sin.(ΔΨ).*Amp

    if (USE_GAUSS) 
        x_re = (Δf/2 * dot( weights,  integs_re))
        x_im = (Δf/2 * dot( weights,  integs_im))
    else
        x_re = trapz(f_vals, integs_re)
        x_im = trapz(f_vals, integs_im)
    end

    return x_re, x_im

end


function calcloglike(dd, fₗ, fₕ, x, dim=7, N_nodes=1000, linear = false)
    
    #Arbitrary constant for the 'initial' value of dₗ_ι
    #when we optimise over dₗ_ι
    α0 = -56.3888135025341

    #Unpack parameters depending on how many dimensions we're using.
    if (dim >= 4)
        if (linear == true)
            γₛ  = x[1]
            #c_f_temp = x[2]
            #c_f = c_f_temp^((11 - 2 * γₛ)/3)
            c_f = x[2]
            ℳ   = x[3]*MSun 
            q   = x[4]            
        else
            γₛ  = exp(x[1])
            c_f = exp(x[2])
            ℳ   = exp(x[3])*MSun
            q   = exp(x[4])
        end
    end

    if (dim <= 4)
        Φ_c = 0.
        t̃_c = 0.
        dₗ_ι = α0  
    end

    if (dim == 5)
        Φ_c = 0.
        t̃_c = x[5]
        dₗ_ι = α0  
    end

    if (dim == 6)
        Φ_c = 0.
        t̃_c = x[5]
        dₗ_ι = x[6] 
    end

    if (dim == 7)
        Φ_c = x[5]
        t̃_c = x[6]
        dₗ_ι = x[7] 
    end


    #Set some parameter boundaries
    if (γₛ < 0)
        return -1e30
    end
    #if (γₛ > 3)
    #    return sgn*1e30
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

    #Initialise the waveform for comparison, h
    dd_alt = DynamicDress(γₛ, c_f, ℳ, q, Φ_c, t̃_c, dₗ_ι)
 
    #Calculate real and imaginary parts of the overlap <d|h>
    dh_re, dh_im = calculate_match_unnorm_r(dd, dd_alt, fₗ, fₕ, N_nodes)

    #If we're optimising over the phase (dim < 7)
    #then take the real value
    if (dim == 7)
        dh = dh_re
    else
        dh = sqrt(dh_re^2 + dh_im^2)
    end

    #Calculate the overlap <h|h>
    hh = calculate_SNR_r(dd_alt, fₗ, fₕ, N_nodes)

    #Optimise over dₗ_ι if necessary
    r = 1.
    if (dim <= 5)
        r = dh/hh
    end

    #Return <d|h> - (1/2)<h|h>
    return r*dh - 0.5*r^2*hh    
    #return 0.5*dh^2/hh
end

# %%
"""
Expected result:

    log L(s|s) = 42.32768093699036
    log L(h|h) = 42.35282361247973
    log L(h|s) = 5.606088315541141e-7
"""
function test_calcloglike(N_nodes=10000)
    fₗ = 0.022607529999065474
    f_c = 4.397009835544328
    dd_ref = DynamicDress(2.3333333333333335, 0.00018806659428775589, 3.151009407916561e31, 0.001, 0.0, 0.0, -56.3888135025341)
    #dd_alt = DynamicDress(2.27843666, 7.38694332e-5, 3.1518396596159997e31, 0.000645246955, 0.0, -198.186232, -56.3888135025341)
    dd_alt = DynamicDress(exp(0.81637266), exp(-9.91926788),  exp(2.76308336)*MSun,  exp(-7.52114998), 0.0,  -227.74995698, -56.3888135025341)
    
    x_ref = [dd_ref.γₛ, dd_ref.c_f, dd_ref.ℳ / MSun, dd_ref.q, dd_ref.t̃_c]
    x_alt = [dd_alt.γₛ, dd_alt.c_f, dd_alt.ℳ / MSun, dd_alt.q, dd_alt.t̃_c]
    #-1.98186232e+02
    
    #2.27843666e+00  7.38694332e-05  1.58499392e+01  6.45246955e-04
    #-198.186232

    println("runMCMC.jl:")

    ll_ss = calcloglike(dd_ref, fₗ, f_c, x_ref, 5, N_nodes, true)
    ll_hh = calcloglike(dd_alt, fₗ, f_c, x_alt, 5, N_nodes, true)
    ll_hs = calcloglike(dd_ref, fₗ, f_c, x_alt, 5, N_nodes, true) 

    println("   log L(s|s) = $(ll_ss)")
    println("   log L(h|h) = $(ll_hh)")
    println("   log L(h|s) = $(ll_hs)")

    println("")
    println("analysis.jl:")

    fc_ref = f_isco(m₁(dd_ref.ℳ, dd_ref.q))
    fc_alt = f_isco(m₁(dd_alt.ℳ, dd_alt.q))

    ll_ss = calculate_loglike(dd_ref, dd_ref, fₗ, f_c, fc_ref, fc_ref, N_nodes=N_nodes)
    ll_hh = calculate_loglike(dd_alt, dd_alt, fₗ, f_c, fc_alt, fc_alt, N_nodes=N_nodes)
    ll_hs = calculate_loglike(dd_ref, dd_alt, fₗ, f_c, fc_ref, fc_alt, N_nodes=N_nodes)


    println("   log L(s|s) = $(ll_ss)")
    println("   log L(h|h) = $(ll_hh)")
    println("   log L(h|s) = $(ll_hs)")
end

test_calcloglike()

# %%
function logliketest(x)
    y = (x .- x0)
    #print(size(y))
    #print(size(sigma))
    #logL = -0.5*sum(transpose(y).*sigma.*y)
    logL = -0.5*y'*inv(errs_cov)*y
    return logL
end


function inv_stable(Γ)
    # Improve stability of inversion
    scales = sqrt(inv(Diagonal(Γ)))
    Γr = scales * Γ * scales

    @assert all(eigvals(Γr) .> 0) eigvals(Γr)

    return scales * inv(Γr) * scales
end

function test_1d(;N_samples=1000, output_label=nothing, temperature = 1.0)

    println("> Testing in 1D...")

    dd = make_dress(DynamicDress, 1000.0 * MSun, 1.0*MSun, 226 * MSun/pc^3, 7.0/3.0, 1e8*pc, 0.0, 0.0, -1e8*pc/c)
    fₗ = f_of_t_to_c(5 * yr, f_c, dd)
    #fₗ = 1e-2
    #println("> WARNING: Using f_l = 1e-2 Hz...")
    #println("> SNR^2 for benchmark system, <d|d>: ", calculate_SNR(dd, fₗ, fₕ, 1000))

    #x_true = [log(dd.γₛ), log(dd.c_f), log(dd.ℳ), log(dd.q), dd.t̃_c]
    x_true = [(dd.γₛ), (dd.c_f), (dd.ℳ), (dd.q), dd.t̃_c]
    inds = [1, 2, 3, 4, 6]
    N = length(inds)
    #errs_cov = fim_cov_sub(fₗ, fₕ, f_c, inds, dd)


    println([x_true[1] x_true[2] x_true[3:end]...])

    x0 = x_true[5]

    loglike(x) = (1/temperature)*calcloglike(dd, fₗ, fₕ, [x_true[1:4]... x ], 5, 500, true)
    loglike_max = loglike(x0)
    println("> Maximum loglike/T: ", loglike_max)

    #Calculate correct covariance matrix:
    #gradL(x) = ForwardDiff.gradient(loglike, x)
    #∂L = ForwardDiff.gradient(loglike, x_true)
    #∂L[∂L .=== nothing] .= 0.
    #∂∂L = 

    ∂L(y) = ForwardDiff.derivative(loglike, y)
    ∂∂L(y) = ForwardDiff.derivative(∂L, y)
    ∂∂∂L(y) = ForwardDiff.derivative(∂∂L, y)
    ∂∂∂∂L(y) = ForwardDiff.derivative(∂∂∂L, y)

    L0 = loglike_max
    ∂L0 = ∂L(x0)
    ∂∂L0 = ∂∂L(x0)
    ∂∂∂L0 = ∂∂∂L(x0)
    ∂∂∂∂L0 = ∂∂∂∂L(x0)
    #∂∂∂L0 = 0.0
    #∂∂∂∂L0 = 0.0

    σ = 1/sqrt(-∂∂L0)

    dxlist = range(-5*σ, stop=5*σ, length=N_samples)
    Lnum = zeros(N_samples)
    Lapprox = zeros(N_samples)
    ProgressMeter.@showprogress "Calculating:" for i = 1:N_samples
        dx = dxlist[i]
        Lnum[i] = loglike(x0 + dx)
        Lapprox[i] = L0 + ∂L0*dx + (1/2)*∂∂L0*dx^2 + (1/6)*∂∂∂L0*dx^3 + (1/6)*∂∂∂L0*dx^3 + (1/24)*∂∂∂∂L0*dx^4
    end
    #FIM_new = -1.0*ForwardDiff.hessian(loglike, x_true[0])
    #FIM_new = -1.0*FIM_new
    #for i=1:5
    #    FIM_new[i,i] = abs(FIM_new[i,i])
    #end

    fig, ax = PyPlot.subplots(1, 1, figsize=(7,7))

    ax.plot(dxlist, Lnum)
    ax.plot(dxlist, Lapprox)

    fig
end

function run_5d(;N_samples=1000, output_label=nothing)

    println("> Running with 5-dimensions...")

    dd = make_dress(DynamicDress, 1000.0 * MSun, 1.0*MSun, 226 * MSun/pc^3, 7.0/3.0, 1e8*pc, 0.0, 0.0, -1e8*pc/c)
    fₗ = f_of_t_to_c(5 * yr, f_c, dd)

    γₛ = 7.0/3.0

    #c0 = (dd.c_f/f_b(1000.0 * MSun, 1.0*MSun, γₛ))^((11-2*γₛ)/3)
    #c0 = (dd.c_f)^((3 / (11 - 2 * dd.γₛ)))
    x_true_full = [(dd.γₛ), dd.c_f, (dd.ℳ)/MSun, dd.q, dd.t̃_c]
    #x_true_full = [(dd.γₛ), dd.c_f, (dd.ℳ)/MSun, dd.q, dd.Φ_c, dd.t̃_c, dd.dₗ_ι]
    #x_true_full = [log(dd.γₛ), log(dd.c_f), log((dd.ℳ)/MSun), log(dd.q), dd.Φ_c, dd.t̃_c, dd.dₗ_ι]
    #x_true_full = [log(dd.γₛ), log(dd.c_f), log((dd.ℳ)/MSun), log(dd.q), log(dd.t̃_c + 1e-10), dd.dₗ_ι]

    DIM = 5
    x_true = x_true_full[1:DIM]
    x_true[1:4] = log.(x_true_full[1:4])
    println(x_true)

    #x_true = [log(dd.γₛ), log(dd.c_f), log(dd.ℳ), log(dd.q), dd.t̃_c]
    #x_true = [(dd.γₛ), (dd.c_f), (dd.ℳ)/MSun, (dd.q), dd.t̃_c]
    #x_true = [(dd.γₛ), (dd.c_f), (dd.ℳ)/MSun, dd.q, dd.t̃_c]
    #x_true = [(dd.γₛ), (dd.c_f), (dd.ℳ)/MSun, dd.q]
    #x_true = [(dd.ℳ)/MSun, dd.q]


    #loglike(x) = calcloglike(dd, fₗ, fₕ, x, 5, 1000, true)
    loglike(x) = like_wrapper(x, dd, fₗ, fₕ)

    loglike_max = loglike(x_true)
    println("> x_true: ", x_true)
    println("> Maximum loglike/T: ", loglike_max)

    #x_test = [ 2.27843666e+00  7.38694332e-05  1.58499392e+01  6.45246955e-04 -1.98186232e+02]
    #L_test =  loglike(x_test)
    #println("TEST LIKELIHOOD: ", L_test)

    #return x_test, L_test

    #loglike_sub(x) = calcloglike(dd, fₗ, fₕ, [x... x[5]], 5, 500, true)

    #Calculate correct covariance matrix:
    #∂L[∂L .=== nothing] .= 0.
    #∂∂L = 
    
    #FIM_new = -1.0*FIM_new
    #for i=1:5
    #    FIM_new[i,i] = abs(FIM_new[i,i])
    #end

    FIM_new = -1.0*ForwardDiff.hessian(loglike, x_true)
    #FIM_new = 0.5*(FIM_new + FIM_new')
    println("det(FIM): ", det(FIM_new))
    println(" ")
    println("FIM: ", FIM_new)
    println(" ")
    println("Condition number: ", cond(FIM_new))
    println(" ")

    #FIM_bf = convert(Array{BigFloat,2}, FIM_new)
    #FIM_bf = BigFloat(FIM_new)

    covmat = inv_stable(FIM_new)

    #Rescale the whole thing by a different (5, 5) matrix...
    #FIM_pert = FIM_new .* (1 .+ 1e-6.*randn(length(x_true)))
    #covmat_pert = inv(FIM_pert)

    #println("Δcovmat/covmat:", (covmat .- covmat_pert)./covmat)
    
    #println("covmat:", covmat)
    #println(" ")
    #covmat = inv(FIM_bf)
    #println("Σ Σ^-1 = ", covmat*FIM_bf)
    #println(" ")
    #println("covmat (Float64):", covmat_old)
    #println(" ")
    #println("covmat (BigFloat):", covmat)
    #println(" ")
    #println("relative difference:", (covmat_old .- covmat)./covmat)


    #scales = sqrt(inv(Diagonal(covmat)))
    #corr = scales * covmat * scales
    #covmat = inv(scales)*R2*(R*corr*R')*R2'*inv(scales)

    #rescale = [1.0 1.0 1.0 1.0]
    #rescale_mat = (rescale'*rescale)
    #println(rescale_mat)
    #covmat = covmat.*0.1

    #println("> Benchmark values: ", x_true)
    errors = sqrt.([covmat[i, i] for i in 1:size(covmat)[1]])
    println("> Errors: ", errors)
    #return 0


    #x_test = [ 0.85719845, -8.17143022, 71.52774677, -6.71501771, 43.93385739]
    #x_test = [ 0.85719845, -8.17143022, 72.52774677, -6.71501771, 43.93385739]
    #println("> Test loglike/T: ", loglike(x_test))

    #quit()

    N_ini = 2000

    x_ini = 1.0*x_true #.+ 1e-0.*(randn(length(x_true)).*errors)
        
    #println(covmat)
    
    #full_chain = RunMCMC(loglike, x_ini, 1000, covmat, output_label=output_label, adjust_steps = 500)
    #chains, likes = RunMCMC(loglike, x_ini, N_samples, errs_cov, output_label=output_label, adjust_steps = 500)
    #println(full_chain[:,3:7] )

    #return full_chain, covmat
    #println("Range: ", minimum(full_chain[:,3]), " ->", maximum(full_chain[:,3]))

    #return full_chain, covmat

    #covmat =  Diagonal(sqrt.([covmat[i, i] for i in 1:size(covmat)[1]]))

    N_refine = 3

    for i_ref = 1:N_refine
        if (i_ref%2 == 0)
            #println("A")
            scale_refine = 2.0
        else
            #println("B")
            scale_refine = 0.5
        end

        if (i_ref == 3)
            scale_refine = 1.0
        end

        scale_refine = 1.0

        full_chain = RandomSample(loglike, x_ini, N_ini, scale_refine*covmat, output_label=output_label)
        y = (full_chain[:,3:end] .- x_true')
        #y = full_chain[:,3:end]

        logBL = [-0.5*y[i,:]'*inv_stable(scale_refine*covmat)*y[i,:] for i in 1:size(y)[1]]
        logBL = logBL .- 0.5*log((2*π)^5*det(scale_refine*covmat))
        ΔlogL = full_chain[:,2] .- maximum(full_chain[:,2]) 

        inds = ΔlogL .> -100
        #NEED TO BE CAREFUL - NEED TO CORRECT THE RESCALING FOR THE FACT THAT I"M NOT INCLUDING ALL!!!

        #chi2 ~ 2*DeltaLogL
        #Compare deltaloglike and chi-squared
        #corr = 1/(1-0.96257)
        #println(ΔlogL)
        #println("Target fraction (chi^2 < 5): 0.584; Current fraction: ", sum(ΔlogL .> -2.5)/N_ini)
        #println("Target fraction (chi^2 < 10): 0.925; Current fraction: ", sum(ΔlogL .> -5)/N_ini)
        #println("Use this to get the shape, then find the bounding ellipse...?")

        #The fraction within the 1 sigma band in 5 dimensions is...

        x_max = maximum(abs.(y[inds,1]))
        #println(minimum(y[inds,1]))
        #println(x_max, ";    ", sqrt(covmat[1,1]))

        wvals = exp.(ΔlogL - logBL)
        #wvals = exp.(ΔlogL)
        weights = ProbabilityWeights(wvals[inds])
        covmat = cov(full_chain[inds,3:end], weights, corrected=false)

        fig, ax = PyPlot.subplots(1, 1, figsize=(7,7))
        xvals = full_chain[:,3]
        yvals = full_chain[:,4]
        s1 = ax.scatter(xvals[inds], yvals[inds], c=ΔlogL[inds])
        plt.colorbar(s1, ax=ax)
        fig.savefig("/Users/bradkav/Code/darkdress.jl/figures/sample" * string(i_ref)* "d.pdf")
        #covmat = covmat*((x_max)^2)/(covmat[1,1])

    end

    #scale = 3.0

    #full_chain = RandomSample(loglike, x_ini, N_ini, covmat, output_label=output_label, precision=false)
    #return full_chain, covmat

    #scale = (2.38)^2/5
    scale = 0.2
    #scale = 10
    #scale = 1
    full_chain = RunMCMC(loglike, x_ini, N_samples, 
                                            scale*covmat, 
                                            output_label = output_label,
                                            adjust_steps = 1000,
                                            verbose=true,
                                            adapt_method=nothing)
    #full_chain_new = RandomSample(loglike, x_ini, N_samples, scale*covmat_new, output_label=output_label, adjust_steps = 500000)

    #y = (full_chain_new[:,3:7] .- x_true')
    #logBL = [-0.5*y[i,:]'*inv(covmat_new)*y[i,:] for i in 1:size(y)[1]]
    #wvals = exp.(full_chain_new[:,2] - logBL)
    #weights = ProbabilityWeights(wvals)
    #covmat_new2 = cov(full_chain_new[:,3:7], weights)

    #full_chain_new2 = RunMCMC(loglike, x_ini, N_samples, scale*covmat_new2, output_label=output_label, adjust_steps = 500000)

    
    return full_chain, covmat
    #return full_chain_new, covmat_new
end

#if (N_args == 1)
#    fileID = ARGS[1]
#else
#    println("> Expecting one command line argument (output file ID)...")
#    println("> Exiting...")
#    exit()
#end

dd = make_dress(DynamicDress, 1000.0 * MSun, 1.0*MSun, 226 * MSun/pc^3, 7.0/3.0, 1e8*pc, 0.0, 0.0, -1e8*pc/c)

#writedlm("chains/chain_" * fileID * ".txt", chains, "   ")


chain1, covmat1 = run_5d(;N_samples=10000, output_label="5d_final_5")
#------------------------------


#fb = f_b(1000.0 * MSun, 1.0*MSun, 7.0/3.0)
#c0 = dd.c_f/((f_b(1000.0 * MSun, 1.0*MSun, 7.0/3.0))^((11-2*7.0/3.0)/3))
#c0 = dd.c_f/((f_b(1000.0 * MSun, 1.0*MSun, 7.0/3.0))^((11-2*7.0/3.0)/3))
#x0 =  [log(dd.γₛ), log(dd.c_f), log(dd.ℳ/MSun), log(dd.q), dd.t̃_c]
#c0 = (dd.c_f)^((3 / (11 - 2 * dd.γₛ)))#/fb
begin
    x0 =  [dd.γₛ, dd.c_f, dd.ℳ/MSun, dd.q, dd.t̃_c]
    x0[1:4] = log.(x0[1:4])
end
#x0 =  [dd.γₛ, dd.c_f, dd.ℳ/(MSun), dd.q, dd.t̃_c,  dd.dₗ_ι]
#x0 =  [dd.γₛ, dd.c_f, dd.ℳ/(MSun)]
#x0 =  [dd.γₛ, dd.c_f, dd.ℳ/(MSun), dd.q, dd.t̃_c,  dd.dₗ_ι]
#begin
    #x0 =  [dd.γₛ, dd.c_f, dd.ℳ/(MSun), dd.q]
    #x0 = log.(x0)
#end
#x0 =  [dd.ℳ/MSun, dd.q]#, dd.t̃_c]
#x0 =  [dd.γₛ, dd.c_f, 1000.0, 1.0]
#CHECK IMPACT OF RESOLUTION!

#TRY LINEAR JUMPS, CREATE BOUNDING ELLIPSOID!!!

#IDEAS:
#   - Try alternating shrink and grow (alternate between rescale = 0.75 and rescale=1.25)
#   - Try truncating the delta-log-likelihood at some value (because we don't trust the values when the match is small)
#   - Calculate the 1-sigma bounding ellipsoid?

#See if it works for linear!!!!

#chain1 = convert(Array{Float64,2}, chain1)
#covmat1 = convert(Array{Float64,2}, covmat1)

begin
    y = (chain1[:,3:end] .- x0')
    logBL = [-0.5*y[i,:]'*inv_stable(covmat1)*y[i,:] for i in 1:size(y)[1]]
    ΔlogL1 = chain1[:,2] .- maximum(chain1[:,2])
    println(maximum(chain1[:,2]))
    #ΔlogL2 = chain2[:,2] .- maximum(chain2[:,2])
    #ΔlogL3 = chain3[:,2] .- maximum(chain3[:,2])

    inds1 = ΔlogL1 .> -100
    inds = sortperm(ΔlogL1[inds1])

    wvals = exp.(ΔlogL1 - logBL)
    weights = ProbabilityWeights(wvals)
    #weights = weights/sum(weights)
    covmat_new = cov(chain1[:,3:end], weights)

    #newlogL = [-0.5*y[i,:]'*inv_stable(covmat_new)*y[i,:] for i in 1:size(y)[1]]
end

#inds = (max(chain[:,2] - chains[:,2]) <

begin
    indx = 1
    indy = 2
    indz = 3
    PyPlot.close("all")
    xvals1 = (chain1[:,indx+2])
    yvals1 = (chain1[:,indy+2])
    zvals1 = (chain1[:,indz+2])

    #zvals = (chains[:,indz])
    #x_list = range(0.5*min(xvals), stop=1.5*max(xvals), length=100)
    #y_list = range(0.5*min(yvals), stop=1.5*max(yvals), length=100)

    fig, ax = PyPlot.subplots(1, 1, figsize=(7,7))

    #println(ΔlogL1[inds])

    #s1 = ax.scatter(xvals1[inds], yvals1[inds], c=log.(wvals[inds]))
    s1 = ax.scatter(xvals1[inds], yvals1[inds], c=ΔlogL1[inds])
    #s1 = ax.scatter(xvals1[inds], yvals1[inds], c=zvals1[inds])
    #s1 = ax[1].scatter(xvals, yvals, c=newlogL)
    #s1 = ax.scatter(xvals1, yvals1, c=logBL)
    
    ax.axvline(x0[indx], linestyle="--", color="grey")
    ax.axhline(x0[indy], linestyle="--", color="grey")
    
    plt.colorbar(s1, ax=ax, label=L"\Delta \log \mathcal{L}")

    plt.xlabel(L"\gamma_\mathrm{sp}")
    plt.ylabel(L"\tilde{c}_f")

    submat = [covmat1[indy,indy] covmat1[indy,indx];
            covmat1[indx,indy] covmat1[indx, indx]]

    eig_errs = sqrt.(eigvals(submat))
    eig_vecs = eigvecs(submat)

                    #V[:,k] is the kth eigenvector
    imax = argmax(eig_errs)
                    #Angle of the major axis with x-axis
    α = -atan(eig_vecs[2,imax], eig_vecs[1,imax])

                    #1-, 2-, 3-sigma contours
    for N = 1:3
        xvals = range(-N*eig_errs[1], stop=N*eig_errs[1], length=100)
        yvals = eig_errs[2]*sqrt.(1e-14 + N^2 .- xvals.^2/eig_errs[1]^2)

        #Flip for the other side of the ellipse
        xvals = vcat(xvals, reverse(xvals))
        yvals = vcat(yvals, -reverse(yvals))

        xrot = cos(α)*xvals - sin(α)*yvals 
        yrot = sin(α)*xvals + cos(α)*yvals 

        ax.plot(xrot .+ x0[indx], yrot .+ x0[indy], color="black")
    end

    #println(">Done plotting.")
    fig.savefig("/Users/bradkav/Code/darkdress.jl/figures/2d_scatter_linear_rescale_v4.pdf")
    fig
end

begin
    PyPlot.close("all")
    fig, ax = PyPlot.subplots(1, 1, figsize=(7,7))
    plt.hist(exp.(chains[:,4]))
    fig
end



lims_Adam = [
    (-0.02, 0.015),
    (-0.2, 0.2),
    (-0.00008, 0.0001),
    (-0.07, 0.1),
    (-300, 250)
]

function my_corner(
    samples; labels=nothing, weights=nothing, base_dim=2, n_bins=15, lims=nothing, covmat =nothing
)
    n = size(samples)[2]
    @assert n < 10  # too slow otherwise
    @assert size(samples)[2] < size(samples)[1]
    PyPlot.close("all")
    fig, axes = PyPlot.subplots(n, n, figsize=(base_dim * n, base_dim * n))
    if lims !== nothing
        bins = [
            range(lims[i][1], lims[i][2], length=n_bins + 1)
            for i in 1:length(lims)
        ]
    else
        bins = [n_bins for i in 1:length(lims)]
    end
    for i in 1:n
        for j in 1:n
            ax = axes[i, j]
            if i == j
                # 1D hist
                ax.hist(
                    samples[:, i], weights=weights, density=true, bins=bins[i]
                )
                lims !== nothing && ax.set_xlim(lims[i])

                if covmat !== nothing
                    ax.axvline(sqrt(covmat[i,i]), linestyle="--", color="black")
                    ax.axvline(-sqrt(covmat[i,i]), linestyle="--", color="black")
                end


            elseif j > i
                # Hide axis
                ax.set_axis_off()
            else
                # 2D plot
                ax.hist2d(
                    samples[:, j], samples[:, i], weights=weights, density=true,
                    bins=[bins[j], bins[i]]
                )
                lims !== nothing && ax.set_xlim(lims[j])
                lims !== nothing && ax.set_ylim(lims[i])

                if covmat !== nothing

                    submat = [covmat[i,i] covmat[j,i];
                             covmat[i,j] covmat[j, j]]

                    eig_errs = sqrt.(eigvals(submat))
                    eig_vecs = eigvecs(submat)

                    #V[:,k] is the kth eigenvector
                    imax = argmax(eig_errs)
                    #Angle of the major axis with x-axis
                    α = -atan(eig_vecs[2,imax], eig_vecs[1,imax])

                    #1-, 2-, 3-sigma contours
                    for N = 1:3
                        xvals = range(-N*eig_errs[1], stop=N*eig_errs[1], length=100)
                        yvals = eig_errs[2]*sqrt.(1e-14 + N^2 .- xvals.^2/eig_errs[1]^2)

                        #Flip for the other side of the ellipse
                        xvals = vcat(xvals, reverse(xvals))
                        yvals = vcat(yvals, -reverse(yvals))

                        xrot = cos(α)*xvals - sin(α)*yvals 
                        yrot = sin(α)*xvals + cos(α)*yvals 

                        ax.plot(xrot, yrot, color="black")
                    end
                end

            end
            if labels !== nothing
                i == n && ax.set_xlabel(labels[j])
                j == 1 && ax.set_ylabel(labels[i])
            end
            ax.set_aspect(1 / ax.get_data_ratio(), adjustable="box")
        end
    end
    fig.tight_layout()
    PyPlot.display(fig)
    return fig, axes
end


labs = [L"\Delta\gamma_\mathrm{sp}/\gamma_\mathrm{sp}", L"\Delta c_f/\tilde{c}_f", L"\Delta \mathcal{M}/\mathcal{M}", L"\Delta q/q", L"\tilde{t}_c"]

#c_f_true = 0.00018806659428775589
c_f_true = 0.017187007242570305
M_chirp_true = 3.151009407916561e31/MSun
x_true = [7.0/3.0, c_f_true, M_chirp_true, 1e-3, 0.0]
x_denom =  [7.0/3.0, c_f_true, M_chirp_true, 1e-3, 1.0]
#true_samps

begin
    fₗ = f_of_t_to_c(5 * yr, f_c, dd)
    inds = [1, 2, 3, 4, 6]
    N = length(inds)
    #errs_cov = fim_cov_sub(fₗ, fₕ, f_c, inds, dd)

    #errs_cov_full = fim_cov(fₗ, fₕ, f_c, dd)
    #errs_cov = zeros(N, N)
    FIM_cov_full = fim_cov(fₗ, fₕ, f_c, dd)
    FIM_cov_sub = zeros(N, N)
    for i = 1:N, j = 1:N
            i1 = inds[i]
            j1 = inds[j]
            FIM_cov_sub[i, j] = FIM_cov_full[i1, j1]
    end

    FIM_full = fim(fₗ, fₕ, f_c, dd)
    FIM_sub = zeros(N, N)
    for i = 1:N, j = 1:N
        i1 = inds[i]
        j1 = inds[j]
        FIM_sub[i, j] = FIM_full[i1, j1]
    end
    # Improve stability of inversion
    #scales = sqrt(inv(Diagonal(FIM)))
    #FIMr = scales * FIM * scales

    #@assert all(eigvals(Γr) .> 0) eigvals(Γr)

    #FIM_covmat = scales * inv(FIMr) * scales

    #FIM_covmat_full = fim_cov_sub(fₗ, fₕ, f_c, inds, dd)

    #println(sqrt(FIM_covmat[2,2]))
    #println("FIM errors:", fim_errs(fₗ, fₕ, f_c, dd))

    FIM_scaled = FIM_sub.*(x_denom*x_denom')

end

chains = readdlm("/Users/bradkav/Code/darkdress.jl/chains/chain_5d_final_full.txt")

x0 =  [dd.γₛ, dd.c_f, dd.ℳ/MSun, dd.q, dd.t̃_c]
x_denom = 0.0.*x0
x_denom[1:4] = x0[1:4]
x_denom[5] = 1.0

my_lims = [
    (-0.06, 0.06),
    (-1.0, 2.0),
    (-0.00025, 0.00025),
    (-1.0, 1.5),
    (-250, 250)
]

begin
    true_samps = 1.0*chains[:,3:7]
    true_samps[:,1:4] = exp.(true_samps[:,1:4])
    scaled_samps = (true_samps .- x0')./x_denom'

    #scaled_samps

    thin = 5

    f, ax = my_corner(
        scaled_samps[1:thin:end,:]; labels=labs, weights=chains[1:thin:end,1], base_dim=5, n_bins=50, lims=my_lims, covmat=nothing
    )
    f.savefig("/Users/bradkav/Code/darkdress.jl/figures/corner_5d.pdf")
    PyPlot.display(f)
end