using Random
import ProgressMeter
import Distributions
#using Base

#rng = MersenneTwister(1212);


function SampleNewPoint(x, dist)
   
    dx = rand(dist,1)

    return x .+ dx #+ beta*dx2
end

function SampleNewPoint(x, chol, errs)

    dx = chol*randn(length(x))

    
    #beta = 1e-3

    #dx2 = errs.*randn(length(x))

    #x_new = x .+ covmat*dx
    #x_new[5] = rand(Float64)*2*pi
    return x .+ dx #+ beta*dx2
end


function RunMCMC(f, x0, N_samples, covmat; output_label=nothing, adjust_steps=1000000, verbose=false, adapt_method=nothing)
        
    errs_curr = sqrt.([covmat[i, i] for i in 1:size(covmat)[1]])

    #https://arxiv.org/pdf/1304.4473.pdf
    
    chol = cholesky((1/2)*(covmat + covmat')).L

    chain = zeros(N_samples,length(x0))
    loglikes = zeros(N_samples)
    multiplicity = zeros(N_samples)

    x_curr = 1.0*x0
    logL_curr = f(x0)

    N_attempts = 0

    ProgressMeter.@showprogress "MCMC:" for i in 1:N_samples
        accepted = false

        #Update to store the rejected samples, or the multiplicities
        logL_new = 0
        x_new = 0
        accept_rate = 0

        chain[i,:] = x_curr
        loglikes[i] = logL_curr 

        while accepted == false
            N_attempts += 1

            multiplicity[i] += 1
            x_new = SampleNewPoint(x_curr, chol, errs_curr)
            logL_new = f(x_new)
            logα = logL_new - logL_curr
            #println(logα)
            u = rand(Float64)
            #Accepted new point?
            if log(u) <= logα
                accepted = true
            end


        end

        accept_rate = i/N_attempts*100
        #accept_rate = ((i-1)%adjust_steps)/N_attempts*100
        if (adapt_method == "rescale")

            #accept_rate = i/N_attempts*100
            if (i%adjust_steps == 0)
                #N_attempts = 0
                #println(accept_rate)
                if (accept_rate < 10)
                    if (verbose)
                        println("> --- Shrinking jumps (", round(accept_rate; digits=2), "% acceptance)...")
                    end
                    chol = chol*0.75
                    errs_curr = errs_curr*0.75
                end
                if (accept_rate > 25)
                    if (verbose)
                        println("> --- Increasing jumps (", round(accept_rate; digits=2), "% acceptance)...")
                    end
                    chol = chol*1.25
                    errs_curr = errs_curr*1.25
                end
            end
        end

        if (adapt_method == "estimate")
            
            if (i%adjust_steps == 0)
                weights =  ProbabilityWeights(multiplicity[1:i])
                #weights = ProbabilityWeights(exp.(loglikes[1:i]))
                #weights = weights/sum(weights)
                covmat_curr = cov(chain[1:i,:], weights)
                println(covmat_curr)
                errs_curr = sqrt.([covmat_curr[j, j] for j in 1:size(covmat_curr)[1]])
                dim = size(covmat_curr)[1]
                covmat_curr *= ((2.38)^2/dim)
                chol = cholesky((1/2)*(covmat_curr + covmat_curr')).L
            end
        end

        x_curr = x_new
        logL_curr = logL_new
       
        if (verbose)
            if (i%adjust_steps == 0)
                println("> Acceptance rate: ", i, "/",N_attempts, " (", round(accept_rate; digits=2), "%); c_f = ", exp(x_new[2]), "; loglike = ", logL_new)
            end
        end
        #if (i%100 == 0)
        #    if (accept_rate < 25)
        #        covmat_curr = covmat_curr*0.75
        #    end
        #    if (accept_rate > 40)
        #        covmat_curr = covmat_curr*1.25
        #    end
        #end

        #chain[i,:] = x0
        #loglikes[i] = logL_curr 

        if (i%100 == 0) 
            #full_chain = hcat(multiplicity[1:i], loglikes[1:i])
            #full_chain = hcat(full_chain, chain[1:i,:])
            #println(size(full_chain))
            full_chain = [multiplicity[1:i] loglikes[1:i] chain[1:i,:]]
            #println(size(full_chain2))
            if (isnothing(output_label) == false)
                writedlm("/Users/bradkav/Code/darkdress.jl/chains/chain_" * output_label * ".txt", full_chain, "   ")
                #writedlm("chains/likes_" * output_label * ".txt", (loglikes[1:i]), "   ")
            end
        end
    end

    full_chain = [multiplicity loglikes chain]
    #full_chain = hcat(full_chain, chain)
    return full_chain

end


function RandomSample(f, x0, N_samples, covmat; output_label=nothing, precision=false)
        
    errs_curr = sqrt.([covmat[i, i] for i in 1:size(covmat)[1]])

    #https://arxiv.org/pdf/1304.4473.pdf
    
    if (precision == false)
        chol = cholesky((1/2)*(covmat + covmat')).L
    else
        U1 = cholesky((1/2)*(covmat + covmat')).U
        scales = sqrt(inv(Diagonal(U1)))
        Γ = scales * U1 * scales
        chol = scales * inv(Γ) * scales
        println(chol*U1)
        #chol_temp = 1.0*chol
        #chol[3,3] = chol_temp[4,3]
        #chol[]

    end

    #println("Running new sampling method...")
    #sample_dist = Distributions.MvNormal((1/2)*(covmat + covmat'))

    chain = zeros(N_samples,length(x0))
    loglikes = zeros(N_samples)
    multiplicity = ones(N_samples)

    N_attempts = 0

    ProgressMeter.@showprogress "Sampling:" for i in 1:N_samples
        x_new = SampleNewPoint(x0, chol, 0.0*errs_curr)
        logL_new = f(x_new)
    
        chain[i,:] = x_new
        loglikes[i] = logL_new


        if (i%100 == 0) 
            full_chain = [multiplicity[1:i] loglikes[1:i] chain[1:i,:]]
            if (isnothing(output_label) == false)
                writedlm("/Users/bradkav/Code/darkdress.jl/chains/chain_" * output_label * ".txt", full_chain, "   ")
                #writedlm("chains/likes_" * output_label * ".txt", (loglikes[1:i]), "   ")
            end
        end
    end

    full_chain = [multiplicity loglikes chain]

    return full_chain

end