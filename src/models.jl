

function ln_like_pma(θ_system, pma::ProperMotionAnom)
    ll = 0.0
    
    for i in eachindex(pma.ra_epoch, pma.dec_epoch)
        pm_ra_star = 0.0
        pm_dec_star = 0.0
        
        # The model can support multiple planets
        for key in keys(θ_system.planets)
            θ_planet = θ_system.planets[key]

            if θ_planet.mass < 0
                return -Inf
            end

            # TODO: we are creating these from scratch for each observation instead of sharing them
            orbit = construct_elements(θ_system, θ_planet)

            # RA and dec epochs are usually slightly different
            # Note the unit conversion here from jupiter masses to solar masses to 
            # make it the same unit as the stellar mass (element.mu)
            pm_ra_star += propmotionanom(orbit, pma.ra_epoch[i], θ_planet.mass*mjup2msol)[1]
            pm_dec_star += propmotionanom(orbit, pma.dec_epoch[i], θ_planet.mass*mjup2msol)[2]
        end

        residx = pm_ra_star - pma.pm_ra[i]
        residy = pm_dec_star - pma.pm_dec[i]
        σ²x = pma.σ_pm_ra[i]^2
        σ²y = pma.σ_pm_dec[i]^2
        χ²x = -0.5residx^2 / σ²x - log(sqrt(2π * σ²x))
        χ²y = -0.5residy^2 / σ²y - log(sqrt(2π * σ²y))

        ll += χ²x + χ²y
    end

    return ll
end

# TODO: image modelling for multi planet systems do not consider how "removing" one planet
# might increase the contrast of another.
# function ln_like_images(θ_system, system)
#     ll = 0.0
#     for key in keys(θ_system.planets)
#         θ_planet = θ_system.planets[key]
#         elements = construct_elements(θ_system, θ_planet)

#         if (elements.a <= 0 ||
#             elements.e < 0 ||
#             elements.plx < 0 ||
#             elements.μ <= 0)
#             ll += NaN
#             continue
#         end


#         ll += ln_like_images_element(elements, θ_planet, system)
#     end

#     # # Connect the flux at each epoch to an overall flux in this band for this planet
#     # # fᵢ = θ_band.epochs
#     # # ll += -1/2 * sum(
#     # #     (fᵢ .- θ_band.f).^2
#     # # ) / (θ_band.σ_f² * mean(fᵢ)^2)

#     # And connect that flux to a modelled Teff and mass
#     # f_model = model_interpolator(θ_planet.Teff, θ_planet.mass)
#     # ll += -1/2 * (f_model - θ_band)^2 /  (θ_planet.σ_f_model² * f_model^2)

#     return ll
# end

function ln_like_images(elements::DirectOrbits.AbstractElements, θ_planet, system)
    images = system.images
    T = eltype(θ_planet)
    ll = zero(T)
    for i in eachindex(images.epoch)
       
        # Calculate position at this epoch
        ra, dec = kep2cart(elements, images.epoch[i])
        # x must be negated to go from sky coordinates (ra increasing to left) to image coordinates (ra increasing to right).
        x = -ra
        y = dec

        # Get the photometry in this image at that location
        # Note in the following equations, subscript x (ₓ) represents the current position (both x and y)
        f̃ₓ = lookup_coord(images.image[i], (x, y), images.platescale[i])

        # Find the uncertainty in that photometry value (i.e. the contrast)
        r = √(x^2 + y^2)
        σₓ = images.contrast[i](r / images.platescale[i])

        # When we get a position that falls outside of our available
        # data (e.g. under the coronagraph) we cannot say anything
        # about the likelihood. This is equivalent to σₓ→∞ or log likelihood 
        # of zero.
        if !isfinite(σₓ) || !isfinite(f̃ₓ)
            continue
        end

        band = images.band[i]

        # Verify the user has specified a prior or model for this band.
        if !hasproperty(θ_planet, band)
            error("No photometry prior for the band $band was specified, and neither was mass.")
        end
        f_band = getproperty(θ_planet, band)

        # Direct imaging likelihood.
        # Notes: we are assuming that the different images fed in are not correlated.
        # The general multivariate Gaussian likleihood is exp(-1/2 (x⃗-μ⃗)ᵀ𝚺⁻¹(x⃗-μ⃗)) + √((2π)ᵏ|𝚺|)
        # Because the images are uncorrelated, 𝚺 is diagonal and we can separate the equation
        # into a a product of univariate Gaussian likelihoods or sum of log-likelihoods.
        # That term for each image is given below.

        # Ruffio et al 2017, eqn (31)
        # Mawet et al 2019, eqn (8)


        σₓ² = σₓ^2
        ll += -1 / (2σₓ²) * (f_band^2 - 2f_band * f̃ₓ) 



        # l = -1 / (2σₓ²) * (f_band^2 - 2f_band * f̃ₓ) - log(sqrt(2π * σₓ²))
        # ll += -1 / (2σₓ²) * (f_band - f̃ₓ)^2 - log(sqrt(2π * σₓ²))
        # n = Normal(f̃ₓ, σₓ)
        # ll += logpdf(n, f_band)
    end

    return ll
end

# # If there is no astrometry atached to the planet, it does not contribute anything to the likelihood function
# function ln_like_astrom(elements, planet::Planet{<:Any,<:Any,Nothing})
#     return 0.0
# end

# Astrometry
function ln_like_astrom(elements, planet::Planet{<:Any,<:Any,<:Astrometry})
    ll = 0.0
    
    astrom = astrometry(planet)
    for i in eachindex(astrom.epoch)
        x, y = kep2cart(elements, astrom.epoch[i])
        residx = astrom.ra[i ] - x
        residy = astrom.dec[i] - y
        σ²x = astrom.σ_ra[i ]^2
        σ²y = astrom.σ_dec[i]^2
        χ²x = -(1/2)*residx^2 / σ²x - log(sqrt(2π * σ²x))
        χ²y = -(1/2)*residy^2 / σ²y - log(sqrt(2π * σ²y))
        ll += χ²x + χ²y
    end
    return ll
end


function ln_like(θ_system, system::System)
    ll = 0.0
    if θ_system.μ <= 0
        return -Inf
    end
    # Go through each planet in the model and add its contribution
    # to the ln-likelihood.
    for (θ_planet, planet) in zip(θ_system.planets, system.planets)
        if θ_planet.a <= 0 || θ_planet.e < 0 || θ_planet.e >= 1
            return -Inf
        end

        # Resolve the combination of system and planet parameters
        # as a KeplerianElements object. This pre-computes
        # some factors used in various calculations.
        kep_elements = construct_elements(θ_system, θ_planet)

        if !isnothing(planet.astrometry)
            ll += ln_like_astrom(kep_elements, planet)
        end

        if !isnothing(system.images)
            ll += ln_like_images(kep_elements, θ_planet, system)
        end

    end

    # TODO: PMA is re-calculating some factores used in kep_elements.
    # Should think of a way to integrate it into the loop above
    if !isnothing(system.propermotionanom)
        ll += ln_like_pma(θ_system, system.propermotionanom)
    end

    return ll
end



    # Hierarchical parameters over multiple planets
    # if haskey(system.priors.priors, :σ_i²)
    #     # If the sampler wanders into negative variances, return early to prevent
    #     # taking square roots of negative values later on
    #     if θ.σ_i² < 0
    #         return -Inf
    #     end

    #     # hierarchical priors here
    #     sum_iᵢ = zero(θ.i)
    #     sum_iᵢθi² = zero(θ.i)
    #     for θ_planet in θ.planets
    #         sum_iᵢ += θ_planet.i
    #         sum_iᵢθi² += (θ_planet.i .- θ.i)^2
    #     end
    #     ll += -1/2 * sum_iᵢθi² / θ.σ_i²  - log(sqrt(2π * θ.σ_i²))
    # end
    # if haskey(system.priors.priors, :σ_Ω²)
    #     # If the sampler wanders into negative variances, return early to prevent
    #     # taking square roots of negative values later on
    #     if θ.σ_Ω² < 0
    #         return -Inf
    #     end

    #     # hierarchical priors here
    #     sum_Ωᵢ = zero(θ.Ω)
    #     sum_ΩᵢθΩ² = zero(θ.Ω)
    #     for θ_planet in θ.planets
    #         _, Ωᵢ, _  = get_ωΩτ(θ_system, θ_planet)
    #         sum_Ωᵢ += Ωᵢ
    #         sum_ΩᵢθΩ² += (Ωᵢ .- θ.Ω)^2
    #     end
    #     ll += -1/2 * sum_ΩᵢθΩ² / θ.σ_Ω²  - log(sqrt(2π * θ.σ_Ω²))
    # end






# # This is a straight forward implementation that unfortunately is not type stable.
# # This is because we are looping over a heterogeneous container
# function make_ln_prior(priors)
#     return function ln_prior(params)
#         lp = zero(first(params))
#         for i in eachindex(params)
#             pd = priors[i]
#             param = params[i]
#             lp += logpdf(pd, param)
#         end
#         return lp 
#     end
# end

function make_ln_prior(system::System)

    # This function uses meta-programming to unroll all the code at compile time.
    # This is a big performance win, since it avoids looping over all the different
    # types of distributions that might be specified as priors.
    # Otherwise we would have to loop through an abstract vector and do runtime dispatch!
    # This way all the code gets inlined into a single tight numberical function in most cases.

    i = 0
    prior_evaluations = Expr[]

    # System priors
    for prior_distribution in values(system.priors.priors)
        i += 1
        ex = :(
            lp += $logpdf($prior_distribution, arr[$i])
        )
        push!(prior_evaluations,ex)
    end

    # Planet priors
    for planet in system.planets
        # for prior_distribution in values(planet.priors.priors)
        for (key, prior_distribution) in zip(keys(planet.priors.priors), values(planet.priors.priors))
            i += 1
            # Work around for Beta distributions.
            # Outside of the range [0,1] logpdf returns -Inf.
            # This works fine, but AutoDiff outside this range causes a DomainError.
            if typeof(prior_distribution) <: Beta
                ex = :(
                    lp += 0 <= arr[$i] < 1 ? $logpdf($prior_distribution, arr[$i]) : -Inf
                )
            else
                ex = :(
                    lp += $logpdf($prior_distribution, arr[$i])
                )
            end
            push!(prior_evaluations,ex)
        end
    end

    # Here is the function we return.
    # It maps an array of parameters into our nested named tuple structure
    # Note: eval() would normally work fine here, but sometimes we can hit "world age problemms"
    # The RuntimeGeneratedFunctions package avoids these in all cases.
    return @RuntimeGeneratedFunction(:(function (arr)
        l = $i
        @boundscheck if length(arr) != l
            error("Expected exactly $l elements in array (got $(length(arr)))")
        end
        lp = zero(first(arr))
        # Add contributions from planet priors
        @inbounds begin
           $(prior_evaluations...) 
        end
        return lp
    end))
end

# Same as above, but assumes the input to the log prior was sampled
# using transformed distributions from Bijectors.jl
# Uses logpdf_with_trans() instead of logpdf to make the necessary corrections.
function make_ln_prior_transformed(system::System)

    i = 0
    prior_evaluations = Expr[]

    # System priors
    for prior_distribution in values(system.priors.priors)
        i += 1
        ex = :(
            lp += $logpdf_with_trans($prior_distribution, arr[$i], true)
        )
        push!(prior_evaluations,ex)
    end

    # Planet priors
    for planet in system.planets
        # for prior_distribution in values(planet.priors.priors)
        for (key, prior_distribution) in zip(keys(planet.priors.priors), values(planet.priors.priors))
            i += 1
            ex = :(
                lp += $logpdf_with_trans($prior_distribution, arr[$i], true)
            )
            push!(prior_evaluations,ex)
        end
    end

    # Here is the function we return.
    # It maps an array of parameters into our nested named tuple structure
    # Note: eval() would normally work fine here, but sometimes we can hit "world age problemms"
    # The RuntimeGeneratedFunctions package avoids these in all cases.
    return @RuntimeGeneratedFunction(:(function (arr)
        l = $i
        @boundscheck if length(arr) != l
            error("Expected exactly $l elements in array (got $(length(arr)))")
        end
        lp = zero(first(arr))
        # Add contributions from planet priors
        @inbounds begin
           $(prior_evaluations...) 
        end
        return lp
    end))
end

function _list_priors(system::System)

    priors_vec = []
    # System priors
    for prior_distribution in values(system.priors.priors)
        push!(priors_vec,prior_distribution)
    end

    # Planet priors
    for planet in system.planets
        # for prior_distribution in values(planet.priors.priors)
        for (key, prior_distribution) in zip(keys(planet.priors.priors), values(planet.priors.priors))
            push!(priors_vec, prior_distribution)
        end
    end

    # narrow the type
    return map(identity, priors_vec)
end


# function ln_prior(θ, system::System)
#     lp = 0.0
#     lp += system.priors.ln_prior(θ)
#     # for (planet, θ_planet) in zip(system.planets, θ.planets)
#     # for (planet, θ_planet) in zip(Tuple(system.planets), Tuple(θ.planets))
#     for key in keys(system.planets)
#         lp += system.planets[key].priors.ln_prior(θ.planets[key])
#     end

#     return lp
# end

function ln_post(θ, system::System)
    return ln_prior(θ, system) + ln_like(θ, system)
end


"""
    make_arr2nt(system::System)

Returns a function that maps an array of parameter values (e.g. sampled from priors,
sampled from posterior, dual numbers, etc.) to a nested named tuple structure suitable
for passing into `ln_like`.
In the process, this evaluates all Deterministic variables defined in the model.

Example:
```julia
julia> arr2nt = DirectDetections.make_arr2nt(system);
julia> params = DirectDetections.sample_priors(system)
9-element Vector{Float64}:
  1.581678216418196
 29.019567489624023
  1.805551918844831
  4.527607604709967
 -2.4432825071288837
 10.165695048003027
 -2.8667829383306063
  0.0006589349005787781
 -1.2600092725864576
julia> arr2nt(params)
(μ = 1.581678216418196, plx = 29.019567489624023, planets = (B = (τ = 1.805551918844831, ω = 4.527607604709967, i = -2.4432825071288837, Ω = 10.165695048003027, loge = -2.8667829383306063, loga = 0.0006589349005787781, logm = -1.2600092725864576, e = 0.001358992505357737, a = 1.0015184052910453, mass = 0.054952914078000334),))
```
"""
function make_arr2nt(system::System)

    # Roll flat vector of variables e.g. drawn from priors or posterior
    # into a nested NamedTuple structure for e.g. ln_like
    i = 0
    body_sys_priors = Expr[]

    # This function uses meta-programming to unroll all the code at compile time.
    # This is a big performance win, since it avoids looping over all the functions
    # etc. In fact, most user defined e.g. Derived callbacks can get inlined right here.

    # Priors
    for key in keys(system.priors.priors)
        i += 1
        ex = :(
            $key = arr[$i]
        )
        push!(body_sys_priors,ex)
    end

    # Deterministic variables for the system
    body_sys_determ = Expr[]
    if !isnothing(system.deterministic)
        for (key,func) in zip(keys(system.deterministic.variables), values(system.deterministic.variables))
            ex = :(
                $key = $func(sys)
            )
            push!(body_sys_determ,ex)
        end
    end

    # Planets: priors & deterministic variables
    body_planets = Expr[]
    for planet in system.planets
        
        # Priors
        body_planet_priors = Expr[]
        for key in keys(planet.priors.priors)
            i += 1
            ex = :(
                $key = arr[$i]
            )
            push!(body_planet_priors,ex)
        end

        if isnothing(planet.deterministic)
            ex = :(
                $(planet.name) = (;
                    $(body_planet_priors...)
                )
            )
        # Resolve deterministic vars.
        else
            body_planet_determ = Expr[]
            for (key,func) in zip(keys(planet.deterministic.variables), values(planet.deterministic.variables))
                ex = :(
                    $key = $func(sys_res, (;$(body_planet_priors...)))
                )
                push!(body_planet_determ,ex)
            end

            ex = :(
                $(planet.name) = (;
                    $(body_planet_priors...),
                    $(body_planet_determ...)
                )
            )
        end
        push!(body_planets,ex)
    end

    # Here is the function we return.
    # It maps an array of parameters into our nested named tuple structure
    # Note: eval() would normally work fine here, but sometimes we can hit "world age problemms"
    # The RuntimeGeneratedFunctions package avoids these in all cases.
    func = @RuntimeGeneratedFunction(:(function (arr)
        l = $i
        @boundscheck if length(arr) != l
            error("Expected exactly $l elements in array (got $(length(arr)))")
        end
        # Expand system variables from priors
        sys = (;$(body_sys_priors...))
        # Resolve deterministic system variables
        sys_res = (;
            sys...,
            $(body_sys_determ...)
        )
        # Get resolved planets
        pln = (;$(body_planets...))
        # Merge planets into resolved system
        sys_res_pln = (;sys..., planets=pln)
        return sys_res_pln
    end))

    return func
end


