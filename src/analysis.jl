
# This file contains funtions for analysing results from chains.
using Dates

"""
    projectpositions(chains.planets[1], mjd("2020-02-02"))

Given the posterior for a particular planet in the model and a modified julian date(s),
return `ra` and `dec` offsets in mas for each sampling in the posterior.
"""
function projectpositions(chains, planet_key, times)

    ras = zeros(size(chains,1),size(chains,3),length(times))
    decs = zeros(size(chains,1),size(chains,3),length(times))
    
    for is in collect(Iterators.partition(1:size(chains,1), 5000))
        for j in 1:size(chains,3)
            els = Octofitter.construct_elements(chains, planet_key, is .* j)
            for (i,el) in zip(is,els)
                for (k, t) in enumerate(times)
                    o = orbitsolve(el, t)
                    ras[i,j,k] = raoff(o)
                    decs[i,j,k] = decoff(o)
                end
            end
        end
    end
    return ras, decs
end
export projectpositions



function bayesfactor(chain, planet, property)
    prior = model.system.planets[planet].priors.priors[property]
    post = chain["$planet.$property"]

    # Fit a histogram to the posterior.
    # TODO: in future this could be a KDE
    nbins = floor(Int,sqrt(length(post))/2)
    h = fit(Histogram, vec(post), range(0,maximum(post),length=nbins))
    hn = normalize(h, mode=:pdf)
    
    i_first_nonzero_bin = findfirst(>(0), hn.weights)
    bin_centre = mean(hn.edges[1][i_first_nonzero_bin:i_first_nonzero_bin+1])
    
    lnbf = logpdf(prior, bin_centre) - log(hn.weights[i_first_nonzero_bin])

    # Now fit a Gaussian to the posterior in order to estimate bayes factors for
    # very strong detections but assuming the posterior is normally distributed
    post = fit(Normal, vec(post))
    lnbf_exrap = logpdf(prior, 1e-6) - logpdf(post, 1e-6)

    return (;lnbf, islowerlimit=i_first_nonzero_bin>1, lnbf_exrap)
end




"""
    plotchains(
        chain, planet_key;
        N=1500,
        ii = rand(1:size(chain,1)*size(chain,3), N),
        color=length(model.system.planets) == 0 || !haskey(chain, string(planet_key)*"_a") ? nothing : string(planet_key)*"_a",
        colorbartitle=color,
        clims=nothing,
        cmap=:plasma,
        alpha=30/length(ii),
        attime=nothing,
        kwargs...,
    )

Draw samples from a posterior chain for a given planet given by name `planet_key` and visualize them in some way.
Use `kind` to control what plot is made. A few options: :astrometry, :radvel, :trueanom, :meananom, :eccanom, :x, :y, :z, (:x, :y), :raoff, :decoff, :pmra, :pmdec, :accra, :accdec, :radvel, :posangle, :projectedseparation.
See PlanetOrbits documentation for more details.

Inputs:
* chain                   The chain to draw from
* planet_key              Planet name in the model (symbol)
* N=1500                  Number of samples to draw for the plot 
* kind=nothing            Specify what kind of plot to make. 
* ii=...                  Specific row numbers to use, if you want to e.g. plot the same 100 samples in a few different plots
* color="planet_key_a"   Column name to to map colors to. Semi-major axis by default but can be any column or an arbitrary array.
* colorbartitle=color     Name for colourbar
* clims=nothing           Tuple of colour limits (min and max)
* cmap=:plasma            Colormap
* alpha=...               Transparency of the lines
"""
function plotchains end


"""
Same as `plotchains` but plots into an existing figure.
"""
function plotchains! end


"""
    octoplot(
        system::Union{Octofitter.System,Octofitter.Model,
        chain::Octofitter.MCMCChains.Chains;
        fname="\$(system.name)-plot-grid",
        color = "\$(first(keys(system.planets)))_e",
        colorbartitle=string(color),
        clims = quantile(vec(chain[color]),(0.01, 0.99)),
        cmap = :plasma,
        dpi=200,
        kwargs...
    )

Given a `System` or `Octofitter.LogDensityModel` and an MCMC Chain (sampled either from 
the posterior or the prior), produce a panel of 9 plots visualizing the orbits.
The output is saved to a file based on the name of the system (`fname`).
"""
function octoplot(model::System, ::Any)
    error("You must load the Plots.jl package (`using Plots`) before calling this function. Then, pass your model and chain result as arguments.")
end
function Octofitter.octoplot(model::Octofitter.LogDensityModel, args...; kwargs...)
    return Octofitter.octoplot(model.system, args...; kwargs...)
end

function octocorner end
function octocorner(model::System, ::Any)
    error("You must load the Makie package (eg `using CairoMakie`) and PairPlots package (eg `using PairPlots`) before calling this function. Then, pass your model and chain result as arguments.")
end
function Octofitter.octocorner(model::Octofitter.LogDensityModel, args...; kwargs...)
    return Octofitter.octocorner(model.system, args...; kwargs...)
end


function timeplotgrid end
export plotchains, plotchains!, octoplot, octocorner
