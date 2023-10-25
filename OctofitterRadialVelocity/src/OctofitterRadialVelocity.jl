module OctofitterRadialVelocity

using Octofitter
using PlanetOrbits
using Tables, TypedTables
using Distributions
using DataDeps
using LoopVectorization
using StrideArrays
using RecipesBase

# Radial Velocity data type
const rv_cols = (:epoch, :rv, :σ_rv)


"""
    RadialVelocityLikelihood(
        (;inst_idx=1, epoch=5000.0,  rv=−6.54, σ_rv=1.30),
        (;inst_idx=1, epoch=5050.1,  rv=−3.33, σ_rv=1.09),
        (;inst_idx=1, epoch=5100.2,  rv=7.90,  σ_rv=.11),
    )

Represents a likelihood function of relative astometry between a host star and a secondary body.
`:epoch` (mjd), `:rv` (km/s), and `:σ_rv` (km/s), and `:inst_idx` are all required.

`:inst_idx` is used to distinguish RV time series between insturments so that they may optionally
be fit with different zero points and jitters.
In addition to the example above, any Tables.jl compatible source can be provided.
"""
struct RadialVelocityLikelihood{TTable<:Table} <: Octofitter.AbstractLikelihood
    table::TTable
    function RadialVelocityLikelihood(observations...)
        table = Table(observations...)
        if !issubset(rv_cols, Tables.columnnames(table))
            error("Ecpected columns $rv_cols")
        end
        return new{typeof(table)}(table)
    end
end
RadialVelocityLikelihood(observations::NamedTuple...) = RadialVelocityLikelihood(observations)
export RadialVelocityLikelihood

"""
Radial velocity likelihood.
"""
function Octofitter.ln_like(rv::RadialVelocityLikelihood, θ_system, elements, num_epochs::Val{L}=Val(length(rv.table))) where L
    T = promote_type(typeof(θ_system.M), typeof(θ_system.plx))
    ll = zero(T)

    epochs = rv.table.epoch
    σ_rvs = rv.table.σ_rv
    inst_idxs = rv.table.inst_idx
    rvs = rv.table.rv

    # # # TODO: This is a debug override. This is forcing omega to be shifted by pi
    # # # to compare with orvara.
    # nt = θ_system.planets.B
    # nt = merge(nt,(;ω=nt.ω+π))
    # elements = (Octofitter.construct_elements(Visual{KepOrbit}, θ_system, nt),)

    # single_instrument_mode = !hasproperty(rv.table, :inst_idx)
    barycentric_rv_inst = (
        hasproperty(θ_system, :rv0_1) ? θ_system.rv0_1 : convert(T, NaN),
        hasproperty(θ_system, :rv0_2) ? θ_system.rv0_2 : convert(T, NaN),
        hasproperty(θ_system, :rv0_3) ? θ_system.rv0_3 : convert(T, NaN),
        hasproperty(θ_system, :rv0_4) ? θ_system.rv0_4 : convert(T, NaN),
        hasproperty(θ_system, :rv0_5) ? θ_system.rv0_5 : convert(T, NaN),
        hasproperty(θ_system, :rv0_6) ? θ_system.rv0_6 : convert(T, NaN),
        hasproperty(θ_system, :rv0_7) ? θ_system.rv0_7 : convert(T, NaN),
        hasproperty(θ_system, :rv0_8) ? θ_system.rv0_8 : convert(T, NaN),
        hasproperty(θ_system, :rv0_9) ? θ_system.rv0_9 : convert(T, NaN),
        hasproperty(θ_system, :rv0_10) ? θ_system.rv0_10 : convert(T, NaN),
    )
    jitter_inst = (
        hasproperty(θ_system, :jitter_1) ? θ_system.jitter_1 : convert(T, NaN),
        hasproperty(θ_system, :jitter_2) ? θ_system.jitter_2 : convert(T, NaN),
        hasproperty(θ_system, :jitter_3) ? θ_system.jitter_3 : convert(T, NaN),
        hasproperty(θ_system, :jitter_4) ? θ_system.jitter_4 : convert(T, NaN),
        hasproperty(θ_system, :jitter_5) ? θ_system.jitter_5 : convert(T, NaN),
        hasproperty(θ_system, :jitter_6) ? θ_system.jitter_6 : convert(T, NaN),
        hasproperty(θ_system, :jitter_7) ? θ_system.jitter_7 : convert(T, NaN),
        hasproperty(θ_system, :jitter_8) ? θ_system.jitter_8 : convert(T, NaN),
        hasproperty(θ_system, :jitter_9) ? θ_system.jitter_9 : convert(T, NaN),
        hasproperty(θ_system, :jitter_10) ? θ_system.jitter_10 : convert(T, NaN),
    )
    # Vector of radial velocity of the star at each epoch. Go through and sum up the influence of
    # each planet and put it into here. 
    # Then loop through and get likelihood.
    # Hopefully this is more efficient than looping over each planet at each epoch and adding up the likelihood.
    rv_star = StrideArray{T}(undef, (StaticInt(num_epochs),))
    fill!(rv_star, 0)
    for planet_i in eachindex(elements)
        orbit = elements[planet_i]
        # Need to structarrays orbit???
        planet_mass = θ_system.planets[planet_i].mass
        # @turbo
        for epoch_i in eachindex(epochs)
            rv_star[epoch_i] += radvel(orbit, epochs[epoch_i], planet_mass*Octofitter.mjup2msol)
        end
    end
    # @turbo 
    for i in eachindex(epochs)
        # Each measurement is tagged with a jitter and rv zero point variable.
        # We then query the system variables for them.
        # A previous implementation used symbols instead of indices but it was too slow.
        inst_idx = inst_idxs[i]
        if isnan(jitter_inst[inst_idx]) || isnan(barycentric_rv_inst[inst_idx])
            error(lazy"`jitter_$inst_idx` and `rv0_$inst_idx` must be provided")
        end
            resid = rv_star[i] + barycentric_rv_inst[inst_idx] - rvs[i]
        σ² = σ_rvs[i]^2 + jitter_inst[inst_idx]^2
        χ² = -0.5resid^2 / σ² - log(sqrt(2π * σ²))
        ll += χ²

        # Leveraging Distributions.jl to make this clearer:
        # ll += logpdf(Normal(0, sqrt(σ²)), resid)
    end

    return ll
end




# Generate new radial velocity observations for a planet
function Octofitter.generate_from_params(like::RadialVelocityLikelihood, θ_planet, elem::PlanetOrbits.AbstractOrbit)

    # Get epochs and uncertainties from observations
    epochs = like.table.epoch 
    σ_rvs = like.table.σ_rv 

    # Generate new planet radial velocity data
    rvs = DirectOribts.radvel.(elem, epochs)
    radvel_table = Table(epoch=epochs, rv=rvs, σ_rv=σ_rvs)

    return RadialVelocityLikelihood(radvel_table)
end




# Generate new radial velocity observations for a star
function Octofitter.generate_from_params(like::RadialVelocityLikelihood, θ_system, orbits::Vector{<:Visual{KepOrbit}})

    # Get epochs, uncertainties, and planet masses from observations and parameters
    epochs = like.table.epoch 
    σ_rvs = like.table.σ_rv 
    planet_masses = [θ_planet.mass for θ_planet in θ_system.planets] .* 0.000954588 # Mjup -> Msun

    # Generate new star radial velocity data
    rvs = radvel.(reshape(orbits, :, 1), epochs, transpose(planet_masses))
    # TODO: Question: is adding jitter like this appropriate in a generative model? I think so.
    noise = randn(length(epochs)) .* θ_system.jitter
    rvs = sum(rvs, dims=2)[:,1] .+ θ_system.rv .+ noise
    radvel_table = Table(epoch=epochs, rv=rvs, σ_rv=σ_rvs)

    return RadialVelocityLikelihood(radvel_table)
end

mjd2jd(mjd) = mjd - 2400000.5
jd2mjd(jd) = jd + 2400000.5



include("harps.jl")
include("hires.jl")
include("lick.jl")
include("radvel.jl")


# Plot recipe for astrometry data
@recipe function f(rv::RadialVelocityLikelihood)
   
    xguide --> "time (mjd)"
    yguide --> "radvel (m/s)"

    multiple_instruments = hasproperty(rv.table,:inst_idx) && 
                           length(unique(rv.table.inst_idx)) > 1
    if !multiple_instruments
        @series begin
            color --> :black
            label := nothing
            seriestype := :scatter
            markersize--> 0
            yerr := rv.table.σ_rv
            rv.table.epoch, rv.table.rv
        end
    else
        for inst_idx in sort(unique(rv.table.inst_idx))
            @series begin
                label := nothing
                seriestype := :scatter
                markersize--> 0
                color-->inst_idx
                markerstrokecolor-->inst_idx
                yerr := rv.table.σ_rv[rv.table.inst_idx.==inst_idx]
                rv.table.epoch[rv.table.inst_idx.==inst_idx], rv.table.rv[rv.table.inst_idx.==inst_idx]
            end
        end
    end


end




function __init__()

    register(DataDep("HARPS_RVBank",
        """
        Dataset:     A public HARPS radial velocity database corrected for systematic errors
        Author:      Trifonov et al.
        License:     CC0-1.0
        Publication: https://www.aanda.org/articles/aa/full_html/2020/04/aa36686-19/aa36686-19.html
        Website:     https://www2.mpia-hd.mpg.de/homes/trifonov/HARPS_RVBank.html

        A public HARPS radial velocity database corrected for systematic errors. (2020)
        
        File size: 132MiB
        """,
        "https://www2.mpia-hd.mpg.de/homes/trifonov/HARPS_RVBank_v1.csv",
        "17b2a7f47569de11ff1747a96997203431c81586ffcf08212ddaa250bb879a40",
    ))

    register(DataDep("HIRES_rvs",
        """
        Dataset:     A public HIRES radial velocity database corrected for systematic errors
        Author:      Butler et al.
        License:     
        Publication: https://ui.adsabs.harvard.edu/abs/2017yCat..51530208B/abstract
        Website:     https://ebps.carnegiescience.edu/data/hireskeck-data

        File size: 3.7MiB
        """,
        "https://drive.google.com/uc?id=10xCy8UIH8wUAnNJ8zCN8kfFzazWnw-f_&export=download",
        "ad68c2edb69150318e8d47e34189fe104f2a5194a4fcd363c78c741755893251",
        post_fetch_method=unpack
    ))


    register(DataDep("Lick_rvs",
        """
        Dataset:     The Twenty-Five Year Lick Planet Search
        Author:      Fischer et al.
        License:     
        Publication: https://iopscience.iop.org/article/10.1088/0067-0049/210/1/5#apjs488421t2

        A public Lick radial velocity database.
        
        File size: 780k
        """,
        "https://content.cld.iop.org/journals/0067-0049/210/1/5/revision1/apjs488421t2_mrt.txt?Expires=1698868925&Signature=YyKJ4p64PeQg2sh~VAYj6aXxH8b-lH0F0lS6GF0YP07V7oaZWzM4sthpMRldUE7cHQZbMkwoW0R-Jq2FymIYqIlAnT1-qs-y~JifD1A1WThaBOEP2gl5JGgDOGXXMCLK4VuKM3ZucSUu9TWIb3vbNqrG7l~V9LIs-K2bW~KcM-syfRzJ1YC6TSiej1PHJVhoxN-SUQRAw2lkLVQ-eea30IFOw9RSmYFqrqUQGnwx7fdkbTd5ZSvQ~BmB0HZjsav890rZpEVWlCs8ITLpKab3aEysIptlezpS90boNDi3CR-p7We2M9WfibcsemIa72HH7cZS~S1Ri8QTQra5nTY8eQ__&Key-Pair-Id=KL1D8TIY3N7T8",
    ))
    
    return
end

end
