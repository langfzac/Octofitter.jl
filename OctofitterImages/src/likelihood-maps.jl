
const likemap_cols = (:map, :epoch, :platescale,)

"""
    LogLikelihoodMap(...)

One or more maps of likelihood vs Δright-ascension and Δdeclination computed with some other tool.
The platescale column maps pixels in the matrix to separation in milliarcseconds.
Pass a vector of named tuples with the following fields:
$likemap_cols

For example:
```julia
LogLikelihoodMap(
    (; epoch=1234.0, map=readfits("abc.fits"), platescale=19.4)
)
```
Contrast can be a function that returns the 1 sigma contrast of the image from a separation in mas to the same units as the image file.
Or, simply leave it out and it will be calculated for you.
Epoch is in MJD.
Band is a symbol which matches the one used in the planet's `Priors()` block.
Platescale is in mas/px.
"""
struct LogLikelihoodMap{TTable<:Table} <: Octofitter.AbstractLikelihood
    table::TTable
    function LogLikelihoodMap(observations...)
        table = Table(observations...)
        # Fallback to calculating contrast automatically
        if !issubset(likemap_cols, columnnames(table))
            error("Expected columns $likemap_cols")
        end
        # Create linear interpolators over the input likemaps
        mapinterp = map(table.map) do img
            LinearInterpolation(parent.(dims(img)), img, extrapolation_bc=convert(eltype(img), NaN))
        end
        table = Table(table; mapinterp)
        return new{typeof(table)}(table)
    end
end
LogLikelihoodMap(observations::NamedTuple...) = LogLikelihoodMap(observations)
export LogLikelihoodMap



"""
Likelihood of there being planets in a sequence of likemaps.
"""
function Octofitter.ln_like(likemaps::LogLikelihoodMap, θ_planet, orbit)
    
    # Resolve the combination of system and planet parameters
    # as a Visual{KepOrbit} object. This pre-computes
    # some factors used in various calculations.
    # elements = construct_elements(θ_system, θ_planet)


    likemaps_table = likemaps.table
    T = eltype(first(θ_planet))
    # ll = zero(T)
    ll = -Inf
    for i in eachindex(likemaps_table.epoch)

        soln = orbitsolve(orbit, likemaps_table.epoch[i])

        # Note the x reversal between RA and image coordinates
        x = -raoff(soln)
        y = +decoff(soln)

        # Get the photometry in this image at that location
        # Note in the following equations, subscript x (ₓ) represents the current position (both x and y)
        platescale = likemaps_table.platescale[i]
        χ² = likemaps_table.mapinterp[i](x/platescale, y/platescale)

        # When we get a position that falls outside of our available
        # data (e.g. under the coronagraph) we cannot say anything
        # about the likelihood. This is equivalent to σₓ→∞ or log likelihood 
        # of zero.
        if !isfinite(χ²)
            continue
        end
        if isfinite(ll)
           ll += χ²
        else
            ll = χ²
        end
    end

    return ll
end



# Generate new likemaps
# This isn't currently possible unless we can eg. find a way to plump the system log-likelihood into here.
# function Octofitter.generate_from_params(like::LogLikelihoodMap, θ_planet,  orbit::PlanetOrbits.AbstractOrbit)

#     newrows = map(like.table) do row
#         (;map, epoch, platescale) = row

#         injected = copy(map)
    
#         # Generate new astrometry point
#         os = orbitsolve(orbit, epoch)

#         ra = raoff(os)
#         dec = decoff(os)

#         phot = θ_planet[band]

#         # TODO: verify signs
#         dx = ra/platescale
#         dy = -dec/platescale
#         translation_tform = Translation(
#             mean(axes(psf,1))-mean(axes(image,1))+mean(dims(image,1))+dx,
#             mean(axes(psf,2))-mean(axes(image,2))+mean(dims(image,2))+dy
#         )
#         # TBD if we want to support rotations for handling negative sidelobes.

#         injected .+= psf_scaled

#         return merge(row, (;image=injected))
#     end

#     return LogLikelihoodMap(newrows)
# end