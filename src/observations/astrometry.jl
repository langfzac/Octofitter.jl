
# Astrometry Data type
const astrom_cols1 = (:epoch, :ra, :dec, :σ_ra, :σ_dec)
const astrom_cols3 = (:epoch, :pa, :sep, :σ_pa, :σ_sep)
struct Astrometry{TTable<:Table} <: AbstractObs
    table::TTable
    function Astrometry(observations...)
        table = Table(observations...)
        if !issubset(astrom_cols1, Tables.columnnames(table)) && 
           !issubset(astrom_cols3, Tables.columnnames(table))
            error("Expected columns $astrom_cols1 or $astrom_cols3")
        end
        return new{typeof(table)}(table)
    end
end
Astrometry(observations::NamedTuple...) = Astrometry(observations)
export Astrometry


# Plot recipe for astrometry data
@recipe function f(astrom::Astrometry)
   
    xflip --> true
    xguide --> "Δ right ascension (mas)"
    yguide --> "Δ declination (mas)"

    if hasproperty(astrom.table, :pa)
        x = astrom.table.sep .* cosd.(astrom.table.pa)
        y = astrom.table.sep .* sind.(astrom.table.pa)
        return -y, -x
    else
        xerror := astrom.table.σ_ra
        yerror := astrom.table.σ_dec

        return astrom.table.ra, astrom.table.dec
    end
end


# Astrometry likelihood function
function ln_like(astrom::Astrometry, θ_planet, orbit,)
    ll = 0.0
    for i in eachindex(astrom.table.epoch)

        star_δra =  0.
        star_δdec = 0.

        o = orbitsolve(orbit, astrom.table.epoch[i])
        # PA and Sep specified
        if hasproperty(astrom.table, :pa) && hasproperty(astrom.table, :sep)
            ρ = projectedseparation(o)
            pa = posangle(o)

            pa_diff = ( astrom.table.pa[i] - pa + π) % 2π - π;
            pa_diff = pa_diff < -π ? pa_diff + 2π : pa_diff;
            resid1 = pa_diff
            resid2 = astrom.table.sep[i] - ρ
            σ²1 = astrom.table.σ_pa[i ]^2
            σ²2 = astrom.table.σ_sep[i]^2
        # RA and DEC specified
        else
            x = raoff(o)# + star_δra
            y = decoff(o)# + star_δdec
            resid1 = astrom.table.ra[i] - x
            resid2 = astrom.table.dec[i] - y
            σ²1 = astrom.table.σ_ra[i ]^2
            σ²2 = astrom.table.σ_dec[i]^2
        end

        χ²1 = -(1/2)*resid1^2 / σ²1 - log(sqrt(2π * σ²1))
        χ²2 = -(1/2)*resid2^2 / σ²2 - log(sqrt(2π * σ²2))
        ll += χ²1 + χ²2
    end
    return ll
end
