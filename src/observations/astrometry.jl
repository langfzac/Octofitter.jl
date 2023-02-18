
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
using LinearAlgebra
@recipe function f(astrom::Astrometry)
   
    xflip --> true
    xguide --> "Δ right ascension (mas)"
    yguide --> "Δ declination (mas)"
    color --> :black

    if hasproperty(astrom.table, :pa)
        x = astrom.table.sep# .* cosd.(astrom.table.pa)
        y = astrom.table.pa# .* sind.(astrom.table.pa)

        if hasproperty(astrom.table, :cor)
            cor = astrom.table.cor
        else
            cor = 0
        end

        
        σ₁ = astrom.table.σ_sep
        σ₂ = astrom.table.σ_pa

        error_ellipses = broadcast(x,y,σ₁,cor,σ₂) do x,y,σ₁,cor,σ₂
            Σ = [
                σ₁^2        cor*σ₁*σ₂
                cor*σ₁*σ₂   σ₂^2
            ]
            vals, vecs = eigen(Σ) # should be real and sorted by real eigenvalue
            length_major = sqrt(vals[2])
            length_minor = sqrt(vals[1])
            λ = vecs[:,2]
            α = atan(λ[2],λ[1])
            @show length_major length_minor α

            xvals = [
                # Major axis
                x - length_major*cos(α),
                x,
                x + length_major*cos(α),
                NaN,
                # Minor axis
                x - length_minor*cos(α+π/2),
                x,
                x + length_minor*cos(α+π/2),
                NaN,
            ]
            yvals = [
                # Major axis
                y - length_major*sin(α),
                y,
                y + length_major*sin(α),
                NaN,
                # Minor axis
                y - length_minor*sin(α+π/2),
                y,
                y + length_minor*sin(α+π/2),
                NaN,
            ]
            xvals, yvals
        end
        xs = Base.vcat(getindex.(error_ellipses,1)...)
        ys = Base.vcat(getindex.(error_ellipses,2)...)
        @show xs ys

        @series begin
            label := nothing
            xs .* cos.(ys), xs .* sin.(ys)
        end
        @series begin
            x = astrom.table.sep .* cos.(astrom.table.pa)
            y = astrom.table.sep .* sin.(astrom.table.pa)
            seriestype:=:scatter
            x, y
        end
    else
        x = astrom.table.ra
        y = astrom.table.dec

        if hasproperty(astrom.table, :cor)
            cor = astrom.table.cor
        else
            cor = 0
        end
        
        σ₁ = astrom.table.σ_ra
        σ₂ = astrom.table.σ_dec

        error_ellipses = broadcast(x,y,σ₁,cor,σ₂) do x,y,σ₁,cor,σ₂
            Σ = [
                σ₁^2        cor*σ₁*σ₂
                cor*σ₁*σ₂   σ₂^2
            ]
            vals, vecs = eigen(Σ) # should be real and sorted by real eigenvalue
            length_major = sqrt(vals[2])
            length_minor = sqrt(vals[1])
            λ = vecs[:,2]
            α = atan(λ[2],λ[1])

            xvals = [
                # Major axis
                x - length_major*cos(α),
                x + length_major*cos(α),
                NaN,
                # Minor axis
                x - length_minor*cos(α+π/2),
                x + length_minor*cos(α+π/2),
                NaN,
            ]
            yvals = [
                # Major axis
                y - length_major*sin(α),
                y + length_major*sin(α),
                NaN,
                # Minor axis
                y - length_minor*sin(α+π/2),
                y + length_minor*sin(α+π/2),
                NaN,
            ]
            xvals, yvals
        end
        xs = Base.vcat(getindex.(error_ellipses,1)...)
        ys = Base.vcat(getindex.(error_ellipses,2)...)

        @series begin
            label := nothing
            xs, ys
        end

    end


end


# Astrometry likelihood function
function ln_like(astrom::Astrometry, θ_planet, orbit,)

    # Note: since astrometry data is stored in a typed-table, the column name
    # checks using `hasproperty` ought to be compiled out completely.

    ll = 0.0
    for i in eachindex(astrom.table.epoch)

        # Covariance between the two dimensions
        cor = 0.0

        o = orbitsolve(orbit, astrom.table.epoch[i])
        # PA and Sep specified
        if hasproperty(astrom.table, :pa) && hasproperty(astrom.table, :sep)
            ρ = projectedseparation(o)
            pa = posangle(o)

            pa_diff = ( astrom.table.pa[i] - pa + π) % 2π - π;
            pa_diff = pa_diff < -π ? pa_diff + 2π : pa_diff;
            resid1 = pa_diff
            resid2 = astrom.table.sep[i] - ρ
            σ₁ = astrom.table.σ_pa[i ]
            σ₂ = astrom.table.σ_sep[i]

        # RA and DEC specified
        else
            x = raoff(o)
            y = decoff(o)
            resid1 = astrom.table.ra[i] - x
            resid2 = astrom.table.dec[i] - y
            σ₁ = astrom.table.σ_ra[i ]
            σ₂ = astrom.table.σ_dec[i]

            # Add non-zero correlation if present
            if hasproperty(astrom.table, :cor)
                cor = astrom.table.cor[i]
            end
        end

        # Manual definition:
        # χ²1 = -(1/2)*resid1^2 / σ²1 - log(sqrt(2π * σ²1))
        # χ²2 = -(1/2)*resid2^2 / σ²2 - log(sqrt(2π * σ²2))
        # ll += χ²1 + χ²2

        # Leveraging Distributions.jl to make this clearer:
        # χ²1 = logpdf(Normal(0, sqrt(σ²1)), resid1)
        # χ²2 = logpdf(Normal(0, sqrt(σ²2)), resid2)
        # ll += χ²1 + χ²2

        # Same as above, with support for covariance:
        dist = MvNormal(@SArray[
            σ₁^2        cor*σ₁*σ₂
            cor*σ₁*σ₂   σ₂^2
        ])
        ll += logpdf(dist, @SArray[resid1, resid2])

    end
    return ll
end

# Generate new astrometry observations
function genobs(obs::Astrometry, elem::VisualOrbit, θ_planet)

    # Get epochs and uncertainties from observations
    epochs = obs.table.epoch
    σ_ras = obs.table.σ_ra 
    σ_decs = obs.table.σ_dec

    # Generate now astrometry data
    ras = raoff.(elem, epochs)
    decs = decoff.(elem, epochs)
    astrometry_table = Table(epoch=epochs, ra=ras, dec=decs, σ_ra=σ_ras, σ_dec=σ_decs)

    return Astrometry(astrometry_table)
end
