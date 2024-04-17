#  Proper motion anomaly functions

# This code provides helpers for loading data from the HIPPARCOS GAIA Catalog of Accelerations.
# 

"""
    gaia_plx(gaia_id=12123)

Get a distribution (truncated Normal) of parallax distance in mas of a source with 
GAIA catalog id `gaia_id`.
"""
function gaia_plx(;gaia_id,catalog=(datadep"HGCA_eDR3")*"/HGCA_vEDR3.fits") 
    
    # Load the Hipparcos-GAIA catalog of accelerations as a table
    hgca = FITS(catalog,"r") do fits
        Table(fits[2])
    end

    idx = findfirst(==(gaia_id), hgca.gaia_source_id)
    return truncated(Normal(Float64(hgca.parallax_gaia[idx,]), Float64(hgca.parallax_gaia_error[idx,])), lower=0)
end
export gaia_plx


# function ghca_pmra(;gaia_id)


const pma_cols = (:ra_epoch, :dec_epoch, :dt, :pm_ra, :pm_dec, :σ_pm_ra, :σ_pm_dec,)
struct ProperMotionAnomLikelihood{TTable<:Table} <: AbstractLikelihood
    table::TTable
    function ProperMotionAnom(observations...)
        table = Table(observations...)
        if !issubset(pma_cols, Tables.columnnames(table))
            error("Expected columns $pma_cols")
        end
        return new{typeof(table)}(table)
    end
end
ProperMotionAnom(observations::NamedTuple...) = ProperMotionAnom(observations)
export ProperMotionAnom

struct HGCALikelihood{TTable<:Table} <: AbstractLikelihood
    table::TTable
    function HGCALikelihood(observations...)
        table = Table(observations...)
        # if !issubset(pma_cols, Tables.columnnames(table))
        #     error("Expected columns $pma_cols")
        # end
        return new{typeof(table)}(table)
    end
end
HGCALikelihood(observations::NamedTuple...) = HGCALikelihood(observations)
export HGCALikelihood


"""
    HGCALikelihood(;gaia_id=1234)

Load proper motion anomaly data from the HIPPARCOS-GAIA Catalog of Accelerations (Brandt et al)
for a star with catalog id `gaia_id`.
The resulting velocities are in mas/yr and have the long term trend between HIPPARCOS and GAIA
already subtracted out. e.g. we would expect 0 pma if there is no companion.
"""
function HGCALikelihood(;gaia_id,catalog=(datadep"HGCA_eDR3")*"/HGCA_vEDR3.fits")

    # Load the Hipparcos-GAIA catalog of accelerations (downloaded automatically with datadeps)
    hgca = FITS(catalog,"r") do fits
        Table(fits[2])
    end

    # Available columns (for reference)
    # chisq            crosscal_pmdec_hg  crosscal_pmdec_hip   crosscal_pmra_hg   crosscal_pmra_hip  epoch_dec_gaia          epoch_dec_hip
    # epoch_ra_gaia    epoch_ra_hip       gaia_dec             gaia_ra            gaia_source_id     hip_id                  nonlinear_dpmdec
    # nonlinear_dpmra  parallax_gaia      parallax_gaia_error  pmdec_gaia         pmdec_gaia_error   pmdec_hg                pmdec_hg_error
    # pmdec_hip        pmdec_hip_error    pmra_gaia            pmra_gaia_error    pmra_hg            pmra_hg_error           pmra_hip
    # pmra_hip_error   pmra_pmdec_gaia    pmra_pmdec_hg        pmra_pmdec_hip     radial_velocity    radial_velocity_error   radial_velocity_source

    # Find the row with a GAIA source id match
    idx = findfirst(==(gaia_id), hgca.gaia_source_id)

    return HGCALikelihood(hgca[idx])

end
export HGCALikelihood




"""
Specific HGCA proper motion modelling. Model the GAIA-Hipparcos/Δt proper motion
using 5 position measurements averaged at each of their epochs.
"""
function ln_like(pma::HGCALikelihood, θ_system, elements, _L=1 #=length of observations: we know this is one=#)
    ll = 0.0

    # This observation type just wraps one row from the HGCA (see hgca.jl)
    hgca = pma.table
    # Roughly over what time period were the observations made?
    dt_gaia = 1038 # EDR3: days between  Date("2017-05-28") - Date("2014-07-25")
    dt_hip = 4*365
    # How many points over Δt should we average the proper motion and stellar position
    # at each epoch? This is because the PM is not an instantaneous measurement.
    N_ave = 25 #5

    # Look at the position of the star around both epochs to calculate 
    # our modelled delta-position proper motion

    # If the user specified a AbsoluteVisual orbit, then all our `raoff` and instantaneous
    # `pmra` calls below will take into account the star's motion in spherical coordinates;
    # however, we still have to account for a couple of things:
    # the RA & DEC position & light travel time

    deg2mas = 60*60*1000 

    # First epoch: Hipparcos
    ra_hip_model = 0.0
    dec_hip_model = 0.0
    pmra_hip_model = 0.0
    pmdec_hip_model = 0.0
    # The model can support multiple planets
    for i in eachindex(elements)
        θ_planet = θ_system.planets[i]
        orbit = elements[i]
        if θ_planet.mass < 0
            return -Inf
        end
        # Average multiple observations over a timescale +- dt/2
        # to approximate what HIPPARCOS would have measured.
        for δt = range(-dt_hip/2, dt_hip/2, N_ave)
            # RA and dec epochs are usually slightly different
            # Note the unit conversion here from jupiter masses to solar masses to 
            # make it the same unit as the total mass (element.mu)
            o_ra = orbitsolve(orbit, years2mjd(hgca.epoch_ra_hip[1])+δt)
            o_dec = orbitsolve(orbit, years2mjd(hgca.epoch_dec_hip[1])+δt)
            ra_hip_model += raoff(o_ra, θ_planet.mass*mjup2msol)
            dec_hip_model += decoff(o_dec, θ_planet.mass*mjup2msol)
            if o_ra isa PlanetOrbits.OrbitSolutionAbsoluteVisual
                # Add a correction from the difference in linear vs non-linearly propagated ra&dec
                dra_deg = o_ra.compensated.ra2 - (θ_system.ra + θ_system.pmra*(
                    o_ra.compensated.epoch2a-o_ra.compensated.epoch1 # [years]
                )/deg2mas)
                ra_hip_model += dra_deg*deg2mas# deg->mas
                ddec_deg = o_ra.compensated.dec2 - (θ_system.dec + θ_system.pmdec*(
                    o_dec.compensated.epoch2a-o_dec.compensated.epoch1 # [years]
                )/deg2mas)
                dec_hip_model += ddec_deg*deg2mas # deg->mas
            end
            pmra_hip_model += pmra(o_ra, θ_planet.mass*mjup2msol)
            pmdec_hip_model += pmdec(o_dec, θ_planet.mass*mjup2msol)
        end
    end
    ra_hip_model/=N_ave
    dec_hip_model/=N_ave
    pmra_hip_model/=N_ave
    pmdec_hip_model/=N_ave

    # Last epoch: GAIA
    ra_gaia_model = 0.0
    dec_gaia_model = 0.0
    pmra_gaia_model = 0.0
    pmdec_gaia_model = 0.0
    # The model can support multiple planets
    for i in eachindex(elements)
        θ_planet = θ_system.planets[i]
        orbit = elements[i]
        if θ_planet.mass < 0
            return -Inf
        end
        # Average multiple observations over a timescale +- dt
        # to approximate what HIPPARCOS and GAIA would have measured.
        for δt = range(-dt_gaia/2, dt_gaia/2, N_ave)
            # RA and dec epochs are usually slightly different
            # Note the unit conversion here from jupiter masses to solar masses to 
            # make it the same unit as the stellar mass (element.M)
            o_ra = orbitsolve(orbit, years2mjd(hgca.epoch_ra_gaia[1])+δt)
            o_dec = orbitsolve(orbit, years2mjd(hgca.epoch_dec_gaia[1])+δt)
            ra_gaia_model += raoff(o_ra, θ_planet.mass*mjup2msol)
            dec_gaia_model += decoff(o_dec, θ_planet.mass*mjup2msol)
            if o_ra isa PlanetOrbits.OrbitSolutionAbsoluteVisual
                # Add a correction from the difference in linear vs non-linearly propagated ra&dec
                dra_deg = o_ra.compensated.ra2 - (θ_system.ra + θ_system.pmra*(
                    o_ra.compensated.epoch2a-o_ra.compensated.epoch1 # [years]
                )/deg2mas)
                ra_hip_model += dra_deg*deg2mas# deg->mas
                ddec_deg = o_ra.compensated.dec2 - (θ_system.dec + θ_system.pmdec*(
                    o_dec.compensated.epoch2a-o_dec.compensated.epoch1 # [years]
                )/deg2mas)
                dec_hip_model += ddec_deg*deg2mas # deg->mas
            end
            pmra_gaia_model += pmra(o_ra, θ_planet.mass*mjup2msol)
            pmdec_gaia_model += pmdec(o_dec, θ_planet.mass*mjup2msol)
        end
    end
    ra_gaia_model/=N_ave
    dec_gaia_model/=N_ave
    pmra_gaia_model/=N_ave
    pmdec_gaia_model/=N_ave

    # Model the GAIA-Hipparcos delta-position velocity
    pmra_hg_model = (ra_gaia_model - ra_hip_model)/(years2mjd(hgca.epoch_ra_gaia[1]) - years2mjd(hgca.epoch_ra_hip[1]))
    pmdec_hg_model = (dec_gaia_model - dec_hip_model)/(years2mjd(hgca.epoch_dec_gaia[1]) - years2mjd(hgca.epoch_dec_hip[1]))

    # Hipparcos epoch
    c = hgca.pmra_pmdec_hip[1]*hgca.pmra_hip_error[1]*hgca.pmdec_hip_error[1]
    dist_hip = MvNormal(@SArray[
        hgca.pmra_hip_error[1]^2   c
        c                       hgca.pmdec_hip_error[1]^2 
    ])
    resids_hip = @SArray[
        pmra_hip_model  +  θ_system.pmra  - hgca.pmra_hip[1],
        pmdec_hip_model +  θ_system.pmdec - hgca.pmdec_hip[1]
    ]
    ll += logpdf(dist_hip, resids_hip)

    # Hipparcos - GAIA epoch
    c = hgca.pmra_pmdec_hg[1]*hgca.pmra_hg_error[1]*hgca.pmdec_hg_error[1]
    dist_hg = MvNormal(@SArray[
        hgca.pmra_hg_error[1]^2   c
        c                      hgca.pmdec_hg_error[1]^2 
    ])

    # TODO: We have to undo the spherical coordinate correction that was done in the HGCA catalog
    # since our calculations use real, not tangent plane, coordinates
    resids_hg = @SArray[
        pmra_hg_model  +  θ_system.pmra  - hgca.pmra_hg[1]# - hgca.nonlinear_dpmra,
        pmdec_hg_model +  θ_system.pmdec - hgca.pmdec_hg[1]# - hgca.nonlinear_dpmdec
    ]
    ll += logpdf(dist_hg, resids_hg)

    # GAIA epoch
    c = hgca.pmra_pmdec_gaia[1]*hgca.pmra_gaia_error[1]*hgca.pmdec_gaia_error[1]
    dist_gaia = MvNormal(@SArray[
        hgca.pmra_gaia_error[1]^2   c
        c                        hgca.pmdec_gaia_error[1]^2 
    ])
    resids_gaia = @SArray[
        pmra_gaia_model  +  θ_system.pmra  - hgca.pmra_gaia[1],
        pmdec_gaia_model +  θ_system.pmdec - hgca.pmdec_gaia[1]
    ]
    ll += logpdf(dist_gaia, resids_gaia)

    # # Old manual version not support covariance
    # # Compute the likelihood at all three epochs (Hipparcos, GAIA-Hip, GAIA)
    # pmra_model = @SVector[pmra_hip_model, pmra_hg_model, pmra_gaia_model]
    # pmdec_model = @SVector[pmdec_hip_model, pmdec_hg_model, pmdec_gaia_model]
    # pmra_meas = @SVector[hgca.pmra_hip[1], hgca.pmra_hg[1], hgca.pmra_gaia[1]]
    # pmdec_meas = @SVector[hgca.pmdec_hip[1], hgca.pmdec_hg[1], hgca.pmdec_gaia[1]]
    # σ_pmra = @SVector[hgca.pmra_hip_error[1], hgca.pmra_hg_error[1], hgca.pmra_gaia_error[1]]
    # σ_pmdec = @SVector[hgca.pmdec_hip_error[1], hgca.pmdec_hg_error[1], hgca.pmdec_gaia_error[1]]
    # # Loop through epochs
    # for i in 1:3
    #     residx = pmra_model[i] + θ_system.pmra - pmra_meas[i]
    #     residy = pmdec_model[i] + θ_system.pmdec - pmdec_meas[i]
    #     σ²x = σ_pmra[i]^2
    #     σ²y = σ_pmdec[i]^2
    #     χ²x = -0.5residx^2 / σ²x - log(sqrt(2π * σ²x))
    #     χ²y = -0.5residy^2 / σ²y - log(sqrt(2π * σ²y))
    #     ll += χ²x + χ²y
    # end

    return ll
end




"""
Specific HGCA proper motion modelling. Model the GAIA-Hipparcos/Δt proper motion
using 25 position measurements averaged at each of their epochs.
"""
function generate_from_params(like::HGCALikelihood, θ_system, orbit::PlanetOrbits.AbstractOrbit)
    ll = 0.0

    # This observation type just wraps one row from the HGCA (see hgca.jl)
    hgca = like.table
    # Roughly over what time period were the observations made?
    dt_gaia = 1038 # EDR3: days between  Date("2017-05-28") - Date("2014-07-25")
    dt_hip = 4*365
    # How many points over Δt should we average the proper motion and stellar position
    # at each epoch? This is because the PM is not an instantaneous measurement.
    N_ave = 25 #5

    # Look at the position of the star around both epochs to calculate 
    # our modelled delta-position proper motion

    # First epoch: Hipparcos
    ra_hip_model = 0.0
    dec_hip_model = 0.0
    pmra_hip_model = 0.0
    pmdec_hip_model = 0.0
    # The model can support multiple planets
    for i in eachindex(orbits)
        θ_planet = θ_system.planets[i]
        orbit = orbits[i]
        if θ_planet.mass < 0
            return -Inf
        end
        # Average multiple observations over a timescale +- dt/2
        # to approximate what HIPPARCOS would have measured.
        for δt = range(-dt_hip/2, dt_hip/2, N_ave)
            # RA and dec epochs are usually slightly different
            # Note the unit conversion here from jupiter masses to solar masses to 
            # make it the same unit as the stellar mass (element.mu)
            # TODO: we can't yet use the orbitsolve interface here for the pmra calls,
            # meaning we calculate the orbit 2x as much as we need.
            o_ra = orbitsolve(orbit, years2mjd(hgca.epoch_ra_hip[1])+δt)
            o_dec = orbitsolve(orbit, years2mjd(hgca.epoch_dec_hip[1])+δt)
            ra_hip_model += -raoff(o_ra) * θ_planet.mass*mjup2msol/totalmass(orbit)
            dec_hip_model += -decoff(o_dec) * θ_planet.mass*mjup2msol/totalmass(orbit)
            pmra_hip_model += pmra(o_ra, θ_planet.mass*mjup2msol)
            pmdec_hip_model += pmdec(o_dec, θ_planet.mass*mjup2msol)
        end
    end
    ra_hip_model/=N_ave
    dec_hip_model/=N_ave
    pmra_hip_model/=N_ave
    pmdec_hip_model/=N_ave

    # Last epoch: GAIA
    ra_gaia_model = 0.0
    dec_gaia_model = 0.0
    pmra_gaia_model = 0.0
    pmdec_gaia_model = 0.0
    # The model can support multiple planets
    for i in eachindex(orbits)
        θ_planet = θ_system.planets[i]
        orbit = orbits[i]
        if θ_planet.mass < 0
            return -Inf
        end
        # Average multiple observations over a timescale +- dt
        # to approximate what HIPPARCOS and GAIA would have measured.
        for δt = range(-dt_gaia/2, dt_gaia/2, N_ave)
            # RA and dec epochs are usually slightly different
            # Note the unit conversion here from jupiter masses to solar masses to 
            # make it the same unit as the stellar mass (element.M)
            o_ra = orbitsolve(orbit, years2mjd(hgca.epoch_ra_gaia[1])+δt)
            o_dec = orbitsolve(orbit, years2mjd(hgca.epoch_dec_gaia[1])+δt)
            ra_gaia_model += -raoff(o_ra) * θ_planet.mass*mjup2msol/totalmass(orbit)
            dec_gaia_model += -decoff(o_dec) * θ_planet.mass*mjup2msol/totalmass(orbit)
            pmra_gaia_model += pmra(o_ra, θ_planet.mass*mjup2msol)
            pmdec_gaia_model += pmdec(o_dec, θ_planet.mass*mjup2msol)
        end
    end
    ra_gaia_model/=N_ave
    dec_gaia_model/=N_ave
    pmra_gaia_model/=N_ave
    pmdec_gaia_model/=N_ave

    # Model the GAIA-Hipparcos delta-position velocity
    pmra_hg_model = (ra_gaia_model - ra_hip_model)/(years2mjd(hgca.epoch_ra_gaia[1]) - years2mjd(hgca.epoch_ra_hip[1]))
    pmdec_hg_model = (dec_gaia_model - dec_hip_model)/(years2mjd(hgca.epoch_dec_gaia[1]) - years2mjd(hgca.epoch_dec_hip[1]))

    # Merge the measurements together into a new observation and add noise according to the sigma
    # we were passed in from the original measurements
    return HGCALikelihood(merge(hgca[1], (;
        pmra_hip=pmra_hip_model+hgca[1].pmra_hip_error,
        pmdec_hip=pmdec_hip_model+hgca[1].pmdec_hip_error,
        pmra_gaia=pmra_gaia_model+hgca[1].pmra_gaia_error,
        pmdec_gaia=pmdec_gaia_model+hgca[1].pmdec_gaia_error,
        pmra_hg=pmra_hg_model+hgca[1].pmra_hg_error,
        pmdec_hg=pmdec_hg_model+hgca[1].pmdec_hg_error,
    )))

end
