using LinearAlgebra
using Interpolations
using BlockArrays
using Distributions


#=
This file implements a variant of InterferometryLikelihood
that accounts for Fiber positioning throughput loss.
=#
const required_cols_fiber = (required_cols...,)

include("GRAVITY-correlation.jl")

struct GRAVITYWideCPLikelihood{TTable<:Table,TInterp} <: AbstractInterferometryLikelihood
    table::TTable
    fiber_coupling_interpolator::TInterp
end
const GRAVITYWideCPLikelihood = GRAVITYWideCPLikelihood
function GRAVITYWideCPLikelihood(observations...)
    input_table = Table(observations...)
    if :filename ∈ Tables.columnnames(input_table)
        rows = map(_prepare_input_row, eachrow(input_table))
        table = Table(rows)
    else
        table = input_table
    end

    if !issubset(required_cols_fiber, Tables.columnnames(table))
        error("Expected columns $vis_cols")
    end

    # Take some additional preparation steps per input-row.
    rows_with_kernel_phases = map(eachrow(table)) do row′
        row = row′[1]

        # Calculate the design matrix
        # TODO: replace hardcoded T with one calculated using cps_index1, 2, and 
        T = Int8[
            1 -1 0 1 0 0
            1 0 -1 0 1 0
            0 1 -1 0 0 1
            0 0 0 1 -1 1
        ]

        # We now generate a unified design matrix that works with all wavelengths.
        # Later, this will allow us to easily deal with correlations between both CPs & wavelengths.
        Λ = length(row.eff_wave)
        Tλ = zeros(Int8, Λ * size(T, 1), Λ * size(T, 2))

        # We now replicate our deign matrix T across wavelengths.
        # We will put each wavelength together, grouped by baseline.

        for baseline_i in axes(T, 1), baseline_j in axes(T, 2)
            for wavelength_i in 1:Λ
                Tλ[wavelength_i+(baseline_i-1)*Λ, wavelength_i+(baseline_j-1)*Λ] = T[baseline_i, baseline_j]
            end
        end

        # Determine a kernel phase basis using the cholesky factorization of the
        # design matrix * design matrixᵀ
        C, U=cholesky(Tλ*Tλ')
        P₁ = collect(C) ./ sqrt.(diag(C*C'))
        i_max = findfirst(<=(1e-5), diag(P₁))-1
        P₁ = P₁[:,1:i_max]'

        return (; row..., Tλ, P₁)
    end
    table = Table(rows_with_kernel_phases)

    # Create an interpolator object that maps separation to fiber coupling efficiency
    @info "Pre-calculating fiber coupling efficiency over grid"
    sep_mas = 0:2:100
    λs = range(extrema(vec(table.eff_wave[1]))..., length=15)
    fiber_coupling = stack([
        fiber_coupling_fraction(sep_mas, λ)
        for λ in λs
    ])
    coupling_interp = LinearInterpolation((sep_mas, λs), fiber_coupling, extrapolation_bc=0.0)

    return GRAVITYWideCPLikelihood(table, coupling_interp)
end
GRAVITYWideCPLikelihood(observations::NamedTuple...) = GRAVITYWideCPLikelihood(observations)
export GRAVITYWideCPLikelihood

# theta = range(0, 100, length=250)
# Credit: W. Balmer, D. Bakely, and others.
function fiber_coupling_fraction(theta, lambda_w=2.2e-6)
    D = 8
    x = range(-D * 2, D * 2, length=500)#0)
    y = range(-D * 2, D * 2, length=500)#0)
    r = LinearAlgebra.norm.(x, y')
    m = r .< D / 2
    # arcseconds
    phase = reshape(x, :, 1, 1) ./ lambda_w .* reshape(theta, 1, 1, :) * 1e-3 / (180 / pi * 3600) * 2pi
    w_0 = 0.32D
    field_pup = @. m * exp(1im * phase)
    field_fiber = @. exp(-1 * r^2 / (2 * w_0^2))
    Inj = abs2.(sum(field_pup .* field_fiber, dims=(1, 2))) / abs(sum(m .* field_fiber))^2
    return Inj[1, 1, :]
end



"""
Visibliitiy modelling likelihood for point sources.
"""
function Octofitter.ln_like(vis::GRAVITYWideCPLikelihood, θ_system, orbits, num_epochs::Val{L}=Val(length(vis.table))) where {L}

    # Convoluted way to get either Float64 normally or a Dual{Float64} if using ForwardDiff
    T = float(typeof(θ_system.M))
    for θ_planet in θ_system.planets
        TT = typeof(first(promote(θ_planet...)))
        T = promote_type(T, TT)
    end
    ll = zero(T)

    # Access the data here: 
    epochs = vis.table.epoch
    band = vis.table.band

    # Add an extra optional uncertainty
    # in quadrature
    # cp_C_y = hasproperty(θ_system, :cp_C_y) ? θ_system.cp_C_y : zero(T)


    # Loop through epochs
    for i_epoch in eachindex(epochs)

        epoch = epochs[i_epoch]
        this_band = band[i_epoch]

        index_cps1 = vis.table.index_cps1[i_epoch]
        index_cps2 = vis.table.index_cps2[i_epoch]
        index_cps3 = vis.table.index_cps3[i_epoch]
        # use_vis2 = vis.table.use_vis2[i_epoch]

        cps_model = zeros(T, size(vis.table.cps_data[i_epoch][:, 1]))
        cvis_model = zeros(complex(T), size(vis.table.u[i_epoch][:, 1]))

        contrasts = T[getproperty(θ_planet, this_band) for θ_planet in θ_system.planets]
        sols = [orbitsolve(orbits[i_planet], epoch) for i_planet in 1:length(θ_system.planets)]

        throughputs = broadcast(sols, contrasts, vis.table.eff_wave[i_epoch]') do sol, flux_ratio, wavelength_m
            # Model the fiber as placed at the photocentre of the two bodies
            secondary_offset_mas = projectedseparation(sol)
            # Now calculate throughput loss on the secondary due to it being offset wrt. the 
            # fiber (assumed to be at photocentre)
            fiber_offset_mas = (flux_ratio * secondary_offset_mas) / (1.0 + flux_ratio)
            coupling = vis.fiber_coupling_interpolator(fiber_offset_mas, wavelength_m)
            return coupling
        end

        Λ = length(vis.table.eff_wave[i_epoch])
        Len = Λ * size(vis.table.cps_data[i_epoch], 1)
        cp_resids = zeros(T, Len) # Fill this in a moment

        # Loop through wavelengths
        # The following is NOT threadsafe. DON'T multithread it!
        for i_wave in axes(vis.table.u[i_epoch], 2)
            u = @views vis.table.u[i_epoch][:, i_wave]
            v = @views vis.table.v[i_epoch][:, i_wave]
            cps_data = @views vis.table.cps_data[i_epoch][:, i_wave]
            σ_cp = @views vis.table.dcps[i_epoch][:, i_wave]
            vis2_data = @views vis.table.vis2_data[i_epoch][:, i_wave]
            dvis2 = @views vis.table.dvis2[i_epoch][:, i_wave]

            # to normalize complex visibilities 
            cvis_model .= 0
            cps_model .= 0
            norm_factor_model = zero(T)

            # Consider all planets
            for i_planet in eachindex(orbits)
                # All parameters relevant to this planet
                # Get model contrast parameter in this band (band provided as a symbol, e.g. :L along with data in table row.)
                contrast = contrasts[i_planet]
                throughput = throughputs[i_planet]
                Δra = raoff(sols[i_planet])  # in mas
                Δdec = decoff(sols[i_planet]) # in mas

                # add complex visibilities from all planets at a single epoch, for this wavelength
                cvis_bin!(cvis_model; Δdec, Δra, contrast=contrast * throughput, u, v)
                norm_factor_model += contrast
            end
            cvis_model .+= 1.0 # add contribution from the primary primary
            cvis_model .*= 1.0 / (1.0 + norm_factor_model)

            # Compute closure phases
            closurephase!(cps_model; vis=cvis_model, index_cps1, index_cps2, index_cps3)

            for i_T3 in eachindex(σ_cp, cps_data, cps_model)
                cp_resids[(i_T3-1)*Λ+i_wave] = cps_data[i_T3] - cps_model[i_T3]
                # if !isfinite(cp_resids[(i_T3-1)*Λ+i_wave] )
                #     @warn "non finite CP calculated"  cps_data[i_T3] cps_model[i_T3]
                # end
            end
        end
        # Done calculating the residuals for this epoch

        # # CP Only:
        # σ_cp = vec(vis.table[i_epoch].dcps) #sqrt.(σ_cp_jitter .^ 2 .+ vec(vis.table[i_epoch].dcps) .^ 2)
        # distribution = MvNormal(Diagonal(σ_cp))
        # ll += logpdf(distribution, cp_resids)

        # ## Diagonalized covariance Kernphases

        if hasproperty(vis.table, :jitter)
            kp_jitter_name =  vis.table.jitter[i_epoch]
            kp_jitter = convert(T, getproperty(θ_system, kp_jitter_name))
            kp_jitter = max(eps(),kp_jitter)
        else
            kp_jitter = zero(T)
        end

        P₁ = vis.table.P₁[i_epoch]
        σ_cp = vis.table[i_epoch].dcps
        σ_kp = kp_jitter

        # Generate the semi-analytic correlation matrix from Jens
        CT3_y = hasproperty(θ_system, :CT3_y) ? float(θ_system.CT3_y) : zero(T)
        C_T3 = CT3(vis.table[i_epoch], CT3_y)

        # Calculate the CP covariance matrix using the uncertainties
        Σ_T3 = C_T3 .* vec(σ_cp) .* vec(σ_cp)'

        # Convert to KP covariance matrix
        Σ_kernphases = P₁ * Σ_T3 * P₁'

        # Convert the CP residuals to KPs
        kernphase_resids = P₁ * vec(cp_resids)

        # Create a multivariate normal distribution
        # Add the kernel phase jitters along the diagonal.
        dist = MvNormal(
            Hermitian(
                Σ_kernphases .+ 
                Diagonal(fill(σ_kp^2,size(Σ_kernphases,1)))
            )
        )
        ll += logpdf(dist,kernphase_resids)
    end


    return ll
end




# Generate new observations for a system of possibly multiple planets
function Octofitter.generate_from_params(vis_input::GRAVITYWideCPLikelihood, θ_system, orbits::Vector{<:AbstractOrbit})

    vis = deepcopy(vis_input)

    T = typeof(θ_system.M)
    ll = zero(T)

    # Access the data here: 
    epochs = vis.table.epoch
    band = vis.table.band

    # Loop through epochs
    for i_epoch in eachindex(epochs)

        epoch = epochs[i_epoch]
        this_band = band[i_epoch]

        index_cps1 = vis.table.index_cps1[i_epoch]
        index_cps2 = vis.table.index_cps2[i_epoch]
        index_cps3 = vis.table.index_cps3[i_epoch]
        use_vis2 = vis.table.use_vis2[i_epoch]

        cps_model = zeros(T, size(vis.table.cps_data[i_epoch][:, 1]))
        cvis_model = zeros(complex(T), size(vis.table.u[i_epoch][:, 1]))

        contrasts = T[getproperty(θ_planet, this_band) for θ_planet in θ_system.planets]
        sols = [orbitsolve(orbits[i_planet], epoch) for i_planet in 1:length(θ_system.planets)]

        throughputs = broadcast(sols, contrasts, vis.table.eff_wave[i_epoch]') do sol, flux_ratio, wavelength_m
            # Model the fiber as placed at the photocentre of the two bodies
            secondary_offset_mas = projectedseparation(sol)
            # Now calculate throughput loss on the secondary due to it being offset wrt. the 
            # fiber (assumed to be at photocentre)
            fiber_offset_mas = (flux_ratio * secondary_offset_mas) / (1.0 + flux_ratio)
            coupling = vis.fiber_coupling_interpolator(fiber_offset_mas, wavelength_m)
            return coupling
        end

        # Loop through wavelengths
        for i_wave in axes(vis.table.u[i_epoch], 2)
            u = @views vis.table.u[i_epoch][:, i_wave]
            v = @views vis.table.v[i_epoch][:, i_wave]
            cps_data = @views vis.table.cps_data[i_epoch][:, i_wave]
            σ_cp = @views vis.table.dcps[i_epoch][:, i_wave]
            vis2_data = @views vis.table.vis2_data[i_epoch][:, i_wave]
            dvis2 = @views vis.table.dvis2[i_epoch][:, i_wave]

            # to normalize complex visibilities 
            cvis_model .= 0
            cps_model .= 0
            norm_factor_model = zero(T)

            # Consider all planets
            for i_planet in eachindex(orbits)
                # All parameters relevant to this planet
                # Get model contrast parameter in this band (band provided as a symbol, e.g. :L along with data in table row.)
                contrast = contrasts[i_planet]
                throughput = throughputs[i_planet]
                Δra = raoff(sols[i_planet])  # in mas
                Δdec = decoff(sols[i_planet]) # in mas

                # add complex visibilities from all planets at a single epoch, for this wavelength
                cvis_bin!(cvis_model; Δdec, Δra, contrast=contrast * throughput, u, v)
                norm_factor_model += contrast
            end
            cvis_model .+= 1.0 #add contribution from the primary primary
            cvis_model .*= 1.0 / (1.0 + norm_factor_model)

            # Compute closure phases
            closurephase!(cps_model; vis=cvis_model, index_cps1, index_cps2, index_cps3)

            cps_data .= cps_model
            vis2_data .= abs.(cvis_model) .^ 2

        end
    end
    return vis
end