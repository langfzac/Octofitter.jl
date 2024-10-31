module OctofitterRadialVelocityMakieExt
using Octofitter
using OctofitterRadialVelocity
using PlanetOrbits
using Makie
using Statistics
using StatsBase
using AbstractGPs
# using TemporalGPs
using Dates
using AstroImages

## Need three panels
# 1) Mean model (orbit + GP) and data (- mean instrument offset)
# 2) Residuals of above
# 3) Phase folded curve
function Octofitter.rvpostplot(
    model,
    results,
    args...;
    fname="$(model.system.name)-rvpostplot.png",
    kwargs...,
)
    fig = Figure()
    Octofitter.rvpostplot!(fig.layout, model, results,args...;kwargs...)

    Makie.save(fname, fig, px_per_unit=3)

    return fig
end
function Octofitter.rvpostplot!(
    gridspec_or_fig,
    model::Octofitter.LogDensityModel,
    results::Chains,
    sample_idx = argmax(results["logpost"][:]),
    planet_key = first(keys(model.system.planets))
)
    gs = gridspec_or_fig


    rv_likes = filter(model.system.observations) do obs
        obs isa StarAbsoluteRVLikelihood || obs isa OctofitterRadialVelocityMakieExt.MarginalizedStarAbsoluteRVLikelihood
    end
    # if length(rv_likes) > 1
    #     error("`rvpostplot` requires a system with only one StarAbsoluteRVLikelihood. Combine the data together into a single likelihood object.")
    # end
    # if length(rv_likes) != 1
    #     error("`rvpostplot` requires a system with a StarAbsoluteRVLikelihood.")
    # end
    # Start filling the RV plot
    els = Octofitter.construct_elements(results,planet_key, :)
    M = (results[string(planet_key)*"_mass"] .* Octofitter.mjup2msol)

    # Sometimes we want the true RV, including eg perspecive acceleration, and
    # sometimes we want just the perturbation orbit. 
    # This function returns an orbit-element's parent orbit-element if its
    # an AbsoluteVisualOrbit, otherwise, just returns the passed in element.
    nonabsvis_parent(element::AbsoluteVisual) = element.parent 
    nonabsvis_parent(element::AbstractOrbit) = element

    # For phase-folded plot
    t_peri = periastron(els[sample_idx])
    T = period(els[sample_idx])

    # Model plot vs raw data
    all_epochs = vec(vcat((rvs.table.epoch for rvs in rv_likes)...))
    tmin, tmax = extrema(all_epochs)
    delta = tmax-tmin
    ts_grid = range(tmin-0.015delta, tmax+0.015delta,length=10000)
    # Ensure the curve has points at exactly our data points. Otherwise for fine structure
    # we might miss them unless we add a very very fine grid.
    ts = sort(vcat(ts_grid, all_epochs))
    # RV = radvel.(els[ii], ts', M[ii])
    # RV_map = radvel.(els[sample_idx], ts, M[sample_idx])

    # For secondary date axis on top
    date_start = mjd2date(ts[begin])
    date_end = mjd2date(ts[end])
    date_start = Date(year(date_start), month(date_start))
    date_end = Date(year(date_end), month(date_end))
    dates = range(date_start, date_end, step=Year(1))
    dates_str = string.(year.(dates))
    if length(dates) == 1
        dates = range(date_start, date_end, step=Month(1))
        dates_str = map(d->string(year(d),"-",lpad(month(d),2,'0')),dates)
    else
        year_step = 1
        while length(dates) > 8
            year_step += 1
            dates = range(date_start, date_end, step=Year(year_step))
        end
        dates_str = string.(year.(dates))
    end
    ax_fit = Axis(
        gs[1,1],
        ylabel="RV [m/s]",
        xaxisposition=:top,
        xticks=(
            mjd.(string.(dates)),
            dates_str
        )
    )
    ax_resid = Axis(
        gs[2,1],
        xlabel="time [MJD]",
        ylabel="Residuals",
    )
    linkxaxes!(ax_fit, ax_resid)
    # hidexdecorations!(ax_fit,grid=false)

    ax_phase = Axis(
        gs[3,1],
        xlabel="Phase",
        ylabel="RV [m/s]",
        xticks=-0.5:0.1:0.5,
    )
    # xlims!(ax_phase, -0.5,0.5)
    rowgap!(gs, 1, 0)

    # Horizontal zero line
    hlines!(ax_resid, 0, color=:black, linewidth=3)

    # Perspective acceleration line
    if els[sample_idx] isa AbsoluteVisual
        lines!(ax_fit, ts_grid, radvel.(els[sample_idx], ts_grid, 0.0), color=:orange)
    end
        

    nt_format = Octofitter.mcmcchain2result(model, results)


    any_models_have_a_gp = false
    for rvs in rv_likes
        any_models_have_a_gp |= hasproperty(rvs, :gaussian_process) && !isnothing(rvs.gaussian_process)
    end


    # Main blue orbit line in top panel
    RV = radvel.(els[sample_idx], ts, M[sample_idx])
    # Use a narrow line if we're overplotting a complicated GP
    if !any_models_have_a_gp
        lines!(ax_fit, ts, RV, color=:blue, linewidth=3)
    end


    # Model plot vs phase-folded data (without any perspective acceleration)
    phases = -0.5:0.005:0.5
    ts_phase_folded = ((phases .+ 0.5) .* T) .+ t_peri .+ T/4
    RV = radvel.(nonabsvis_parent(els[sample_idx]), ts_phase_folded, M[sample_idx])
    Makie.lines!(
        ax_phase,
        phases,
        RV,
        color=:blue,
        linewidth=5
    )
    Makie.xlims!(ax_phase, -0.5,0.5)


    # Calculate RVs minus the median instrument-specific offsets.
    
    # For the phase-folded binned data, 
    # we also collect all data minus perspective acceleration and any GP,
    # as well as the data uncertainties, and data + jitter + GP quadrature uncertainties
    N = 0
    masks = []
    for rvs in rv_likes
        push!(masks, N .+ (1:length(rvs.table.rv)))
        N += length(rvs.table.rv)
    end
    epochs_all = zeros(N)
    rvs_all_minus_accel_minus_perspective = zeros(N)
    errs_all_data_jitter_gp = zeros(N)

    rv_like_idx = 0
    for rvs in rv_likes
        rv_like_idx += 1 
        mask = masks[rv_like_idx]
        rvs_off_sub = collect(rvs.table.rv)
        jitters = zeros(length(rvs_off_sub))


        if hasproperty(rvs,:offset_symbol)
            barycentric_rv_inst = nt_format[sample_idx][rvs.offset_symbol]
            jitter = nt_format[sample_idx][rvs.jitter_symbol]
        else
            barycentric_rv_inst = _find_rv_zero_point_maxlike(rvs, nt_format[sample_idx], (els[sample_idx],))
            jitter = nt_format[sample_idx][rvs.jitter_symbol]
        end

        # Apply barycentric rv offset correction for this instrument
        # using the MAP parameters
        rvs_off_sub .-= barycentric_rv_inst
        jitters .= jitter

        # Calculate the residuals minus the orbit model and any perspecive acceleration
        model_at_data = radvel.(els[sample_idx], rvs.table.epoch, M[sample_idx]) 
        resids = rvs_off_sub .- model_at_data 
        errs_all = zeros(length(resids))
        data_minus_off_and_gp  = zeros(length(resids))
        perspective_accel_to_remove = radvel.(els[sample_idx], rvs.table.epoch, 0.0)


        # plot!(ax_fit, ts, posterior_gp; bandscale=1, color=(:black,0.3))
        # barycentric_rv_inst = results["rv0_$inst_idx"][sample_idx]
        data = rvs.table

        ts_inst = sort(vcat(
            vec(data.epoch),
            range((extrema(data.epoch) )...,step=step(ts_grid)
        )))


        # Plot a gaussian process per-instrument
        # If not using a GP, we fit a GP with a "ZeroKernel"
        map_gp = nothing
        if hasproperty(rvs, :gaussian_process) && !isnothing(rvs.gaussian_process)
            row = results[sample_idx,:,:];
            nt = (Table((row)))[1]
            map_gp = rvs.gaussian_process(nt)
        end
        if isnothing(map_gp)
            map_gp = GP(ZeroKernel())
        # Drop TemporalGPs for now due to compilation failures
        # elseif map_gp isa TemporalGPs.LTISDE
        #     # Unwrap the temporal GPs wrapper so that we can calculate mean_and_var
        #     # We don't need the speed up provided by LTISDE for plotting once.
        #     map_gp = map_gp.f
        end

        fx = map_gp(
            # x
            vec(rvs.table.epoch),
            # y-err
            vec(
                sqrt.(rvs.table.σ_rv.^2 .+ jitters.^2)
            )
        )
        # condition GP on residuals (data - orbit - inst offsets)
        map_gp_posterior = posterior(fx, vec(resids))
        y, var = mean_and_var(map_gp_posterior, ts_inst)

        # Subtract MAP GP from residuals
        resids = resids .-= mean(map_gp_posterior, vec(data.epoch))
        data_minus_off_and_gp .= rvs_off_sub .- mean(map_gp_posterior, vec(data.epoch))
        y_inst, var = mean_and_var(map_gp_posterior, ts_inst)

        errs_data_jitter = sqrt.(
            data.σ_rv.^2 .+
            jitter.^2
        )
        errs_data_jitter_gp = sqrt.(
            data.σ_rv.^2 .+
            jitter.^2 .+
            mean_and_var(map_gp_posterior, vec(data.epoch))[2]
        )

        
        epochs_all[mask] = vec(rvs.table.epoch)
        rvs_all_minus_accel_minus_perspective[mask] = rvs_off_sub .- mean(map_gp_posterior, vec(data.epoch))
        errs_all_data_jitter_gp[mask] .= errs_data_jitter_gp

        RV_sample_idxnst =  radvel.(els[sample_idx], ts_inst, M[sample_idx])
        obj = band!(ax_fit, ts_inst,
            vec(y_inst .+ RV_sample_idxnst .- sqrt.(var)),# .+ jitter^2)),
            vec(y_inst .+ RV_sample_idxnst .+ sqrt.(var)),# .+ jitter^2)),
            color=(Makie.wong_colors()[rv_like_idx], 0.35)
        )
        # Try to put bands behind everything else
        translate!(obj, 0, 0, -10)

        # Draw the full model ie. RV + perspective + GP
        # We darken the colour by plotting a faint black line under it
        lines!(
            ax_fit,
            ts_inst,
            radvel.(els[sample_idx], ts_inst, M[sample_idx]) .+ y,
            color=(:black,1),
            linewidth=0.3
        )
        lines!(
            ax_fit,
            ts_inst,
            radvel.(els[sample_idx], ts_inst, M[sample_idx]) .+ y,
            color=(Makie.wong_colors()[rv_like_idx],0.8),
            # color=:blue,
            linewidth=0.3
        )
        


        # Model plot vs raw data
        errorbars!(
            ax_fit,
            data.epoch,
            rvs_off_sub,
            errs_data_jitter_gp,
            linewidth=1,
            color="#CCC",
        )
        errorbars!(
            ax_fit,
            data.epoch,
            rvs_off_sub,
            data.σ_rv,
            # linewidth=1,
            color=Makie.wong_colors()[rv_like_idx]
        )

        errorbars!(
            ax_resid,
            data.epoch,
            resids,
            errs_data_jitter_gp,
            linewidth=1,
            color="#CCC",
        )
        errorbars!(
            ax_resid,
            data.epoch,
            resids,
            data.σ_rv,
            color=Makie.wong_colors()[rv_like_idx]
        )

        # Phase-folded plot
        phase_folded = mod.(data.epoch .- t_peri .- T/4, T)./T .- 0.5
        errorbars!(
            ax_phase,
            phase_folded,
            data_minus_off_and_gp.-perspective_accel_to_remove,
            errs_data_jitter_gp,
            linewidth=1,
            color="#CCC",
        )
        errorbars!(
            ax_phase,
            phase_folded,
            data_minus_off_and_gp.-perspective_accel_to_remove,
            data.σ_rv,
            # linewidth=1,
            color=Makie.wong_colors()[rv_like_idx]
        )
        
        # scatter!(
        #      ax_phase,
        #     phase_folded,
        # #     rvs_off_sub .- mean(map_gp_posterior, vec(data.epoch)),
        #     data_minus_off_and_gp[thisinst_mask],
        #     color=Makie.wong_colors()[rv_like_idx],
        #     markersize=4
        # )

        # barycentric_rv_inst = results["rv0_$inst_idx"][sample_idx]
        # thisinst_mask = vec(rvs.table.inst_idx.==inst_idx)
        # jitter = jitters_all[thisinst_mask]
        # data = rvs.table#[thisinst_mask]
        # rvs_off_sub = rvs_off_sub#[thisinst_mask]

        Makie.scatter!(
            ax_fit,
            data.epoch,
            rvs_off_sub,
            color=Makie.wong_colors()[rv_like_idx],
            markersize=4,
            strokecolor=:black,strokewidth=0.1,
        )

        Makie.scatter!(
            ax_resid,
            data.epoch,
            resids,
            color=Makie.wong_colors()[rv_like_idx],
            markersize=4,
            strokecolor=:black,strokewidth=0.1,
        )
        phase_folded = mod.(data.epoch .- t_peri .- T/4, T)./T .- 0.5
        Makie.scatter!( 
            ax_phase,
            phase_folded,
            data_minus_off_and_gp.-perspective_accel_to_remove,
            color=Makie.wong_colors()[rv_like_idx],
            markersize=4,
            strokecolor=:black,strokewidth=0.1,
        )


        Makie.xlims!(ax_resid, extrema(ts))
    end



    # Binned values on phase folded plot
    # Noise weighted (including jitter and GP)
    # bins = -0.45:0.1:0.45
    bins = -0.495:0.05:0.495
    binned = zeros(length(bins))
    binned_unc = zeros(length(bins))
    phase_folded = mod.(epochs_all .- t_peri .- T/4, T)./T .- 0.5
    
    for (i,bin_cent) in enumerate(bins)
        mask = bin_cent - step(bins)/2 .<= phase_folded .<= bin_cent + step(bins/2)
        if count(mask) == 0
            binned[i] = NaN
            continue
        end
        binned[i] = mean(
            rvs_all_minus_accel_minus_perspective[mask],
            ProbabilityWeights(1 ./ errs_all_data_jitter_gp[mask].^2)
        )
        binned_unc[i] = std(
            rvs_all_minus_accel_minus_perspective[mask],
            ProbabilityWeights(1 ./ errs_all_data_jitter_gp[mask].^2)
        )
    end
    errorbars!(
        ax_phase,
        bins,
        binned,
        binned_unc,
        color=:black,
        linewidth=2,
    )
    scatter!(
        ax_phase,
        bins,
        binned,
        color=:red,
        markersize=10,
        strokecolor=:black,
        strokewidth=2,
    )

    Legend(
        gs[1:2,2],
        [
          MarkerElement(color = Makie.wong_colors()[i], marker=:circle, markersize = 15)
          for i in 1:length(rv_likes)
        ],
        [rv.instrument_name for rv in rv_likes],
        "instrument",
        valign=:top,
        halign=:left,
        # width=Relative(1),
        # height=Relative(1),
    )
    markers =  [
        [
            LineElement(color = Makie.wong_colors()[i], linestyle = :solid,
            points = Point2f[((-1+i)/length(rv_likes), 0), ((-1+i)/length(rv_likes), 1)])
            for i in 1:length(rv_likes)
        ],
        LineElement(color = "#CCC", linestyle = :solid,
            points = Point2f[(0.0, 0), (0.0, 1)]),
        LineElement(color = :blue,linewidth=4,),
        MarkerElement(color = :red, strokecolor=:black, strokewidth=2, marker=:circle, markersize = 15),   
    ]
    labels = [
        "data uncertainty",
        any_models_have_a_gp ? "data, jitter, and\nmodel uncertainty" : "data and jitter\nuncertainty",
        "orbit model",
        "binned",
    ]
    if nonabsvis_parent(first(els)) != first(els)
        push!(markers, LineElement(color = :orange,linewidth=4,))
        push!(labels, "perspective")
    end

    Legend(gs[3,2], markers, labels, valign=:top, halign=:left)
    Makie.rowsize!(gs, 1, Auto(2))
    Makie.rowsize!(gs, 2, Auto(1))
    Makie.rowsize!(gs, 3, Auto(2))

end

function Octofitter.rvpostplot_animated(model, chain; framerate=4,compression=0, fname="rv-posterior.mp4", N=min(size(chain,1),50))
    imgs = []
    print("generating plots")
    for i in rand(1:size(chain,1), N)
        print(".")
        f = tempname()*".png"
        Octofitter.rvpostplot(model,chain,i,fname=f)
        push!(imgs, load(f))
        rm(f)
    end
    println()
    fig = Figure()
    ax = Axis(fig[1,1],autolimitaspect=1)
    hidedecorations!(ax)
    i = Observable(imgs[1])
    image!(ax, @lift(rotr90($i)))
    print("animating")
    Makie.record(fig, fname, imgs; framerate, compression) do img
        print(".")
        i[] = img
    end
    println()
    return fname
end


function _find_rv_zero_point_maxlike(
    rvlike,#::MarginalizedStarAbsoluteRVLikelihood,
    θ_system,
    planet_orbits::Tuple,
)
    T = Octofitter._system_number_type(θ_system)

    # Data for this instrument:
    epochs = rvlike.table.epoch
    σ_rvs = rvlike.table.σ_rv
    rvs = rvlike.table.rv

    jitter = getproperty(θ_system, rvlike.jitter_symbol)

    # RV residual calculation: measured RV - model
    resid = zeros(T, length(rvs))
    resid .+= rvs
    # Start with model ^

    # Go through all planets and subtract their modelled influence on the RV signal:
    for planet_i in eachindex(planet_orbits)
        orbit = planet_orbits[planet_i]
        planet_mass = θ_system.planets[planet_i].mass
        for i_epoch in eachindex(epochs)
            sol = orbitsolve(orbit, epochs[i_epoch])
            resid[i_epoch] -= radvel(sol, planet_mass*Octofitter.mjup2msol)
        end
    end
    
    # Marginalize out the instrument zero point using math from the Orvara paper
    A = zero(T)
    B = zero(T)
    for i_epoch in eachindex(epochs)
        # The noise variance per observation is the measurement noise and the jitter added
        # in quadrature
        var = σ_rvs[i_epoch]^2 + jitter^2
        A += 1/var
        B -= 2resid[i_epoch]/var
    end

    rv0 = B/2A

    return -rv0
end


end