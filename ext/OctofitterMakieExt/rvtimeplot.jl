


##################################################
# RV vs time plot
# Note that the "full" plot requires to use
# rvpostplot(). It shows the jitter, residuals,
# and phase folded curves properly for a single sample.
# This just gives a global view.


# If `relative_rv=true`, then plot the relative RV between secondary 
# and primary bodies.
function rvtimeplot(
    model,
    results,
    fname="$(model.system.name)-rvplot.png",
    args...;
    figure=(;),
    relative_rv=false,
    kwargs...
)
    fig = Figure(;
        size=(700,600),
        figure...
    )
    rvtimeplot!(fig.layout, model, results, args...; kwargs...)

    Makie.save(fname, fig, px_per_unit=3)

    return fig
end
function rvtimeplot!(
    gridspec_or_fig,
    model::Octofitter.LogDensityModel,
    results::Chains;
    # If less than 500 samples, just show all of them
    N  = min(size(results, 1)*size(results, 3), 500),
    # If showing all samples, include each sample once.
    # Otherwise, sample randomly with replacement
    ii = (
        N == size(results, 1)*size(results, 3) ? 
        (1:size(results, 1)*size(results, 3)) :
        rand(1:size(results, 1)*size(results, 3),N)
    ),
    axis=(;),
    colormap=:plasma,
    colorbar=true,
    top_time_axis=true,
    bottom_time_axis=true,
    relative_rv=nothing,
    kwargs...
)
    gs = gridspec_or_fig

    # Find the minimum time step size
    periods = Float64[]
    for planet_key in keys(model.system.planets)
        orbs = Octofitter.construct_elements(results, planet_key, :)
        append!(periods, period.(orbs))

        if isnothing(relative_rv)
            relative_rv = haskey(chain, Symbol("$(planet_key)_mass"))
        end
    end
    if isnothing(relative_rv)
        relative_rv = false
    end
    min_period = minimum(periods)
    med_period = quantile(periods, 0.35)

    # Always try and show at least one full period
    # decide t_start and t_end based on epochs of data if astrometry 
    # available, otherwise other data, otherwise arbitrary
    t_start = t_stop = mjd()
    for like_obj in model.system.observations
        if hasproperty(like_obj, :table) && hasproperty(like_obj.table, :epoch)
            t_start, t_stop = extrema(like_obj.table.epoch,)
            d = t_stop - t_start
            t_start -= 0.015d
            t_stop += 0.015d
        end
    end
    for planet in model.system.planets
        for like_obj in planet.observations
            if nameof(typeof(like_obj)) == :PlanetRelAstromLikelihood
                t_start, t_stop = extrema(like_obj.table.epoch)
                d = t_stop - t_start
                t_start -= 0.015d
                t_stop += 0.015d
                break
            end
            if hasproperty(like_obj, :table) && hasproperty(like_obj.table, :epoch)
                t_start, t_stop = extrema(like_obj.table.epoch)
                d = t_stop - t_start
                t_start -= 0.015d
                t_stop += 0.015d
            end
        end
    end
    t_cur = t_stop-t_start
    if t_cur < med_period
        t_extend = med_period - t_cur
        t_start -= t_extend/2
        t_stop += t_extend/2
    end
    ts = range(t_start, t_stop, step=min_period / 150)
    if length(ts) > 1500
        # Things can get very slow if we have some very short period 
        # solutions
        ts = range(t_start, t_stop, length=1500)
    end

    date_pos, date_strs = _date_ticks(ts)
    ax= Axis(
        gs[1, 1];
        ylabel=relative_rv ? "relative rv [m/s]" : "rv [m/s]",
        xaxisposition=:top,
        xticks=(date_pos, date_strs),
        xgridvisible=false,
        ygridvisible=false,
        xticksvisible=top_time_axis,
        xticklabelsvisible=top_time_axis,
        axis...
    )
    ax_secondary = Axis(
        gs[1, 1];
        xlabel="MJD",
        xgridvisible=false,
        ygridvisible=false,
        yticksvisible=false,
        yticklabelsvisible=false,
        ylabelvisible=false,
        xticksvisible=bottom_time_axis,
        xticklabelsvisible=bottom_time_axis,
        xlabelvisible=bottom_time_axis,
        axis...
    )
    linkxaxes!(ax, ax_secondary)
    xlims!(ax, extrema(ts))
    xlims!(ax_secondary, extrema(ts))



    rv_model_t = zeros(length(ii), length(ts))
    color_model_t = zeros(length(ii), length(ts))
    for (i_planet, planet_key) in enumerate(keys(model.system.planets))
      
        orbs = Octofitter.construct_elements(results, planet_key, ii)
        sols = orbitsolve.(orbs, ts')

        # Plot each relative RV separately
        if relative_rv
            orbs = Octofitter.construct_elements(results, planet_key, ii)

            sols = orbitsolve.(orbs, ts')
            
            au_per_yr_2_m_per_s = 4744 
            rv_model_t .= velz.(sols) .* au_per_yr_2_m_per_s
            color_model_t .= rem2pi.(meananom.(sols), RoundDown) .+ 0 .* ii

            lines!(
                ax,
                concat_with_nan(ts' .+ 0 .* ii),
                concat_with_nan(rv_model_t),
                color=concat_with_nan(color_model_t),
                colormap=:plasma,
                alpha=min.(1, 100 / length(ii)),
            )

        # Or the absolute RV of the primary
        else
            # Draws from the posterior
            mass = results["$(planet_key)_mass"][ii] .* Octofitter.mjup2msol

    
            rv_model_t .+= radvel.(sols, mass) 

                
            # We subtract off the "average" instrument RV offset here.
            # This varies by instrument, and isn't centred around a net hierarchical value.
            # For now just take the average of the different zero points per row.
            if i_planet == 1 && haskey(results, :rv0_1)
                rv0s = []
                for i in 1:10
                    k = Symbol("rv0_$i")
                    if haskey(results, k)
                        push!(rv0s, results[k][ii])
                    end
                end
                ave_rv0 = mean(stack(rv0s),dims=2)
                rv_model_t .-= ave_rv0

                color_model_t .+= rem2pi.(
                    meananom.(sols), RoundDown) .+ 0 .* ii
            end
        end
    end

    if !relative_rv
        lines!(ax,
            concat_with_nan(ts' .+ 0 .* rv_model_t),
            concat_with_nan(rv_model_t);
            alpha=min.(1, 100 / length(ii)),
            color=concat_with_nan(color_model_t),
            colorrange=(0,2pi),
            colormap
        )
    end



    # Now overplot the data points, if any.
    for planet_key in keys(model.system.planets)
        planet = getproperty(model.system.planets, planet_key)
        for like_obj in planet.observations
            if nameof(typeof(like_obj)) != :PlanetRelativeRVLikelihood || !relative_rv
                continue
            end
            epoch = vec(like_obj.table.epoch)
            rv = vec(like_obj.table.rv)
            σ_rv = vec(like_obj.table.σ_rv)
            inst_idx = vec(like_obj.table.inst_idx)
            if any(!=(1), inst_idx)
                @warn "instidx != 1 data plotting not yet implemented"
                continue
            end
            jitter = results["$(planet_key)_jitter_1"]
            σ_tot = median(sqrt.(σ_rv .^2 .+ jitter' .^2),dims=2)[:]
            Makie.errorbars!(
                ax, epoch, rv, σ_tot;
                color=:black,
                linewidth=3,
            )
            Makie.scatter!(
                ax, epoch, rv;
                color=:white,
                strokewidth=0.5,
                strokecolor=:black,
                markersize=2,
            )
        end
    end

    if colorbar
        Colorbar(
            gs[1,2];
            colormap,
            label="mean anomaly",
            colorrange=(0,2pi),
            ticks=(
                [0,pi/2,pi,3pi/2,2pi],
                ["0", "π/2", "π", "3π/2", "2π"]
            )
        )
    end
end
