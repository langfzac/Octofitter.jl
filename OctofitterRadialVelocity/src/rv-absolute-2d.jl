using Bumper

const rv_2d_cols = (:epoch, :order, :rv, :σ_rv)

struct StarAbsolute2DRVLikelihood{TTable<:Table, GP, TF, offset_symbol, jitter_symbol, ET, OT} <: Octofitter.AbstractLikelihood
    table::Table
    instrument_name::String
    gaussian_process::GP
    trend_function::TF
    offset_symbol::Symbol
    jitter_symbol::Symbol
    epochs::ET
    orders::OT
    function StarAbsolute2DRVLikelihood(
        observations...; 
        offset,
        jitter,
        trend_function = (θ_system, epoch) -> zero(Octofitter._system_number_type(θ_system)),
        instrument_name="", 
        gaussian_process=nothing
    )
        table = Table(observations...)
        if !issubset(rv_2d_cols, Tables.columnnames(table))
            error("Expected columns $rv_2d_cols")
        end
        rows = map(eachrow(table)) do row′
            row = (;row′[1]..., rv=float(row′[1].rv[1]))
            return row
        end
        table = Table(rows)
        
        # Sort by order, then epoch
        df = table |> DataFrame
        sort!(df, [order(:order), order(:epoch)])
        table = df |> Table

        epochs = Tuple(unique(table.epoch))
        orders = Tuple(unique(table.order))
        return new{typeof(table), typeof(gaussian_process), typeof(trend_function), offset, jitter, typeof(epochs), typeof(orders)}(table, instrument_name, gaussian_process, trend_function, offset, jitter, epochs, orders)
    end
end
StarAbsolute2DRVLikelihood(observations::NamedTuple...;kwargs...) = StarAbsolute2DRVLikelihood(observations; kwargs...)
function Octofitter.likeobj_from_epoch_subset(obs::StarAbsolute2DRVLikelihood, obs_inds)
    return StarAbsolute2DRVLikelihood(
        obs.table[obs_inds,:,1]...;
        offset_symbol=obs.offset_symbol,
        jitter_symbol=obs.jitter_symbol,
        trend_function=obs.trend_function,
        instrument_name=obs.instrument_name,
        gaussian_process=obs.gaussian_process,
    )
end
export StarAbsolute2DRVLikelihood

function _getparams(::StarAbsolute2DRVLikelihood{TTable, GP, TF, offset_symbol, jitter_symbol}, θ_system) where {TTable, GP, TF, offset_symbol, jitter_symbol}
    offset = getproperty(θ_system, offset_symbol)
    jitter = getproperty(θ_system, jitter_symbol)
    return (;offset, jitter)
end

function Octofitter.ln_like(
    rvlike::StarAbsolute2DRVLikelihood,
    θ_system,
    planet_orbits::Tuple,
    orbit_solutions,
    orbit_solutions_i_epoch_start
)
    L = length(rvlike.table.epoch)
    LE = length(rvlike.epochs)
    LO = length(rvlike.orders)
    T = Octofitter._system_number_type(θ_system)
    ll = zero(T)

    # Get offset and jitter values from the input
    (;offset, jitter) = _getparams(rvlike, θ_system)

    @no_escape begin
        # pre allocate arrays N epochs x N orders
        rv_buf = @alloc(T, LE, LO)
        rv_var_buf = @alloc(T, LE, LO)

        # rv data - offset - trend (if any)
        @views rv_buf[:] .= rvlike.table.rv .- offset .- rvlike.trend_function(θ_system, rvlike.table.epoch)

        # Now get the barycentric rv at each epoch
        # Assumes data is sorted by order first, then epoch
        for planet_i in eachindex(planet_orbits)
            planet_mass = θ_system.planets[planet_i].mass
            for epoch_i in eachindex(rvlike.epochs)
                @views rv_buf[epoch_i, :] .-= radvel(
                    orbit_solutions[planet_i][epoch_i+orbit_solutions_i_epoch_start],
                    planet_mass*Octofitter.mjup2msol
                )
            end
        end

        # White noise contributions to the variance
        @views rv_var_buf[:] .= rvlike.table.σ_rv.^2 .+ jitter^2

        # Now, if no GP, just compute Gaussian likelihood
        if isnothing(rvlike.gaussian_process)
            fx = MvNormal(Diagonal(vec(rv_var_buf)))
            ll += logpdf(fx, vec(rv_buf))
        else
            # If we have a GP
            local gp
            try
                gp = @inline rvlike.gaussian_process(θ_system)
            catch err
                if err isa PosDefException
                    @warn "err" exception=(err, catch_backtrace()) maxlog=1
                    ll = convert(T, -Inf)
                elseif err isa ArgumentError
                    @warn "err" exception=(err, catch_backtrace()) maxlog=1
                    ll = convert(T, -Inf)
                else
                    rethrow(err)
                end
            end

            if isfinite(ll)
                epochs = @alloc(T, LE)
                epochs .= rvlike.epochs
                # 1) GP for each order
                if gp isa Vector{GP}
                    for order_i in eachindex(rvlike.orders)
                        fx_i = gp[order_i](epochs, rv_var_buf[:, order_i])
                        try 
                            ll += logpdf(fx_i, rv_buf[:, order_i])::T
                        catch err
                            if err isa PosDefException || err isa DomainError
                                @warn "err" exception=(err, catch_backtrace()) θ_system
                                ll = convert(T, -Inf)
                            else
                                rethrow(err)
                            end
                        end
                        # If one of the orders fail, break out of the loop
                        if ~isfinite(ll); break; end
                    end
                end
            end
        end
    end
    return ll
end

function Octofitter.generate_from_params(like::StarAbsolute2DRVLikelihood, θ_system, orbits::Vector{<:RadialVelocityOrbit})
    epochs = like.table.epoch
    orders = like.table.order
    σ_rvs = like.table.σ_rv
    planet_masses = [θ_planet.mass for θ_planet in θ_system.planets] .* 0.000954588 # Mjup -> Msun

    rvs = radvel.(reshape(orbits, :, 1), epochs, transpose(planet_masses))
    rvs = sum(rvs, dims=2)[:,1] .+ θ_system.rv
    radvel_table = Table(epoch=epochs, rv=rvs, order=orders, σ_rv=σ_rvs)    

    return StarAbsolute2DRVLikelihood(radvel_table)
end