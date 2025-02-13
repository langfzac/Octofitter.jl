using KroneckerProductKernels

const rv_2d_cols = (:epoch, :wavelength, :order, :rv)

struct StarAbsolute2DRVLikelihood{TTable<:Table, GP, TF, OS, JS, ET, WT, XT} <: Octofitter.AbstractLikelihood
    table::Table
    instrument_name::String
    gaussian_process::GP
    trend_function::TF
    offset_symbol::OS
    jitter_symbol::JS
    epochs::ET
    wavelengths::WT
    X ::XT
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
        #df = table |> DataFrame
        #sort!(df, [order(:wavelength), order(:epoch)])
        #table = df |> Table
        table = sort(table, by=x->(x.wavelength, x.epoch))

        epochs = Tuple(unique(table.epoch))
        wavelengths = Tuple(unique(table.wavelength))
        X = RowVecs(vcat([[λ t] for (λ,t) in zip(table.wavelength, table.epoch)]...)) # Input for GP evaluation

        # Offset and jitter symbols
        # Assumes each order's symbols just have an "_i" appended in numerical order
        offsets = Symbol.(Tuple("$(offset)_" .* string.(collect(1:length(wavelengths)))))
        jitters = Symbol.(Tuple("$(jitter)_" .* string.(collect(1:length(wavelengths)))))

        return new{typeof(table), typeof(gaussian_process), typeof(trend_function), typeof(offsets), typeof(jitters), typeof(epochs), typeof(wavelengths), typeof(X)}(table, instrument_name, gaussian_process, trend_function, offsets, jitters, epochs, wavelengths, X)
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

function _getparams(rvlike::StarAbsolute2DRVLikelihood{TTable, GP, TF, OS, JS, ET, WT, XT}, θ_system) where {TTable, GP, TF, OS, JS, ET, WT, XT}
    offsets = [getproperty(θ_system, sym) for sym in rvlike.offset_symbol]
    jitters = [getproperty(θ_system, sym) for sym in rvlike.jitter_symbol]
    return (;offsets, jitters)
end


function Octofitter.ln_like(
    rvlike::StarAbsolute2DRVLikelihood,
    θ_system,
    planet_orbits::Tuple,
    orbit_solutions,
    orbit_solutions_i_epoch_start
)
    LE = length(rvlike.epochs)
    LO = length(rvlike.wavelengths)
    T = Octofitter._system_number_type(θ_system)
    ll = zero(T)

    # Get offset and jitter values from the input
    #offsets = getproperty.(Ref(θ_system), rvlike.offset_symbol)
    #jitters = getproperty.(Ref(θ_system), rvlike.jitter_symbol)
    #(;offsets, jitters) = _getparams(rvlike, θ_system)

    @no_escape begin
        # pre allocate arrays N epochs x N orders
        rv_buf = @alloc(T, LE, LO)
        offsets = @alloc(T, LO)
        jitters = @alloc(T, LO)

        # Get the offsets and jitters
        for i in eachindex(offsets)
            offsets[i] = getproperty(θ_system, rvlike.offset_symbol[i])
            jitters[i] = getproperty(θ_system, rvlike.jitter_symbol[i])
        end

        # rv data - offset - trend (if any)
        @views rv_buf[:] .= rvlike.table.rv
        for i in eachindex(offsets)
            @views rv_buf[:, i] .-= offsets[i] # .- rvlike.trend_function(θ_system, rvlike.table.epoch)
        end

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

        # Assume each jitter term is the diagonal of Σλ
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
            Σy = KroneckerProductKernels.:⊗(Diagonal(collect(jitters.^2)), Diagonal(ones(T, LE)))
            fx = gp(rvlike.X, Σy)
            try
                ll += logpdf(fx, vec(rv_buf))
            catch err
                if err isa PosDefException || err isa DomainError
                    @warn "err" exception=(err, catch_backtrace()) θ_system
                    ll = convert(T, -Inf)
                else
                    rethrow(err)
                end
            end
        end
    end
    return ll
end

function Octofitter.generate_from_params(like::StarAbsolute2DRVLikelihood, θ_system, orbits::Vector{<:RadialVelocityOrbit})
    epochs = like.table.epoch
    orders = like.table.order
    wavelengths = like.table.wavelength
    planet_masses = [θ_planet.mass for θ_planet in θ_system.planets] .* 0.000954588 # Mjup -> Msun

    rvs = zeros(length(unique(epochs)), length(unique(orders)), length(orbits))

    # Add barycentric RVs to each order
    for (i,orbit) in enumerate(orbits)
        rvs_bary = radvel.(orbit, unique(epochs), planet_masses[i])
        for j in eachindex(rvs_bary)
            rvs[j,:,i] .= rvs_bary[j]
        end
    end

    # Now add the offsets to each order
    offsets = getproperty.(Ref(θ_newsystem), rvlike.offset_symbol)

    radvel_table = Table(epoch=epochs, rv=rvs[:], order=orders, wavelength=wavelengths)    

    return StarAbsolute2DRVLikelihood(radvel_table)
end