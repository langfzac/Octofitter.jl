using KroneckerProductKernels

const rv_2d_cols = (:epoch, :wavelength, :order, :rv)

struct StarAbsolute2DRVLikelihood{TTable<:Table, GP, TF, ET, WT, XT} <: Octofitter.AbstractLikelihood
    table::Table
    priors::Octofitter.Priors
    derived::Octofitter.Derived
    held_out_table::TTable
    name::String
    gaussian_process::GP
    trend_function::TF
    epochs::ET
    wavelengths::WT
    X::XT
end
function StarAbsolute2DRVLikelihood(
    observations;
    variables::Union{Nothing, Tuple{Octofitter.Priors, Octofitter.Derived}}=nothing, 
    trend_function = (θ_obs, epoch) -> zero(Octofitter._system_number_type(θ_obs)),
    name::String,
    gaussian_process=nothing
)
    (priors,derived)=variables

    table = Table(observations)[:,:,1]
    if !Octofitter.equal_length_cols(table)
        error("The columns in the input data do not all have the same length")
    end
    if !issubset(rv_2d_cols, Tables.columnnames(table))
        error("Expected columns $rv_2d_cols")
    end
    rows = map(eachrow(table)) do row′
        row = (;row′[1]..., rv=float(row′[1].rv[1]))
        return row
    end
    table = Table(rows)
    
    if any(>=(mjd("2050")),  table.epoch) || any(<=(mjd("1950")),  table.epoch)
        @warn "The data you entered fell outside the range year 1950 to year 2050. The expected input format is MJD (modified julian date). We suggest you double check your input data!"
    end

    # Sort by order, then epoch
    table = sort(table, by=x->(x.wavelength, x.epoch))

    epochs = Tuple(unique(table.epoch))
    wavelengths = Tuple(unique(table.wavelength))
    X = RowVecs(vcat([[λ t] for (λ,t) in zip(table.wavelength, table.epoch)]...)) # Input for GP evaluation

    # We need special book keeping for computing cross-validataion scores
    # We keep a table of "held out" data if needed for that purpose.
    # Here we leave it empty.
    held_out_table = empty(table)

    return StarAbsolute2DRVLikelihood{typeof(table), typeof(gaussian_process), typeof(trend_function), typeof(epochs), typeof(wavelengths), typeof(X)}(
        table, priors, derived, held_out_table, name, gaussian_process, trend_function, epochs, wavelengths, X
    )
end
export StarAbsolute2DRVLikelihood

# In-place simulation logic for StarAbsolute2DRVLikelihood (performance-critical)
function Octofitter.simulate!(rv_model_buf, rvlike::StarAbsolute2DRVLikelihood,  θ_system, θ_obs, planet_orbits::Tuple, orbit_solutions, orbit_solutions_i_epoch_start)
    L = length(rvlike.epochs)

    # Compute the model RV values (what we expect to observe)
    rv_model_buf .= rvlike.trend_function.(Ref(θ_obs), rvlike.epochs)

    # Add RV constribution from all planets:
    for planet_i in eachindex(planet_orbits)
        orbit = planet_orbits[planet_i]
        planet_mass = θ_system.planets[planet_i].mass
        for epoch_i in eachindex(rvlike.epochs)
            rv_model_buf[epoch_i] += radvel(
                orbit_solutions[planet_i][epoch_i+orbit_solutions_i_epoch_start],
                planet_mass*Octofitter.mjup2msol
            )
        end
    end

    return (rv_model = rv_model_buf, epochs = rvlike.epochs)
end

# Allocating simulation logic for StarAbsoluteRVLikelihood (convenience method)
function Octofitter.simulate(rvlike::StarAbsolute2DRVLikelihood, θ_system, θ_obs, planet_orbits::Tuple, orbit_solutions, orbit_solutions_i_epoch_start)
    T = Octofitter._system_number_type(θ_system)
    L = length(rvlike.epochs)
    rv_model_buf = Vector{T}(undef, L)
    return Octofitter.simulate!(rv_model_buf, rvlike, θ_system, θ_obs, planet_orbits, orbit_solutions, orbit_solutions_i_epoch_start)
end

function Octofitter.ln_like(
    rvlike::StarAbsolute2DRVLikelihood,
    θ_system,
    θ_obs,
    planet_orbits::Tuple,
    orbit_solutions,
    orbit_solutions_i_epoch_start
)
    LE = length(rvlike.epochs)
    LO = length(rvlike.wavelengths)
    T = Octofitter._system_number_type(θ_system)
    ll = zero(T)

    @no_escape begin
        # pre allocate arrays N epochs x N orders
        rv_model_buf = @alloc(T, LE, LO)
        rv_residuals = @alloc(T, LE*LO)
        rv_bary_buf = @alloc(T, LE)

        # Use in-place simulation method to get model values
        sim = Octofitter.simulate!(rv_bary_buf, rvlike, θ_system, θ_obs, planet_orbits, orbit_solutions, orbit_solutions_i_epoch_start)
        rv_model_buf .= rv_bary_buf # Broadcast across each column

        # Compute residuals
        rv_residuals .= rvlike.table.rv .- vec(rv_model_buf) # Flatten buffer

        # Compute the 2D GP
        local gp, fx
        try
            gp = @inline rvlike.gaussian_process(θ_obs)
            fx = gp(rvlike.X, θ_obs.σ_data^2)
        catch err
            @info "bad gp: " θ_obs
            if err isa DomainError
                ll = convert(T, -Inf)
            elseif err isa PosDefException
                ll = convert(T, -Inf)
            elseif err isa ArgumentError
                ll = convert(T, -Inf)
            else
                rethrow(err)
            end
        end

        if isfinite(ll)
            try
                # Normal path: evaluate likelihood against all data
                if isempty(rvlike.held_out_table)
                    ll += logpdf(fx, vec(rv_model_buf))
                else
                    # Implement cross validation later...
                    throw(NotImplementedException())
                end
            catch err
                @info "bad ll: " θ_obs
                if err isa PosDefException || err isa DomainError
                    @warn "err" exception=(err, catch_backtrace()) θ_system
                    ll = convert(T, -Inf)
                else
                    rethrow(err)
                end
            end
        end
    end
    ll isa Float64 ? (@info "Internal ll: " ll) : nothing
    return ll
end

#=function Octofitter.generate_from_params(like::StarAbsolute2DRVLikelihood, θ_system,  θ_obs, orbits, orbit_solutions, orbit_solutions_i_epoch_start; add_noise)
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
end=#