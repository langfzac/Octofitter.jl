using Bumper




function make_ln_like(system::System, θ_system)

    # We want to gather up all observation epochs in a standardized order.
    # That way we can loop through and solve each orbit at each epoch.

    # We assume that all planets will need to be solved at every general system observation epoch
    # We assume that each planet will need to be solved at every epoch it has attached observations
    # We'll solve them once in a hot (possibly multi-threaded) loop, and pass views into each likelihood
    # function.

    # I want a vector of solution types per object
    # The solutions will be ordered first, each epoch in the system observations,
    # then, each epoch in that planet's observations.
    # Note that in practice MA is always fixed when solving; it's only e that we might want the gradient for.
    epochs_system_obs = Float64[]    
    for obs in system.observations
        if hasproperty(obs, :table) && hasproperty(obs.table, :epoch)
            # TODO: deal with HGCA
            append!(epochs_system_obs, obs.table.epoch)
        end
    end
    
    planet_keys = Symbol[]
    planet_construction_exprs = Expr[]
    planet_like_exprs = Expr[]
    planet_orbit_solution_exprs = Expr[]
    planet_sol_keys = Symbol[]
    j = 0
    for i in eachindex(system.planets)
        planet = system.planets[i]
        OrbitType = _planet_orbit_type(planet)
        # θ_planet = θ_system.planets[i]
        key = Symbol("planet_$i")
        sols_key = Symbol("sols_planet_$i")

        epochs_planet_i = copy(epochs_system_obs)
        i_epoch_start = length(epochs_planet_i) + 1
        likelihood_exprs = map(enumerate(planet.observations)) do (i_like, like)
            num_epochs_this_obs = 0
            if hasproperty(like, :table) && hasproperty(like.table, :epoch)
                num_epochs_this_obs = length(like.table.epoch)
                append!(epochs_planet_i, like.table.epoch)
            end
            i_epoch_end = i_epoch_start + num_epochs_this_obs - 1
            expr = :(
                $(Symbol("ll$(j+1)")) = $(Symbol("ll$j")) + ln_like(
                    system.planets[$(Meta.quot(i))].observations[$i_like],
                    # $(system.planets[i].observations[like]),
                    θ_system.planets.$i,
                    $(key),
                    $sols_key, $(i_epoch_start-1)
                );
                # if !isfinite($(Symbol("ll$(j+1)")))
                #     println("invalid likelihood value encountered")
                # end
            )
            i_epoch_start = i_epoch_end + 1
            j+=1
            return expr
        end
        likelihood_expr = quote
            $(likelihood_exprs...)
        end
        # epochs_planet_i = tuple(epochs_planet_i...)

        planet_contruction = quote
            $key = $(OrbitType)(;merge(θ_system, θ_system.planets.$i)...)
        end

        if isempty(epochs_planet_i)  
            orbit_sol_expr = quote
                $sols_key = ()
            end
        else
            orbit_sol_expr = quote
                # Pre-solve kepler's equation for all epochs
                # epochs = Vector{Float64}(undef, $(length(epochs_planet_i)))
                epochs = @alloc(Float64, $(length(epochs_planet_i)))
                $((
                    :(epochs[$j] = $(epochs_planet_i[j]))
                    for j in 1:length(epochs_planet_i)
                )...)

                sol0 = orbitsolve($key, first(epochs))
                # $sols_key = Vector{typeof(sol0)}(undef, length(epochs))
                $sols_key = @alloc(typeof(sol0), length(epochs))
                $sols_key[begin] = sol0
                $_kepsolve_all!(view($sols_key, 2:length(epochs)), $key, view(epochs, 2:length(epochs)))
            end
        end

        push!(planet_keys, key)
        push!(planet_construction_exprs, planet_contruction)
        push!(planet_like_exprs, likelihood_expr)
        push!(planet_orbit_solution_exprs, orbit_sol_expr)
        push!(planet_sol_keys, sols_key)
    end

    # TODO: now need to pass solved epochs in through system Likelihoods, then deal with HGCA.

    i_epoch_start = 1
    # TODO: this seems way overcomplicated? Just need a list of symbols
    # interpolated in.
    solutions_list = :(tuple($((:($sols_key) for sols_key in planet_sol_keys)...)))
    sys_exprs = map(eachindex(system.observations)) do i
        like = system.observations[i]
        num_epochs_this_obs = 0
        if hasproperty(like, :table) && hasproperty(like.table, :epoch)
            num_epochs_this_obs = length(like.table.epoch)
        end
        i_epoch_end = i_epoch_start + num_epochs_this_obs - 1
        # Provide the number of observations as a compile time constant 
        expr = :(
            $(Symbol("ll$(j+1)")) = $(Symbol("ll$j")) + ln_like(
                system.observations[$i], θ_system, elems,
                ($solutions_list),
                $(i_epoch_start-1)
            );
            # if !isfinite($(Symbol("ll$(j+1)")))
            #     println("invalid likelihood value encountered")
            # end
        )
        i_epoch_start = i_epoch_end + 1
        j+=1
        return expr
    end

    return @RuntimeGeneratedFunction(:(function (system::System, θ_system)
        ll0 = zero(_system_number_type(θ_system))

        @no_escape begin

            # Construct all orbit elements
            $(planet_construction_exprs...)

            # Solve all orbits
            $(planet_orbit_solution_exprs...)

            # evaluate all their individual observation likelihoods
            $(planet_like_exprs...)

            # Construct a tuple of existing planet orbital elements
            elems = tuple($(planet_keys...))
            
            $(sys_exprs...)

        end

        return $(Symbol("ll$j"))
    end))
end

const _kepsolve_use_threads = Ref(false)
function _kepsolve_all!(solutions, orbit, epochs)
    if _kepsolve_use_threads[]
        return _kepsolve_all_multithread!(solutions, orbit, epochs)
    else
        return _kepsolve_all_singlethread!(solutions, orbit, epochs)
    end
end
function _kepsolve_all_singlethread!(solutions, orbit, epochs)
    for epoch_i in eachindex(epochs)
        solutions[epoch_i] = orbitsolve(orbit, epochs[epoch_i])
    end
    return solutions
end

function _kepsolve_all_multithread!(solutions, orbit, epochs)
    Threads.@threads for epoch_i in 1:length(epochs)
        solutions[epoch_i] = orbitsolve(orbit, epochs[epoch_i])
    end
    return solutions
end


# Generate calibration data
"""
    generate_from_params(system, θ=drawfrompriors(system))

Generate a new system and observations from an existing model.
"""
function generate_from_params(system::System, θ_newsystem = drawfrompriors(system))

    # Generate new orbits for each planet in the system
    orbits = map(eachindex(system.planets)) do i
        planet = system.planets[i]
        θ_newplanet = θ_newsystem.planets[i]
        neworbit = Octofitter.construct_elements(Octofitter.orbittype(planet), θ_newsystem, θ_newplanet)
        return neworbit
    end

    # Generate new observations for each planet in the system
    newplanets = map(1:length(system.planets)) do i
        planet = system.planets[i]
        orbit = orbits[i]
        θ_newplanet = θ_newsystem.planets[i]
        newplanet_obs = map(planet.observations) do obs
            return generate_from_params(obs, θ_newplanet, orbit)
        end
        newplanet = Planet{Octofitter.orbittype(planet)}(planet.priors, planet.derived, newplanet_obs..., name=planet.name)
        return newplanet
    end

    # Generate new observations for the star
    newstar_obs = map(system.observations) do obs
        return generate_from_params(obs, θ_newsystem, collect(orbits))
    end

    # Generate new system
    newsystem = System(system.priors, system.derived, newstar_obs..., newplanets..., name=system.name)

    return newsystem
end
export generate_from_params

