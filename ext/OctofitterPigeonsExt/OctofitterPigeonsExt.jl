module OctofitterPigeonsExt
using Random
using Octofitter
using Pigeons
using MCMCChains
using LinearAlgebra
using Logging
using Pathfinder

function (model::Octofitter.LogDensityModel)(θ)
    return model.ℓπcallback(θ)
end
function Pigeons.initialization(model::Octofitter.LogDensityModel, rng::AbstractRNG, chain_no::Int)

    Octofitter.get_starting_point!!(rng, model)

    initial_θ_t = collect(model.starting_points[chain_no])
    # initial_θ_t = collect(Octofitter.get_starting_point!!(rng, model))
    initial_θ = model.invlink(initial_θ_t)
    initial_logpost = model.ℓπcallback(initial_θ_t)

    if any(!isfinite, initial_θ_t) || any(!isfinite, initial_θ) || !isfinite(initial_logpost)
        error("Could not find a starting point with finite arguments initial_logpost=$initial_logpost, initial_θ_t=$initial_θ_t, initial_θ=$(model.arr2nt(initial_θ))")
    end

    # @info "Determined initial position" chain_no initial_θ initial_θ_nt=model.arr2nt(initial_θ) initial_logpost
    # @info "Determined initial position" chain_no initial_logpost
    
    return initial_θ_t
end

# Valid for reference model only
function Pigeons.sample_iid!(model_reference::Octofitter.LogDensityModel, replica, shared)
    # This could in theory be done without any array allocations
    θ = model_reference.sample_priors(replica.rng)
    θ_t = model_reference.link(θ)
    replica.state .= θ_t
end

function Pigeons.default_reference(target::Octofitter.LogDensityModel)
    reference_sys = prior_only_model(target.system)
    # Note we could run into issues if their priors aren't well handled by the default
    # autodiff backend
    reference = Octofitter.LogDensityModel(reference_sys)
    return reference
end


function Pigeons.default_explorer(target::Octofitter.LogDensityModel)
    return Pigeons.Compose(
        Pigeons.SliceSampler(),
        Pigeons.AutoMALA(default_autodiff_backend=target.autodiff_backend_symbol)
    )
end


"""
octofit_pigeons(model; nrounds, n_chains=[auto])

Use Pigeons.jl to sample from intractable posterior distributions.

```julia
model = Octofitter.LogDensityModel(System, autodiff=:ForwardDiff, verbosity=4)
chain, pt = octofit_pigeons(model)
```
"""
Base.@nospecializeinfer function Octofitter.octofit_pigeons(
    target::Octofitter.LogDensityModel;
    n_rounds::Int,
    n_chains::Int=16,
    n_chains_variational::Int=16,
    checkpoint::Bool=false,
    pigeons_kw...
)
    @nospecialize
    inputs = Pigeons.Inputs(;
        target,
        record = [traces; round_trip; record_default(); index_process],
        multithreaded=true,
        show_report=true,
        n_rounds,
        n_chains,
        n_chains_variational,
        variational = GaussianReference(first_tuning_round = 5),
        checkpoint,
        pigeons_kw...
    )
    return octofit_pigeons(inputs)
end

Base.@nospecializeinfer function Octofitter.octofit_pigeons(
    pt::Pigeons.PT
)
    @nospecialize

    start_time = time()
    pt = pigeons(pt)
    stop_time = time()

    mcmcchains = Chains(pt.inputs.target, pt)
    mcmcchains_with_info = MCMCChains.setinfo(
        mcmcchains,
        (;
            start_time,
            stop_time,
            model_name=pt.inputs.target.system.name
        )
    )
    return (;chain=mcmcchains_with_info, pt)
end
Base.@nospecializeinfer function Octofitter.octofit_pigeons(
    inputs::Pigeons.Inputs
)
    @nospecialize

    start_time = time()
    pt = pigeons(inputs)
    stop_time = time()

    mcmcchains = Chains(inputs.target, pt)
    mcmcchains_with_info = MCMCChains.setinfo(
        mcmcchains,
        (;
            start_time,
            stop_time,
            model_name=inputs.target.system.name
        )
    )
    return (;chain=mcmcchains_with_info, pt)
end
Base.@nospecializeinfer function MCMCChains.Chains(
    model::Octofitter.LogDensityModel,
    pt::Pigeons.PT,
    chain_num::Int=pt.inputs.n_chains
)
    ln_prior = Octofitter.make_ln_prior_transformed(model.system)
    ln_like = Octofitter.make_ln_like(model.system, model.arr2nt(model.sample_priors(Random.default_rng())))

    # Resolve the array back into the nested named tuple structure used internally.
    # Augment with some internal fields
    samples = get_sample(pt, chain_num)
    chain_res = map(samples) do sample 
        θ_t = @view(sample[begin:begin+model.D-1])
        logpot = sample[model.D+1]
        # Map the variables back to the constrained domain and reconstruct the parameter
        # named tuple structure.
        θ = model.invlink(θ_t)
        resolved_namedtuple = model.arr2nt(θ)
        # Add log posterior, tree depth, and numerical error reported by
        # the sampler.
        # Also recompute the log-likelihood and add that too.
        ll = ln_like(model.system, resolved_namedtuple)
        lp = ln_prior(θ,true)
        # logpot does not equal ll + lp, so I'm not fully sure what it is.
        return merge((;
            loglike=ll,
            logprior=lp,
            logpost=ll+lp,
            pigeons_logpotential = logpot
        ), resolved_namedtuple)
    end
    # Then finally flatten and convert into an MCMCChain object / table.
    # Mark the posterior, likelihood, numerical error flag, and tree depth as internal
    mcmcchains = Octofitter.result2mcmcchain(
        chain_res,
        Dict(:internals => [
            :loglike,
            :logpost,
            :logprior,
            :pigeons_logpotential,
        ])
    )
    return mcmcchains
end

end