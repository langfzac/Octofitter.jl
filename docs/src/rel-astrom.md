# [Basic Astrometry Fit](@id fit-astrometry)

Here is a worked example of a one-planet model fit to relative astrometry (positions measured between the planet and the host star). 

Start by loading the Octofitter and Distributions packages:
```@example 1
using Octofitter, Distributions
```

### Specifying the data
We will create a likelihood object to contain our relative astrometry data. We can specify this data in several formats. It can be listed in the code or loaded from a file (eg. a CSV file, FITS table, or SQL database).

```@example 1
astrom_like = PlanetRelAstromLikelihood(
    #        MJD         mas                       mas                        mas         mas
    (epoch = 50000, ra = -505.7637580573554, dec = -66.92982418533026, σ_ra = 10, σ_dec = 10, cor=0),
    (epoch = 50120, ra = -502.570356287689, dec = -37.47217527025044, σ_ra = 10, σ_dec = 10, cor=0),
    (epoch = 50240, ra = -498.2089148883798, dec = -7.927548139010479, σ_ra = 10, σ_dec = 10, cor=0),
    (epoch = 50360, ra = -492.67768482682357, dec = 21.63557115669823, σ_ra = 10, σ_dec = 10, cor=0),
    (epoch = 50480, ra = -485.9770335870402, dec = 51.147204404903704, σ_ra = 10, σ_dec = 10, cor=0),
    (epoch = 50600, ra = -478.1095526888573, dec = 80.53589069730698, σ_ra = 10, σ_dec = 10, cor=0),
    (epoch = 50720, ra = -469.0801731788123, dec = 109.72870493064629, σ_ra = 10, σ_dec = 10, cor=0),
    (epoch = 50840, ra = -458.89628893460525, dec = 138.65128697876773, σ_ra = 10, σ_dec = 10, cor=0),
    instrument_name = "GPI" # optional -- name for this group of data
)
```

You can also specify it in separation (mas) and positon angle (rad):
```julia
astrom_like_2 = PlanetRelAstromLikelihood(
    (epoch = 50000, sep = 505.7637580573554, pa = deg2rad(24.1), σ_sep = 10, σ_pa =deg2rad(1.2), cor=0),
    # ...etc.
    instrument_name = "GPI" # optional -- name for this group of data
)
```

Another way we could specify the data is by column:
```@example 1
astrom_like = PlanetRelAstromLikelihood(
    Table(
        epoch= [50000,  50120, 50240, 50360,50480, 50600, 50720, 50840,], # MJD
        ra = [-505.764, -502.57, -498.209, -492.678,-485.977, -478.11, -469.08, -458.896,], # mas
        dec = [-66.9298, -37.4722, -7.92755, 21.6356, 51.1472,  80.5359,  109.729,  138.651, ], # mas
        σ_ra = fill(10.0, 8),
        σ_dec = fill(10.0, 8),
        cor = fill(0.0, 8),
    ),
    instrument_name = "GPI" # optional -- name for this group of data
)
nothing # hide
```

Finally we could also load the data from a file somewhere. Here is an example 
of loading a CSV:
```julia
using CSV # must install CSV.jl first
astrom_data = CSV.read("mydata.csv", Table)
astrom_like = PlanetRelAstromLikelihood(
    astrom_data,
    instrument_name="GPI"
)
```

You can also pass a DataFrame or any other table format.

In Octofitter, `epoch` is always the modified Julian date (measured in days). If you're not sure what this is, you can get started by just putting in arbitrary time offsets measured in days.

In this case, we specified `ra` and `dec` offsets in milliarcseconds. We could instead specify `sep` (projected separation) in milliarcseconds and `pa` in radians. You cannot mix the two formats in a single `PlanetRelAstromLikelihood` but you can create two different likelihood objects, one for each format.

### Creating a planet

We now create our first planet model. Let's name it planet `b`. 
The name of the planet will be used in the output results.

In Octofitter, we specify planet and system models using a "probabilistic
programming language". Quantities with a `~` are random variables. The distributions on the right hand sides are **priors**. You must specify a 
proper prior for any quantity which is allowed to vary. 

We now create our planet `b` model using the `@planet` macro.
```@example 1
@planet b Visual{KepOrbit} begin
    a ~ truncated(Normal(10, 4), lower=0.1, upper=100)
    e ~ Uniform(0.0, 0.5)
    i ~ Sine()
    ω ~ UniformCircular()
    Ω ~ UniformCircular()
    θ ~ UniformCircular()
    tp = θ_at_epoch_to_tperi(system,b,50420)
end astrom_like
nothing # hide
```

In the model definition, `b` is the name of the planet (it can be anything), `Visual{KepOrbit}` is the type of orbit parameterization ([see the PlanetOrbits.jl documentation page](https://sefffal.github.io/PlanetOrbits.jl/dev/api/)).


After the `begin` comes our variable specification. Quantities with a `~` are random variables aka. **our priors**. You must specify a proper prior for any quantity which is allowed to vary. 
"Uninformative" priors like `1/x` must be given bounds, and can be specified with `LogUniform(lower, upper)`.

!!! warning
    Make sure that variables like mass and eccentricity can't be negative. You can pass a distribution to `truncated` to prevent this, e.g. `M ~ truncated(Normal(1, 0.1),lower=0)`.

Priors can be any univariate distribution from the Distributions.jl package.

For a `KepOrbit` you must specify the following parameters:
* `a`: Semi-major axis, astronomical units (AU)
* `i`: Inclination, radians
* `e`: Eccentricity in the range [0, 1)
* `ω`: Argument of periastron, radius
* `Ω`: Longitude of the ascending node, radians.
* `tp`: Epoch of periastron passage

Many different distributions are supported as priors, including `Uniform`, `LogNormal`, `LogUniform`, `Sine`, and `Beta`. See the section on [Priors](@ref priors) for more information.
The parameters can be specified in any order.

You can also hardcode a particular value for any parameter if you don't want it to vary. Simply replace eg. `e ~ Uniform(0, 0.999)` with `e = 0.1`.
This `=` syntax works for arbitrary mathematical expressions and even functions. We use it here to reparameterize `tp`.

`tp` is a date which sets the location of the planet around its orbit. It repeats every orbital period and the orbital period depends on the scale of the orbit. This makes it quite hard to sample from. We therefore reparameterize using `θ` parameter, representing the position angle of the planet at a given reference epoch. This parameterization speeds up sampling quite a bit!

After the variables block are zero or more `Likelihood` objects. These are observations specific to a given planet that you would like to include in the model. If you would like to sample from the priors only, don't pass in any observations.

For this example, we specify `PlanetRelAstromLikelihood` block. This is where you can list the position of a planet relative to the star at different epochs.


### Creating a system

A system represents a host star with one or more planets. Properties of the whole system are specified here, like parallax distance and mass of the star. This is also where you will supply data like images, astrometric acceleration, or stellar radial velocity since they don't belong to any planet in particular.

```@example 1
@system Tutoria begin
    M ~ truncated(Normal(1.2, 0.1), lower=0.1)
    plx ~ truncated(Normal(50.0, 0.02), lower=0.1)
end b
nothing #hide
```

`Tutoria` is the name we have given to the system. It could be eg `PDS70`, or anything that will help you keep track of the results.

The variables block works just like it does for planets. Here, the two parameters you must provide are:
* `M`: Gravitational parameter for the total mass of the system, expressed in units of Solar mass.
* `plx`: Distance to the system expressed in milliarcseconds of parallax.

Make sure to truncate the priors to prevent unphysical negative masses or parallaxes.


After that, just list any planets that you want orbiting the star. Here, we pass planet `b`.
This is also where we could pass likelihood objects for system-wide data like stellar radial velocity.


### Prepare model
We now convert our declarative model into efficient, compiled code:
```@example 1
model = Octofitter.LogDensityModel(Tutoria)
```

This type implements the julia LogDensityProblems.jl interface and can be passed to a wide variety of samplers.


### Initialize starting points for chains

Run the `initialize!` function to find good starting points for the chain. You can provide guesses for parameters if you want to.
```julia
init_chain = initialize!(model) # No guesses provided, slower global optimization will be used
```

```@example 1
init_chain = initialize!(model, (;
    plx = 50,
    M = 1.21,
    planets = (;
        b=(;
            a = 10.0,
            e = 0.01,
            # note! Never initialize a value on the bound, exactly 0 eccentricity is disallowed by the `Uniform(0,1)` prior
        )
    )
))
```

### Visualize the starting points

Plot the inital values to make sure that they are reasonable, and match your data. This is a great time to confirm that your data were entered in correctly.

```@example 1
using CairoMakie
octoplot(model, init_chain)
```

The starting points for sampling look reasonable!

!!! note
    The return value from `initialize!` is a "variational approximation". You can pass that chain to any function expecting a `chain` argument, like `Octofitter.savechain` or `octocorner`. It gives a rough approximation of the posterior we expect. The central values are probably close, but the uncertainties are unreliable.

### Sampling
Now we are ready to draw samples from the posterior:
```@example 1
octofit(model, verbosity = 0,iterations=2,adaptation=2,); # hide
chain = octofit(model)
```

You will get an output that looks something like this with a progress bar that updates every second or so. You can reduce or completely silence the output by reducing the `verbosity` value down to 0.

Once complete, the `chain` object will hold the sampler results. Displaying it prints out a summary table like the one shown above.
For a basic model like this, sampling should take less than a minute on a typical laptop.

Sampling can take much longer when you have measurements with very small uncertainties.

### Diagnostics
The first thing you should do with your results is check a few diagnostics to make sure the sampler converged as intended.

A few things to watch out for: check that you aren't getting many numerical errors (`ratio_divergent_transitions`). 
This likely indicates a problem with your model: either invalid values of one or more parameters are encountered (e.g. the prior on semi-major axis includes negative values) or that there is a region of very high curvature that is failing to sample properly. This latter issue can lead to a bias in your results.

One common mistake is to use a distribution like `Normal(10,3)` for semi-major axis. This left tail of this distribution includes negative values, and our orbit model is not defined for negative semi-major axes. A better choice is a `truncated(Normal(10,3), lower=0.1)` distribution (not including zero, since a=0 is not defined).

You may see some warnings during initial step-size adaptation. These are probably nothing to worry about if sampling proceeds normally afterwards.

You should also check the acceptance rate (`mean_accept`) is reasonably high and the mean tree depth (`mean_tree_depth`) is reasonable (~4-8). 
Lower than this and the sampler is taking steps that are too large and encountering a U-turn very quicky. Much larger than this and it might be being too conservative. 

Next, you can make a trace plot of different variabes to visually inspect the chain:
```@example 1
using CairoMakie
lines(
    chain["b_a"][:],
    axis=(;
        xlabel="iteration",
        ylabel="semi-major axis (AU)"
    )
)
```

And an auto-correlation plot:
```@example 1
using StatsBase
using CairoMakie
lines(
    autocor(chain["b_e"][:], 1:500),
    axis=(;
        xlabel="lag",
        ylabel="autocorrelation",
    )
)
```
This plot shows that these samples are not correlated after only about 5 iterations. No thinning is necessary.

To confirm convergence, you may also examine the `rhat` column from chains. This diagnostic approaches 1 as the chains converge and should at the very least equal `1.0` to one significant digit (3 recommended).

Finaly, you might consider running multiple chains. Simply run `octofit` multiple times, and store the result in different variables. Then you can combine the chains using `chainscat` and run additional inter-chain convergence diagnostics:
```@example 1
using MCMCChains
chain1 = octofit(model)
chain2 = octofit(model)
chain3 = octofit(model)
merged_chains = chainscat(chain1, chain2, chain3)
gelmandiag(merged_chains)
```

This will check that the means and variances are similar between chains that were initialized at different starting points.

### Analysis
As a first pass, let's plot a sample of orbits drawn from the posterior.
The function `octoplot` is a conveninient way to generate a 9-panel plot of velocities and position:
```@example 1
using CairoMakie
octoplot(model,chain)
```
This function draws orbits from the posterior and displays them in a plot. Any astrometry points are overplotted. 

You can control what panels are displayed, the time range, colourscheme, etc. See the documentation on `octoplot` for more details.

### Pair Plot
A very useful visualization of our results is a pair-plot, or corner plot. We can use the `octocorner` function and our PairPlots.jl package for this purpose:
```@example 1
using CairoMakie
using PairPlots
octocorner(model, chain, small=true)
```
Remove `small=true` to display all variables.

In this case, the sampler was able to resolve the complicated degeneracies between eccentricity, the longitude of the ascending node, and argument of periapsis.


### Saving your chain

Variables can be retrieved from the chains using the following sytnax: `sma_planet_b = chain["b_a",:,:]`. The first index is a string or symbol giving the name of the variable in the model. Planet variables are prepended by the name of the planet and an underscore.

You can save your chain in FITS table format by running:
```julia
Octofitter.savechain("mychain.fits", chain)
```

You can load it back via:
```julia
chain = Octofitter.loadchain("mychain.fits")
```

### Saving your model

You may choose to save your model so that you can reload it later to make plots, etc:
```@example 1
using Serialization
serialize("model1.jls", model)
```

Which can then be loaded at a later time using:
```julia
using Serialization
using Octofitter # must include all the same imports as your original script
model = deserialize("model1.jls")
```

!!! warning
    Serialized models are only loadable/restorable on the same computer, version of Octofitter, and version of Julia. They are not intended for long-term archiving. For reproducibility, make sure to keep your original model definition script.


### Comparing chains
We can compare two different chains by passing them both to `octocorner`. Let's compare the `init_chain` with the full results from `octofit`:
```@example 1
octocorner(model, chain, init_chain, small=true)
```