# Posterior Predictive Checks

A posterior predictive check compares our true data with simulated data drawn from the posterior. This allows us to evaluate if the model is able to reproduce our observations appropriately. Samples drawn from the posterior predictive distribution should match the locations of the original data.

To demonstrate, we will fit a model to relative astrometry data:

```@example 1
using Octofitter
using Plots:Plots
using Distributions

astrom_like = PlanetRelAstromLikelihood(Table(;
    epoch= [50000,50120,50240,50360,50480,50600,50720,50840,],
    ra = [-505.764,-502.57,-498.209,-492.678,-485.977,-478.11,-469.08,-458.896,],
    dec = [-66.9298,-37.4722,-7.92755,21.6356, 51.1472, 80.5359, 109.729, 138.651,],
    σ_ra = fill(10.0, 8),
    σ_dec = fill(10.0, 8),
    cor = fill(0.0, 8)
))
@planet b Visual{KepOrbit} begin
    a ~ truncated(Normal(10, 4), lower=0, upper=100)
    e ~ Uniform(0.0, 0.5)
    i ~ Sine()
    ω ~ UniformCircular()
    Ω ~ UniformCircular()
    τ ~ UniformCircular(1.0)
    P = √(b.a^3/system.M)
    tp =  b.τ*b.P*365.25 + 50420 # reference epoch for τ. Choose an MJD date near your data.
end astrom_like
@system Tutoria begin
    M ~ truncated(Normal(1.2, 0.1), lower=0)
    plx ~ truncated(Normal(50.0, 0.02), lower=0)
end b
model = Octofitter.LogDensityModel(Tutoria)

using Random
Random.seed!(0)
chain = octofit(model)
```

We now have our posterior as approximated by the MCMC chain. Convert these posterior samples into orbit objects:
```@example 1
# Instead of creating orbit objects for all rows in the chain, just pick
# every twentieth row.
ii = 1:20:1000
orbits = Octofitter.construct_elements(chain, :b, ii)
```

Calculate and plot the location the planet would be at each observation epoch:
```@example 1
epochs = astrom_like.table.epoch' # transpose

x = raoff.(orbits, epochs)
y = decoff.(orbits, epochs)

Plots.plot(orbits, kind=(:raoff, :decoff), color=:lightgrey)
Plots.scatter!(
    x, y,
    lims=:symmetric,
    markerstrokewidth=0,
    markersize=3,
    legend=:topleft,
    label=["epoch $i" for  i in epochs]
)
Plots.plot!(astrom_like, linewidth=2, color=:black, label="observed")

Plots.xlims!(-700,400)
Plots.ylims!(-300,300)
```

Looks like a great match to the data! Notice how the uncertainty around the middle point is lower than the ends. That's because the orbit's posterior location at that epoch is also constrained by the surrounding data points. We can know the location of the planet in hindsight better than we could measure it!

You can follow this same procedure for any kind of data modelled with Octofitter.