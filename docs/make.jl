using Documenter, DirectDetections


makedocs(
    sitename="DirectDetections.jl",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Tutorials" => [
            "Fit Astrometry" => "modelling.md",
            "Fit Astrometric Acceleration" => "pma.md",
            "Fit Images" => "images.md",
            "Fit Radial Velocity" => "rv.md",
            "Connecting Mass with Photometry" => "mass-photometry.md",
            "Loading and Saving Data" => "loading-saving.md",
            "Custom Likelihoods" => "custom-likelihood.md",
            # "Multiple Planets" => "multi-planets.md",
            # "" => "multi-planets.md",
        ],
        "Documentation" => [
            "Priors" => "priors.md",
            "Derived Variables" => "derived.md",
            "Sampler" => "samplers.md",
            "Chains" => "chains.md",
            "Kepler Solver" => "kepler.md",
            "API" => "api.md"
        ]
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)


deploydocs(
    repo = "github.com/sefffal/DirectDetections.jl.git",
    devbranch = "main"
)
