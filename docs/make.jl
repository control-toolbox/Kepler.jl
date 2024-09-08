using Documenter

makedocs(;
    warnonly=[:cross_references, :autodocs_block],
    sitename="kepler",
    format=Documenter.HTML(;
        prettyurls=false,
        assets=[
            asset("https://control-toolbox.org/assets/css/documentation.css"),
            asset("https://control-toolbox.org/assets/js/documentation.js"),
        ],
    ),
    pages=["Orbit transfer" => "index.md"],
    checkdocs=:none,
)

deploydocs(; repo="github.com/control-toolbox/Kepler.jl.git", devbranch="main")
