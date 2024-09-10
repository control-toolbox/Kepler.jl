using Documenter

repo_url = "github.com/control-toolbox/Kepler.jl"

makedocs(;
    warnonly=[:cross_references, :autodocs_block],
    sitename="kepler",
    format=Documenter.HTML(;
        repolink = "https://"*repo_url,
        prettyurls=false,
        assets=[
            asset("https://control-toolbox.org/assets/css/documentation.css"),
            asset("https://control-toolbox.org/assets/js/documentation.js"),
        ],
    ),
    pages=["Orbit transfer" => "index.md"],
    checkdocs=:none,
)

deploydocs(;
    repo=repo_url*".git", devbranch="main"
)
