using Documenter

makedocs(;
    warnonly = [:cross_references, :autodocs_block],
    sitename = "kepler",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Orbit transfer" => "index.md",
    ],
    checkdocs=:none,
)

deploydocs(
    repo = "github.com/control-toolbox/kepler.git",
    devbranch = "main"
)
