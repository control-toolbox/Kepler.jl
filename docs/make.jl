using Documenter

makedocs(;
    warnonly = [:cross_references, :autodocs_block],
    sitename = "Kepler.jl",
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Introduction" => "index.md",
        "Kepler" => "application-kepler.md",
    ],
    checkdocs=:none,
)

deploydocs(
    repo = "github.com/control-toolbox/Kepler.jl.git",
    devbranch = "main"
)
