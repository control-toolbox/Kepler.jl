# to run the documentation generation:
# julia --project=. docs/make.jl
pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))
pushfirst!(LOAD_PATH, @__DIR__)

using Documenter

# For reproducibility
mkpath(joinpath(@__DIR__, "src", "assets"))
cp(
    joinpath(@__DIR__, "Manifest.toml"),
    joinpath(@__DIR__, "src", "assets", "Manifest.toml");
    force=true,
)
cp(
    joinpath(@__DIR__, "Project.toml"),
    joinpath(@__DIR__, "src", "assets", "Project.toml");
    force=true,
)

repo_url = "github.com/control-toolbox/Kepler.jl"

makedocs(;
    warnonly=[:cross_references, :autodocs_block],
    sitename="Kepler",
    format=Documenter.HTML(;
        repolink="https://" * repo_url,
        prettyurls=false,
        size_threshold_ignore=["index.md"],
        assets=[
            asset("https://control-toolbox.org/assets/css/documentation.css"),
            asset("https://control-toolbox.org/assets/js/documentation.js"),
        ],
    ),
    pages=["Orbit transfer" => "index.md"],
    checkdocs=:none,
)

# PR previews are opt-in: CTActions sets DOCUMENTER_PUSH_PREVIEW to "true" on a pull request
# carrying the `documentation push preview` label.
push_preview = get(ENV, "DOCUMENTER_PUSH_PREVIEW", "false") == "true"

deploydocs(; repo=repo_url * ".git", devbranch="main", push_preview=push_preview)
