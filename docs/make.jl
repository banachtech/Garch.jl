using Garch
using Documenter

DocMeta.setdocmeta!(Garch, :DocTestSetup, :(using Garch); recursive=true)

makedocs(;
    modules=[Garch],
    authors="banachtech <balaji@banach.tech> and contributors",
    repo="https://github.com/banachtech/Garch.jl/blob/{commit}{path}#{line}",
    sitename="Garch.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://banachtech.github.io/Garch.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/banachtech/Garch.jl",
    devbranch="main",
)
