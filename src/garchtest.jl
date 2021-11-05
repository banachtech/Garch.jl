using Turing, StatsBase, MCMCChains, CSV, DataFrames
using Garch

# generate fake data
x, h = Garch.simulate(0.5, 0.7, 0.0, n=1000)

# fit Garch(1,1) with mle
mle_fit = Garch.fitMLE(x, isZD=false)
mle_fit.pars ./ mle_fit.serr

# load btc data
btc = DataFrame(CSV.File("data/btc.csv", missingstrings=["NaN",""," "]))
dropmissing!(btc)
btc = log.(btc[2:end]) .- log.(btc[1:end - 1])
btc .= btc .- mean(btc)

# fit Garch(1,1) with mle
btcfit = Garch.fitMLE(btc[1:1000])

