
using CSV, DataFrames, Plots, Statistics
df = DataFrame(CSV.File("data/btc.csv"))
x = log.(df.cl[2:2001]) .- log.(df.cl[1:2000])
x .= x .- mean(x)

m1 = Std()
m2 = GJR()

res1 = fit!(m1, x)
res2 = fit!(m2, x)

v1 = predict(m1, 200, 240)
v2 = predict(m2, 200, 240)
