using Pkg
Pkg.activate(dirname(@__DIR__))

using LaProteina
using NPZ

weights = npzread(joinpath(dirname(@__DIR__), "weights", "score_network.npz"))

# Create model
sn = ScoreNetwork(n_layers=14, token_dim=768, pair_dim=256, n_heads=12, dim_cond=256, latent_dim=8)

# Check shapes
py_w = Float32.(weights["init_repr_factory.linear_out.weight"])
py_w_t = py_w'
jl_w = sn.init_repr_factory.projection.weight

println("Python weight shape: ", size(py_w))
println("Python weight shape (transposed): ", size(py_w_t))
println("Julia weight shape: ", size(jl_w))
