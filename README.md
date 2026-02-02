# LaProteina.jl

Julia port of NVIDIA's [la-proteina](https://github.com/NVIDIA/la-proteina) protein generation model with exact numerical parity to the Python implementation.

## Overview

LaProteina.jl implements flow matching on protein structure space for unconditional protein generation. The model consists of:

- **ScoreNetwork**: 14-layer transformer (160M parameters) that predicts velocity fields for flow matching on CA coordinates and local latent representations
- **DecoderTransformer**: 12-layer VAE decoder that converts latent representations to all-atom protein structures

The implementation achieves <1e-5 numerical parity with the Python reference and generates valid protein structures with correct CA-CA distances (~3.8Å).

## Dependencies

### Required: Flowfusion.jl (rdn-flow branch)

This package requires the `rdn-flow` branch of Flowfusion.jl which adds the `RDNFlow` process for flow matching on (R^d)^n:

```julia
using Pkg
Pkg.add(url="https://github.com/MurrellGroup/Flowfusion.jl", rev="rdn-flow")
```

Or in your `Project.toml`:
```toml
[deps]
Flowfusion = "..."

[sources]
Flowfusion = {url = "https://github.com/MurrellGroup/Flowfusion.jl", rev = "rdn-flow"}
```

### Other Dependencies

The package also requires (installed automatically):
- Flux.jl
- NNlib.jl
- NPZ.jl (for loading weights)
- Distributions.jl

## Weights

You need to download the pretrained weights and place them in a `weights/` directory:

### Required Files

1. **Score Network weights**: `score_network.npz` (~634 MB)
2. **Decoder weights**: `decoder.npz` (~512 MB)

### Extracting Weights from Python Checkpoint

If you have the original Python checkpoint (`LD1_ucond_notri_512.ckpt`), use the provided script:

```bash
python scripts/extract_weights.py \
    --checkpoint /path/to/LD1_ucond_notri_512.ckpt \
    --output-dir weights/
```

This will create `score_network.npz` and `decoder.npz`.

### Weight Format

Weights are stored in NumPy `.npz` format with keys matching the Python model structure. The Julia weight loading functions handle the necessary transpositions for Julia's column-major order.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/MurrellGroup/LaProteina.jl")
```

## Quick Start

```julia
using LaProteina

# Load models
score_net = ScoreNetwork(
    n_layers=14, token_dim=768, n_heads=12,
    latent_dim=8, dim_cond=256, pair_dim=256
)
load_score_network_weights!(score_net, "weights/score_network.npz")

decoder = DecoderTransformer(
    n_layers=12, token_dim=768, n_heads=12,
    latent_dim=8, dim_cond=128, pair_dim=256
)
load_decoder_weights!(decoder, "weights/decoder.npz")

# Generate a 50-residue protein using Flowfusion's gen() API
L = 50   # sequence length
B = 3    # batch size (number of samples)

flow_samples = generate_with_flowfusion(score_net, L, B;
    nsteps=100,
    self_cond=true,
    schedule_mode=:power,
    schedule_p=2.0
)

# Decode to all-atom structure
dec_input = Dict(
    :z_latent => flow_samples[:local_latents],
    :ca_coors => flow_samples[:bb_ca],
    :mask => flow_samples[:mask]
)
dec_out = decoder(dec_input)

# Save to PDB
samples = Dict(
    :ca_coords => flow_samples[:bb_ca],
    :latents => flow_samples[:local_latents],
    :all_atom_coords => dec_out[:coors],
    :aatype => dec_out[:aatype_max],
    :atom_mask => dec_out[:atom_mask],
    :mask => flow_samples[:mask]
)
samples_to_pdb(samples, "output/"; prefix="generated")
```

## Flowfusion Integration

This package uses [Flowfusion.jl](https://github.com/MurrellGroup/Flowfusion.jl)'s `gen()` API for flow matching sampling. The integration provides:

- **RDNFlow process**: Flow matching on (R^d)^n for CA coordinates (3D with zero COM) and latent representations (8D)
- **gen() sampling**: Uses Flowfusion's standardized generation interface
- **Model wrapper**: `MutableScoreNetworkWrapper` adapts the ScoreNetwork to Flowfusion's expected `model(t, Xt) -> X1_hat` signature

### Using the Flowfusion API directly

```julia
using LaProteina
using Flowfusion: RDNFlow, gen, ContinuousState, tensor

# Create processes
P_ca = RDNFlow(3; zero_com=true)      # CA coordinates
P_ll = RDNFlow(8; zero_com=false)     # Latents
P = (P_ca, P_ll)

# Sample initial noise
x0_ca = Flowfusion.sample_rdn_noise(P_ca, L, B)
x0_ll = Flowfusion.sample_rdn_noise(P_ll, L, B)
X0 = (ContinuousState(x0_ca), ContinuousState(x0_ll))

# Create model wrapper
model = MutableScoreNetworkWrapper(score_net, L, B; self_cond=true)

# Generate with Flowfusion's gen()
steps = Float32.(range(0, 1, length=101))  # Time steps
X_final = gen(P, X0, model, steps)

# Extract results
ca_coords = tensor(X_final[1])
latents = tensor(X_final[2])
```

### Legacy API

The original `full_simulation` function is still available for backwards compatibility:

```julia
flow_samples = full_simulation(score_net, L, B;
    nsteps=100,
    self_cond=true,
    schedule_mode=:power,
    schedule_p=2.0
)
```

## Architecture Details

### Tensor Convention

Julia uses column-major order, so tensors are transposed from Python:
- Python: `[Batch, Length, Dim]`
- Julia: `[Dim, Length, Batch]`

Conversion utilities are provided:
```julia
julia_tensor = python_to_julia(python_tensor)  # [B,L,D] -> [D,L,B]
python_tensor = julia_to_python(julia_tensor)  # [D,L,B] -> [B,L,D]
```

### PyTorchLayerNorm

A key difference from Flux.jl: this package uses `PyTorchLayerNorm` which computes `sqrt(var + eps)` instead of Flux's `sqrt(var + eps²)`. This is critical for numerical parity when variance is small.

### Key Components

| Component | Description |
|-----------|-------------|
| `ScoreNetwork` | Flow matching velocity predictor |
| `DecoderTransformer` | VAE decoder for all-atom coords |
| `PairBiasAttention` | AF3-style attention with pair bias |
| `ProteINAAdaLN` | Adaptive LayerNorm with sigmoid gating |
| `SwiGLUTransition` | SwiGLU feedforward with adaptive scaling |
| `generate_with_flowfusion` | Flowfusion gen() API sampling |
| `MutableScoreNetworkWrapper` | Model wrapper for gen() |
| `full_simulation` | Legacy ODE/SDE integration |

## Numerical Parity

The implementation has been verified against the Python reference:

| Output | Max Difference |
|--------|---------------|
| CA coordinates (v) | 2.5e-6 |
| Local latents (v) | 3.5e-6 |

## File Structure

```
LaProteina.jl/
├── src/
│   ├── JuProteina.jl          # Main module
│   ├── constants.jl           # Amino acid constants
│   ├── utils.jl               # Tensor utilities, PyTorchLayerNorm
│   ├── inference.jl           # Legacy flow matching sampling
│   ├── flowfusion_sampling.jl # Flowfusion gen() API integration
│   ├── weights.jl             # Weight loading
│   ├── features/
│   │   ├── feature_factory.jl # Feature extraction
│   │   ├── time_embedding.jl  # Sinusoidal embeddings
│   │   └── pair_features.jl   # Distance/separation features
│   ├── nn/
│   │   ├── adaptive_ln.jl     # ProteINAAdaLN, AdaptiveOutputScale
│   │   ├── pair_bias_attention.jl
│   │   ├── transition.jl      # SwiGLU transition
│   │   ├── transformer_block.jl
│   │   ├── score_network.jl
│   │   └── decoder.jl
│   └── data/
│       └── pdb_loading.jl     # PDB I/O
├── scripts/
│   ├── extract_weights.py     # Extract weights from checkpoint
│   └── run_inference.jl       # Example inference script
├── test/
│   └── ...                    # Parity tests
└── weights/                   # Place weights here (not in git)
    ├── score_network.npz
    └── decoder.npz
```

## Citation

If you use this code, please cite the original la-proteina paper:

```bibtex
@article{laproteina2024,
  title={La-Proteina: ...},
  author={NVIDIA},
  year={2024}
}
```

## License

This Julia port follows the same license as the original la-proteina repository.
