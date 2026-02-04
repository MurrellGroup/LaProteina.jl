# LaProteina

Julia port of NVIDIA's [la-proteina](https://github.com/NVIDIA/la-proteina) protein generation model with exact numerical parity to the Python implementation.

## Overview

LaProteina implements flow matching on protein structure space for unconditional protein generation. The model architecture consists of:

- **VAE Encoder**: 12-layer transformer that encodes all-atom protein structures into per-residue latent representations (mean and log_scale for 8D latents)
- **ScoreNetwork**: 14-layer transformer (160M parameters) that predicts velocity fields for flow matching on CA coordinates and local latent representations
- **VAE Decoder**: 12-layer transformer that decodes latent representations back to all-atom protein structures

The implementation achieves <1e-5 numerical parity with the Python reference and generates valid protein structures with correct CA-CA distances (~3.8Å).

## Training Data

### Data Source: AlphaFold Database

Training data comes from the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/), which provides predicted structures for proteins from various proteomes.

The training pipeline expects mmCIF files in a directory structure:
```
~/shared_data/afdb_laproteina/raw/
├── AF-A0A000-F1-model_v4.cif
├── AF-A0A001-F1-model_v4.cif
└── ... (~289k files for full dataset)
```

### Data Filtering

Proteins are filtered during preprocessing:
- **Minimum length**: 30 residues
- **Maximum length**: 256 residues
- Files that fail to parse are skipped

Typical filtering results: ~80% of proteins pass length filter, ~0% parse failures.

## Training Pipeline

Training uses a two-stage approach for efficiency:

### Stage 1: Precompute VAE Encoder Outputs

Since the VAE encoder is frozen during ScoreNetwork training, we precompute encoder outputs once:

```bash
cd LaProteina
julia scripts/precompute_all_training_data.jl
```

This script:
1. Loads each protein structure from mmCIF files
2. Extracts CA coordinates and centers them (zero center of mass)
3. Runs the frozen VAE encoder to get `z_mean` and `z_log_scale` for each residue
4. Saves results to sharded JLD2 files for efficient loading

**Output format**: 10 sharded files (~230k proteins total, ~2.5 GB):
```
~/shared_data/afdb_laproteina/precomputed_shards/
├── train_shard_01.jld2  (~23k proteins, ~268 MB)
├── train_shard_02.jld2
└── ...
├── train_shard_10.jld2
```

**Resume support**: If interrupted, set `START_SHARD` environment variable:
```bash
START_SHARD=5 julia scripts/precompute_all_training_data.jl
```

**Processing time**: ~70 minutes per shard on A100 GPU (~11.5 hours total).

### Stage 2: Flow Matching Training

Train the ScoreNetwork on precomputed data:

```julia
using LaProteina
using Flux, CUDA, Optimisers
import Flowfusion: RDNFlow

# Load precomputed shard
proteins = load_precomputed_shard("train_shard_01.jld2")

# Load ScoreNetwork
score_net = ScoreNetwork(n_layers=14, token_dim=768, pair_dim=256,
                         n_heads=12, dim_cond=256, latent_dim=8)
load_score_network_weights!(score_net, "weights/score_network.npz")
score_net_gpu = score_net |> gpu

# Define flow processes (must match inference settings)
P_ca = RDNFlow(3; zero_com=true, schedule=:log, schedule_param=2.0f0,
               sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0)
P_ll = RDNFlow(8; zero_com=false, schedule=:power, schedule_param=2.0f0,
               sde_gt_mode=:tan, sde_gt_param=1.0f0)
P = (P_ca, P_ll)

# Training loop
opt_state = Optimisers.setup(Adam(1e-5), score_net_gpu)
for epoch in 1:num_epochs
    for batch_indices in batches
        # Create batch with z sampling (reparameterization trick)
        batch = batch_from_precomputed(proteins, batch_indices, P)

        # Compute flow matching loss and gradients
        loss, grads = Flux.withgradient(score_net_gpu) do m
            efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll,
                                    batch.x1_ca, batch.x1_ll,
                                    batch.t_ca, batch.t_ll, batch.t_model, batch.mask)
        end
        Optimisers.update!(opt_state, score_net_gpu, grads[1])
    end
end
```

### Precomputed Data Format

Each `PrecomputedProtein` contains:
- `ca_coords::Matrix{Float32}` - [3, L] centered CA coordinates in nanometers
- `z_mean::Matrix{Float32}` - [8, L] VAE encoder mean output
- `z_log_scale::Matrix{Float32}` - [8, L] VAE encoder log-scale output
- `mask::Vector{Float32}` - [L] residue mask (1.0 for valid residues)

At training time, latents are sampled using the reparameterization trick:
```julia
z = z_mean + randn(size(z_mean)) .* exp.(z_log_scale)
```

This matches the Python training behavior and ensures the ScoreNetwork sees the full distribution of encoder outputs.

## Dependencies

### Required: Flowfusion.jl (rdn-flow branch)

This package requires the `rdn-flow` branch of Flowfusion.jl which adds the `RDNFlow` process for flow matching on (R^d)^n:

```julia
using Pkg
Pkg.add(url="https://github.com/MurrellGroup/Flowfusion.jl", rev="rdn-flow")
```

### Other Dependencies

The package also requires (installed automatically):
- Flux.jl, CUDA.jl - Neural networks and GPU support
- NNlib.jl - Neural network primitives
- NPZ.jl - Loading weights from NumPy format
- JLD2.jl - Saving/loading precomputed data
- Distributions.jl

## Weights

You need pretrained weights in the `weights/` directory:

### Required Files

| File | Size | Description |
|------|------|-------------|
| `encoder.npz` | ~340 MB | VAE encoder weights |
| `score_network.npz` | ~634 MB | ScoreNetwork weights |
| `decoder.npz` | ~512 MB | VAE decoder weights |

### Extracting Weights from Python Checkpoint

If you have the original Python checkpoints:

```bash
# Extract ScoreNetwork and Decoder from flow model checkpoint
python scripts/extract_weights.py \
    --checkpoint /path/to/LD1_ucond_notri_512.ckpt \
    --output-dir weights/

# Extract Encoder from VAE checkpoint
python scripts/extract_encoder_weights.py \
    --checkpoint /path/to/AE1_ucond_512.ckpt \
    --output weights/encoder.npz
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/MurrellGroup/LaProteina.jl")
```

## Quick Start: Inference

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

# Generate a 50-residue protein
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
- **Schedule transforms**: Log schedule for CA, power schedule for latents
- **gen() sampling**: Uses Flowfusion's standardized generation interface
- **Model wrapper**: `MutableScoreNetworkWrapper` adapts the ScoreNetwork to Flowfusion's expected `model(t, Xt) -> X1_hat` signature

### Flow Matching Schedule

The flow uses different schedules for CA coordinates and latents:

| Component | Schedule | Parameter | SDE Mode |
|-----------|----------|-----------|----------|
| CA coords | Log | 2.0 | 1/t |
| Latents | Power | 2.0 | tan |

These schedules are baked into the `RDNFlow` process and must match between training and inference.

## Architecture Details

### Tensor Convention

Julia uses column-major order, so tensors are transposed from Python:
- Python: `[Batch, Length, Dim]`
- Julia: `[Dim, Length, Batch]`

### PyTorchLayerNorm

This package uses `PyTorchLayerNorm` which computes `sqrt(var + eps)` instead of Flux's `sqrt(var + eps²)`. This is critical for numerical parity.

### Key Components

| Component | Description |
|-----------|-------------|
| `EncoderTransformer` | VAE encoder (frozen during flow training) |
| `ScoreNetwork` | Flow matching velocity predictor |
| `DecoderTransformer` | VAE decoder for all-atom coords |
| `PairBiasAttention` | AF3-style attention with pair bias |
| `ProteINAAdaLN` | Adaptive LayerNorm with sigmoid gating |
| `SwiGLUTransition` | SwiGLU feedforward with adaptive scaling |
| `PrecomputedProtein` | Storage format for precomputed encoder outputs |
| `batch_from_precomputed` | Create training batches with z sampling |
| `efficient_flow_loss_gpu` | GPU-optimized flow matching loss |

## File Structure

```
LaProteina/
├── src/
│   ├── LaProteina.jl              # Main module
│   ├── constants.jl               # Amino acid constants
│   ├── utils.jl                   # Tensor utilities, PyTorchLayerNorm
│   ├── inference.jl               # Legacy flow matching sampling
│   ├── flowfusion_sampling.jl     # Flowfusion gen() API integration
│   ├── weights.jl                 # Weight loading functions
│   ├── training.jl                # Training utilities
│   ├── features/
│   │   ├── feature_factory.jl     # Feature extraction
│   │   ├── time_embedding.jl      # Sinusoidal embeddings
│   │   ├── pair_features.jl       # Distance/separation features
│   │   └── geometry.jl            # Geometric calculations
│   ├── nn/
│   │   ├── adaptive_ln.jl         # ProteINAAdaLN, AdaptiveOutputScale
│   │   ├── pair_bias_attention.jl # Attention with pair bias
│   │   ├── transition.jl          # SwiGLU transition
│   │   ├── transformer_block.jl   # Transformer blocks
│   │   ├── encoder.jl             # VAE encoder
│   │   ├── encoder_efficient.jl   # Efficient encoder for precomputation
│   │   ├── score_network.jl       # ScoreNetwork
│   │   ├── score_network_efficient.jl  # GPU-optimized ScoreNetwork
│   │   └── decoder.jl             # VAE decoder
│   ├── training/
│   │   └── precompute_encoder.jl  # Precomputed training data utilities
│   └── data/
│       ├── pdb_loading.jl         # PDB/mmCIF I/O
│       └── transforms.jl          # Data transforms
├── scripts/
│   ├── extract_weights.py         # Extract weights from checkpoint
│   ├── extract_encoder_weights.py # Extract encoder weights
│   ├── precompute_all_training_data.jl  # Full dataset precomputation
│   └── run_inference.jl           # Example inference script
├── test/
│   ├── train_on_shard.jl          # Training test on precomputed data
│   ├── test_precomputed_training.jl
│   ├── gpu_utilization_test.jl
│   └── smol_demo_training_data.jld2  # Small demo dataset
└── weights/                       # Place weights here (not in git)
    ├── encoder.npz
    ├── score_network.npz
    └── decoder.npz
```

## Numerical Parity

The implementation has been verified against the Python reference:

| Output | Max Difference |
|--------|---------------|
| CA coordinates (v) | 2.5e-6 |
| Local latents (v) | 3.5e-6 |
| Encoder outputs | ~1e-5 |

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
