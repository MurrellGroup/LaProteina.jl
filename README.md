# LaProteina

Julia port of NVIDIA's [la-proteina](https://github.com/NVIDIA/la-proteina) protein generation model with exact numerical parity to the Python implementation, extended with [Branching Flows](https://github.com/MurrellGroup/BranchingFlows.jl) for variable-length protein generation.

## Overview

LaProteina implements flow matching on protein structure space for unconditional protein generation. The model architecture consists of:

- **VAE Encoder**: 12-layer transformer that encodes all-atom protein structures into per-residue latent representations (mean and log_scale for 8D latents)
- **ScoreNetwork**: 14-layer transformer (160M parameters) that predicts velocity fields for flow matching on CA coordinates and local latent representations
- **VAE Decoder**: 12-layer transformer that decodes latent representations back to all-atom protein structures

The implementation achieves <1e-5 numerical parity with the Python reference and generates valid protein structures with correct CA-CA distances (~3.8 Angstrom).

### Branching Flows Extension

The **BranchingScoreNetwork** extends the base ScoreNetwork with split and deletion heads, enabling variable-length protein generation via [Branching Flows](https://github.com/MurrellGroup/BranchingFlows.jl). During generation, residues can split (creating new positions) or be deleted, allowing the model to produce proteins of varying lengths from a single initial noise state.

See [docs/branching_flows.md](docs/branching_flows.md) for full details.

## How It Works

### Flow Matching on Protein Structure

La-proteina generates proteins by learning to reverse a noising process. At t=0 we have pure noise; at t=1 we have a valid protein. The ScoreNetwork predicts the velocity field `v(x, t)` that transports noise to structure:

```
x_t = (1 - t) * x_0 + t * x_1       # Linear interpolation (bridge)
v(x_t, t) = x_1 - x_0                # Target velocity
x_1_hat = x_t + (1 - t) * v_hat      # Predicted endpoint from velocity
```

The model operates on two modalities simultaneously:
- **CA coordinates** (`bb_ca`): 3D positions of alpha-carbon atoms, in nanometers
- **Local latents** (`local_latents`): 8D per-residue latent vectors from the VAE encoder

Each modality has its own schedule transform (see below), but they share the same transformer backbone.

### Schedule Transforms

The flow uses non-uniform time schedules that accelerate interpolation at different rates:

| Component | Schedule | Formula | Effect |
|-----------|----------|---------|--------|
| CA coords | Log | `tau(s) = (1 - 10^(-2s)) / (1 - 10^(-2))` | Fast early interpolation; 90% done by s=0.48 |
| Latents | Power | `tau(s) = s^2` | Slow early, fast late; 90% done by s=0.95 |

The model receives schedule-transformed times `tau(s)` for each modality, while the generation process steps through uniform `s` in [0, 1]. This separation means:
- The process handles interpolation and noise injection in `tau`-space
- The model is conditioned on `tau_ca` and `tau_ll` (different per modality)
- Loss scaling uses `1/(1-tau)^2` per modality

### SDE Sampling

During generation, an SDE adds controlled noise along the trajectory for better sample quality:

```
dx = [v + g(tau) * score_scale * score] * d_tau + sqrt(2 * g(tau) * noise_scale) * dW
```

Where `g(tau) = 1/(tau + eps)` for CA and `g(tau) = tan(tau * pi/2)` for latents. The `sc_scale_noise=0.1` parameter controls noise magnitude. Above `t_lim_ode=0.98`, the SDE switches to pure ODE for clean final convergence.

### Self-Conditioning

The model uses self-conditioning: the predicted `x_1` from the previous timestep is fed back as an additional input. This significantly improves generation quality. The self-conditioning features (`x_sc`) are processed through the same feature extraction pipeline as the noisy input.

## Architecture

### ScoreNetwork

The core model is a 14-layer transformer with:
- **Token dimension**: 768
- **Pair dimension**: 256 (for pair representation)
- **Attention heads**: 12
- **Conditioning dimension**: 256 (for time embedding)
- **Output parameterization**: v (velocity)

Each transformer layer consists of:
1. **PairBiasAttention**: Multi-head attention with pair bias (AF3-style), using ProteINAAdaLN (adaptive LayerNorm with sigmoid gating)
2. **SwiGLU Transition**: Feedforward with SwiGLU activation and adaptive output scaling
3. Optional **PairUpdate**: Outer product update of pair representation

Input features include:
- Sequence features: time embedding, position embedding, noisy CA coords, noisy latents, self-conditioning predictions
- Pair features: pairwise CA distances (binned), relative sequence separation

### DecoderTransformer

The VAE decoder converts latent representations back to all-atom structures:
- Input: `z_latent` [8, L, B] + `ca_coors` [3, L, B]
- Output: all-atom coordinates [3, 37, L, B], sequence logits [20, L, B], atom masks [37, L, B]

### EncoderTransformer

The VAE encoder extracts latent representations from all-atom structures:
- Input: all-atom coordinates + residue types + masks
- Output: mean [8, L, B] + log_scale [8, L, B]
- Frozen during flow matching training; outputs are precomputed

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
        batch = batch_from_precomputed(proteins, batch_indices, P)
        loss, grads = Flux.withgradient(score_net_gpu) do m
            efficient_flow_loss_gpu(m, batch.xt_ca, batch.xt_ll,
                                    batch.x1_ca, batch.x1_ll,
                                    batch.t_ca, batch.t_ll, batch.t_model, batch.mask)
        end
        Optimisers.update!(opt_state, score_net_gpu, grads[1])
    end
end
```

## Quick Start: Inference

### Fixed-Length Generation (Base Model)

```julia
using LaProteina
using Flux: gpu

# Load models
score_net = ScoreNetwork(
    n_layers=14, token_dim=768, n_heads=12,
    latent_dim=8, dim_cond=256, pair_dim=256,
    qk_ln=true, update_pair_repr=false, output_param=:v
)
load_score_network_weights!(score_net, "weights/score_network.npz")

decoder = DecoderTransformer(
    n_layers=12, token_dim=768, n_heads=12,
    latent_dim=8, dim_cond=128, pair_dim=256,
    qk_ln=true, update_pair_repr=false
)
load_decoder_weights!(decoder, "weights/decoder.npz")

# Generate (see test/test_gpu_sampling.jl for full example)
samples = sample_with_flowfusion(gpu(score_net), decoder, 100, 1;
    nsteps=400, self_cond=true, dev=gpu)

# Save to PDB
samples_to_pdb(samples, "output/"; prefix="generated", save_all_atom=true)
```

### Variable-Length Generation (Branching Flows)

```julia
using LaProteina
using BranchingFlows
using Flux: gpu

# Load BranchingScoreNetwork
base = ScoreNetwork(n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
                    dim_cond=256, latent_dim=8, output_param=:v,
                    qk_ln=true, update_pair_repr=false)
model = BranchingScoreNetwork(base)

# Load fine-tuned weights
weights = load("weights/branching_full.jld2")
Flux.loadmodel!(model.base, weights["base"])
Flux.loadmodel!(model.indel_time_proj, weights["indel_time_proj"])
Flux.loadmodel!(model.split_head, weights["split_head"])
Flux.loadmodel!(model.del_head, weights["del_head"])
model = gpu(model)

# Generate variable-length protein
result = generate_with_branching(model, 100;
    nsteps=400, latent_dim=8, self_cond=true, dev=gpu, verbose=true)

println("Generated protein: $(result.final_length) residues")
```

See [test/test_branching_full_sampling.jl](test/test_branching_full_sampling.jl) for a complete example with cosine time steps and decoder output.

## Dependencies

### Required Packages

| Package | Branch | Purpose |
|---------|--------|---------|
| [Flowfusion.jl](https://github.com/MurrellGroup/Flowfusion.jl) | rdn-flow | RDNFlow process, gen() API |
| [BranchingFlows.jl](https://github.com/MurrellGroup/BranchingFlows.jl) | - | Variable-length generation |
| [ForwardBackward.jl](https://github.com/MurrellGroup/ForwardBackward.jl) | - | State representations |
| Flux.jl | - | Neural networks |
| CUDA.jl | - | GPU support |
| NPZ.jl | - | Loading weights from NumPy format |
| JLD2.jl | - | Saving/loading precomputed data |

## Weights

You need pretrained weights in the `weights/` directory:

| File | Size | Description |
|------|------|-------------|
| `encoder.npz` | ~340 MB | VAE encoder weights |
| `score_network.npz` | ~634 MB | ScoreNetwork weights |
| `decoder.npz` | ~512 MB | VAE decoder weights |
| `branching_full.jld2` | ~650 MB | Fine-tuned BranchingScoreNetwork (optional) |

### Extracting Weights from Python Checkpoint

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

## File Structure

```
LaProteina/
├── src/
│   ├── LaProteina.jl              # Main module
│   ├── constants.jl               # Amino acid and atom constants
│   ├── utils.jl                   # Tensor utilities, PyTorchLayerNorm
│   ├── inference.jl               # Legacy flow matching sampling
│   ├── flowfusion_sampling.jl     # Flowfusion gen() API integration
│   ├── weights.jl                 # Weight loading functions
│   ├── training.jl                # Training utilities
│   ├── features/
│   │   ├── feature_factory.jl     # Feature extraction (29 feature types)
│   │   ├── time_embedding.jl      # Sinusoidal embeddings
│   │   ├── pair_features.jl       # Distance/separation features
│   │   └── geometry.jl            # Dihedral and bond angle calculations
│   ├── nn/
│   │   ├── adaptive_ln.jl         # ProteINAAdaLN, AdaptiveOutputScale
│   │   ├── pair_bias_attention.jl # AF3-style attention with pair bias
│   │   ├── transition.jl          # SwiGLU transition blocks
│   │   ├── transformer_block.jl   # Transformer blocks, PairUpdate
│   │   ├── encoder.jl             # VAE encoder
│   │   ├── encoder_efficient.jl   # Efficient encoder for precomputation
│   │   ├── score_network.jl       # ScoreNetwork (main model)
│   │   ├── score_network_efficient.jl  # GPU-optimized ScoreNetwork
│   │   └── decoder.jl             # VAE decoder
│   ├── training/
│   │   └── precompute_encoder.jl  # Precomputed training data utilities
│   ├── data/
│   │   ├── pdb_loading.jl         # PDB/mmCIF file I/O
│   │   └── transforms.jl          # Data transforms
│   └── branching/
│       ├── branching_score_network.jl  # BranchingScoreNetwork
│       ├── branching_inference.jl      # Generation with branching
│       ├── branching_training.jl       # Training utilities
│       └── branching_states.jl         # State conversion utilities
├── scripts/
│   ├── train_branching_full.jl    # Branching flows training (full)
│   ├── plot_training.jl           # Training loss visualization
│   ├── precompute_all_training_data.jl  # Dataset precomputation
│   └── extract_weights.py         # Weight extraction from Python
├── test/
│   ├── test_gpu_sampling.jl       # Base model sampling test
│   ├── test_branching_full_sampling.jl  # Branching model sampling
│   └── ...                        # Parity tests, feature tests, etc.
├── weights/                       # Place weights here (not in git)
└── docs/
    ├── branching_flows.md         # Branching Flows integration guide
    └── branching_flows_conversion.md  # Conversion reference
```

## Architecture Details

### Tensor Convention

Julia uses column-major order, so tensors are transposed from Python:
- Python: `[Batch, Length, Dim]`
- Julia: `[Dim, Length, Batch]`

### PyTorchLayerNorm

This package uses `PyTorchLayerNorm` which computes `sqrt(var + eps)` instead of Flux's `sqrt(var + eps^2)`. This is critical for numerical parity.

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
