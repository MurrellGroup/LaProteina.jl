# LaProteina.jl

Julia implementation of NVIDIA's [La-Proteina](https://github.com/NVIDIA-Digital-Bio/la-proteina) protein generation model with exact numerical parity to the Python reference, extended with [Branching Flows](https://github.com/MurrellGroup/BranchingFlows.jl) for variable-length protein generation.

## Overview

LaProteina implements flow matching on protein structure space for atomistic protein generation. The model architecture consists of:

- **VAE Encoder**: 12-layer transformer that encodes all-atom protein structures into per-residue latent representations (mean and log_scale for 8D latents)
- **ScoreNetwork**: 14-layer transformer (160M parameters) that predicts velocity fields for flow matching on CA coordinates and local latent representations
- **VAE Decoder**: 12-layer transformer that decodes latent representations back to all-atom protein structures

The implementation achieves <1e-5 numerical parity with the Python reference and generates valid protein structures with correct CA-CA distances (~3.8 Angstrom).

### Model Variants

Seven pretrained model variants (LD1-LD7) are available, covering unconditional generation, long proteins, and motif scaffolding. See [docs/model_variants.md](docs/model_variants.md) for full details.

| Model | Description | Max Length |
|-------|-------------|-----------|
| LD1 | Unconditional, no triangle update | 512 |
| LD2 | Unconditional, with triangle update | 512 |
| LD3 | Long proteins | 800 |
| LD4 | Indexed motif scaffolding, all-atom | 256 |
| LD5 | Indexed motif scaffolding, tip-atom | 256 |
| LD6 | Unindexed motif scaffolding, all-atom | 256 |
| LD7 | Unindexed motif scaffolding, tip-atom | 256 |

### Branching Flows Extension

The **BranchingScoreNetwork** extends the base ScoreNetwork with split and deletion heads, enabling variable-length protein generation via [Branching Flows](https://github.com/MurrellGroup/BranchingFlows.jl). During generation, residues can split (creating new positions) or be deleted, allowing the model to produce proteins of varying lengths from a single initial noise state.

See [docs/branching_flows.md](docs/branching_flows.md) for full details.

## How It Works

### Flow Matching on Protein Structure

La-Proteina generates proteins by learning to reverse a noising process. At t=0 we have pure noise; at t=1 we have a valid protein. The ScoreNetwork predicts the velocity field `v(x, t)` that transports noise to structure:

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

See [docs/model_architecture.md](docs/model_architecture.md) for a full deep dive including AdaLN variants, SwiGLU details, and BranchingScoreNetwork internals.

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
3. Optional **TriangularUpdate**: Outer product update of pair representation (LD2)

Input features include:
- Sequence features: time embedding, position embedding, noisy CA coords, noisy latents, self-conditioning predictions
- Pair features: pairwise CA distances (binned), relative sequence separation

See [docs/feature_system.md](docs/feature_system.md) for the complete feature catalog (29 feature types) and pre-configured factory functions.

### DecoderTransformer

The VAE decoder converts latent representations back to all-atom structures:
- Input: `z_latent` [8, L, B] + `ca_coors` [3, L, B]
- Output: all-atom coordinates [3, 37, L, B], sequence logits [20, L, B], atom masks [37, L, B]

### EncoderTransformer

The VAE encoder extracts latent representations from all-atom structures:
- Input: all-atom coordinates + residue types + masks
- Output: mean [8, L, B] + log_scale [8, L, B]
- Frozen during flow matching training; outputs are precomputed

## Weights

Pretrained weights are available as SafeTensors files, converted from the original NVIDIA PyTorch checkpoints:

### Score Networks

| File | Model | Size |
|------|-------|------|
| `LD1_ucond_notri_512.safetensors` | Unconditional, no triangle update | ~1.6 GB |
| `LD2_ucond_tri_512.safetensors` | Unconditional, triangle update | ~1.6 GB |
| `LD3_ucond_notri_800.safetensors` | Long proteins (300-800 res) | ~1.6 GB |
| `LD4_motif_idx_aa.safetensors` | Indexed motif, all-atom | ~1.6 GB |
| `LD5_motif_idx_tip.safetensors` | Indexed motif, tip-atom | ~1.6 GB |
| `LD6_motif_uidx_aa.safetensors` | Unindexed motif, all-atom | ~1.6 GB |
| `LD7_motif_uidx_tip.safetensors` | Unindexed motif, tip-atom | ~1.6 GB |

### Decoders (Autoencoders)

| File | Model | Size |
|------|-------|------|
| `AE1_ucond_512.safetensors` | Unconditional (up to 512 res) | ~978 MB |
| `AE2_ucond_800.safetensors` | Long proteins (up to 800 res) | ~978 MB |
| `AE3_motif.safetensors` | Motif scaffolding | ~978 MB |

Weights are licensed under the [NVIDIA Open Model License](https://github.com/NVIDIA-Digital-Bio/la-proteina/blob/main/LICENSE/license_weights.txt).

## Quick Start: Inference

See [docs/inference_sampling.md](docs/inference_sampling.md) for the full sampling guide including SDE parameters, self-conditioning details, and PDB output.

### Fixed-Length Generation

```julia
using LaProteina
using OnionTile  # GPU kernel overrides (loaded at runtime, not a package dep)
using Flux: gpu

# Load score network and decoder from SafeTensors
score_net = ScoreNetwork(
    n_layers=14, token_dim=768, n_heads=12,
    latent_dim=8, dim_cond=256, pair_dim=256,
    qk_ln=true, update_pair_repr=false, output_param=:v
)
load_score_network_weights_st!(score_net, "checkpoints/LD1_ucond_notri_512.safetensors")

decoder = DecoderTransformer(
    n_layers=12, token_dim=768, n_heads=12,
    latent_dim=8, dim_cond=128, pair_dim=256,
    qk_ln=true, update_pair_repr=false
)
load_decoder_weights_st!(decoder, "checkpoints/AE1_ucond_512.safetensors")

# Generate 100-residue protein
samples = sample_with_flowfusion(gpu(score_net), decoder, 100, 1;
    nsteps=400, self_cond=true, dev=gpu)

# Save to PDB
samples_to_pdb(samples, "output/"; prefix="generated", save_all_atom=true)
```

### All Model Variants

To run inference across all 7 model variants:

```bash
julia --project=../run -t 4 scripts/infer_all_variants.jl
```

### Variable-Length Generation (Branching Flows)

```julia
using LaProteina
using OnionTile
using BranchingFlows
using Flux: gpu
using JLD2

# Load BranchingScoreNetwork
base = ScoreNetwork(n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
                    dim_cond=256, latent_dim=8, output_param=:v,
                    qk_ln=true, update_pair_repr=false, cropped_flag=true)
model = BranchingScoreNetwork(base)

# Load fine-tuned weights (JLD2 from Julia training)
weights = load("branching_full.jld2")
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

See `scripts/sample_branching_full_OU.jl` for a complete sampling script with comparison modes and OUBridgeExpVar processes.

## Training

See [docs/training_guide.md](docs/training_guide.md) for full training configuration, monitoring, and hyperparameter reference. See [docs/data_pipeline.md](docs/data_pipeline.md) for data loading details.

Training uses a two-stage approach for efficiency:

### Stage 1: Precompute VAE Encoder Outputs

Since the VAE encoder is frozen during ScoreNetwork training, we precompute encoder outputs once:

```bash
julia --project=../run scripts/precompute_all_training_data.jl
```

### Stage 2: Flow Matching Training

```bash
julia --project=../run -t 4 scripts/train_branching_full_OU.jl
```

## Dependencies

### Package Dependencies

| Package | Purpose |
|---------|---------|
| [Flowfusion.jl](https://github.com/MurrellGroup/Flowfusion.jl) | RDNFlow process, flow matching API |
| [BranchingFlows.jl](https://github.com/MurrellGroup/BranchingFlows.jl) | Variable-length generation |
| [ForwardBackward.jl](https://github.com/MurrellGroup/ForwardBackward.jl) | State representations (OUBridgeExpVar) |
| [Onion.jl](https://github.com/MurrellGroup/Onion.jl) | Transformer architecture, GPU dispatch hooks |
| Flux.jl | Neural networks |
| CUDA.jl | GPU support |
| SafeTensors.jl | Loading pretrained weights |
| JLD2.jl | Saving/loading Julia-trained checkpoints |

### Runtime Dependencies (not in Project.toml)

| Package | Purpose |
|---------|---------|
| [OnionTile.jl](https://github.com/MurrellGroup/OnionTile.jl) | cuTile GPU kernels (flash attention, fused layernorm) |

OnionTile is loaded at runtime via `using OnionTile` in scripts. It activates CuArray method overrides for Onion's dispatch hooks. The run environment (`run/Project.toml`) includes OnionTile, but LaProteina's package Project.toml intentionally does not.

## File Structure

```
LaProteina.jl/
├── src/
│   ├── LaProteina.jl              # Main module
│   ├── constants.jl               # Amino acid and atom constants (from OpenFold)
│   ├── utils.jl                   # Tensor utilities, PyTorchLayerNorm
│   ├── inference.jl               # Time schedules, PDB saving utilities
│   ├── flowfusion_sampling.jl     # Flowfusion sampling API integration
│   ├── weights_safetensors.jl     # SafeTensors weight loading
│   ├── features/
│   │   ├── feature_factory.jl     # Feature extraction (29 feature types)
│   │   ├── time_embedding.jl      # Sinusoidal and index embeddings
│   │   ├── pair_features.jl       # Distance and separation features
│   │   └── geometry.jl            # Dihedral and bond angle calculations
│   ├── nn/
│   │   ├── adaptive_ln.jl         # ProteINAAdaLN, AdaptiveOutputScale
│   │   ├── pair_bias_attention.jl # AF3-style attention with pair bias
│   │   ├── transition.jl          # SwiGLU transition blocks
│   │   ├── triangular_update.jl   # Triangle multiplicative pair updates (LD2)
│   │   ├── transformer_block.jl   # Transformer blocks with optional pair update
│   │   ├── encoder.jl             # VAE encoder
│   │   ├── encoder_efficient.jl   # GPU-optimized encoder for precomputation
│   │   ├── score_network.jl       # ScoreNetwork (main model, all variants)
│   │   ├── score_network_efficient.jl  # GPU-native ScoreNetwork forward
│   │   └── decoder.jl             # VAE decoder
│   ├── gpu/
│   │   ├── gpu.jl                 # GPU mode selection and Onion hook imports
│   │   ├── layers.jl              # CuArray method overrides, fused GPU ops
│   │   ├── checkpointing.jl       # Gradient checkpointing
│   │   ├── utils_nocutile.jl      # Utilities for no-cuTile fallback
│   │   └── stubs.jl               # Stubs when GPU unavailable
│   ├── motif/
│   │   ├── contig_parser.jl       # Parse motif/scaffold segment specs
│   │   ├── motif_extraction.jl    # Extract motif features from PDB
│   │   └── motif_batch.jl         # Batch preparation for motif conditioning
│   ├── training/
│   │   └── precompute_encoder.jl  # Precomputed training data utilities
│   ├── data/
│   │   └── pdb_loading.jl         # PDB/mmCIF file I/O via BioStructures
│   └── branching/
│       ├── branching_score_network.jl  # BranchingScoreNetwork (split + del heads)
│       ├── branching_inference.jl      # Generation with branching flows
│       ├── branching_training.jl       # Training utilities and loss
│       └── branching_states.jl         # State conversion utilities
├── scripts/
│   ├── infer_all_variants.jl           # Inference for all LD1-LD7 variants
│   ├── sample_branching_full_OU.jl     # Variable-length sampling (OUBridgeExpVar)
│   ├── train_branching_full_OU.jl      # Branching flows training
│   ├── integration_test.jl             # Comprehensive integration tests
│   ├── precompute_all_training_data.jl # Dataset precomputation
│   ├── convert_shards_to_namedtuples.jl # Data format conversion
│   ├── convert_weights_to_safetensors.py # Convert .ckpt to SafeTensors
│   ├── extract_weights.py              # Extract weights from PyTorch .ckpt
│   └── extract_encoder_weights.py      # Extract encoder weights from .ckpt
└── docs/
    ├── model_variants.md          # LD1-LD7 and AE1-AE3 variant guide
    ├── model_architecture.md      # Neural network architecture deep dive
    ├── feature_system.md          # Feature extraction pipeline (29 features)
    ├── gpu_optimizations.md       # GPU dispatch, Onion hooks, modes
    ├── training_guide.md          # Training configuration and monitoring
    ├── data_pipeline.md           # Data loading, precomputation, sharding
    ├── inference_sampling.md      # Generation and sampling guide
    ├── branching_flows.md         # Branching Flows integration guide
    └── branching_flows_conversion.md  # State conversion reference
```

## Architecture Details

### Tensor Convention

Julia uses column-major order, so tensors are transposed from Python:
- Python: `[Batch, Length, Dim]`
- Julia: `[Dim, Length, Batch]`

### GPU Acceleration

GPU dispatch routes through [Onion.jl](https://github.com/MurrellGroup/Onion.jl) dispatch hooks, which are overridden at runtime by [OnionTile.jl](https://github.com/MurrellGroup/OnionTile.jl) with cuTile kernels (flash attention, fused LayerNorm). See [docs/gpu_optimizations.md](docs/gpu_optimizations.md) for details.

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

If you use this code, please cite the original La-Proteina paper:

```bibtex
@article{geffner2025laproteina,
  title={La-Proteina: Atomistic Protein Generation via Partially Latent Flow Matching},
  author={Geffner, Tomas and Didi, Kieran and Cao, Zhonglin and Reidenbach, Danny and Zhang, Zuobai and Dallago, Christian and Kucukbenli, Emine and Kreis, Karsten and Vahdat, Arash},
  journal={arXiv preprint arXiv:2507.09466},
  year={2025}
}
```

## License

This Julia implementation follows the same license as the original [La-Proteina repository](https://github.com/NVIDIA-Digital-Bio/la-proteina). Model weights are licensed under the [NVIDIA Open Model License](https://github.com/NVIDIA-Digital-Bio/la-proteina/blob/main/LICENSE/license_weights.txt).
