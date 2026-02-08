# Training Guide

Complete reference for training LaProteina's BranchingScoreNetwork.

## Prerequisites

- NVIDIA GPU with >= 40 GB VRAM (A100 recommended)
- Julia with 8+ threads (`julia -t 8`)
- cuTile mode enabled (default; do NOT set `LAPROTEINA_NOCUTILE` or `LAPROTEINA_NO_OVERRIDES`)
- Precomputed encoder shards (see Data Preparation below)
- Pretrained weights: `score_network.npz` in `weights/`

## Data Preparation

### Step 1: Obtain AlphaFold DB Structures

Training data comes from the AlphaFold Protein Structure Database. Place mmCIF files in:
```
~/shared_data/afdb_laproteina/raw/
├── AF-A0A000-F1-model_v4.cif
├── AF-A0A001-F1-model_v4.cif
└── ... (~289k files for full dataset)
```

### Step 2: Precompute VAE Encoder Outputs

Since the VAE encoder is frozen during ScoreNetwork training, we precompute and cache its outputs:

```bash
julia scripts/precompute_all_training_data.jl
```

**Configuration** (in script):
- `afdb_dir`: `~/shared_data/afdb_laproteina/raw`
- `output_dir`: `~/shared_data/afdb_laproteina/precomputed_shards`
- `n_shards`: 10
- `min_length`: 30 residues
- `max_length`: 256 residues

**What it does:**
1. Loads each protein structure from mmCIF
2. Filters by length (30-256 residues, ~80% pass)
3. Extracts CA coordinates and centers them (zero center of mass)
4. Runs the frozen VAE encoder to get `z_mean` and `z_log_scale`
5. Saves as `PrecomputedProteinNT` NamedTuples to sharded JLD2 files

**Output:**
```
~/shared_data/afdb_laproteina/precomputed_shards/
├── train_shard_01.jld2  (~23k proteins, ~268 MB)
├── train_shard_02.jld2
└── train_shard_10.jld2
```

**Resume support:** Set `START_SHARD=N` environment variable to resume from shard N.

**Processing time:** ~70 minutes per shard on A100 GPU (~11.5 hours total).

### Precomputed Data Format

```julia
const PrecomputedProteinNT = NamedTuple{
    (:ca_coords, :z_mean, :z_log_scale, :mask),
    Tuple{Matrix{Float32}, Matrix{Float32}, Matrix{Float32}, Vector{Float32}}
}
```

| Field | Shape | Description |
|-------|-------|-------------|
| `ca_coords` | [3, L] | Zero-centered CA coordinates in nm |
| `z_mean` | [8, L] | VAE encoder mean output |
| `z_log_scale` | [8, L] | VAE encoder log scale output |
| `mask` | [L] | Residue validity (1.0 = valid) |

During training, latents are sampled: `z = z_mean + randn() * exp(z_log_scale)`.

## Training Script

The main training script is `scripts/train_branching_full.jl`:

```bash
julia -t 8 scripts/train_branching_full.jl
```

### Current Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 8 | Proteins per batch |
| Total iterations | 200,000 | ~2-3 days on A100 |
| Warmdown start | Batch 180,000 | (n_batches - warmdown_batches) |
| Warmdown duration | 20,000 batches | Linear decay to 1e-9 |
| Optimizer | Muon | Momentum-based optimizer |
| LR schedule | burnin_learning_schedule | start=1e-5, target=2.5e-4, growth=1.05, decay=0.99995 |
| LR update frequency | Every 10 batches | |
| CA loss scale | 2.0 | Upweights CA coordinate loss |
| LL loss scale | 0.1 | Downweights latent loss |
| Softclamp threshold | 3.5 | Per-component soft clamp |
| Softclamp hardcap | 5.0 | Per-component hard cap |
| Total loss hardcap | 20.0 | Maximum total loss |
| X0 mean length | 100 | Poisson mean for initial noise length |
| deletion_pad | 1.1 | 10% extra deletion positions |
| Tile padding | 64 | Sequences padded to multiples of 64 |

### Optimizer: Muon

The training uses Muon (from CannotWaitForTheseOptimisers.jl), a momentum-based optimizer. The learning rate follows a burnin schedule:

```julia
sched = burnin_learning_schedule(1e-5, 2.5e-4, 1.05, 0.99995)
```

- **Warmup phase**: LR multiplied by `growth=1.05` each step until reaching `target=2.5e-4`
- **Steady phase**: LR multiplied by `decay=0.99995` each step
- **Warmdown phase** (last 20k batches): Linear decay from current LR to `1e-9`

The LR is updated every 10 batches via `Flux.adjust!(opt_state, next_rate(sched))`.

### Process Setup

```julia
P_ca = RDNFlow(3;
    zero_com=false, schedule=:log, schedule_param=2.0f0,
    sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0,
    sc_scale_noise=0.1f0, sc_scale_score=1.0f0, t_lim_ode=0.98f0)

P_ll = RDNFlow(8;
    zero_com=false, schedule=:power, schedule_param=2.0f0,
    sde_gt_mode=:tan, sde_gt_param=1.0f0,
    sc_scale_noise=0.1f0, sc_scale_score=1.0f0, t_lim_ode=0.98f0)

P = CoalescentFlow((P_ca, P_ll), Beta(1.0, 2.0))
```

Note: `zero_com=false` for branching (unlike the base model which uses `zero_com=true`). This avoids issues with single-position bridges.

## Loss Functions

Four loss components, each independently softclamped:

### 1. CA Loss (Flow Matching MSE)

```julia
# v-parameterization: predict velocity, convert to x1
x1_ca_pred = xt_ca + (1 - t_ca) * v_ca_pred
t_scale_ca = 1 / max(1 - t_ca, 0.1)^2
ca_loss = mean((x1_ca_pred - x1_ca_target)^2 * t_scale_ca * mask)
```

Scaled by `ca_loss_scale = 2.0`.

### 2. Latent Loss (Flow Matching MSE)

Same as CA but with latent schedule: `t_scale_ll = 1 / max(1 - t_ll, 0.1)^2`.

Scaled by `ll_loss_scale = 0.1`.

### 3. Split Loss (Bregman Poisson)

```julia
indel_scale = scalefloss(P, t_raw, 1, 0.2f0)  # 1/(1.2 - t) scaling
split_loss = floss(P, split_pred, split_target, combined_mask, indel_scale)
```

The split target is the expected number of future splits from the coalescent tree. Loss: `exp(pred) - target * pred` (Bregman Poisson divergence).

### 4. Deletion Loss (Logistic BCE)

```julia
del_loss = floss(P.deletion_policy, del_pred, del_target, combined_mask, indel_scale)
```

The deletion target is binary (1 = deleted). Loss: standard binary cross-entropy on logits.

### Softclamp

All four loss components pass through a softclamp:

```julia
function softclamp(loss; threshold=3.5f0, hardcap=5.0f0)
    if loss > threshold
        return min(threshold + log(loss - threshold + 1), hardcap)
    else
        return loss
    end
end
```

Non-finite losses (NaN/Inf) are set to `0.0f0`. Total loss is hard-capped at `20.0f0`.

## Self-Conditioning During Training

Self-conditioning passes are sampled from `Poisson(1)`:

```julia
n_sc_passes = rand(Poisson(1))  # 0, 1, 2, ... (mean=1)
```

For each SC pass:
1. Run the model forward (no gradients) to get velocity predictions
2. Convert to x1 predictions: `x1_sc = xt + (1 - t) * v_pred`
3. Update raw features in-place on GPU via `update_sc_raw_features!()`
4. The next forward pass sees the updated SC features

The in-place GPU update avoids the expensive CPU round-trip that would otherwise be needed to re-extract all features.

## Data Loading

### Batch Construction

```julia
function prepare_training_batch(indices, proteins, P, X0_sampler, base_model; pad_to=64)
    # 1. Convert proteins to BranchingStates
    X1s = [protein_to_X1_simple(proteins[i]) for i in indices]

    # 2. Sample bridged states via BranchingFlows
    bat = branching_bridge(P, X0_sampler, X1s, t_dist;
        coalescence_factor=1.0, use_branching_time_prob=0.5,
        length_mins=Poisson(100), deletion_pad=1.1)

    # 3. Extract raw features
    raw_features = extract_raw_features(base_model, cpu_batch)

    # 4. Pad to tile size
    # All tensors padded to next multiple of 64
end
```

### Parallel Loading

CPU-intensive batch preparation runs in a background thread:

```julia
dataset = BatchDataset(batch_indices, proteins, P, base_model_cpu, TILE_SIZE)
dataloader = Flux.DataLoader(dataset; batchsize=-1, parallel=true)

for (batch_idx, bd_cpu) in enumerate(dataloader)
    bd = dev(bd_cpu)  # Transfer to GPU
    # ... training step ...
end
```

Julia must be started with multiple threads (`julia -t 8`) for parallel loading to work.

## Monitoring

### Log Format

Training writes to a log file with columns:
```
batch, shard, lr, total_loss, ca_scaled, ll_scaled, split, del, t_min, t_max, time_ms, seq_len
```

### What to Watch For

| Metric | Healthy Range | Problem |
|--------|--------------|---------|
| CA loss (scaled) | 0.5-3.0 | > 4.0 consistently = learning rate too high |
| LL loss (scaled) | 0.05-0.5 | Very high = latent schedule issue |
| Split loss | 0.1-1.0 | > 2.0 = branching not learning |
| Del loss | 0.1-0.5 | > 1.0 = deletion prediction unstable |
| Total loss | 1.0-5.0 | > 10.0 = something wrong |
| Per-batch time | ~1500 ms | >> 2000 ms = possible GPU memory thrashing |
| NaN in loss | Never | = gradient explosion, check cuTile backward |

## Checkpointing

- Checkpoints saved every `sample_every=1000` batches
- Format: `checkpoint_batch000001.jld2`, `checkpoint_batch001000.jld2`, etc.
- Final model saved as `branching_full_final.jld2`
- Final model also copied to `weights/branching_full.jld2`

Each checkpoint contains all model parameters (base + indel heads).

## Two-Stage Training Strategy

For training branching heads from a pretrained base:

### Stage 1: Freeze Base, Train Indel Heads

```julia
freeze_base!(model)
# Only indel_time_proj, split_head, del_head receive gradients
# Shorter training (e.g., 5k-10k iterations)
# Saves to: branching_indel_stage1.jld2
```

### Stage 2: Full Fine-Tuning

```julia
# Unfreeze all parameters
# Full training with all loss components
# This is what train_branching_full.jl does
```

Stage 1 prevents the randomly-initialized indel heads from destabilizing the pretrained flow matching. The 0.05x weight initialization also helps stability.

## Key Files

| File | Purpose |
|------|---------|
| `scripts/train_branching_full.jl` | Main training script |
| `scripts/precompute_all_training_data.jl` | Data precomputation |
| `scripts/plot_training.jl` | Training loss visualization |
| `src/branching/branching_training.jl` | softclamp, loss utilities |
| `src/branching/branching_states.jl` | protein_to_branching_state, X0_sampler_laproteina |
| `src/training/precompute_encoder.jl` | PrecomputedProteinNT, sharding |
| `src/nn/score_network.jl` | extract_raw_features, update_sc_raw_features! |
