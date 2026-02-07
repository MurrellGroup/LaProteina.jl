# Branching Flows for Variable-Length Protein Generation

This document describes how Branching Flows is integrated into LaProteina for variable-length protein generation. For the general theory, see the [BranchingFlows.jl](https://github.com/MurrellGroup/BranchingFlows.jl) package.

## Overview

Standard flow matching generates fixed-length sequences: you specify L=100 and get exactly 100 residues. Branching Flows extends this to variable-length generation by allowing residues to **split** (creating new positions) or be **deleted** during the generative flow. The model learns when and where these events should occur.

### What the Model Predicts

At each timestep, the BranchingScoreNetwork predicts four things:

1. **CA velocity** (`v_ca`): velocity field for alpha-carbon coordinates (same as base model)
2. **Latent velocity** (`v_ll`): velocity field for local latent vectors (same as base model)
3. **Split logits** (`split`): log expected number of future splits per position (Poisson rate)
4. **Deletion logits** (`del`): deletion probability per position (logistic)

The base model handles (1) and (2); branching adds (3) and (4) via two small MLP heads.

### How Generation Works

Starting from noise at t=0, the CoalescentFlow process steps forward in time:

1. The model predicts X1 targets and split/del logits
2. The RDNFlow processes interpolate CA coords and latents toward predicted X1
3. The CoalescentFlow samples splits from the predicted Poisson rates
4. The CoalescentFlow samples deletions from the predicted deletion hazard
5. New positions inherit their parent's state; deleted positions are removed
6. The sequence length changes dynamically

By t=1, the flow has converged to a valid protein of variable length.

## Architecture

### BranchingScoreNetwork

Wraps the base ScoreNetwork with additional heads:

```julia
struct BranchingScoreNetwork
    base::ScoreNetwork           # Pretrained 14-layer transformer (160M params)
    indel_time_proj::Dense       # Time conditioning for indel heads (dim_cond -> token_dim)
    split_head::Chain            # Predicts log expected splits [L, B]
    del_head::Chain              # Predicts deletion logits [L, B]
end
```

The split/del heads share the same transformer features as the base model. After the final transformer layer, the token embedding is combined with a time-conditioned signal and fed through each head:

```
indel_cond = indel_time_proj(time_embedding)    # [token_dim, 1, B]
split_logits = split_head(seqs .+ indel_cond)   # [1, L, B] -> squeeze -> [L, B]
del_logits = del_head(seqs .+ indel_cond)       # [1, L, B] -> squeeze -> [L, B]
```

Heads are initialized with 0.05x weight scaling for stable training from a pretrained base.

### Key Files

| File | Purpose |
|------|---------|
| `src/branching/branching_score_network.jl` | BranchingScoreNetwork struct and forward pass |
| `src/branching/branching_inference.jl` | BranchingScoreNetworkWrapper, create_branching_processes, generate_with_branching |
| `src/branching/branching_training.jl` | softclamp, loss utilities |
| `src/branching/branching_states.jl` | proteins_to_X1_states, X0_sampler_laproteina |
| `scripts/train_branching_full.jl` | Training script (the one that works) |
| `test/test_branching_full_sampling.jl` | Sampling from fine-tuned model with cosine steps |

### Process Setup

The branching flow wraps the base RDNFlow processes in a CoalescentFlow:

```julia
# Base processes (same as standard la-proteina)
P_ca = RDNFlow(3; zero_com=false, schedule=:log, schedule_param=2.0,
               sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0,
               sc_scale_noise=0.1, sc_scale_score=1.0, t_lim_ode=0.98)

P_ll = RDNFlow(8; zero_com=false, schedule=:power, schedule_param=2.0,
               sde_gt_mode=:tan, sde_gt_param=1.0,
               sc_scale_noise=0.1, sc_scale_score=1.0, t_lim_ode=0.98)

# Index tracking process (NullProcess — doesn't evolve, just tracks position origins)
P_idx = NullProcess()

# Wrap in CoalescentFlow with Beta(1, 2) branch time distribution
P = CoalescentFlow((P_ca, P_ll, P_idx), Beta(1.0, 2.0))
```

The `create_branching_processes()` function in `branching_inference.jl` creates this setup with all defaults.

### Branch Time Distribution: Beta(1, 2)

The branch time distribution controls when splitting/deletion events are likely to occur. We use `Beta(1, 2)` which:
- Has mode at 0 (events more likely early in the flow)
- Has mean at 1/3
- Matches the convention used in [BranchChain.jl](https://github.com/MurrellGroup/BranchChain.jl)

This means most branching events happen early, when the state is still noisy and structural changes are less disruptive.

### Self-Conditioning with Variable Length

Self-conditioning in branching flows is more complex than in fixed-length generation because the sequence length changes at each step. The `BranchingScoreNetworkWrapper` handles this using an **index tracking state**:

1. A `NullProcess` (3rd component) carries integer indices `[1, 2, 3, ..., L]`
2. When position `i` splits, both children get index `i`
3. When position `j` is deleted, its index disappears
4. At the next step, the wrapper uses these indices to expand/contract the previous self-conditioning predictions to match the new length

```
# Before: sc_ca has shape [3, L_old, B], indices were [1, 2, 3, 4]
# After split at position 2: indices become [1, 2, 2, 3, 4], L_new = 5
# Expansion: sc_ca_new[:, i, :] = sc_ca_old[:, indices[i], :]
# After deletion at position 3: indices become [1, 2, 4], L_new = 3
```

After using the indices for expansion, the wrapper resets them to `[1, 2, ..., L_new]` for the next step.

### Schedule Transform Handling

The model was trained with per-modality schedule transforms: the CA modality sees `tau_ca = log_schedule(s)` and the latent modality sees `tau_ll = power_schedule(s)`, where `s` is the raw uniform time. The wrapper applies these transforms before conditioning the model:

```julia
# Raw uniform progress s ∈ [0, 1]
t_ca = schedule_transform(P_ca, s)   # Log schedule: fast early interpolation
t_ll = schedule_transform(P_ll, s)   # Power schedule: slow early, fast late

# Model receives different times per modality
batch[:t] = Dict(:bb_ca => [t_ca], :local_latents => [t_ll])
```

This is critical — passing raw `s` to the model produces broken samples because the model was trained on schedule-transformed times.

## Training

### Training Script

The working training script is `scripts/train_branching_full.jl`. Run with:

```bash
julia -t 8 scripts/train_branching_full.jl
```

Key hyperparameters (current working values):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 2.5e-4 | With Adam optimizer |
| Batch size | 4 | Proteins per batch |
| Total iterations | 20,000 | ~10 hours on A100 |
| Warmdown | Last 2,000 iters | Linear decay to 1e-9 |
| CA loss scale | 2.0 | Upweights CA loss |
| LL loss scale | 0.1 | Downweights latent loss |
| Softclamp threshold | 3.5 | Per-component loss clamp |
| Softclamp hardcap | 5.0 | Maximum per-component loss |
| Total loss hardcap | 20.0 | Maximum total loss |
| deletion_pad | 1.1 | 10% extra deletion positions |
| X0 mean length | 100 | Mean of Poisson for initial noise length |

### Data Preparation

Training uses `branching_bridge` from BranchingFlows.jl to create training samples:

```julia
bat = branching_bridge(P, X0_sampler, X1s, t_dist;
    coalescence_factor = 1.0,
    use_branching_time_prob = 0.5,
    merger = BranchingFlows.canonical_anchor_merge,
    length_mins = Poisson(100),
    deletion_pad = 1.1
)
```

This function:
1. Takes target protein structures (`X1s`) as BranchingStates
2. Samples coalescent forest structures connecting X0 noise to X1 targets
3. Samples a random time `t` and computes the bridged state `Xt`
4. Returns `Xt`, X1 targets, split targets, and deletion targets

### Loss Functions

Four loss components, each independently softclamped:

#### 1. CA Loss (flow matching MSE)
```julia
# v-parameterization: predict velocity, convert to x1
x1_ca_pred = xt_ca + (1 - t_ca) * v_ca_pred
ca_loss = scaledmaskedmean((x1_ca_pred - x1_ca_target)^2, 1/(1-t_ca)^2, mask)
```

#### 2. Latent Loss (flow matching MSE)
Same as CA but with latent schedule: `1/(1-t_ll)^2` scaling.

#### 3. Split Loss (Bregman Poisson)
Uses `floss` from Flowfusion with CoalescentFlow:
```julia
indel_scale = scalefloss(P, t_raw, 1, 0.2f0)  # 1/(1.2 - t) scaling
split_loss = floss(P, split_pred, split_target, combined_mask, indel_scale)
```
The split target is the expected number of future splits (from the coalescent tree). The loss is Bregman Poisson divergence: `exp(pred) - target * pred`.

#### 4. Deletion Loss (logistic BCE)
Uses `floss` from Flowfusion with the deletion policy:
```julia
del_loss = floss(P.deletion_policy, del_pred, del_target, combined_mask, indel_scale)
```
The deletion target is binary (1 = this position will be deleted). The loss is standard binary cross-entropy on logits.

#### Softclamp
All loss components pass through a softclamp that transitions to log growth above a threshold:
```julia
function softclamp(loss; threshold=3.5, hardcap=5.0)
    if loss > threshold
        return min(threshold + log(loss - threshold + 1), hardcap)
    else
        return loss
    end
end
```

### Warmdown Schedule

The learning rate decays linearly over the last 2000 iterations. Because the LR is only updated every 10 batches, the warmdown schedule must account for this:

```julia
# warmdown_batches = 2000, but next_rate() called every 10 batches
# So create schedule with 200 steps (not 2000)
sched = linear_decay_schedule(current_lr, 1e-9, warmdown_batches / 10)
```

### Parallel Data Loading

CPU-intensive work (`branching_bridge` + `extract_raw_features`) runs in a background thread while the GPU trains:

```julia
dataset = BatchDataset(batch_indices, proteins, P, cpu(model.base))
dataloader = Flux.DataLoader(dataset; batchsize=-1, parallel=true)

for (batch_idx, bd_cpu) in enumerate(dataloader)
    bd = dev(bd_cpu)  # Transfer entire batch to GPU
    # ... training step ...
end
```

Run Julia with multiple threads (`julia -t 8`) for this to work.

## Generation / Inference

### Quick Generation

```julia
result = generate_with_branching(model, 100;
    nsteps=400, latent_dim=8, self_cond=true, dev=gpu, verbose=true)
# Returns: (ca_coords, latents, final_length, trajectory_lengths)
```

### Cosine Time Steps (Better Quality)

For higher quality, use cosine-spaced time steps with more steps:

```julia
step_func(t) = Float32(1 - (cos(t * pi) + 1) / 2)
step_number = 500
steps = step_func.(0f0:Float32(1/step_number):1f0)
```

This gives denser steps near t=0 and t=1 where the flow changes most rapidly.

See `test/test_branching_full_sampling.jl` for a complete example.

### Sampling Results

With the fine-tuned model (20k iterations, Beta(1,2), ca_scale=2.0):
- Starting from Poisson(100) initial lengths
- Final lengths: 113-173 residues (healthy growth via splits)
- Mean CA-CA distances: 0.372-0.381 nm (excellent, expected ~0.38 nm)
- Generation time: ~2 minutes per sample (500 cosine steps on GPU)

### Full Pipeline

```julia
# 1. Generate CA coords + latents
result = generate_with_branching(model, initial_length; ...)

# 2. Decode to all-atom structure
dec_input = Dict(
    :z_latent => reshape(result.latents, 8, L, 1),
    :ca_coors => reshape(result.ca_coords, 3, L, 1),
    :mask => ones(Float32, L, 1)
)
dec_out = decoder(dec_input)

# 3. Save PDB
samples = Dict(
    :ca_coords => ..., :latents => ...,
    :all_atom_coords => dec_out[:coors],
    :aatype => dec_out[:aatype_max],
    :atom_mask => dec_out[:atom_mask],
    :mask => ...
)
samples_to_pdb(samples, output_dir; prefix="sample", save_all_atom=true)
```

## Lessons Learned

### Critical Parameter Choices

1. **Beta(1, 2) for branch times** — not Beta(2, 2). The asymmetric distribution pushes branching events earlier in the flow, matching the convention from BranchChain.jl. With Beta(2, 2), the model severely over-deleted.

2. **Softclamp at 3.5** — not 2.0. With threshold=2.0, the CA loss (scaled at 2x) was hitting the clamp too often, preventing the model from learning CA structure properly.

3. **CA loss scale = 2.0** — the raw CA loss is small relative to other components. Without upweighting, the model underprioritizes CA coordinate accuracy.

4. **Warmdown matters** — a proper linear warmdown from 2.5e-4 to 1e-9 over the last 2000 iterations significantly improves final sample quality.

5. **Per-modality schedule transforms** — the model must receive `tau_ca = log_schedule(s)` and `tau_ll = power_schedule(s)` for conditioning, not raw uniform `s`. The wrapper handles this, but getting it wrong produces broken samples.

6. **zero_com=false for branching** — CA coordinates use `zero_com=false` in the branching setting (unlike the base model which uses `zero_com=true`). This avoids issues with single-position bridges that can zero out coordinates.

### Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| Samples over-delete (L shrinks) | Wrong Beta distribution | Use Beta(1, 2) |
| Loss doesn't decrease at end | Warmdown not working | Account for LR update interval |
| CA-CA distances wrong (~0.5 nm) | Raw time conditioning | Apply schedule_transform per modality |
| Loss spikes | Softclamp too tight | Increase threshold to 3.5 |
| Training diverges | No loss clamping | Use softclamp + total hardcap |
| Self-conditioning breaks after splits | Length mismatch | Use index tracking state |

## Dependencies

| Package | Purpose |
|---------|---------|
| BranchingFlows.jl | CoalescentFlow, branching_bridge, BranchingState |
| Flowfusion.jl (rdn-flow) | RDNFlow, gen(), step(), floss(), scalefloss() |
| ForwardBackward.jl | ContinuousState, DiscreteState, tensor() |
| Distributions.jl | Beta, Poisson, Uniform for sampling |
