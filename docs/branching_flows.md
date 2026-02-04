# Branching Flows Integration for LaProteina

This document describes the integration of Branching Flows into LaProteina for variable-length protein generation.

## Overview

Branching Flows extends the standard flow matching framework to handle variable-length sequences. Instead of generating a fixed-length protein, the model learns to predict:
1. **State predictions** (CA coordinates + local latents) - same as standard flow matching
2. **Split predictions** - expected number of future splits per position
3. **Deletion predictions** - probability of deletion per position

During generation, residues can split (creating new residues) or be deleted, allowing the model to generate proteins of varying lengths from a single initial state.

## Architecture

### BranchingScoreNetwork

Wraps the base `ScoreNetwork` with additional heads:

```julia
struct BranchingScoreNetwork
    base::ScoreNetwork           # Pretrained flow matching model
    indel_time_proj::Dense       # Time conditioning for indel heads
    split_head::Chain            # Predicts log expected splits [L, B]
    del_head::Chain              # Predicts deletion logits [L, B]
end
```

The split/del heads are initialized with scaled-down weights (0.05x) for stable training.

### Key Files

- `src/branching/branching_score_network.jl` - BranchingScoreNetwork definition
- `src/branching/branching_inference.jl` - Generation pipeline
- `src/branching/branching_training.jl` - Training utilities
- `scripts/train_branching_parallel.jl` - Training script with parallel data loading

## Training

### Two-Stage Training

1. **Stage 1**: Freeze base ScoreNetwork, train only split/del heads
2. **Stage 2**: Unfreeze and fine-tune entire model with lower learning rate

### Data Preparation with `branching_bridge`

Training data is prepared using `branching_bridge` from BranchingFlows.jl:

```julia
batch = branching_bridge(
    P,                              # CoalescentFlow process
    X0_sampler,                     # Samples initial noise states
    X1s,                            # Target protein states (BranchingState)
    t_dist;                         # Time distribution (Uniform(0, 1))
    coalescence_factor = 1.0,       # Controls coalescence rate
    use_branching_time_prob = 0.5,  # Probability of sampling at branch times
    length_mins = Poisson(100),     # X0 length distribution
    deletion_pad = 1.1              # Padding factor for length changes
)
```

Parameters should match BranchChain.jl for consistency.

### Losses

Three loss components:

#### 1. State Loss (CA + Latents)
Standard flow matching MSE loss with time scaling:

```julia
t_scale = 1f0 ./ max.(1f0 .- t, 0.1f0).^2  # eps = 0.1
```

**Important**: Use eps=0.1, not smaller values like 1e-5 which cause loss spikes near t=1.

#### 2. Split Loss (Bregman Poisson)
Uses `floss` from Flowfusion with CoalescentFlow:

```julia
indel_scale = scalefloss(P, t, 1, 0.2f0)  # eps = 0.2
split_loss = floss(P, split_pred, split_target, mask, indel_scale)
```

#### 3. Deletion Loss (BCE)
Uses `floss` from Flowfusion with Deletion policy:

```julia
del_loss = floss(P.deletion_policy, del_pred, del_target, mask, indel_scale)
```

**Important**: Always use `floss` from Flowfusion rather than rolling your own loss functions - they handle the Bregman divergence and numerical stability correctly.

### Loss Weighting

In Stage 1 (frozen base), weight state loss down to focus on learning split/del:
```julia
total = ca_loss * 0.1 + ll_loss * 0.1 + split_loss + del_loss
```

In Stage 2 (full fine-tune), use equal weighting:
```julia
total = ca_loss + ll_loss + split_loss + del_loss
```

## Parallel Data Loading

### The Problem

`branching_bridge` and `extract_raw_features` are CPU-intensive. Running them synchronously in the training loop leaves the GPU idle.

### The Solution

Following the pattern from BranchChain.jl, we use Flux's DataLoader with parallel workers:

```julia
# BatchDataset prepares full batches in getindex
struct BatchDataset
    batch_indices::Vector{Vector{Int}}
    proteins::Vector
    process::CoalescentFlow
    base_model::ScoreNetwork
end

Base.getindex(x::BatchDataset, i) = prepare_training_batch(
    x.batch_indices[i], x.proteins, x.process, X0_sampler, x.base_model
)

# Create dataloader with parallel=true, batchsize=-1
dataset = BatchDataset(batch_indices, proteins, P, cpu(model.base))
dataloader = Flux.DataLoader(dataset; batchsize=-1, parallel=true)
```

Key points:
- `batchsize=-1`: Each `getindex` returns a complete batch, no further batching
- `parallel=true`: Prepare next batch in background while GPU trains on current
- Run Julia with multiple threads: `julia -t 8 script.jl`

### GPU Transfer

Transfer the entire batch struct to GPU at once in the training loop:

```julia
for (batch_idx, bd_cpu) in enumerate(dataloader)
    bd = dev(bd_cpu)  # Single transfer of entire NamedTuple
    # ... training ...
end
```

**Don't** call `dev()` on each field individually - transfer the whole struct.

### Memory Management

Required override for proper GPU memory cleanup with DataLoader:

```julia
Flux.MLDataDevices.Internal.unsafe_free!(x) = (
    Flux.fmapstructure(Flux.MLDataDevices.Internal.unsafe_free_internal!, x);
    return nothing
)
```

### Performance

With 8 threads and batch size 4:
- Without parallel loading: ~1026 ms/batch
- With parallel loading: ~896 ms/batch (~13% speedup)

GPU utilization improves significantly as batch preparation happens concurrently.

## Generation

Use `generate_with_branching` for inference:

```julia
result = generate_with_branching(
    model,
    initial_length;           # Starting sequence length
    nsteps = 400,             # Integration steps
    latent_dim = 8,
    self_cond = true,         # Self-conditioning
    schedule = :log,          # Time schedule (:log or :linear)
    verbose = true
)

# Returns:
# - result.ca_coords: [3, L_final] CA coordinates
# - result.latents: [latent_dim, L_final] local latents
# - result.final_length: Final sequence length
# - result.trajectory_lengths: Length at each step
```

## Dependencies

- `BranchingFlows.jl` - Branching Flows framework
- `Flowfusion.jl` - Flow matching utilities, `floss`, `scalefloss`
- `ForwardBackward.jl` - State representations

## Common Issues

### Loss Spikes
- Check eps values: state loss should use eps=0.1, indel losses use eps=0.2
- Use `floss` from Flowfusion, not custom loss implementations

### OOM Errors
- Reduce batch size
- Don't wrap DataLoader with `dev()` (keeps two batches on GPU)
- Transfer batches inside loop with `bd = dev(bd_cpu)`

### Slow Training
- Ensure Julia is running with multiple threads (`-t 8`)
- Verify `parallel=true` in DataLoader
- Check that `branching_bridge` and `extract_raw_features` happen in `prepare_training_batch`
