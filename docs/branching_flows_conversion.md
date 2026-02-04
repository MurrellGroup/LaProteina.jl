# Converting a Flowfusion Model to Branching Flows

This document summarizes the key differences between a standard Flowfusion-based model (like ChainStorm/LaProteina) and a Branching Flows model (like BranchChain), focusing on the minimal changes needed to add variable-length generation capabilities.

## Overview

Branching Flows extends standard flow matching to handle variable-length sequences. Instead of a fixed-size output, the model learns when to:
1. **Split** elements (create new positions)
2. **Delete** elements (remove positions)

This requires adding two new prediction heads to the model and changing how training bridges are constructed.

---

## Key Architectural Changes

### 1. Model Changes: Add Split and Delete Heads

**ChainStorm (standard Flowfusion):**
```julia
struct ChainStormV1{L}
    layers::L
end

function ChainStormV1(dim::Int = 384, depth::Int = 6, f_depth::Int = 6)
    layers = (;
        # ... standard layers ...
        AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias=false)),
    )
    return ChainStormV1(layers)
end

# Forward returns: (frames, aa_logits)
```

**BranchChain (Branching Flows):**
```julia
function BranchChainV1(dim::Int = 384, depth::Int = 6, f_depth::Int = 6)
    layers = (;
        # ... same standard layers ...
        AAdecoder = Chain(StarGLU(dim, 3dim), Dense(dim => 21, bias=false)),

        # NEW: Split/deletion prediction heads
        indelpre_t_encoding = Dense(dim => 3dim),
        count_decoder = StarGLU(Dense(3dim => 2dim), Dense(2dim => 1), ...),  # Predicts expected splits
        del_decoder = StarGLU(Dense(3dim => 2dim), Dense(2dim => 1), ...),    # Predicts deletion prob
    )
    return BranchChainV1(layers)
end

# Forward returns: (frames, aa_logits, split_logits, del_logits)
```

**Key differences:**
- `count_decoder`: Outputs 1 value per position = expected number of future splits (log-space)
- `del_decoder`: Outputs 1 value per position = deletion logit (probability position gets deleted)
- BranchChain uses concatenated intermediate representations from multiple layers, but for simplicity we'll use just the final embedding

### 2. LaProteina Equivalent Changes

For LaProteina's ScoreNetwork, we add simple heads off the final token embedding:
```julia
# In ScoreNetwork constructor, add:
indel_time_proj = Dense(dim_cond => token_dim),
split_head = Chain(Dense(token_dim => token_dim), swish, Dense(token_dim => 1)),
del_head = Chain(Dense(token_dim => token_dim), swish, Dense(token_dim => 1)),

# In forward, after transformer blocks:
# Use final token embedding directly (simpler than BranchChain's multi-layer concat)
indel_cond = indel_time_proj(t_embedding)  # [token_dim, 1, B]
split_logits = split_head(x .+ indel_cond)  # [1, L, B] -> [L, B]
del_logits = del_head(x .+ indel_cond)      # [1, L, B] -> [L, B]
```

---

## State Representation Changes

### Standard Flowfusion: Batched States

In standard Flowfusion, states have a batch dimension and fixed length within a batch:

```julia
# ChainStorm state construction
function compound_state(b)
    L, B = size(b.aas)
    cmask = b.aas .< 100  # valid mask
    X1locs = MaskedState(ContinuousState(b.locs), cmask, b.padmask)
    X1rots = MaskedState(ManifoldState(rotM, ...), cmask, b.padmask)
    X1aas = MaskedState(DiscreteState(21, ...), cmask, b.padmask)
    return (X1locs, X1rots, X1aas)
end

# Training sample
function training_sample(b)
    X0 = zero_state(b)  # Batched [3, 1, L, B] tensors
    X1 = compound_state(b)
    t = rand(Float32, 1, B)  # Same t for all positions
    Xt = bridge(P, X0, X1, t)  # Standard bridge
    return (; t, Xt, X1, ...)
end
```

### Branching Flows: Unbatched States with BranchingState

In Branching Flows, states are passed as **arrays of unbatched samples**, and `branching_bridge` handles batching internally because different samples may have different lengths at time t:

```julia
# BranchChain state construction - returns BranchingState (NOT batched)
function compoundstate(rec)
    L = length(rec.AAs)
    cmask = rand_mask(rec.chainids)

    X1locs = MaskedState(ContinuousState(rec.locs), cmask, cmask)  # No batch dim!
    X1rots = MaskedState(ManifoldState(rotM, eachslice(rec.rots, dims=3)), cmask, cmask)
    X1aas = MaskedState(DiscreteState(21, rec.AAs), cmask, cmask)
    index_state = MaskedState(DiscreteState(0, [1:L;]), cmask, cmask)  # For tracking

    # Wrap in BranchingState with masks
    X1 = BranchingState(
        (X1locs, X1rots, X1aas, index_state),
        rec.chainids,
        flowmask = cmask,      # Which positions evolve in the flow
        branchmask = cmask     # Which positions can branch/delete
    )
    return X1, breaks, pdb_id, chain_labels
end

# Training prep - pass array of unbatched states to branching_bridge
function training_prep(batch_indices, data, ...)
    sampled = compoundstate.(data[batch_indices])  # Array of (X1, breaks, ...) tuples
    X1s = [s[1] for s in sampled]  # Array of BranchingStates (not batched!)

    # branching_bridge does the batching internally
    bat = branching_bridge(P, X0sampler, X1s, t,
        coalescence_factor = 1.0,
        use_branching_time_prob = 0.5,
        merger = BranchingFlows.canonical_anchor_merge,
        length_mins = Poisson(mean_length),
        deletion_pad = deletion_pad,
        X1_modifier = X1_modifier,
    )
    return bat
end
```

**Key insight:** Because splits/deletions change sequence length, `branching_bridge` samples a random forest structure for each sample, which may result in different lengths at time t. It then batches these variable-length sequences with padding.

---

## Training Target Changes

### Standard Flowfusion: Just X1 Prediction

```julia
function losses(hatframes, aalogits, ts)
    # Location loss: predict X1 locations
    l_loc = floss(P[1], hatloc, ts.X1[1], scalefloss(...))

    # Rotation loss: predict rotation tangent to X1
    l_rot = floss(P[2], hatrot, ts.rotξ, scalefloss(...))

    # AA loss: predict X1 amino acids
    l_aas = floss(P[3], hataas, ts.X1[3], scalefloss(...))

    return l_loc, l_rot, l_aas
end
```

### Branching Flows: X1 + Splits + Deletions

`branching_bridge` returns additional training targets:

```julia
bat = branching_bridge(P, X0sampler, X1s, t, ...)

# bat contains:
bat.Xt              # BranchingState at time t (may have different length than X1!)
bat.t               # Time values [B]
bat.X1_locs_target  # Target locations (batched, padded)
bat.X1aas_target    # Target AAs
bat.rotξ_target     # Rotation guide
bat.splits_target   # Expected split counts per position [L, B]
bat.del             # Deletion indicators [L, B] (1 = deleted, 0 = survives)

# Loss function now has 5 terms:
function losses(P, X1hat, ts)
    hat_frames, hat_aas, hat_splits, hat_del = X1hat

    # Standard flow matching losses
    l_loc = floss(P.P[1], hat_loc, ts.X1_locs_target, ...)
    l_rot = floss(P.P[2], hat_rot, ts.rotξ_target, ...)
    l_aas = floss(P.P[3], hat_aas, onehot(ts.X1aas_target), ...)

    # NEW: Split/deletion losses
    splits_loss = floss(P, hat_splits, ts.splits_target,
                        ts.Xt.padmask .* ts.Xt.branchmask, ...)
    del_loss = floss(P.deletion_policy, hat_del, ts.del,
                     ts.Xt.padmask .* ts.Xt.branchmask, ...)

    return l_loc, l_rot, l_aas, splits_loss, del_loss
end
```

**Split target:** Expected number of descendants (coalescent count) - uses Bregman Poisson loss
**Deletion target:** Binary indicator - uses logistic BCE loss

---

## Process Changes: CoalescentFlow Wrapper

### Standard Flowfusion Process

```julia
const P = (
    FProcess(BrownianMotion(0.2f0), schedule_f),     # CA coordinates
    FProcess(ManifoldProcess(0.2f0), schedule_f),    # Rotations
    NoisyInterpolatingDiscreteFlow(0.2f0, K=2, dummy_token=21)  # AAs
)

# Or for LaProteina:
P_ca = RDNFlow(3; zero_com=true, schedule=:log, ...)
P_ll = RDNFlow(latent_dim; zero_com=false, schedule=:power, ...)
```

### Branching Flows: CoalescentFlow Wrapper

```julia
using BranchingFlows: CoalescentFlow

# Wrap the base processes
const P = CoalescentFlow(
    (
        FProcess(BrownianMotion(0.2f0), schedule_f),
        FProcess(ManifoldProcess(0.2f0), schedule_f),
        NoisyInterpolatingDiscreteFlow(0.2f0, K=2, dummy_token=21)
    );
    branch_time = ...,        # Distribution for split times
    split_transform = ...,    # Maps logits -> split intensity
    coalescence_policy = ..., # How to merge states
    deletion_hazard = ...,    # Deletion time distribution
)
```

`CoalescentFlow` adds:
- Sampling of coalescent forests during training
- Split/deletion dynamics during generation
- Appropriate loss functions for count/deletion targets

---

## Mask Handling

Branching Flows requires explicit masks on all states:

```julia
# BranchingState has three masks:
BranchingState(
    state_tuple,
    group_ids,
    flowmask = ...,   # [L] - which positions evolve (flow matching applies)
    branchmask = ..., # [L] - which positions can split/delete
    padmask = ...     # [L] - which positions are valid (not padding)
)

# Masks are critical because:
# 1. branching_bridge needs to know which positions can branch
# 2. Variable-length batching requires padding masks
# 3. Loss masking must respect both pad and branch masks
```

---

## X0 Sampling Changes

### Standard: Sample noise with fixed shape

```julia
function zero_state(b)
    L, B = size(b.aas)
    X0locs = MaskedState(ContinuousState(randn(Float32, 3, 1, L, B)), ...)
    X0rots = MaskedState(ManifoldState(rotM, rand_rotations(L*B)), ...)
    X0aas = MaskedState(DiscreteState(21, dummy_tokens), ...)
    return (X0locs, X0rots, X0aas)
end
```

### Branching: X0sampler returns single-element state

```julia
function X0sampler(root)
    # Returns state for a SINGLE element (branching_bridge will expand)
    return (
        ContinuousState(randn(Float32, 3, 1, 1)),      # Single location
        ManifoldState(rotM, [rand_rotation()]),        # Single rotation
        DiscreteState(21, [21]),                       # Dummy AA
        DiscreteState(0, [1])                          # Index tracker
    )
end
```

The key difference: in Branching Flows, generation starts from a **variable number of noise elements** (determined by `length_mins` distribution), which then split/delete to reach the target length.

---

## Generation/Inference Changes

### Standard Flowfusion

```julia
function flow_quickgen(b, model; steps = ...)
    X0 = zero_state(b)  # Fixed-length noise
    X1pred = flowX1predictor(X0, b, model)
    return gen(P, X0, X1pred, steps)  # Fixed length throughout
end
```

### Branching Flows

```julia
function design(model, X1_template, ...)
    # Initialize with branching_bridge at t=0
    bat = branching_bridge(P, X0sampler, [X1_template], t=0, ...)
    X0 = bat.Xt  # May have different length than X1!

    # Generation with step function that handles splits/deletions
    samp = gen(P, X0, step_spec(model, ...), steps)
    return samp
end

# During generation, Flowfusion.step for CoalescentFlow:
# 1. Predicts split counts and deletion probs
# 2. Samples splits from Poisson(predicted_count * branch_density)
# 3. Samples deletions from hazard function
# 4. Updates state with new elements / removed elements
```

---

## Self-Conditioning with Branching (Critical)

Self-conditioning in Branching Flows is more complex than standard Flowfusion because **the sequence length changes during generation**. When splits/deletions happen, the self-conditioning predictions from the previous step must be expanded/contracted to match the new state size.

### The Problem

In standard Flowfusion self-conditioning:
```julia
# Standard: length is fixed, so sc_state has same shape as Xt
mutable struct Wrapper
    sc_state  # Previous X1 prediction, shape [D, L, B]
end

function (w::Wrapper)(t, Xt)
    # Xt and sc_state always have the same L
    pred = model(t, Xt, sc_state=w.sc_state)
    w.sc_state = pred  # Update for next step
    return pred
end
```

In Branching Flows, after a step:
- Some positions may have split (L increases)
- Some positions may have been deleted (L decreases)
- The new `Xt` has a different length than `sc_state`!

### The Solution: Index Tracking State

BranchChain uses an **index tracking state** (4th component of the state tuple) to track which original positions correspond to which current positions:

```julia
# State includes an index tracker
X1 = BranchingState(
    (X1locs, X1rots, X1aas, index_state),  # index_state tracks original indices
    groupings,
    ...
)

# index_state.S.state contains [1, 2, 3, 4, ...] initially
# After splits/deletions, it might be [1, 1, 2, 4, ...] (pos 1 split, pos 3 deleted)
```

### Self-Conditioning Wrapper for Branching

```julia
mutable struct BranchingWrapper
    model
    sc_x1_ca::Union{Nothing, AbstractArray}   # Previous CA prediction [3, L_prev, B]
    sc_x1_ll::Union{Nothing, AbstractArray}   # Previous latent prediction [D, L_prev, B]
end

function (w::BranchingWrapper)(t, Xt_branching::BranchingState)
    # Extract current state
    Xt_ca = tensor(Xt_branching.state[1])   # [3, L_current, B]
    Xt_ll = tensor(Xt_branching.state[2])   # [D, L_current, B]

    # Get index state - tells us which original position each current position came from
    current_indices = Xt_branching.state[4].S.state  # [L_current, B] or [L_current]

    # Expand self-conditioning to match current length
    if !isnothing(w.sc_x1_ca)
        # Index into previous predictions using current_indices
        # If position i came from original position j, use sc_x1[:,j,:]
        sc_ca_expanded = w.sc_x1_ca[:, current_indices, :]
        sc_ll_expanded = w.sc_x1_ll[:, current_indices, :]
    else
        sc_ca_expanded = nothing
        sc_ll_expanded = nothing
    end

    # Run model with expanded self-conditioning
    x1_ca, x1_ll, split_logits, del_logits = model(t, Xt_ca, Xt_ll,
                                                    sc_ca=sc_ca_expanded,
                                                    sc_ll=sc_ll_expanded)

    # Store predictions at CURRENT indices for next step
    # (BranchingFlows.step will handle the split/delete and update indices)
    w.sc_x1_ca = x1_ca
    w.sc_x1_ll = x1_ll

    return (x1_ca, x1_ll, split_logits, del_logits)
end
```

### Key Insight: Adjacent Insertions

When BranchingFlows performs a split at position `i`, the new element is inserted **immediately after** position `i`. The index state is updated so both positions have the same original index. This means:

```julia
# Before split at position 2:
# indices = [1, 2, 3, 4]
# sc_x1 has predictions for positions 1, 2, 3, 4

# After split at position 2:
# indices = [1, 2, 2, 3, 4]  # Position 2 duplicated
# New Xt has length 5
# When expanding sc_x1: both new positions 2 and 3 get sc_x1[:,2,:] (the split source)
```

### Handling Deletions

Deletions are simpler - the deleted positions just disappear from the index state:

```julia
# Before deletion at position 3:
# indices = [1, 2, 3, 4]

# After deletion at position 3:
# indices = [1, 2, 4]  # Position 3 gone
# sc_x1 expansion: just index normally, position 3 no longer needed
```

### BranchChain's step_spec Implementation

BranchChain's actual implementation:

```julia
function step_spec(model, ...)
    sc_frames = nothing  # Stores (Translation ∘ Rotation) from previous prediction

    function step_fn(t, Xt)
        # Get current residue indices from index state
        frominds = Xt.state[4].S.state[:]  # Which original position each came from

        # Reconstruct self-cond frames at current indices
        if !isnothing(sc_frames)
            # Index into stored frames using frominds
            sc_frames_expanded = Translation(sc_frames.translation[:, :, frominds, :]) ∘
                                 Rotation(sc_frames.rotation[:, :, frominds, :])
        else
            sc_frames_expanded = nothing
        end

        # Run model
        frames, aas, splits, dels = model(t, Xt, sc_frames=sc_frames_expanded)

        # Store for next step (at current indices, will be re-indexed next step)
        sc_frames = frames

        return (frames, aas, splits, dels)
    end

    return step_fn
end
```

### LaProteina Adaptation

For LaProteina (CA coords + latents instead of frames):

```julia
mutable struct BranchingScoreNetworkWrapper
    score_net::ScoreNetwork
    dev  # GPU device function
    # Self-conditioning state (stored at previous step's indices)
    sc_ca::Union{Nothing, AbstractArray}
    sc_ll::Union{Nothing, AbstractArray}
end

function (w::BranchingScoreNetworkWrapper)(t, Xt::BranchingState)
    # Unpack state components
    ca_state = Xt.state[1]   # MaskedState wrapping ContinuousState
    ll_state = Xt.state[2]   # MaskedState wrapping ContinuousState
    idx_state = Xt.state[3]  # Index tracking (DiscreteState with original indices)

    x_ca = tensor(ca_state.S)  # [3, L, B]
    x_ll = tensor(ll_state.S)  # [D, L, B]
    current_indices = idx_state.S.state  # [L, B] - which original pos each came from

    # Expand self-conditioning to match current length
    if !isnothing(w.sc_ca)
        # current_indices tells us: for each current position, which original position?
        # Use this to index into the stored predictions
        sc_ca = expand_by_indices(w.sc_ca, current_indices)
        sc_ll = expand_by_indices(w.sc_ll, current_indices)
    else
        sc_ca, sc_ll = nothing, nothing
    end

    # Build batch and run model
    mask = Xt.padmask
    batch = build_batch(x_ca, x_ll, t, mask, sc_ca, sc_ll)
    output = w.score_net(w.dev(batch))

    # Extract predictions
    x1_ca, x1_ll = extract_x1(output, x_ca, x_ll, t)
    split_logits = output[:split]   # [L, B]
    del_logits = output[:del]       # [L, B]

    # Store predictions for next step (will be re-indexed when splits/dels happen)
    w.sc_ca = cpu(x1_ca)
    w.sc_ll = cpu(x1_ll)

    # Return in format expected by BranchingFlows.step
    return (ContinuousState(x1_ca), ContinuousState(x1_ll), split_logits, del_logits)
end

function expand_by_indices(arr, indices)
    # arr: [D, L_old, B], indices: [L_new, B]
    # For each position in L_new, indices tells us which L_old position to copy
    D, L_old, B = size(arr)
    L_new = size(indices, 1)
    expanded = similar(arr, D, L_new, B)
    for b in 1:B
        for i in 1:L_new
            src_idx = indices[i, b]
            expanded[:, i, b] = arr[:, src_idx, b]
        end
    end
    return expanded
end
```

---

## Training Strategy: Staged Approach

BranchChain uses a staged training approach when adding branching to a pretrained model:

1. **Stage 1: Freeze base model, train only split/del heads**
   - Load pretrained weights for all standard layers
   - Freeze everything except `count_decoder`, `del_decoder`, `indelpre_t_encoding`
   - Train until split/del predictions stabilize

2. **Stage 2: Unfreeze and fine-tune everything**
   - Unfreeze all parameters
   - Continue training with lower learning rate
   - All losses contribute to gradients

This prevents the new heads from destabilizing the pretrained flow matching.

---

## Summary: Minimal Changes for LaProteina -> BranchingLaProteina

### 1. Add to ScoreNetwork (or create BranchingScoreNetwork)
   - `indel_time_proj`: Dense layer for time conditioning on split/del heads
   - `split_head`: MLP outputting [1, L, B] split logits (off final embedding)
   - `del_head`: MLP outputting [1, L, B] deletion logits (off final embedding)
   - Forward returns `(x1_ca, x1_ll, split_logits, del_logits)` instead of just `(x1_ca, x1_ll)`

### 2. Add state construction
   - `compoundstate(protein)`: Convert precomputed protein to unbatched BranchingState with index tracking
   - `X0sampler()`: Return single-element noise state for CA and latents

### 3. Change training
   - Replace `batch_from_precomputed` with call to `branching_bridge`
   - Add split loss (Bregman Poisson) and deletion loss (BCE)
   - Mask handling with `padmask .* branchmask`

### 4. Add dependencies
   - `using BranchingFlows`
   - Wrap `(P_ca, P_ll)` in `CoalescentFlow`

### 5. Update generation/inference
   - Create `BranchingScoreNetworkWrapper` that handles self-conditioning with index expansion
   - Use `branching_bridge` to initialize X0
   - `gen` with CoalescentFlow handles variable-length dynamics via `step`
   - Self-conditioning must expand predictions using index state when splits/deletions occur

### 6. Training strategy
   - Stage 1: Freeze pretrained weights, train only split/del heads
   - Stage 2: Unfreeze all, fine-tune with lower LR

---

## Code References

- **ChainStorm (baseline):** https://github.com/MurrellGroup/ChainStorm.jl
- **BranchChain (branching):** https://github.com/MurrellGroup/BranchChain.jl
- **BranchingFlows package:** https://github.com/MurrellGroup/BranchingFlows.jl
