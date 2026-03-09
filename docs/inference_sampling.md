# Inference and Sampling

Guide to generating proteins with LaProteina.

## Fixed-Length Generation (Base Model)

### Using `sample_with_flowfusion`

The simplest way to generate fixed-length proteins:

```julia
using LaProteina
using Flux: gpu

# Load models
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

# Generate: L=100 residues, B=1 sample
samples = sample_with_flowfusion(gpu(score_net), decoder, 100, 1;
    nsteps=400, self_cond=true, dev=gpu)
```

### Process Setup

Two RDNFlow processes handle CA coordinates and latents independently:

```julia
P_ca = RDNFlow(3;
    zero_com=true,                    # Zero center of mass constraint
    schedule=:log, schedule_param=2.0, # Log time schedule
    sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0,  # SDE noise: g(t) = 1/(t+eps)
    sc_scale_noise=0.1, sc_scale_score=1.0,
    t_lim_ode=0.98)                   # Switch to ODE above t=0.98

P_ll = RDNFlow(8;
    zero_com=false,                    # No COM constraint for latents
    schedule=:power, schedule_param=2.0, # Power time schedule: tau = s^2
    sde_gt_mode=:tan, sde_gt_param=1.0,  # SDE noise: g(t) = tan(t*pi/2)
    sc_scale_noise=0.1, sc_scale_score=1.0,
    t_lim_ode=0.98)
```

### MutableScoreNetworkWrapper

Wraps the ScoreNetwork for use with `Flowfusion.gen()`:

```julia
mutable struct MutableScoreNetworkWrapper
    model::ScoreNetwork
    dev                    # GPU device function
    sc_ca::Union{Nothing, AbstractArray}   # Previous CA prediction
    sc_ll::Union{Nothing, AbstractArray}   # Previous latent prediction
end
```

The wrapper:
1. Receives uniform time `u` from the flow process
2. Applies schedule transforms: `t_ca = schedule_transform(P_ca, u)`, `t_ll = schedule_transform(P_ll, u)`
3. Builds the batch Dict with transformed times
4. Runs the model on GPU
5. Stores x1 predictions for next step's self-conditioning

### SDE Sampling

During generation, an SDE adds noise along the trajectory for better sample diversity:

```
dx = [v + g(tau) * score_scale * score] * d_tau + sqrt(2 * g(tau) * noise_scale) * dW
```

| Component | g(tau) | Effect |
|-----------|--------|--------|
| CA | `1 / (tau + eps)` | Strong noise early (when tau ≈ 0), weak late |
| Latents | `tan(tau * pi/2)` | Weak noise early, strong late |

Above `t_lim_ode=0.98`, the SDE switches to pure ODE (no noise) for clean final convergence.

## Variable-Length Generation (Branching Flows)

### Using `generate_with_branching`

```julia
using LaProteina
using BranchingFlows
using Flux: gpu

# Load BranchingScoreNetwork
base = ScoreNetwork(n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
                    dim_cond=256, latent_dim=8, output_param=:v,
                    qk_ln=true, update_pair_repr=false)
model = BranchingScoreNetwork(base)

# Load weights (JLD2 from Julia training)
weights = load("branching_full.jld2")
Flux.loadmodel!(model.base, weights["base"])
Flux.loadmodel!(model.indel_time_proj, weights["indel_time_proj"])
Flux.loadmodel!(model.split_head, weights["split_head"])
Flux.loadmodel!(model.del_head, weights["del_head"])
model = gpu(model)

# Generate
result = generate_with_branching(model, 100;
    nsteps=400, latent_dim=8, self_cond=true, dev=gpu, verbose=true)

println("Generated $(result.final_length) residues")
```

### Process Setup for Branching

```julia
P_ca = RDNFlow(3;
    zero_com=false,   # NOTE: false for branching (not true like base model)
    schedule=:log, schedule_param=2.0,
    sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0,
    sc_scale_noise=0.1, sc_scale_score=1.0, t_lim_ode=0.98)

P_ll = RDNFlow(8;
    zero_com=false,
    schedule=:power, schedule_param=2.0,
    sde_gt_mode=:tan, sde_gt_param=1.0,
    sc_scale_noise=0.1, sc_scale_score=1.0, t_lim_ode=0.98)

P_idx = NullProcess()  # Index tracking (doesn't evolve, just tracks positions)

P = CoalescentFlow((P_ca, P_ll, P_idx), Beta(1.0, 2.0))
```

**Key difference from base model:** `zero_com=false` for CA in branching. This avoids issues with single-position bridges that can zero out coordinates.

### BranchingScoreNetworkWrapper

Handles self-conditioning with variable sequence lengths:

```julia
mutable struct BranchingScoreNetworkWrapper
    model::BranchingScoreNetwork
    dev
    sc_ca::Union{Nothing, AbstractArray}
    sc_ll::Union{Nothing, AbstractArray}
end
```

At each step:
1. Extract current indices from the index tracking state
2. Expand previous SC predictions to match current length using indices
3. Run model to get (x1_ca, x1_ll, split_logits, del_logits)
4. Store predictions for next step
5. Reset index tracking to `[1, 2, ..., L_current]`

### Self-Conditioning with Splits and Deletions

When splits/deletions change the sequence length, the wrapper uses the index tracking state to expand/contract SC predictions:

```
Before split at position 2:
  indices = [1, 2, 3, 4], sc_ca has shape [3, 4, B]

After split at position 2:
  indices = [1, 2, 2, 3, 4], L_new = 5
  sc_ca_new[:, i, :] = sc_ca_old[:, indices[i], :]
  (Both new positions 2 and 3 get the prediction from old position 2)

After deletion at position 3:
  indices = [1, 2, 4], L_new = 3
  (Position 3 simply disappears)
```

### Branching Dynamics

During generation, the CoalescentFlow at each step:
1. Model predicts x1 targets + split logits + deletion logits
2. RDNFlow interpolates CA and latents toward predicted x1
3. Splits sampled from predicted Poisson rates
4. Deletions sampled from predicted hazard function
5. New positions inherit parent's state
6. Sequence length changes dynamically

### Branch Time Distribution

`Beta(1, 2)` controls when branching events occur:
- Mode at 0 (events more likely early in the flow)
- Mean at 1/3
- Most structural changes happen early when the state is still noisy

## Cosine Time Steps (Better Quality)

For higher quality, use cosine-spaced time steps with more steps:

```julia
step_func(t) = Float32(1 - (cos(t * pi) + 1) / 2)
step_number = 500
steps = step_func.(0f0:Float32(1/step_number):1f0)
```

This gives denser steps near t=0 and t=1 where the flow changes most rapidly.

See `scripts/sample_branching_full_OU.jl` for a complete example with OUBridgeExpVar processes and comparison modes.

## Decoding to All-Atom Structures

After generating CA coordinates and latents, decode to full atom37 structures:

```julia
# Prepare decoder input
dec_input = Dict(
    :z_latent => reshape(result.latents, 8, L, 1),
    :ca_coors => reshape(result.ca_coords, 3, L, 1),
    :mask => ones(Float32, L, 1)
)

# Run decoder
dec_out = decoder(dev(dec_input))

# Output keys:
# :coors      → [3, 37, L, B]  all-atom coordinates
# :seq_logits → [20, L, B]     amino acid logits
# :aatype_max → [L, B]         predicted amino acid types
# :atom_mask  → [37, L, B]     which atoms are present
```

## Saving to PDB

```julia
# Collect all outputs
samples = Dict(
    :ca_coords => cpu(ca_coords),          # [3, L, B]
    :latents => cpu(latents),              # [8, L, B]
    :all_atom_coords => cpu(dec_out[:coors]),  # [3, 37, L, B]
    :aatype => cpu(dec_out[:aatype_max]),   # [L, B]
    :atom_mask => cpu(dec_out[:atom_mask]), # [37, L, B]
    :mask => mask                          # [L, B]
)

# Save PDB files
samples_to_pdb(samples, "output_dir/"; prefix="sample", save_all_atom=true)
```

`samples_to_pdb` writes one PDB file per sample in the batch.

For individual proteins:
```julia
save_pdb("output.pdb", coords, aatype; atom_mask=atom_mask)
```

Coordinates are automatically converted from nm back to Angstroms during PDB writing.

## Quality Tips

1. **More steps = better**: 400-500 steps is recommended. Below 200 gives noticeably worse quality.
2. **Cosine scheduling**: Use cosine time steps for better allocation of steps near flow boundaries.
3. **SDE noise**: The default `sc_scale_noise=0.1` provides good diversity. Set to 0 for deterministic ODE sampling.
4. **Self-conditioning**: Always enable (`self_cond=true`). Quality degrades significantly without it.
5. **CA-CA distances**: Healthy proteins have mean CA-CA distance ~0.38 nm (3.8 Angstrom). Check this as a quality metric.
6. **Branching results**: Use `--start-length 0` (grow from L=1) for best results. Starting from Poisson(100) can cause length explosion with some checkpoints.

## Sampling Results

With the fine-tuned BranchingScoreNetwork (200k iterations, BS=8, Muon):
- Starting from Poisson(100) initial lengths
- Final lengths: 100-180 residues (variable via splits and deletions)
- Mean CA-CA distances: 0.37-0.38 nm (expected ~0.38 nm)
- Generation time: ~2 minutes per sample (500 cosine steps on GPU)

## Key Files

| File | Purpose |
|------|---------|
| `src/flowfusion_sampling.jl` | `MutableScoreNetworkWrapper`, `generate_with_flowfusion`, `sample_with_flowfusion` |
| `src/inference.jl` | Time schedules, PDB saving utilities |
| `src/branching/branching_inference.jl` | `BranchingScoreNetworkWrapper`, `create_branching_processes`, `generate_with_branching` |
| `src/data/pdb_loading.jl` | `save_pdb` |
| `scripts/infer_all_variants.jl` | Inference for all LD1-LD7 model variants |
| `scripts/sample_branching_full_OU.jl` | Variable-length sampling with OUBridgeExpVar |
