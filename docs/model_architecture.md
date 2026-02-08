# Model Architecture

Deep dive into the neural network architecture of LaProteina.

## ScoreNetwork

The core generative model. Predicts velocity fields for flow matching on CA coordinates and local latent representations.

### Configuration

```julia
ScoreNetwork(
    n_layers=14,           # Transformer layers
    token_dim=768,         # Sequence embedding dimension
    pair_dim=256,          # Pair representation dimension
    n_heads=12,            # Attention heads (head_dim = 768/12 = 64)
    dim_cond=256,          # Time conditioning dimension
    latent_dim=8,          # Local latent dimension
    qk_ln=true,           # LayerNorm on Q/K in attention
    update_pair_repr=false, # Whether to update pair repr between layers
    output_param=:v        # :v (velocity) or :x1 (endpoint)
)
```

### Forward Pass

```
Input batch Dict:
  :x_t → Dict(:bb_ca => [3,L,B], :local_latents => [8,L,B])
  :t   → Dict(:bb_ca => [B], :local_latents => [B])
  :mask → [L,B]
  :x_sc → Dict(:bb_ca => [3,L,B], :local_latents => [8,L,B])  # optional

1. Feature extraction:
   seq_features  = seq_factory(batch)          # [46, L, B] → Dense → [768, L, B]
   cond_features = cond_factory(batch)         # [512, L, B] → Dense → [256, L, B]
   pair_features = pair_factory(batch)         # [217, L, L, B] → Dense → [256, L, L, B]

2. Conditioning transitions:
   cond = transition_c_1(cond)                 # SwiGLU: [256, L, B] → [256, L, B]
   cond = transition_c_2(cond)                 # SwiGLU: [256, L, B] → [256, L, B]

3. Transformer stack (×14):
   seqs = transformer_layer(seqs, pair_rep, cond, mask)

4. Output projections:
   ca_out      = ca_proj(seqs)                 # LayerNorm → Dense → [3, L, B]
   latents_out = local_latents_proj(seqs)      # LayerNorm → Dense → [8, L, B]

Output Dict:
  :bb_ca => Dict(:v => [3,L,B])
  :local_latents => Dict(:v => [8,L,B])
```

### Separated Feature Extraction

For training efficiency, feature extraction is separated from the gradient-tracked forward pass:

```julia
# Outside gradient (CPU or GPU, no AD)
raw = extract_raw_features(model, batch)  # → ScoreNetworkRawFeatures

# Inside gradient (GPU, AD-tracked)
output = forward_from_raw_features(model, raw)
```

This allows in-place self-conditioning updates between passes without re-extracting all features.

## Transformer Block

Each of the 14 layers is a `TransformerBlock`:

```
x ─→ [ProteINAAdaLN] → [PairBiasAttention] → [AdaptiveOutputScale] → + residual
  ─→ [ProteINAAdaLN] → [SwiGLUTransition]  → [AdaptiveOutputScale] → + residual
  ─→ output
```

Both sub-layers use adaptive normalization conditioned on time embeddings.

### PairBiasAttention

AF3-style multi-head attention with pair feature bias and sigmoid gating.

```julia
struct PairBiasAttention
    node_norm::PyTorchLayerNorm     # Pre-norm on input
    to_qkv::Dense                   # (token_dim → 3 * n_heads * head_dim)
    to_g::Dense                     # (token_dim → n_heads * head_dim) sigmoid gate
    q_norm::PyTorchLayerNorm        # Q LayerNorm (when qk_ln=true)
    k_norm::PyTorchLayerNorm        # K LayerNorm (when qk_ln=true)
    pair_norm::PyTorchLayerNorm     # Pair feature normalization
    to_bias::Dense                  # (pair_dim → n_heads) pair bias projection
    to_out::Dense                   # (n_heads * head_dim → token_dim) output projection
end
```

**Computation:**
1. Project input to Q, K, V (each [head_dim, L, n_heads, B])
2. Apply Q/K LayerNorm if enabled
3. Compute pair bias: `bias = to_bias(pair_norm(pair_rep))` → [n_heads, L, L, B]
4. Flash attention: `attn = softmax((Q @ K^T) / sqrt(d) + bias) @ V`
5. Gate: `output = to_out(sigmoid(g) * attn)`

The sigmoid gating is a key difference from standard attention — it provides a learned scaling per head that improves gradient flow.

**Wrapped in `MultiHeadBiasedAttentionADALN`:**
```julia
struct MultiHeadBiasedAttentionADALN
    adaln::ProteINAAdaLN            # Input adaptive normalization
    mha::PairBiasAttention          # Core attention
    scale_output::AdaptiveOutputScale  # Output adaptive scaling
end
```

### AdaLN Variants

**ProteINAAdaLN** — Per-position adaptive LayerNorm:
```
output = LayerNorm(x) * sigmoid(gamma(LayerNorm(cond))) + beta(LayerNorm(cond))
```
- `gamma`: Dense(dim_cond → dim) + sigmoid activation
- `beta`: Dense(dim_cond → dim, bias=false)
- Conditioning is per-position [dim_cond, L, B]

**AdaptiveOutputScale** — Output gating:
```
output = x * sigmoid(linear(cond))
```
- `linear`: Dense(dim_cond → dim), zero-initialized weights, bias initialized to -2.0
- Initial scale ≈ sigmoid(-2) ≈ 0.12 (near-identity residual at start of training)

**AdaptiveLayerNormIdentical** — Batch-level conditioning:
- Used for pair representation conditioning
- Broadcasts conditioning [dim_cond, B] to match [dim, L, L, B]
- Same gamma/beta parameterization as ProteINAAdaLN

### SwiGLU Transition

```julia
struct SwiGLUTransition
    ln::PyTorchLayerNorm           # Optional input LayerNorm
    linear_in::Dense               # dim → 2 * dim_inner (bias=false)
    swiglu::SwiGLU                 # swish(gate) * value
    linear_out::Dense              # dim_inner → dim (bias=false)
end
```

Expansion factor = 4 (so dim_inner = 4 * 768 = 3072 for the ScoreNetwork).

SwiGLU activation: `linear_in` produces `[x; g]`, then `output = swish(g) * x`.

**Wrapped in `TransitionADALN`:**
```julia
struct TransitionADALN
    adaln::ProteINAAdaLN
    transition::SwiGLUTransition
    scale_output::AdaptiveOutputScale
end
```

### PairUpdate (Optional)

When `update_pair_repr=true`, pair representations are updated between transformer layers:

```julia
struct PairUpdate
    outer_proj::Dense           # (2 * token_dim → pair_dim)
    pair_ln::PyTorchLayerNorm   # Normalization after update
end
```

Computation: For each (i,j) pair, concatenate `[seq_i; seq_j]` and project to pair space. Add to existing pair_rep and normalize.

Not used in the default ScoreNetwork configuration (`update_pair_repr=false`).

## BranchingScoreNetwork

Extends ScoreNetwork with split and deletion prediction heads.

```julia
struct BranchingScoreNetwork
    base::ScoreNetwork           # Pretrained 14-layer transformer
    indel_time_proj::Dense       # Dense(dim_cond → token_dim) time conditioning
    split_head::Chain            # Dense(token_dim → hdim) → swish → Dense(hdim → 1)
    del_head::Chain              # Dense(token_dim → hdim) → swish → Dense(hdim → 1)
end
```

### Head Architecture

After the final transformer layer, the token embedding is combined with time conditioning:

```
indel_cond = indel_time_proj(mean(cond, dims=2))   # [token_dim, 1, B]
seqs_with_time = seqs + indel_cond                  # [token_dim, L, B]
split_logits = split_head(seqs_with_time)           # [1, L, B] → squeeze → [L, B]
del_logits = del_head(seqs_with_time)               # [1, L, B] → squeeze → [L, B]
```

### Initialization

Heads are initialized with 0.05x weight scaling (1/20 of default) for stable training from a pretrained base. This ensures the split/del predictions start near zero and don't destabilize the pretrained flow matching.

### Key Functions

- `freeze_base!(model)`: Freeze base model parameters for stage 1 training
- `trainable_indel_params(model)`: Get only indel head parameters
- `load_base_weights!(model, path)`: Load pretrained ScoreNetwork weights
- `save_branching_weights(model, path)`: Save indel heads to JLD2
- `forward_branching_from_raw_features(model, raw)`: Forward with separated features
- `forward_branching_from_raw_features_gpu(model, raw)`: GPU-optimized with pre-normalized pairs

### Output

```julia
Dict(
    :bb_ca => Dict(:v => [3, L, B]),
    :local_latents => Dict(:v => [8, L, B]),
    :split => [L, B],     # Log expected future splits (Poisson rate)
    :del => [L, B]        # Deletion logits (logistic probability)
)
```

## VAE Encoder (EncoderTransformer)

12-layer transformer that encodes all-atom protein structures to per-residue latent representations.

```julia
EncoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
```

**Input features** (468 dims → 768):
- ChainBreak (1) + ResidueType (20) + Atom37Coord absolute (148) + Atom37Coord relative (148) + BackboneTorsion (63) + SidechainAngle (88)

**Pair features** (316 dims → 256):
- RelSeqSep (127) + BackbonePairDist (84) + ResidueOrientation (105)

**Conditioning**: Empty (zeros) — no time conditioning for the encoder.

**Output projection**: `LayerNorm → Dense(768 → 16)` → split into `mean [8, L, B]` + `log_scale [8, L, B]`

**Reparameterization**: `z = mean + randn() * exp(log_scale)`

The encoder is frozen during ScoreNetwork training. Outputs are precomputed once and saved to JLD2 shards.

## VAE Decoder (DecoderTransformer)

12-layer transformer that decodes latent representations back to all-atom structures.

```julia
DecoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
```

**Input features** (11 dims → 768):
- CACoord (3) + Latent (8)

**Pair features** (157 dims → 256):
- RelSeqSep (127) + CAPairDist (30)

**Conditioning**: Empty (zeros).

**Output projections:**
- `logit_proj`: LayerNorm → Dense(768 → 20) → amino acid logits
- `struct_proj`: LayerNorm → Dense(768 → 111) → atom coordinates (37 atoms * 3 dims)

**Output Dict:**
```julia
Dict(
    :coors => [3, 37, L, B],        # All-atom coordinates (nm)
    :seq_logits => [20, L, B],      # Amino acid logits
    :aatype_max => [L, B],          # Argmax amino acid indices
    :atom_mask => [37, L, B],       # Per-atom validity mask
    :residue_mask => [L, B]         # Per-residue mask
)
```

Atom coordinates are relative to the input CA position: `final_coords = predicted_offset + ca_coors`.

## PyTorchLayerNorm

Custom LayerNorm matching PyTorch's numerical behavior:

```julia
struct PyTorchLayerNorm
    scale::Vector{Float32}
    bias::Vector{Float32}
    eps::Float32
    size::NTuple{N,Int}
end
```

**Key difference:** Uses `sqrt(var + eps)` instead of Flux's `sqrt(var + eps^2)`.

When variance is tiny (< eps), Flux's squaring of eps gives `sqrt(var + eps^2) ≈ eps`, while PyTorch gives `sqrt(var + eps) ≈ sqrt(eps)`. This difference causes massive numerical divergence and is critical for achieving parity with the Python reference.

## Key Files

| File | Contents |
|------|----------|
| `src/nn/score_network.jl` | ScoreNetwork, PairReprBuilder, feature extraction, SC utilities |
| `src/nn/score_network_efficient.jl` | EfficientScoreNetworkBatch, GPU-native forward pass |
| `src/nn/pair_bias_attention.jl` | PairBiasAttention, MultiHeadBiasedAttentionADALN |
| `src/nn/adaptive_ln.jl` | ProteINAAdaLN, AdaptiveOutputScale, AdaptiveLayerNormIdentical |
| `src/nn/transition.jl` | SwiGLU, SwiGLUTransition, TransitionADALN, ConditioningTransition |
| `src/nn/transformer_block.jl` | TransformerBlock, PairUpdate |
| `src/nn/encoder.jl` | EncoderTransformer |
| `src/nn/decoder.jl` | DecoderTransformer |
| `src/branching/branching_score_network.jl` | BranchingScoreNetwork |
| `src/utils.jl` | PyTorchLayerNorm, pytorch_normalise |
