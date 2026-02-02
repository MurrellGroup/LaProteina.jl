module JuProteina

using LinearAlgebra
using Statistics
using Random

using Flux
using NNlib
using ChainRulesCore
using Distributions
using Printf
using CUDA
using cuDNN

# Re-export Flowfusion types for convenience
using Flowfusion
using ForwardBackward: ContinuousState, tensor

# Re-export Flux's gpu/cpu for device management
using Flux: gpu, cpu

# Constants and utilities
include("constants.jl")
include("utils.jl")

# Feature extraction
include("features/time_embedding.jl")
include("features/pair_features.jl")
include("features/feature_factory.jl")

# Neural network layers
include("nn/adaptive_ln.jl")
include("nn/pair_bias_attention.jl")
include("nn/transition.jl")
include("nn/transformer_block.jl")
include("nn/score_network.jl")
include("nn/decoder.jl")

# Data loading
include("data/pdb_loading.jl")

# Inference (legacy)
include("inference.jl")

# Flowfusion-based sampling
include("flowfusion_sampling.jl")

# Weight loading
include("weights.jl")

# Exports
export
    # Constants
    ATOM_TYPES, ATOM_TYPE_NUM, CA_INDEX,
    RESTYPES, RESTYPE_ATOM37_MASK,
    aa_to_index, index_to_aa,
    RESTYPE_3TO1, RESTYPE_1TO3, ATOM_ORDER,

    # Tensor utilities
    python_to_julia, julia_to_python,
    python_to_julia_pair, julia_to_python_pair,
    python_to_julia_mask, julia_to_python_mask,
    center_of_mass, zero_center_of_mass, expand_mask,
    masked_mean, masked_mse,
    pytorch_normalise, PyTorchLayerNorm,

    # Time embedding
    get_time_embedding, get_index_embedding,
    sample_t_uniform, sample_t_beta, sample_t_mix_unif_beta,
    gt_schedule, inference_time_steps,
    broadcast_time_embedding,

    # Pair features
    bin_pairwise_distances, relative_sequence_separation,
    pairwise_distances, bin_values,

    # Feature factory
    Feature, FeatureFactory,
    ZeroFeature, TimeFeature, TimePairFeature, PositionFeature,
    XtBBCAFeature, XtLocalLatentsFeature,
    XscBBCAFeature, XscLocalLatentsFeature,
    OptionalCACoorsFeature, OptionalResTypeFeature, CroppedFlagFeature,
    LatentFeature, CACoordFeature,
    DistanceBinFeature, XtBBCAPairDistFeature, XscBBCAPairDistFeature,
    OptionalCAPairDistFeature, CAPairDistFeature, RelSeqSepFeature,
    score_network_seq_features, score_network_cond_features,
    score_network_pair_features, score_network_pair_cond_features,
    encoder_seq_features, encoder_cond_features, encoder_pair_features,
    decoder_seq_features, decoder_pair_features, decoder_cond_features,

    # Layers
    ProteINAAdaLN, AdaptiveOutputScale, AdaptiveLayerNormIdentical,
    PairBiasAttention, MultiHeadBiasedAttentionADALN,
    SwiGLU, SwiGLUTransition, TransitionADALN, ConditioningTransition,
    TransformerBlock, PairUpdate,

    # Score Network
    ScoreNetwork, PairReprBuilder,
    score_network_forward,
    v_to_x1, x1_to_v,
    self_condition_input,

    # Decoder
    DecoderTransformer, decode,
    get_atom_mask_from_aatype,

    # Inference (legacy)
    get_schedule, get_gt,
    sample_rdn_noise,
    rdn_interpolate, rdn_ode_step, rdn_sde_step,
    vf_to_score, score_to_vf,
    full_simulation, sample, samples_to_pdb,
    fm_loss,

    # Flowfusion-based sampling
    RDNFlow, gen, bridge, step,
    ContinuousState, tensor,
    ScoreNetworkWrapper, MutableScoreNetworkWrapper,
    generate_with_flowfusion, sample_with_flowfusion,

    # GPU support
    gpu, cpu,

    # PDB I/O
    load_pdb, extract_ca_coords, batch_pdb_data, save_pdb,

    # Weight loading
    load_score_network_weights!, load_decoder_weights!

end # module
