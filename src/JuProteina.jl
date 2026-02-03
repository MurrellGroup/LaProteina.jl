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
using JLD2

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
include("features/geometry.jl")
include("features/feature_factory.jl")

# Neural network layers
include("nn/adaptive_ln.jl")
include("nn/pair_bias_attention.jl")
include("nn/transition.jl")
include("nn/transformer_block.jl")
include("nn/score_network.jl")
include("nn/score_network_efficient.jl")
include("nn/encoder.jl")
include("nn/encoder_efficient.jl")
include("nn/decoder.jl")

# Data loading
include("data/pdb_loading.jl")

# Inference (legacy)
include("inference.jl")

# Flowfusion-based sampling
include("flowfusion_sampling.jl")

# Weight loading
include("weights.jl")

# Training utilities
include("training/precompute_encoder.jl")

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

    # Geometry utilities
    normalize_last_dim, signed_dihedral_angle, bond_angle,
    backbone_torsion_angles, sidechain_torsion_angles,
    bin_angles, MAX_CHI_ANGLES, CHI_ATOM_INDICES,

    # Feature factory
    Feature, FeatureFactory, get_dim,
    ZeroFeature, TimeFeature, TimePairFeature, PositionFeature,
    XtBBCAFeature, XtLocalLatentsFeature,
    XscBBCAFeature, XscLocalLatentsFeature,
    OptionalCACoorsFeature, OptionalResTypeFeature, CroppedFlagFeature,
    LatentFeature, CACoordFeature,
    # Encoder sequence features
    ChainBreakFeature, ResidueTypeFeature, Atom37CoordFeature,
    BackboneTorsionFeature, SidechainAngleFeature, ChainIdxSeqFeature,
    # Pair features
    DistanceBinFeature, XtBBCAPairDistFeature, XscBBCAPairDistFeature,
    OptionalCAPairDistFeature, CAPairDistFeature, RelSeqSepFeature,
    # Encoder pair features
    BackbonePairDistFeature, ResidueOrientationFeature, ChainIdxPairFeature,
    # Feature factory constructors
    score_network_seq_features, score_network_cond_features,
    score_network_pair_features, score_network_pair_cond_features,
    encoder_seq_features, encoder_cond_features, encoder_pair_features,
    encoder_seq_features_legacy, encoder_pair_features_legacy,
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
    # Separated feature extraction for training
    ScoreNetworkFeatures, extract_features, forward_from_features,
    ScoreNetworkRawFeatures, extract_raw_features, forward_from_raw_features,
    # Efficient GPU-native forward pass
    EfficientScoreNetworkBatch, forward_efficient, to_efficient_batch,
    compute_time_embedding_gpu, compute_pairwise_distances_gpu,

    # Encoder
    EncoderTransformer, encode,
    # Efficient frozen encoder for training
    EncoderRawFeatures, extract_encoder_features, encode_from_features_gpu,
    encode_frozen_efficient, prepare_encoder_batch_cpu, flow_matching_batch_gpu,
    # Pre-computed encoder outputs for fast training
    PrecomputedSample, precompute_encoder_outputs, flow_matching_batch_from_precomputed,
    efficient_flow_loss_gpu,

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
    load_score_network_weights!, load_decoder_weights!, load_encoder_weights!,

    # Training utilities - precomputed encoder
    PrecomputedProtein, precompute_single_protein,
    precompute_dataset_sharded, precompute_dataset_single,
    load_precomputed_shard, batch_from_precomputed

end # module
