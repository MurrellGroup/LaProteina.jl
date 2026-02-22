module LaProteina

using LinearAlgebra
using Statistics
using Random

using Flux
using NNlib
using ChainRulesCore
using Distributions
using Printf
using CUDA
using cuDNN  # Required: Flux GPU backend detection + softmax accuracy fix
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

# Motif scaffolding (data pipeline — before nn/ since features reference constants)
include("motif/contig_parser.jl")
include("motif/motif_extraction.jl")
include("motif/motif_batch.jl")

# Neural network layers
include("nn/adaptive_ln.jl")
include("nn/pair_bias_attention.jl")
include("nn/transition.jl")
include("nn/triangular_update.jl")  # Must be before transformer_block.jl
include("nn/transformer_block.jl")
include("nn/score_network.jl")
include("nn/score_network_efficient.jl")
include("nn/encoder.jl")
include("nn/encoder_efficient.jl")
include("nn/decoder.jl")

# Data loading
include("data/pdb_loading.jl")

# Inference utilities (get_schedule, samples_to_pdb)
include("inference.jl")

# Flowfusion-based sampling
include("flowfusion_sampling.jl")

# Weight loading
include("weights.jl")
include("weights_safetensors.jl")

# Training utilities
include("training/precompute_encoder.jl")

# GPU optimization (Onion dispatch hooks → OnionTile cuTile kernels, CuArray method overrides)
# Must be included after all layer definitions so CuArray methods override defaults
include("gpu/gpu.jl")

# Branching Flows (variable-length generation via coalescent flows)
# Must be after gpu/gpu.jl since branching_score_network.jl uses _transformer_block_prenormed
include("branching/branching_score_network.jl")
include("branching/branching_states.jl")
include("branching/branching_training.jl")
include("branching/branching_inference.jl")

# Exports
export
    # Constants
    ATOM_TYPES, ATOM_TYPE_NUM, CA_INDEX,
    RESTYPES, RESTYPE_ATOM37_MASK, SIDECHAIN_TIP_ATOMS,
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
    # Motif features
    MotifMaskFeature, MotifAbsCoordsFeature, MotifRelCoordsFeature, MotifSeqFeature,
    MotifSidechainAngleFeature, MotifBackboneTorsionFeature, BulkAllAtomXmotifFeature,
    MotifPairDistFeature,
    # Feature factory constructors
    score_network_seq_features, score_network_cond_features,
    score_network_pair_features, score_network_pair_cond_features,
    score_network_seq_features_motif_tip, score_network_seq_features_motif_aa,
    score_network_pair_features_motif_aa,
    encoder_seq_features, encoder_cond_features, encoder_pair_features,
    encoder_seq_features_legacy, encoder_pair_features_legacy,
    decoder_seq_features, decoder_pair_features, decoder_cond_features,

    # Layers
    ProteINAAdaLN, AdaptiveOutputScale, AdaptiveLayerNormIdentical,
    PairBiasAttention, MultiHeadBiasedAttentionADALN,
    SwiGLU, SwiGLUTransition, TransitionADALN, ConditioningTransition,
    TransformerBlock, PairUpdate,
    # Triangular updates
    TriangleMultiplication, PairTransition,

    # Score Network
    ScoreNetwork, PairReprBuilder,
    score_network_forward,
    v_to_x1, x1_to_v,
    self_condition_input,
    # Separated feature extraction for training
    ScoreNetworkFeatures, extract_features, forward_from_features,
    ScoreNetworkRawFeatures, extract_raw_features, forward_from_raw_features,
    compute_sc_feature_offsets, update_sc_raw_features!,
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

    # Inference utilities
    get_schedule, samples_to_pdb,

    # Flowfusion-based sampling
    RDNFlow, gen, bridge, step,
    ContinuousState, tensor,
    ScoreNetworkWrapper, MutableScoreNetworkWrapper,
    generate_with_flowfusion, sample_with_flowfusion,

    # GPU support
    gpu, cpu,
    enable_tf32!, within_gradient, safe_checkpointed,

    # PDB I/O
    load_pdb, extract_ca_coords, batch_pdb_data, save_pdb,

    # Weight loading (NPZ)
    load_score_network_weights!, load_decoder_weights!, load_encoder_weights!,
    # Weight loading (SafeTensors)
    load_score_network_weights_st!, load_decoder_weights_st!, load_encoder_weights_st!,

    # Motif scaffolding
    ContigSegment, ScaffoldSegment, MotifSegment,
    parse_contig, generate_scaffold_lengths, compute_motif_indices,
    extract_motif_from_pdb, prepare_motif_batch,

    # Training utilities - precomputed encoder
    PrecomputedProteinNT, precompute_single_protein,
    precompute_dataset_sharded, precompute_dataset_single,
    load_precomputed_shard, batch_from_precomputed,

    # Branching Flows (variable-length generation)
    BranchingScoreNetwork, BranchingScoreNetworkWrapper, NullProcess,
    forward_branching_from_raw_features, forward_branching_from_raw_features_gpu,
    save_branching_weights, load_branching_weights!, load_base_weights!,
    trainable_indel_params, freeze_base!,
    protein_to_branching_state, X0_sampler_laproteina,
    proteins_to_X1_states, extract_state_tensors, expand_by_indices,
    branching_training_batch, branching_flow_loss, indel_only_loss,
    staged_training_step!, freeze_base_in_state!, thaw_base_in_state!,
    setup_optimizer,
    create_branching_processes, create_initial_state,
    generate_with_branching, reset_self_conditioning!

end # module
