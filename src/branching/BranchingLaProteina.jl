# BranchingLaProteina: Variable-length protein generation using Branching Flows
#
# This module extends LaProteina with split/deletion capabilities for
# generating proteins of variable length.
#
# Key components:
# - BranchingScoreNetwork: ScoreNetwork extended with split/del heads
# - State utilities: Convert precomputed proteins to BranchingState format
# - Training: branching_bridge integration and loss functions
# - Inference: Self-conditioning wrapper handling variable-length states

module BranchingLaProteina

using Reexport

# Import base LaProteina for ScoreNetwork, weight loading, etc.
using LaProteina
@reexport using LaProteina

# Required dependencies
using Flux
using Statistics
using Random

# Include branching-specific code
include("branching_score_network.jl")
include("branching_states.jl")
include("branching_training.jl")
include("branching_inference.jl")

# Exports
export
    # Branching Score Network
    BranchingScoreNetwork,
    forward_branching_from_raw_features,
    trainable_indel_params,
    load_base_weights!,

    # State construction
    protein_to_branching_state,
    X0_sampler_laproteina,
    proteins_to_X1_states,
    extract_state_tensors,
    expand_by_indices,

    # Training
    branching_training_batch,
    branching_flow_loss,
    indel_only_loss,
    bregman_poisson_loss,
    logistic_bce_loss,
    staged_training_step!,
    freeze_base_in_state!,
    thaw_base_in_state!,
    setup_optimizer,

    # Inference
    BranchingScoreNetworkWrapper,
    create_branching_processes,
    generate_with_branching,
    reset_self_conditioning!

end # module
