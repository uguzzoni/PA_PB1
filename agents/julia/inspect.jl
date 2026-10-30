# Report the structure of a trained model as the Julia side understands it.
#
# Input : ArchSpec  (agents/core/schemas.py)
# Output: ArchSpec  (echoed back, reconciled with the actual stored model)
#
# Purpose: give the Architect agent ground truth from the source of truth.
# Python reads the exported weights; this reads the model as PhageNegBinom.jl
# sees it. A disagreement between the two is exactly the bug class worth
# catching early (alphabet order, matrix orientation, layer count).
#
# =============================================================================
# TODO(wire-up): replace the block below with a call into PhageNegBinom.jl to
# load the stored model and report its real layer sizes and activations.
# =============================================================================

include(joinpath(@__DIR__, "common.jl"))

# using PhageNegBinom

run_entrypoint() do cfg
    layer_units = [Int(l.units) for l in cfg.layers]
    seq_length  = Int(cfg.seq_length)
    input_dim   = seq_length * length(AA_ALPHABET)

    @info "inspecting" name=cfg.name layer_units input_dim

    # ==== BEGIN wire-up block ================================================
    #
    # weights = ...  # locate the stored weights for cfg.name
    # model   = PhageNegBinom.load_weights(weights)
    # actual_units = PhageNegBinom.layer_sizes(model)
    # actual_acts  = PhageNegBinom.activations(model)
    #
    # actual_units == layer_units || @warn "stored model disagrees with ArchSpec" actual_units layer_units
    #
    # ==== END wire-up block ==================================================

    # Until wired up, echo the spec back unchanged so the contract is testable.
    return cfg
end
