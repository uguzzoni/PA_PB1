# Train an energy model with PhageNegBinom.jl.
#
# Input : TrainConfig   (agents/core/schemas.py)
# Output: TrainResult   (agents/core/schemas.py)
#
# =============================================================================
# TODO(wire-up): this is the ONE place that calls PhageNegBinom.jl.
#
# Replace the marked block with the real training call. What is needed from
# the package, in the order used below:
#
#   1. a loader for the sequencing counts file         -> training data
#   2. a model constructor taking layer sizes,
#      activations, n_rounds and the likelihood        -> model
#   3. a fit/train! function taking hyperparameters    -> trained model
#   4. an exporter writing the weights as JSON in the
#      same layout data/energy_model_params/*.json uses
#
# Keep the exported layout identical to the existing files: the Python fast
# path in agents/core/tools/fitness.py reads both, and agreement between the
# two is the correctness check for the whole stack.
# =============================================================================

include(joinpath(@__DIR__, "common.jl"))

# using PhageNegBinom

run_entrypoint() do cfg
    model_name = String(cfg.model_name)
    arch = cfg.arch
    dataset = cfg.dataset
    hyper = cfg.hyperparams

    layer_units = [Int(l.units) for l in arch.layers]
    layer_acts  = [String(l.activation) for l in arch.layers]
    seq_length  = Int(arch.seq_length)
    input_dim   = seq_length * length(AA_ALPHABET)

    @info "training" model_name layer_units n_rounds=arch.n_rounds likelihood=arch.likelihood

    # Paths in the config are repository-relative; resolve against the repo root
    # (agents/julia/ -> agents/ -> repo root).
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    counts_path = joinpath(repo_root, String(dataset.counts_file))
    isfile(counts_path) || error("counts file not found: $counts_path")

    # The agent system only ever writes under agents/store/.
    store = joinpath(repo_root, "agents", "store")
    model_dir = joinpath(store, "models", model_name)
    mkpath(model_dir)
    weights_file = joinpath(model_dir, "$(model_name)_energy_model_weights.json")

    t0 = time()

    # ==== BEGIN wire-up block ================================================
    #
    # data  = PhageNegBinom.load_counts(counts_path)
    # model = PhageNegBinom.build_model(layer_units;
    #             activations = layer_acts,
    #             n_rounds    = Int(arch.n_rounds),
    #             likelihood  = Symbol(arch.likelihood),
    #             input_dim   = input_dim,
    #             l2          = get(arch.regularization, :l2, 1e-4))
    # PhageNegBinom.train!(model, data;
    #             epochs        = Int(hyper.epochs),
    #             lr            = Float64(hyper.learning_rate),
    #             batchsize     = Int(hyper.batch_size),
    #             patience      = Int(hyper.early_stopping_patience),
    #             val_fraction  = Float64(dataset.val_fraction),
    #             seed          = Int(hyper.seed))
    # PhageNegBinom.export_weights(model, weights_file)
    # val_loss = PhageNegBinom.validation_loss(model, data)
    # n_params = PhageNegBinom.nparams(model)
    #
    error("""
          agents/julia/train.jl is not wired to PhageNegBinom.jl yet.
          Replace the wire-up block with the real calls (see the TODO at the
          top of this file), then delete this error.
          """)
    # ==== END wire-up block ==================================================

    # Parameter count, biases included -- matches ArchSpec.n_params() in Python.
    n_params = begin
        total = 0; prev = input_dim
        for u in layer_units
            total += prev * u + u
            prev = u
        end
        total
    end

    return Dict(
        "model_name"   => model_name,
        # store-relative, as TrainResult expects
        "weights_path" => relpath(weights_file, store),
        "arch"         => arch,
        "metrics"      => Dict("val_loss" => val_loss, "n" => 0),
        "n_params"     => n_params,
        "duration_s"   => round(time() - t0; digits = 2),
    )
end
