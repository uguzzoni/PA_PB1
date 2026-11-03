# Report the structure of a trained model as the Julia side understands it.
#
# Input : ArchSpec  (agents/core/schemas.py)
# Output: ArchSpec  (echoed back, reconciled with the actual stored model)
#
# Purpose: give the Architect agent ground truth from the source of truth.
# Python reads the exported weights; this reads the model as Flux rebuilds it.
# A disagreement between the two is exactly the bug class worth catching early
# (alphabet order, matrix orientation, layer count).
#
# Note that activations are not recoverable from weight matrices. Files written
# by `export_weights` carry an `arch` block that records them; for the older
# files in data/energy_model_params/ the loader falls back to the convention
# the campaign trained with (relu hidden, identity output) and this script says
# so in `notes` rather than pretending to have measured it.

include(joinpath(@__DIR__, "common.jl"))

"""Find the weights for `name`: the agent store first, then the promoted
   models the ColabDesign pipeline reads."""
function locate_weights(name::AbstractString)
    candidates = String[model_weights_path(name)]
    params_dir = joinpath(repo_root(), "data", "energy_model_params")
    if isdir(params_dir)
        for f in sort(readdir(params_dir))
            endswith(f, ".json") || continue
            (startswith(f, name) || occursin(name, f)) && push!(candidates, joinpath(params_dir, f))
        end
    end
    for c in candidates
        isfile(c) && return c
    end
    return nothing
end

run_entrypoint() do cfg
    name        = String(cfg.name)
    spec_units  = [Int(l.units) for l in cfg.layers]
    spec_acts   = [String(l.activation) for l in cfg.layers]
    L           = Int(cfg.seq_length)

    @info "inspecting" name spec_units seq_length=L

    path = locate_weights(name)
    if path === nothing
        @warn "no stored weights found; echoing the spec back unverified" name
        return cfg
    end

    loaded = load_chain(path; seq_length = L)
    actual_units = loaded.units
    actual_acts  = loaded.activations

    notes = String[]
    push!(notes, "verified against $(relpath(path, repo_root()))")
    push!(notes, "input width $(loaded.n_aa)*$(loaded.seq_length) = $(loaded.n_aa * loaded.seq_length) " *
                 "($(loaded.n_aa == 21 ? "gap alphabet, as the campaign's models" : "no-gap alphabet"))")

    raw = JSON3.read(read(path, String))
    if !haskey(raw, :arch)
        push!(notes, "activations not stored in this file; reported values are the campaign default")
    end
    if actual_units != spec_units
        @warn "stored model disagrees with ArchSpec" actual_units spec_units
        push!(notes, "UNITS MISMATCH: spec said $spec_units, stored model has $actual_units")
    end
    if actual_acts != spec_acts
        @warn "stored activations disagree with ArchSpec" actual_acts spec_acts
        push!(notes, "ACTIVATION MISMATCH: spec said $spec_acts, stored model has $actual_acts")
    end

    return Dict(
        "name"           => name,
        "n_rounds"       => Int(cfg.n_rounds),
        "layers"         => [Dict("units" => u, "activation" => a)
                             for (u, a) in zip(actual_units, actual_acts)],
        "likelihood"     => String(cfg.likelihood),
        "input_encoding" => String(cfg.input_encoding),
        "seq_length"     => loaded.seq_length,
        "n_aa"           => loaded.n_aa,
        "regularization" => cfg.regularization,
        "notes"          => join(notes, "; "),
    )
end
