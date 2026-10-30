# Generate sequences from a trained energy model with PhageMC.jl.
#
# Input : SampleConfig  (agents/core/schemas.py)
# Output: SampleResult  (agents/core/schemas.py)
#
# =============================================================================
# TODO(wire-up): the ONE place that calls PhageMC.jl.
#
# A pure-Python reference MCMC already exists in
# agents/core/tools/fitness.py (sample_mcmc_python), so phase 1 runs without
# this file. Wire this up when you want the samplers PhageMC.jl already
# provides -- in particular the 'gradient' and 'greedy' methods, which the
# Python side deliberately does not reimplement.
#
# Once this works, flip `prefer_julia=True` in
# agents/core/tools/fitness.py::sample_sequences.
#
# Sign convention: `objective` in the config says whether good sequences have
# LOW ("minimize", the default, matching the energy term in
# model_energy_guidance.py) or HIGH model output. Honour it here rather than
# flipping signs anywhere else in the stack.
# =============================================================================

include(joinpath(@__DIR__, "common.jl"))

# using PhageMC
# using PhageNegBinom

run_entrypoint() do cfg
    model_name = String(cfg.model_name)
    n          = Int(cfg.n)
    len        = Int(cfg.length)
    method     = Symbol(cfg.method)
    objective  = Symbol(cfg.objective)
    temp       = Float64(cfg.temperature)
    n_steps    = Int(cfg.n_steps)
    seed       = Int(cfg.seed)
    seeds      = [String(s) for s in cfg.seeds]

    forbidden = Set(collect(String(cfg.constraints.forbidden_aa)))
    allowed   = [a for a in AA_ALPHABET if !(a in forbidden)]
    isempty(allowed) && error("constraints forbid every amino acid")

    fixed = Dict{Int,Char}()
    for (k, v) in pairs(cfg.constraints.fixed_positions)
        # Python is 0-based, Julia is 1-based.
        fixed[parse(Int, String(k)) + 1] = first(String(v))
    end

    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    store = joinpath(repo_root, "agents", "store")

    @info "sampling" model_name n len method objective temp n_steps

    # ==== BEGIN wire-up block ================================================
    #
    # weights = joinpath(store, "models", model_name,
    #                    "$(model_name)_energy_model_weights.json")
    # model   = PhageNegBinom.load_weights(weights)
    # energy  = seq -> PhageNegBinom.energy(model, seq)
    #
    # result = PhageMC.sample(energy;
    #             n          = n,
    #             length     = len,
    #             method     = method,
    #             objective  = objective,
    #             temperature= temp,
    #             nsteps     = n_steps,
    #             alphabet   = allowed,
    #             fixed      = fixed,
    #             seeds      = seeds,
    #             rng        = MersenneTwister(seed))
    #
    # sequences = [Dict("seq" => s, "energy" => e, "source" => "phagemc:$(model_name)")
    #              for (s, e) in zip(result.sequences, result.energies)]
    # accepted  = result.acceptance
    #
    error("""
          agents/julia/sample.jl is not wired to PhageMC.jl yet.
          Until then use the Python reference sampler:
          fitness.sample_sequences(config)  with prefer_julia=False.
          """)
    # ==== END wire-up block ==================================================

    return Dict(
        "config"            => cfg,
        "sequences"         => sequences,
        "accepted_fraction" => accepted,
    )
end
