# Generate sequences from a trained energy model with PhageMC.jl.
#
# Input : SampleConfig  (agents/core/schemas.py)
# Output: SampleResult  (agents/core/schemas.py)
#
# =============================================================================
# The ONE place that calls PhageMC.jl (whose module is named `SeqModelMC`).
#
# What that package actually provides is narrower than the stub assumed:
#
#   Δenergy(model, s, a, i)   the single-mutation energy difference
#   descent!(s, model)        unconstrained first-improvement descent
#   anneal!(s, model, sched)  unconstrained annealing
#   anneal_batch!             requires `model.A` / `model.L`, which a plain
#                             Flux Chain does not have
#
# `anneal!` and `descent!` draw mutations with `rand(1:A)` over the whole
# alphabet, so neither can honour `forbidden_aa` or `fixed_positions`, and the
# constraints in `SeqConstraints` are not optional (cysteine is excluded by
# default). The Metropolis and greedy loops below are therefore written here
# against the package's `Δenergy` primitive, and `descent!` is called directly
# in the one case where it is applicable -- no constraints at all.
#
# There is no `gradient` sampler in the package; that method is implemented
# here with Zygote, which is what makes it worth having on the Julia side at
# all (the Python fast path deliberately does not reimplement it).
#
# Sign convention: `objective` says whether good sequences have LOW
# ("minimize", the default, matching the energy term in
# model_energy_guidance.py) or HIGH model output. It is honoured here, and the
# energies reported back are always the RAW model output.
# =============================================================================

include(joinpath(@__DIR__, "common.jl"))

import SeqModelMC

# Flux re-exports Zygote's `gradient`. Going through Flux keeps this script
# inside PD_energy_model's declared dependencies -- Zygote is only an indirect
# one, so importing it directly would mean editing that project's Project.toml.
using Flux: gradient

"""Indices into the `n_aa` alphabet that a sampler may place at a position.

The gap is always excluded: a designed binder has no gaps, and the 21st column
exists only because the model was trained on gapped data.
"""
function allowed_indices(n_aa::Integer, forbidden::AbstractString)
    alpha = alphabet(n_aa)
    bad = Set(collect(uppercase(forbidden)))
    push!(bad, '-')
    idx = [i for (i, a) in enumerate(alpha) if !(a in bad)]
    isempty(idx) && error("constraints forbid every amino acid")
    return idx
end

"""Set the one-hot column at position `i` of chain `n` to alphabet index `a`."""
function set_residue!(X::AbstractArray{<:Real,3}, n::Integer, i::Integer, a::Integer)
    X[:, i, n] .= 0
    X[a, i, n] = 1
    return X
end

"""Initial states: seeds where given, random over the allowed alphabet otherwise."""
function initial_states(cfg_seeds, n::Integer, L::Integer, n_aa::Integer,
                        allowed::AbstractVector{<:Integer},
                        fixed::AbstractDict{Int,Int}, rng)
    X = zeros(Float32, n_aa, L, n)
    alpha = alphabet(n_aa)
    index = Dict(a => i for (i, a) in enumerate(alpha))
    for k in 1:n
        if k <= length(cfg_seeds)
            s = collect(String(cfg_seeds[k]))
            length(s) == L || error("seed $k has length $(length(s)), expected $L")
            for (l, c) in enumerate(s)
                haskey(index, c) || error("seed $k position $l: residue '$c' not in alphabet")
                set_residue!(X, k, l, index[c])
            end
        else
            for l in 1:L
                set_residue!(X, k, l, rand(rng, allowed))
            end
        end
        for (pos, a) in fixed
            set_residue!(X, k, pos, a)
        end
    end
    return X
end

"""Constrained Metropolis over point mutations, one chain per sequence.

`sign` is +1 when minimising the model output and -1 when maximising, so the
acceptance rule below is always "downhill in `sign * energy`".
"""
function sample_metropolis(chain, X::AbstractArray{<:Real,3}, allowed, mutable,
                           n_steps::Integer, β::Real, sign::Real, rng)
    A, L, N = size(X)
    proposals = 0
    accepted = 0
    best = copy(X)
    best_e = [sign * Float64(chain(X[:, :, n:n])[1]) for n in 1:N]
    cur_e = copy(best_e)

    for _ in 1:n_steps, n in 1:N
        i = rand(rng, mutable)
        a = rand(rng, allowed)
        X[a, i, n] == 1 && continue          # no-op proposal
        s = X[:, :, n:n]
        # PhageMC's primitive: model(mutated) - model(original), one mutation.
        ΔE = sign * Float64(SeqModelMC.Δenergy(chain, s, a, i))
        proposals += 1
        if ΔE <= 0 || rand(rng) < exp(-β * ΔE)
            set_residue!(X, n, i, a)
            accepted += 1
            cur_e[n] += ΔE
            if cur_e[n] < best_e[n]
                best_e[n] = cur_e[n]
                best[:, :, n] = X[:, :, n]
            end
        end
    end
    return (states = best, proposals = proposals, accepted = accepted)
end

"""Constrained first-improvement descent -- `SeqModelMC.descent!` with the
   mutable positions and allowed residues restricted."""
function sample_greedy(chain, X::AbstractArray{<:Real,3}, allowed, mutable, sign::Real,
                       max_sweeps::Integer)
    A, L, N = size(X)
    moves = 0
    for n in 1:N
        for _ in 1:max_sweeps
            improved = false
            for i in mutable, a in allowed
                X[a, i, n] == 1 && continue
                s = X[:, :, n:n]
                ΔE = sign * Float64(SeqModelMC.Δenergy(chain, s, a, i))
                if ΔE < 0
                    set_residue!(X, n, i, a)
                    improved = true
                    moves += 1
                    break
                end
            end
            improved || break
        end
    end
    return (states = X, moves = moves)
end

"""Gradient sampler: descend on soft one-hot logits, then discretise.

Not in PhageMC -- implemented here because it needs the model to be
differentiable, which is exactly what the Flux chain gives us and what the
numpy fast path cannot do.
"""
function sample_gradient(chain, X::AbstractArray{<:Real,3}, allowed, mutable, sign::Real,
                         n_steps::Integer, temp::Real, rng; lr::Real = 1.0, n_seeded::Integer = 0)
    A, L, N = size(X)
    mask = fill(-Inf32, A, L)                 # disallowed (residue, position) pairs
    for i in 1:L, a in allowed
        mask[a, i] = 0
    end

    # Start (near-)uniform over the allowed residues rather than at the random
    # one-hot: initialised on a vertex of the simplex the argmax almost never
    # moves, and the sampler just returns its own initialisation. Chains the
    # caller seeded start biased towards their seed instead, so seeds are still
    # honoured.
    logits = zeros(Float32, A, L, N)
    for n in 1:N, i in 1:L
        if i in mutable
            logits[:, i, n] .= mask[:, i] .+ 0.01f0 .* randn(rng, Float32, A)
            n <= n_seeded && (logits[:, i, n] .+= 3f0 .* Float32.(view(X, :, i, n)))
        else
            logits[:, i, n] .= -Inf32
            logits[argmax(view(X, :, i, n)), i, n] = 0
        end
    end

    function soft(lg)
        z = lg ./ Float32(temp)
        e = exp.(z .- maximum(z; dims = 1))
        return e ./ sum(e; dims = 1)
    end

    for _ in 1:n_steps
        g = only(gradient(lg -> sign * sum(chain(soft(lg))), logits))
        g === nothing && break
        logits .-= Float32(lr) .* g
        for n in 1:N, i in 1:L
            i in mutable && (logits[:, i, n] .+= mask[:, i])
        end
    end

    out = zeros(Float32, A, L, N)
    for n in 1:N, i in 1:L
        a = i in mutable ? argmax(view(logits, :, i, n)) : argmax(view(X, :, i, n))
        out[a, i, n] = 1
    end
    return out
end

"""Mean energy of `n` random sequences -- the baseline a sampler must beat."""
function random_baseline(chain, n::Integer, L::Integer, n_aa::Integer, allowed, rng)
    X = zeros(Float32, n_aa, L, n)
    for k in 1:n, i in 1:L
        set_residue!(X, k, i, rand(rng, allowed))
    end
    return sum(Float64.(vec(chain(X)))) / n
end

run_entrypoint() do cfg
    model_name = String(cfg.model_name)
    n          = Int(cfg.n)
    L          = Int(cfg.length)
    method     = Symbol(cfg.method)
    objective  = Symbol(cfg.objective)
    temp       = Float64(cfg.temperature)
    n_steps    = Int(cfg.n_steps)
    seed       = Int(cfg.seed)
    seeds      = [String(s) for s in cfg.seeds]

    objective in (:minimize, :maximize) || error("objective must be minimize or maximize, got $objective")
    sign = objective === :minimize ? 1.0 : -1.0

    loaded = load_chain(model_weights_path(model_name); seq_length = L)
    chain, n_aa = loaded.chain, loaded.n_aa
    loaded.seq_length == L ||
        error("model $model_name was trained at length $(loaded.seq_length), config asks for $L")

    forbidden = String(cfg.constraints.forbidden_aa)
    allowed = allowed_indices(n_aa, forbidden)

    alpha = alphabet(n_aa)
    aa_index = Dict(a => i for (i, a) in enumerate(alpha))
    fixed = Dict{Int,Int}()
    for (k, v) in pairs(cfg.constraints.fixed_positions)
        # Python is 0-based, Julia is 1-based.
        pos = parse(Int, String(k)) + 1
        1 <= pos <= L || error("fixed position $(pos - 1) out of range for length $L")
        c = first(String(v))
        haskey(aa_index, c) || error("fixed position $(pos - 1): residue '$c' not in alphabet")
        fixed[pos] = aa_index[c]
    end
    mutable_positions = [i for i in 1:L if !haskey(fixed, i)]
    isempty(mutable_positions) && error("every position is fixed; nothing to sample")

    rng = Random.MersenneTwister(seed)
    Random.seed!(seed)   # SeqModelMC's own helpers use the global RNG
    X = initial_states(seeds, n, L, n_aa, allowed, fixed, rng)

    @info "sampling" model_name n L method objective temp n_steps n_aa

    accepted_fraction = nothing
    if method === :mcmc
        β = 1.0 / temp
        got = sample_metropolis(chain, X, allowed, mutable_positions, n_steps, β, sign, rng)
        X = got.states
        accepted_fraction = got.proposals == 0 ? 0.0 : got.accepted / got.proposals
    elseif method === :greedy
        unconstrained = isempty(fixed) && length(allowed) == n_aa
        if unconstrained
            # No constraints to honour: use the package's own descent.
            for k in 1:n
                s = X[:, :, k:k]
                SeqModelMC.descent!(s, chain)
                X[:, :, k] = s[:, :, 1]
            end
        else
            sample_greedy(chain, X, allowed, mutable_positions, sign, max(1, n_steps))
        end
    elseif method === :gradient
        X = sample_gradient(chain, X, allowed, mutable_positions, sign, n_steps, temp, rng;
                            n_seeded = min(length(seeds), n))
    else
        error("unknown method $method (expected mcmc, greedy or gradient)")
    end

    seqs = decode_onehot(X)
    energies = energies_of(chain, seqs, L, n_aa)

    sequences = [Dict("seq" => s, "energy" => e, "source" => "phagemc:$(method):$(model_name)")
                 for (s, e) in zip(seqs, energies)]

    return Dict(
        "config"            => cfg,
        "sequences"         => sequences,
        "accepted_fraction" => accepted_fraction,
    )
end
