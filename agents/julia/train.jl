# Train an energy model with PhageNegBinom.jl.
#
# Input : TrainConfig   (agents/core/schemas.py)
# Output: TrainResult   (agents/core/schemas.py)
#
# =============================================================================
# This is the ONE place that calls PhageNegBinom.jl.
#
# It reproduces the recipe the campaign actually trained with, in
#   PD_energy_model/training/training_2rounds/first_training_phase/
#       model_training_7_PNB_negbinom_ll_3layers*.ipynb
# and exports the selection state in the layout
#   PD_energy_model/analysis/save_model_json.jl
# produces, so the output is a drop-in for data/energy_model_params/.
#
# Note on the package version: PD_energy_model's Manifest pins PhageNegBinom to
# the `changing_ll` branch of SelectionScreeningNB.jl, whose `train!` takes
# grouped `data_params` / `distr_params` NamedTuples. The copy in
# ~/Projects/julia_packages/PhageNegBinom.jl is the older positional-argument
# version. We call the grouped form, matching the notebooks.
# =============================================================================

include(joinpath(@__DIR__, "common.jl"))

import PhageNegBinom
const PNB = PhageNegBinom

# ===========================================================================
# Counts
# ===========================================================================

"""Read a two-column `Sequence,Count` CSV into a `Dict{String,Int}`.

This is the format of `PD_energy_model/data/4_counts/*.csv`. Sequences whose
length differs from `L`, or which contain residues outside the alphabet, are
dropped and counted as rejects rather than aborting the run -- raw NGS tables
routinely carry a tail of junk.
"""
function read_counts_csv(path::AbstractString, L::Integer, n_aa::Integer)
    isfile(path) || error("counts file not found: $path")
    allowed = Set(alphabet(n_aa))
    counts = Dict{String,Int}()
    rejected = 0
    open(path, "r") do io
        header = readline(io)
        occursin(",", header) || error("$path: expected a comma-separated header, got $(repr(header))")
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(strip(line), ',')
            length(parts) < 2 && (rejected += 1; continue)
            seq = String(strip(parts[1]))
            c = tryparse(Int, strip(parts[2]))
            if c === nothing || length(seq) != L || !all(ch -> ch in allowed, seq)
                rejected += 1
                continue
            end
            counts[seq] = get(counts, seq, 0) + c
        end
    end
    return (counts = counts, rejected = rejected)
end

"""Resolve the dataset described by `counts_file` into `(seqs, counts)`.

Two accepted forms, because `DatasetSpec` carries a single path:

1. **A JSON manifest** (`.json`) describing the experiment tree:

       {"branches": [{"label": "F", "files": ["...F_2...csv", "...F_3...csv"]},
                     {"label": "R", "files": ["...R_2...csv", "...R_3...csv"]}]}

   Each branch is one selection lineage; its files are the rounds in order.
   This is the shape the campaign's data actually has (two replicate
   libraries, two rounds each) and the only form that can express it.

2. **A single wide CSV**: `Sequence` followed by one count column per node,
   already merged. Convenient for small or synthetic datasets.

Paths inside a manifest are repository-relative, like `DatasetSpec` itself.
`counts` comes back as `(n_sequences, n_nodes)`, column order = tree order.
"""
function load_dataset(counts_file::AbstractString, L::Integer, n_aa::Integer)
    path = joinpath(repo_root(), counts_file)
    isfile(path) || error("counts file not found: $path")

    if endswith(lowercase(counts_file), ".json")
        spec = JSON3.read(read(path, String))
        haskey(spec, :branches) || error("$counts_file: manifest needs a \"branches\" array")
        branch_labels = String[]
        branch_tables = Vector{Vector{Dict{String,Int}}}()
        total_rejected = 0
        for (b, branch) in enumerate(spec[:branches])
            label = haskey(branch, :label) ? String(branch[:label]) : "branch$b"
            haskey(branch, :files) || error("$counts_file: branch $label has no \"files\"")
            tables = Dict{String,Int}[]
            for f in branch[:files]
                got = read_counts_csv(joinpath(repo_root(), String(f)), L, n_aa)
                total_rejected += got.rejected
                push!(tables, got.counts)
            end
            isempty(tables) && error("$counts_file: branch $label lists no files")
            push!(branch_labels, label)
            push!(branch_tables, tables)
        end
        tables = reduce(vcat, branch_tables)
        rounds_per_branch = [length(t) for t in branch_tables]

        # Outer join on sequence: a sequence seen in any round is a row.
        seqs = sort(collect(union(Set{String}(), (keys(t) for t in tables)...)))
        isempty(seqs) && error("$counts_file: no usable sequences of length $L")
        counts = zeros(Float32, length(seqs), length(tables))
        for (j, t) in enumerate(tables), (i, s) in enumerate(seqs)
            counts[i, j] = get(t, s, 0)
        end
        return (seqs = seqs, counts = counts, labels = branch_labels,
                rounds_per_branch = rounds_per_branch, rejected = total_rejected)
    end

    # Wide CSV
    allowed = Set(alphabet(n_aa))
    seqs = String[]
    rows = Vector{Float32}[]
    rejected = 0
    open(path, "r") do io
        header = split(strip(readline(io)), ',')
        length(header) >= 2 || error("$counts_file: need a Sequence column plus at least one count column")
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(strip(line), ',')
            length(parts) != length(header) && (rejected += 1; continue)
            seq = String(strip(parts[1]))
            if length(seq) != L || !all(ch -> ch in allowed, seq)
                rejected += 1
                continue
            end
            vals = map(p -> something(tryparse(Float32, strip(p)), -1f0), parts[2:end])
            any(v -> v < 0, vals) && (rejected += 1; continue)
            push!(seqs, seq)
            push!(rows, collect(vals))
        end
    end
    isempty(seqs) && error("$counts_file: no usable sequences of length $L")
    counts = permutedims(reduce(hcat, rows))
    n_nodes = size(counts, 2)
    return (seqs = seqs, counts = counts, labels = ["branch1"],
            rounds_per_branch = [n_nodes], rejected = rejected)
end

# ===========================================================================
# Experiment tree and states
# ===========================================================================

"""Build one chained lineage per branch: round 1 -> round 2 -> ...

Node indices are assigned in pre-order across all roots, which is what
`PhageNegBinom.check_index` requires and what makes column `t` of `counts`
correspond to node `t`.
"""
function build_tree(rounds_per_branch::AbstractVector{<:Integer})
    roots = PNB.Node[]
    idx = 0
    for r in rounds_per_branch
        r >= 1 || error("a branch needs at least one round")
        idx += 1
        root = PNB.Node(idx, "branch$(length(roots) + 1) round1")
        node = root
        for k in 2:r
            idx += 1
            node = PNB.Node(idx, node, "branch$(length(roots) + 1) round$k")
        end
        push!(roots, root)
    end
    PNB.check_index(roots) || error("internal error: tree node indices are not pre-order")
    return roots
end

"""States and chemical potentials, following the campaign's notebooks.

One *library* state per branch (the initial composition of that replicate),
one shared *selection* state -- the energy model we are actually after -- and
a *wash* state acting as the negative-selection reference for every node.

`μs[state, node] = 0` switches a state on for that node, `-Inf` switches it
off. The first node of each branch sees only its library; every later node of
that branch sees the shared selection state.
"""
function build_states(rounds_per_branch::AbstractVector{<:Integer},
                      units::AbstractVector{<:Integer}, acts::AbstractVector{<:AbstractString},
                      L::Integer, n_aa::Integer)
    nb = length(rounds_per_branch)
    pairs = Pair{Symbol,Any}[]
    for b in 1:nb
        push!(pairs, Symbol("library$b") => build_chain(units, acts, L, n_aa))
    end
    push!(pairs, :selection => build_chain(units, acts, L, n_aa))
    # Negative-selection reference: a single affine map forced to zero, exactly
    # as in the notebooks. `zero` makes it a fixed baseline, not a free state.
    push!(pairs, :wash => Chain(flatten, Dense(n_aa * L, 1), zero))
    states = NamedTuple(pairs)

    index = Dict(s => i for (i, s) in enumerate(keys(states)))
    n_nodes = sum(rounds_per_branch)
    μs = fill(Float32(-Inf), length(states), n_nodes)
    μn = fill(Float32(-Inf), length(states), n_nodes)

    t = 0
    for (b, r) in enumerate(rounds_per_branch)
        for k in 1:r
            t += 1
            if k == 1
                μs[index[Symbol("library$b")], t] = 0
            else
                μs[index[:selection], t] = 0
            end
        end
    end
    μn[index[:wash], :] .= 0
    return (states = states, μs = μs, μn = μn, index = index)
end

"""L2 on every Dense layer of every learnable state.

The notebooks regularise `state[2]` only; this covers all Dense layers, which
is the usual reading of `regularization = {"l2": ...}` in `ArchSpec`.

The `changing_ll` branch calls `regularizer(ps)` with the whole parameter
NamedTuple -- not the splatted form the older package uses -- so this takes a
single argument and reaches in for `ps.states`.
"""
function make_regularizer(λ::Real)
    λ <= 0 && return Returns(false)
    return function (ps)
        states = ps.states
        total = zero(Float32)
        for name in keys(states)
            for layer in states[name].layers
                if layer isa Dense
                    total += λ * (sum(abs2, layer.weight) + sum(abs2, layer.bias))
                end
            end
        end
        return total
    end
end

# ===========================================================================
# Entry point
# ===========================================================================

run_entrypoint() do cfg
    model_name = String(cfg.model_name)
    arch  = cfg.arch
    dataset = cfg.dataset
    hyper = cfg.hyperparams

    units = [Int(l.units) for l in arch.layers]
    acts  = [String(l.activation) for l in arch.layers]
    L     = Int(arch.seq_length)
    # The campaign's exported models use the 21-letter gap alphabet. ArchSpec
    # has no field for this, so follow that convention; `arch` in the exported
    # file records it for inspect.jl and the Python loader.
    n_aa  = 21
    λ     = haskey(arch.regularization, :l2) ? Float64(arch.regularization[:l2]) : 0.0

    last(units) == 1 || error("final layer must have 1 unit (got $(last(units))): the model emits a scalar energy")

    @info "training" model_name units n_rounds=arch.n_rounds likelihood=arch.likelihood n_aa

    t0 = time()
    data = load_dataset(String(dataset.counts_file), L, n_aa)
    n_nodes = size(data.counts, 2)
    @info "dataset" n_sequences=length(data.seqs) n_nodes rejected=data.rejected branches=data.labels

    roots = build_tree(data.rounds_per_branch)
    PNB.nodecount(roots) == n_nodes ||
        error("tree has $(PNB.nodecount(roots)) nodes but counts has $n_nodes columns")

    built = build_states(data.rounds_per_branch, units, acts, L, n_aa)
    states, μs, μn = built.states, built.μs, built.μn

    # ---- train / validation split -----------------------------------------
    rng = Random.MersenneTwister(Int(dataset.split_seed))
    perm = Random.randperm(rng, length(data.seqs))
    n_val = floor(Int, Float64(dataset.val_fraction) * length(data.seqs))
    val_idx = perm[1:n_val]
    trn_idx = perm[(n_val + 1):end]
    isempty(trn_idx) && error("val_fraction=$(dataset.val_fraction) leaves no training data")

    X = encode_onehot(data.seqs, L, n_aa)
    X_trn = X[:, :, trn_idx]; C_trn = data.counts[trn_idx, :]
    X_val = X[:, :, val_idx]; C_val = data.counts[val_idx, :]

    # ---- likelihood parameters --------------------------------------------
    log_A = vec(log.(max.(sum(data.counts; dims = 1), 1f0)))
    data_params = (log_A = log_A,)
    distr_params = if String(arch.likelihood) == "negbinom"
        (log_α = fill(0f0, n_nodes), γ = fill(1f0, n_nodes))
    elseif String(arch.likelihood) == "poisson"
        NamedTuple()
    else
        error("likelihood $(arch.likelihood) is not supported by PhageNegBinom (use negbinom or poisson)")
    end

    regularizer = make_regularizer(λ)
    optim = Flux.Optimisers.AdaBelief(Float32(hyper.learning_rate))
    batchsize = min(Int(hyper.batch_size), length(trn_idx))
    total_iters = Int(hyper.epochs)

    val_loss() = n_val == 0 ? nothing :
        -Float64(PNB.log_likelihood(X_val, C_val, roots, states, μs, μn, data_params, distr_params))

    function run_stage(iters, learnable, label)
        iters <= 0 && return
        @info "stage" label iters batchsize
        # train! is called in chunks so validation can be checked and early
        # stopping applied. The optimiser state is rebuilt per chunk, so chunks
        # are kept large enough that the lost momentum does not matter.
        chunk = max(1, min(iters, cld(iters, 10)))
        done = 0
        best = Inf
        stale = 0
        while done < iters
            this = min(chunk, iters - done)
            PNB.train!(states, μs, μn, data_params, distr_params,
                       X_trn, C_trn, roots;
                       iters = this,
                       optim = optim,
                       regularizer = regularizer,
                       callback = Returns(nothing),
                       batchsize = batchsize,
                       learnable = learnable)
            done += this
            v = val_loss()
            v === nothing && continue
            @info "  checkpoint" label done val_loss=v
            if Int(hyper.early_stopping_patience) > 0
                if v < best - 1e-9
                    best = v; stale = 0
                else
                    stale += 1
                    if stale >= Int(hyper.early_stopping_patience)
                        @info "  early stop" label done
                        break
                    end
                end
            end
        end
    end

    # Libraries first, then selection. Fitting them jointly from scratch lets
    # the selection state absorb the libraries' composition, which is exactly
    # the confound the energy model must not learn. The notebooks stage it the
    # same way.
    library_states = NamedTuple(Symbol("library$b") => states[Symbol("library$b")]
                                for b in 1:length(data.rounds_per_branch))
    run_stage(cld(total_iters, 2), (; states = library_states), "libraries")
    run_stage(total_iters - cld(total_iters, 2),
              (; states = (selection = states.selection,), μs, μn), "selection")

    # ---- export ------------------------------------------------------------
    store = store_root()
    weights_file = model_weights_path(model_name)
    export_weights(states.selection, weights_file;
                   seq_length = L, n_aa = n_aa, activations = acts,
                   extra = Dict(
                       "model_name" => model_name,
                       "likelihood" => String(arch.likelihood),
                       "n_rounds"   => Int(arch.n_rounds),
                       "trained_on" => String(dataset.counts_file),
                   ))

    final_val = val_loss()
    n_params = sum(length, Flux.trainables(states.selection))

    @info "done" weights_file val_loss=final_val n_params duration=round(time() - t0; digits = 2)

    return Dict(
        "model_name"   => model_name,
        # store-relative, as TrainResult expects
        "weights_path" => relpath(weights_file, store),
        "arch"         => arch,
        "metrics"      => Dict("val_loss" => final_val, "n" => n_val),
        "n_params"     => n_params,
        "duration_s"   => round(time() - t0; digits = 2),
    )
end
