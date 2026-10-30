# Shared helpers for the agent-layer Julia entry points.
#
# Contract with the Python side (agents/core/execution.py):
#
#     julia --project=<PD_energy_model> agents/julia/<script>.jl \
#           --config <path/to/config.json> --out <path/to/result.json>
#
# The script reads config.json, does the work, and writes result.json matching
# the corresponding pydantic model in agents/core/schemas.py. On failure it
# writes {"error": "..."} and exits non-zero, so the Python side always has
# something to report.
#
# These entry points are adapters only. They live here, not in
# PD_energy_model/, so that nested git repository stays untouched.

using JSON3
using Flux
using Flux: Chain, Dense, flatten
using Random
using LinearAlgebra

"""Parse `--config` / `--out` from ARGS."""
function parse_cli(args = ARGS)
    config_path = nothing
    out_path = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--config" && i < length(args)
            config_path = args[i + 1]; i += 2
        elseif args[i] == "--out" && i < length(args)
            out_path = args[i + 1]; i += 2
        else
            i += 1
        end
    end
    config_path === nothing && error("missing --config")
    out_path === nothing && error("missing --out")
    return (config = config_path, out = out_path)
end

read_config(path) = JSON3.read(read(path, String))

function write_result(path, payload)
    open(path, "w") do io
        JSON3.write(io, payload)
    end
    return path
end

"""Run `body(config)` and write its return value, turning any error into the
   error envelope the Python side understands."""
function run_entrypoint(body)
    cli = parse_cli()
    try
        result = body(read_config(cli.config))
        write_result(cli.out, result)
    catch err
        bt = sprint(showerror, err, catch_backtrace())
        write_result(cli.out, Dict("error" => bt))
        @error "entry point failed" exception = (err, catch_backtrace())
        exit(1)
    end
end

"""Repository root (agents/julia/ -> agents/ -> repo root)."""
repo_root() = normpath(joinpath(@__DIR__, "..", ".."))

"""The one writable location. Mirrors `workspace.STORE` on the Python side."""
store_root() = joinpath(repo_root(), "agents", "store")

# ===========================================================================
# Alphabet
# ===========================================================================

"""Canonical amino-acid order.

!!! warning
    Must match `AA_ALPHABET` in agents/core/schemas.py. If these disagree, the
    Python fast path silently returns nonsense rather than failing.
"""
const AA_ALPHABET = collect("ACDEFGHIKLMNPQRSTVWY")

"""The same order plus the gap, as `PhageNegBinom.AMINO_ACIDS_GAP`.

The models already in `data/energy_model_params/` were trained on this
21-letter alphabet: `dense1_W` is 315x45 = (21*15) x (3*15). The gap column is
never occupied by a designed binder -- `colabdesign/energy_model/model_3layer.py`
pads a zero column for it -- but it still has weights, so the input width must
match or every number downstream is wrong.
"""
const AA_ALPHABET_GAP = vcat(AA_ALPHABET, '-')

"""Alphabet of width `n_aa`, 20 (no gap) or 21 (with gap)."""
function alphabet(n_aa::Integer)
    n_aa == 20 && return AA_ALPHABET
    n_aa == 21 && return AA_ALPHABET_GAP
    error("alphabet width must be 20 or 21, got $n_aa")
end

"""One-hot encode sequences as `(A, L, N)` Float32 -- the layout
`PhageNegBinom.aa2onehot` produces and `Flux.flatten` expects.

`Flux.flatten` reduces this to `(A*L, N)` column-major, i.e. position-major
with the amino acid varying fastest. That is the same flattening
`model_3layer.py` does with `x.reshape(1, -1)` on `(1, L, 21)`, which is why
the two forward passes agree.
"""
function encode_onehot(seqs::AbstractVector{<:AbstractString}, L::Integer, n_aa::Integer)
    alpha = alphabet(n_aa)
    index = Dict(a => i for (i, a) in enumerate(alpha))
    X = zeros(Float32, n_aa, L, length(seqs))
    for (n, s) in enumerate(seqs)
        chars = collect(s)
        length(chars) == L || error("sequence $n has length $(length(chars)), expected $L")
        for (l, c) in enumerate(chars)
            i = get(index, c, nothing)
            i === nothing && error("sequence $n position $l: residue '$c' is not in the $(n_aa)-letter alphabet")
            X[i, l, n] = 1
        end
    end
    return X
end

"""Decode an `(A, L, N)` one-hot array back to strings."""
function decode_onehot(X::AbstractArray{<:Real,3})
    A, L, N = size(X)
    alpha = alphabet(A)
    return [String([alpha[argmax(view(X, :, l, n))] for l in 1:L]) for n in 1:N]
end

# ===========================================================================
# Model construction, export and loading
# ===========================================================================

const ACTIVATIONS = Dict(
    "relu"     => relu,
    "tanh"     => tanh,
    "identity" => identity,
    "sigmoid"  => sigmoid,
    "softplus" => softplus,
)

function activation(name::AbstractString)
    haskey(ACTIVATIONS, name) || error("unknown activation $(repr(name)); known: $(sort(collect(keys(ACTIVATIONS))))")
    return ACTIVATIONS[name]
end

"""Build the energy MLP as `Chain(flatten, Dense..., Dense...)`.

Matches the architecture trained in
`PD_energy_model/training/.../model_training_7_PNB_negbinom_ll_3layers*.ipynb`:

    Chain(Flux.flatten, Dense(A*L, 3L, relu), Dense(3L, L, relu), Dense(L, 1, identity))

The layer after `flatten` is index 2, which is why `save_model_json.jl` exports
`model[2]`, `model[3]`, `model[4]`.
"""
function build_chain(units::AbstractVector{<:Integer}, acts::AbstractVector{<:AbstractString},
                     L::Integer, n_aa::Integer)
    length(units) == length(acts) || error("$(length(units)) units but $(length(acts)) activations")
    layers = Any[flatten]
    in_dim = n_aa * L
    for (u, a) in zip(units, acts)
        push!(layers, Dense(in_dim, u, activation(a)))
        in_dim = u
    end
    return Chain(layers...)
end

"""The `Dense` layers of a chain, skipping `flatten` and any parameter-free layer."""
dense_layers(chain) = [l for l in chain.layers if l isa Dense]

"""Export weights in the layout `data/energy_model_params/*.json` already uses.

`save_model_json.jl` writes `JSON.print` of a Flux `(out, in)` matrix, which
serialises column-major and therefore *arrives in Python transposed*, as
`(in, out)`. `model_3layer.py` relies on that: it computes `x @ W`, not
`x @ W.T`. We reproduce the same orientation explicitly rather than by
accident, so the file stays a drop-in for the ColabDesign pipeline.

An `arch` block is added alongside. The existing consumers index by the
`denseN_W` / `denseN_b` keys and ignore anything else, so this is additive:
it lets `inspect.jl` report activations, which are not recoverable from the
weight matrices alone.
"""
function export_weights(chain, path::AbstractString;
                        seq_length::Integer, n_aa::Integer,
                        activations::AbstractVector{<:AbstractString},
                        extra::AbstractDict = Dict())
    denses = dense_layers(chain)
    payload = Dict{String,Any}()
    for (k, layer) in enumerate(denses)
        W = Array(layer.weight)          # Flux: (out, in)
        out_dim, in_dim = size(W)
        # Write as (in, out): outer index = input unit, matching what Python reads.
        payload["dense$(k)_W"] = [[Float64(W[o, i]) for o in 1:out_dim] for i in 1:in_dim]
        payload["dense$(k)_b"] = Float64.(Array(layer.bias))
    end
    payload["arch"] = merge(Dict{String,Any}(
        "seq_length"  => Int(seq_length),
        "n_aa"        => Int(n_aa),
        "units"       => [size(l.weight, 1) for l in denses],
        "activations" => collect(String.(activations)),
    ), Dict{String,Any}(String(k) => v for (k, v) in extra))

    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.write(io, payload)
    end
    return path
end

"""Rebuild a Flux `Chain` from an exported weights file.

Inverse of [`export_weights`](@ref), and tolerant of the files that predate it:
if there is no `arch` block, `n_aa` is inferred from the first layer's input
width and `seq_length`, and the activations fall back to the convention the
campaign actually trained with (relu hidden, identity output).
"""
function load_chain(path::AbstractString; seq_length::Union{Nothing,Integer} = nothing)
    isfile(path) || error("no weights file at $path")
    raw = JSON3.read(read(path, String))

    keys_w = sort([String(k) for k in keys(raw) if occursin(r"^dense\d+_W$", String(k))],
                  by = s -> parse(Int, match(r"\d+", s).match))
    isempty(keys_w) && error("""
        $path has no denseN_W keys. Expected the layout written by
        export_weights / PD_energy_model/analysis/save_model_json.jl.
        Found: $(collect(keys(raw)))""")

    Ws = Matrix{Float32}[]
    bs = Vector{Float32}[]
    for kw in keys_w
        rows = raw[Symbol(kw)]                        # (in, out) nested arrays
        in_dim = length(rows); out_dim = length(rows[1])
        W = Matrix{Float32}(undef, out_dim, in_dim)   # back to Flux (out, in)
        for i in 1:in_dim, o in 1:out_dim
            W[o, i] = Float32(rows[i][o])
        end
        push!(Ws, W)
        kb = replace(kw, "_W" => "_b")
        haskey(raw, Symbol(kb)) || error("$path has $kw but no $kb")
        push!(bs, Float32.(collect(raw[Symbol(kb)])))
    end

    arch = haskey(raw, :arch) ? raw[:arch] : nothing
    in_dim = size(Ws[1], 2)

    L = seq_length !== nothing ? Int(seq_length) :
        arch !== nothing && haskey(arch, :seq_length) ? Int(arch[:seq_length]) :
        error("cannot determine seq_length for $path; pass seq_length=")

    n_aa = if arch !== nothing && haskey(arch, :n_aa)
        Int(arch[:n_aa])
    elseif in_dim == 21 * L
        21
    elseif in_dim == 20 * L
        20
    else
        error("""
              first layer input width $in_dim matches neither 20*$L nor 21*$L.
              Check seq_length and the alphabet used at training time.""")
    end

    acts = if arch !== nothing && haskey(arch, :activations)
        String.(collect(arch[:activations]))
    else
        # The campaign's 3-layer model: relu hidden, identity output.
        vcat(fill("relu", length(Ws) - 1), ["identity"])
    end
    length(acts) == length(Ws) || error("$(length(acts)) activations for $(length(Ws)) layers in $path")

    layers = Any[flatten]
    for (W, b, a) in zip(Ws, bs, acts)
        d = Dense(size(W, 2) => size(W, 1), activation(a))
        d.weight .= W
        d.bias .= b
        push!(layers, d)
    end
    chain = Chain(layers...)
    return (chain = chain, seq_length = L, n_aa = n_aa, activations = acts,
            units = [size(W, 1) for W in Ws])
end

"""Where a registered model's weights live inside the store."""
function model_weights_path(model_name::AbstractString)
    return joinpath(store_root(), "models", model_name, "$(model_name)_energy_model_weights.json")
end

"""Score sequences with a chain. Returns a `Vector{Float64}` of energies."""
function energies_of(chain, seqs::AbstractVector{<:AbstractString}, L::Integer, n_aa::Integer)
    isempty(seqs) && return Float64[]
    X = encode_onehot(seqs, L, n_aa)
    return Float64.(vec(chain(X)))
end
