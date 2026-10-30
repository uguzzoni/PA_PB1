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

"""Canonical amino-acid order.

!!! warning
    Must match `AA_ALPHABET` in agents/core/schemas.py. If these disagree, the
    Python fast path silently returns nonsense rather than failing.
"""
const AA_ALPHABET = collect("ACDEFGHIKLMNPQRSTVWY")
