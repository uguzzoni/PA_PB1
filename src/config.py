"""
config.py — machine-independent path configuration.

Set these environment variables before running any script, e.g. by
sourcing a local .env file:

    export COLABDESIGN_PDB="/path/to/your.pdb"
    export COLABDESIGN_PARAMS_DIR="/path/to/af_params"
    export COLABDESIGN_ENERGY_WEIGHTS_PATH="/path/to/model_weights.json"

Or copy .env.example → .env, fill in your paths, and run:

    source .env && python generate_sequences.py
"""

import os

def _require(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{var}' is not set. "
            "Copy .env.example → .env, fill in your paths, and `source .env`."
        )
    return val

PDB          = _require("COLABDESIGN_PDB")
PARAMS_DIR   = _require("COLABDESIGN_PARAMS_DIR")
WEIGHTS_PATH = _require("COLABDESIGN_ENERGY_WEIGHTS_PATH")
HOTSPOT_RESIDUES = [408, 411, 412, 415, 594, 595, 599,
                    617, 618, 619, 620, 621, 622, 623,
                    629, 631, 639, 640, 643, 666, 667,
                    670, 673, 706, 709, 710, 713, 714]

print(WEIGHTS_PATH)