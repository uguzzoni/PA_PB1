#!/bin/bash
# Usage: bash submit_boltz.sh [results_dir]
# Extracts all peptide sequences from JSON files in results_dir,
# skips any whose boltz output already exists, then submits a boltz array job.

RESULTS_DIR=${1:-results}
SHARED=/home/share_nfs/342-Projets_BGE/342.3-Gen-Chem/342.3.2-BioInfo/342.3.2.8-SeqModels
BOLTZ_OUTDIR=${SHARED}/protein_design/boltz/results

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "ERROR: results directory '$RESULTS_DIR' not found"
    exit 1
fi

JSON_FILES=("$RESULTS_DIR"/*.json)
if [[ ! -e "${JSON_FILES[0]}" ]]; then
    echo "ERROR: no JSON files found in '$RESULTS_DIR'"
    exit 1
fi

# Extract sequences from all JSONs, skipping those already run.
# Output format: name<TAB>seq  (matches run_boltz_peptides.sh $1/$2)
# Name: <protocol>_<index>  →  boltz output dir: ${BOLTZ_OUTDIR}/pa_<name>
PEPTIDE_FILE= ${BOLTZ_OUTDIR}/ 'inputs/designed_peptides_X.tsv'

python3 - "$RESULTS_DIR" "$PEPTIDE_FILE" "$BOLTZ_OUTDIR" << 'PYEOF'
import sys, json, pathlib

results_dir  = pathlib.Path(sys.argv[1])
out_file     = sys.argv[2]
boltz_outdir = pathlib.Path(sys.argv[3])

rows, skipped = [], 0
for jf in sorted(results_dir.glob("*.json")):
    proto = jf.stem
    data  = json.loads(jf.read_text())
    for i, entry in enumerate(data.get("results", [])):
        seq = entry.get("seq", "")
        if not seq:
            continue
        name = f"{proto}_{i}"
        expected_out = boltz_outdir / f"pa_{name}"
        if expected_out.exists():
            print(f"  [skip] {name} — output already exists", flush=True)
            skipped += 1
            continue
        rows.append(f"{name}\t{seq}")

with open(out_file, "w") as fh:
    fh.write("\n".join(rows) + ("\n" if rows else ""))

print(f"New: {len(rows)}  |  Already done: {skipped}  |  Dir: {results_dir}", flush=True)
PYEOF

if [[ $? -ne 0 ]]; then
    echo "ERROR: failed to extract sequences"
    rm -f "$PEPTIDE_FILE"
    exit 1
fi

N=$(grep -c . "$PEPTIDE_FILE" || true)
if [[ "$N" -eq 0 ]]; then
    echo "Nothing to submit — all sequences already processed."
    rm -f "$PEPTIDE_FILE"
    exit 0
fi

echo "Peptide list : $PEPTIDE_FILE"
echo "Submitting ${N} boltz job(s) ..."

batch --array=0-$((N-1)) run_boltz_peptides.sh "$PEPTIDE_FILE"
