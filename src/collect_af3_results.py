#!/usr/bin/env python3
"""
collect_af3_results.py
~~~~~~~~~~~~~~~~~~~~~~
Scan AF3_RESULTS_DIR for prediction folders, extract summary confidence
metrics for the best model (by ranking_score), and write / incrementally
update results/af3/af3_summary.json.

Usage
-----
# First run — process all folders:
    python src/collect_af3_results.py

# Incremental update after adding new AF3 predictions:
    python src/collect_af3_results.py

# Force re-process every folder (e.g. after re-running AF3):
    python src/collect_af3_results.py --force

Output
------
AF3_RESULTS_DIR/af3_summary.json — list of dicts, one per sequence.
Fields: seq, af3_folder, af3_iptm, af3_ptm, af3_ranking_score,
        af3_has_clash, af3_frac_disordered,
        af3_ptm_PA, af3_ptm_pep, af3_iptm_PA, af3_iptm_pep,
        af3_iptm_cross, af3_pae_min_cross
"""

import argparse
import json
import os
import pathlib
import sys

# ── Load .env and config ──────────────────────────────────────────────────────
_root = pathlib.Path(__file__).parent.parent
_env_path = _root / '.env'
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line.startswith('export '):
            _line = _line[7:]
        if '=' in _line and not _line.startswith('#'):
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(_root / 'src'))
import config

AF3_DIR = pathlib.Path(config.AF3_RESULTS_DIR)

# AF3_SKIP: colon-separated list of folder names to skip (from .env)
_skip_env = os.environ.get('AF3_SKIP', '')
AF3_SKIP = set(_skip_env.split(':')) if _skip_env else set()

OUTPUT_JSON = AF3_DIR / 'af3_summary.json'


# ── Extraction helpers ────────────────────────────────────────────────────────

def read_chain_b_seq(msa_dir: pathlib.Path) -> str:
    """Extract peptide sequence (chain B) from unpaired MSA a3m file."""
    a3m_files = list(msa_dir.rglob('*_unpaired_msa_chains_b.a3m'))
    if not a3m_files:
        return ''
    lines = a3m_files[0].read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith('>query') or (i == 0 and line.startswith('>')):
            if i + 1 < len(lines):
                return lines[i + 1].strip().upper()
    return ''


def process_folder(folder: pathlib.Path) -> dict | None:
    """
    Extract summary metrics from one AF3 prediction folder.
    Returns None if the folder cannot be parsed.
    """
    seq = read_chain_b_seq(folder / 'msas')
    if not seq:
        print(f'  [skip — no chain B MSA] {folder.name}')
        return None

    conf_files = sorted(folder.glob('*_summary_confidences_*.json'))
    if not conf_files:
        print(f'  [skip — no confidences] {folder.name}')
        return None

    best, best_score = None, float('-inf')
    for cf in conf_files:
        c = json.loads(cf.read_text())
        score = c.get('ranking_score', float('-inf'))
        if score > best_score:
            best_score, best = score, c

    cpi = best.get('chain_pair_iptm',    [[None, None], [None, None]])
    cpm = best.get('chain_pair_pae_min', [[None, None], [None, None]])
    cpt = best.get('chain_ptm',  [None, None])
    cia = best.get('chain_iptm', [None, None])
    cross = cpi[1][0] if len(cpi) > 1 and len(cpi[1]) > 0 else None

    return {
        'seq'                : seq,
        'af3_folder'         : folder.name,
        'af3_iptm'           : best.get('iptm'),
        'af3_ptm'            : best.get('ptm'),
        'af3_ranking_score'  : best.get('ranking_score'),
        'af3_has_clash'      : best.get('has_clash'),
        'af3_frac_disordered': best.get('fraction_disordered'),
        'af3_ptm_PA'         : cpt[0] if len(cpt) > 0 else None,
        'af3_ptm_pep'        : cpt[1] if len(cpt) > 1 else None,
        'af3_iptm_PA'        : cia[0] if len(cia) > 0 else None,
        'af3_iptm_pep'       : cia[1] if len(cia) > 1 else None,
        'af3_iptm_cross'     : cross,
        'af3_pae_min_cross'  : cpm[1][0] if len(cpm) > 1 and len(cpm[1]) > 0 else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(force: bool = False) -> None:
    print(f'AF3_DIR  : {AF3_DIR}')
    print(f'AF3_SKIP : {AF3_SKIP or "(none)"}')
    print(f'Output   : {OUTPUT_JSON}')
    print()

    # Load existing summary
    existing: dict[str, dict] = {}   # keyed by af3_folder name
    if OUTPUT_JSON.exists() and not force:
        for entry in json.loads(OUTPUT_JSON.read_text()):
            existing[entry['af3_folder']] = entry
        print(f'Existing summary: {len(existing)} entries')
    elif force:
        print('--force: re-processing all folders')
    else:
        print('No existing summary — processing all folders')

    # Scan for new folders
    new_rows, n_skipped, n_existing = [], 0, 0
    for folder in sorted(AF3_DIR.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name in AF3_SKIP:
            n_skipped += 1
            continue
        if folder.name in existing and not force:
            n_existing += 1
            continue
        row = process_folder(folder)
        if row:
            cross_str = f'{row["af3_iptm_cross"]:.3f}' if row['af3_iptm_cross'] is not None else 'N/A'
            print(f'  [new] {row["seq"]}  iptm={row["af3_iptm"]:.3f}'
                  f'  iptm_cross={cross_str}  ranking={row["af3_ranking_score"]:.3f}')
            new_rows.append(row)

    print()
    if not new_rows:
        print(f'Nothing new to add  ({n_existing} already in summary, {n_skipped} skipped).')
        return

    # Merge existing + new, write output
    merged = list(existing.values()) + new_rows
    OUTPUT_JSON.write_text(json.dumps(merged, indent=2))
    print(f'Wrote {len(merged)} entries → {OUTPUT_JSON}')
    print(f'  {len(new_rows)} new  |  {n_existing} existing kept  |  {n_skipped} skipped')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect AF3 prediction summary metrics into af3_summary.json.'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Re-process all folders, replacing any existing entries.',
    )
    args = parser.parse_args()
    main(force=args.force)
