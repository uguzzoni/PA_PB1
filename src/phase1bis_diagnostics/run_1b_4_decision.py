"""
run_1b_4_decision.py

Fase 1-bis (Workplan_v2_phase1bis_instantiated.md §1b.4): sintesi della
decisione sull'uso dello stimatore di incertezza, letta dai summary.json delle
sottofasi precedenti. E' l'input diretto per l'istanziazione di Fase 3 §3.5
(Workplan_v2_fasi_1bis_2_3.md).

Usage:
    uv run python src/phase1bis_diagnostics/run_1b_4_decision.py
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEVERAGE_SUMMARY = os.path.join(REPO_ROOT, "results", "phase1bis", "1b_1_leverage_check", "summary.json")
VALIDATION_SUMMARY = os.path.join(REPO_ROOT, "results", "phase1bis", "1b_3_validation", "summary.json")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase1bis", "1b_4_decision")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LEVERAGE_SUMMARY) as f:
        leverage = json.load(f)

    if leverage["verdict_degenerate"]:
        decision = {
            "outcome": "degenerazione_1b1",
            "decision": ("Nessuno stimatore di incertezza epistemica adottato (§1b.1 ha rilevato "
                         "degenerazione: la leva dei design di Fase 0 non supera in modo apprezzabile "
                         "quella del training set). Caratterizzazione rimandata a Fase 4 (modello "
                         "multilineare, che fornisce incertezza in forma chiusa senza questo regime di "
                         "degenerazione). §3.5 non include soglie su sigma ne' quota riservata ad alta sigma."),
            "evidence": {
                "h_bar_theoretical": leverage["h_bar_theoretical"],
                "by_population_median_ratio": {
                    k: v["h_over_hbar_median"] for k, v in leverage["by_population"].items()},
                "by_population_frac_over_100x": {
                    k: v["frac_over_100x"] for k, v in leverage["by_population"].items()},
                "fase0_designs_n": leverage["by_population"]["fase0_designs"]["n"],
            },
            "source": LEVERAGE_SUMMARY,
            "sottofasi_non_eseguite": ["§1b.2 (stima)", "§1b.3 (validazione)"],
        }
    elif os.path.exists(VALIDATION_SUMMARY):
        with open(VALIDATION_SUMMARY) as f:
            validation = json.load(f)
        if validation.get("criterion_monotonic_pass"):
            decision = {
                "outcome": "adottato",
                "decision": ("Stimatore adottato per i tre usi dichiarati in apertura di Fase 1-bis "
                             "(diagnostica sui design, criterio di selezione in §3.5, annotazione dei "
                             "candidati finali). §3.5 include una soglia su sigma e una quota riservata "
                             "a candidati ad alta sigma."),
                "source": VALIDATION_SUMMARY,
            }
        else:
            decision = {
                "outcome": "diagnostica_descrittiva",
                "decision": ("Stimatore riportato come diagnostica descrittiva sui design prodotti, "
                             "senza ruolo decisionale in §3.5 (il criterio di monotonia di §1b.3 non e' "
                             "stato soddisfatto)."),
                "source": VALIDATION_SUMMARY,
            }
    else:
        raise FileNotFoundError(
            f"{VALIDATION_SUMMARY} non trovato e §1b.1 non e' risultato degenere -- "
            "eseguire §1b.2/§1b.3 prima di questo script.")

    out_path = os.path.join(OUT_DIR, "decision.json")
    with open(out_path, "w") as f:
        json.dump(decision, f, indent=2)
    print(json.dumps(decision, indent=2))
    print(f"\nOutput scritto in {out_path}")


if __name__ == "__main__":
    main()
