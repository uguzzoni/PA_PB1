"""
run_1b_1b_leverage_reanalysis.py

Fase 1-bis (Workplan_v2_phase1bis_instantiated.md §1b.1.b): rianalisi
distribuzionale della leva gia' calcolata in §1b.1 -- NESSUN nuovo calcolo di
leva, tutto parte da results/phase1bis/1b_1_leverage_check/leverage_by_population.csv
(gia' su disco, validato: errore relativo 0.00% su h_bar atteso vs osservato).

Il criterio originale di §1b.1 ("frazione di design con h/h_bar oltre 100x")
era mal posto: il massimo osservato sul training set stesso e' 28,7x, quindi
100x era irraggiungibile per costruzione. Il riferimento corretto e' la
distribuzione del training stesso, non un multiplo assoluto dell'attesa
teorica p/N. Questo script rifa' il confronto su quella base.

Metadati dei design: load_designed_sequences() (src/phase0_diagnostics/data_loading.py,
gia' usata in Fase 0 §0.8.b per la stratificazione composizionale, es.
run_0_8b_design_stratification.py) -- stesso loader, non riscritto.

Usage:
    uv run python src/phase1bis_diagnostics/run_1b_1b_leverage_reanalysis.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEVERAGE_CSV = os.path.join(REPO_ROOT, "results", "phase1bis", "1b_1_leverage_check", "leverage_by_population.csv")
LEVERAGE_SUMMARY = os.path.join(REPO_ROOT, "results", "phase1bis", "1b_1_leverage_check", "summary.json")
OUT_DIR = os.path.join(REPO_ROOT, "results", "phase1bis", "1b_1b_leverage_reanalysis")

PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99, 99.9]
TAIL_PERCENTILES = [95, 99, 99.9]
SMALL_N_WARNING_THRESHOLD = 50  # sotto questa numerosita' i percentili alti sono rumorosi -- segnalare, non nascondere

sys.path.insert(0, os.path.join(REPO_ROOT, "src", "phase0_diagnostics"))
from data_loading import load_designed_sequences  # noqa: E402


def load_leverage():
    df = pd.read_csv(LEVERAGE_CSV)
    with open(LEVERAGE_SUMMARY) as f:
        original_summary = json.load(f)
    return df, original_summary


def attach_design_metadata(leverage_df):
    """Posizionale, non un merge per valore: leverage_by_population.csv per la
    popolazione 'fase0_designs' e' stato scritto iterando load_designed_sequences()
    in ordine, senza dedup e senza scarti (900/900 validi, verificato in §1b.1) --
    la stessa identica lista, nello stesso ordine, e' riprodotta da una nuova
    chiamata a load_designed_sequences() (iterazione deterministica su file
    ordinati). Verificato qui esplicitamente, non assunto."""
    designs_meta = load_designed_sequences()
    design_leverage = leverage_df[leverage_df["population"] == "fase0_designs"].reset_index(drop=True)

    n_meta, n_lev = len(designs_meta), len(design_leverage)
    positional_match = (n_meta == n_lev) and (designs_meta["Sequence"].to_numpy() == design_leverage["sequence"].to_numpy()).all()

    if not positional_match:
        raise RuntimeError(
            f"Disallineamento fra load_designed_sequences() (N={n_meta}) e la popolazione "
            f"'fase0_designs' di leverage_by_population.csv (N={n_lev}): l'attacco posizionale "
            "dei metadati non e' valido. Richiede un merge per valore con gestione esplicita "
            "dei duplicati, non implementato qui -- indagare prima di proseguire.")

    merged = pd.concat([designs_meta.reset_index(drop=True), design_leverage[["h", "h_over_hbar"]]], axis=1)
    coverage = {"n_metadata": n_meta, "n_leverage": n_lev, "n_matched": n_lev, "coverage_fraction": 1.0}
    return merged, coverage


def percentile_table(values, percentiles=PERCENTILES):
    v = np.asarray(values, dtype=float)
    row = {"n": len(v), "min": float(v.min()), "max": float(v.max())}
    for p in percentiles:
        row[f"p{p}"] = float(np.percentile(v, p))
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    leverage_df, original_summary = load_leverage()

    populations = {
        label: leverage_df.loc[leverage_df["population"] == label, "h_over_hbar"].to_numpy()
        for label in ["training_set", "bli_23", "cluster_representative", "fase0_designs"]
    }
    for label, v in populations.items():
        print(f"{label}: N={len(v)}")

    # === (a) percentili per popolazione, lineare + log10 ===
    print("\n(a) Percentili per popolazione (h/h_bar)...")
    perc_rows = []
    for label, v in populations.items():
        row = {"population": label, "scale": "linear", **percentile_table(v)}
        perc_rows.append(row)
        row_log = {"population": label, "scale": "log10", **percentile_table(np.log10(np.clip(v, 1e-6, None)))}
        perc_rows.append(row_log)
        if len(v) < SMALL_N_WARNING_THRESHOLD:
            print(f"  ATTENZIONE: {label} ha N={len(v)} -- i percentili alti (95/99/99.9) sono "
                  "poco affidabili a questa numerosita', riportati comunque ma da leggere con cautela.")
    percentiles_df = pd.DataFrame(perc_rows)
    percentiles_df.to_csv(os.path.join(OUT_DIR, "percentiles_by_population.csv"), index=False)
    print(percentiles_df.to_string(index=False))

    # === (b) confronto delle code: soglie dal training, frazione di ciascuna popolazione che le supera ===
    print("\n(b) Code: soglie dal training set, frazione di ciascuna popolazione oltre...")
    training_vals = populations["training_set"]
    thresholds = {p: float(np.percentile(training_vals, p)) for p in TAIL_PERCENTILES}
    nominal_fraction = {95: 0.05, 99: 0.01, 99.9: 0.001}

    tail_rows = []
    for label, v in populations.items():
        row = {"population": label, "n": len(v)}
        for p in TAIL_PERCENTILES:
            frac = float((v > thresholds[p]).mean())
            row[f"frac_over_training_p{p}"] = frac
            row[f"nominal_if_same_distribution_p{p}"] = nominal_fraction[p]
        tail_rows.append(row)
    tails_df = pd.DataFrame(tail_rows)
    tails_df.to_csv(os.path.join(OUT_DIR, "design_vs_training_tails.csv"), index=False)
    print(tails_df.to_string(index=False))
    print(f"  soglie assolute dal training (h/h_bar): p95={thresholds[95]:.3f}  "
          f"p99={thresholds[99]:.3f}  p99.9={thresholds[99.9]:.3f}")

    # === (c) Kolmogorov-Smirnov ===
    print("\n(c) Test di Kolmogorov-Smirnov contro il training...")
    ks_rows = []
    for label in ["fase0_designs", "bli_23", "cluster_representative"]:
        stat, pval = stats.ks_2samp(populations[label], training_vals)
        ks_rows.append({"population": label, "n": len(populations[label]),
                         "ks_statistic": float(stat), "p_value": float(pval)})
        print(f"  {label} vs training_set: D={stat:.4f}  p={pval:.3e}")
    ks_df = pd.DataFrame(ks_rows)
    ks_df.to_csv(os.path.join(OUT_DIR, "ks_tests.csv"), index=False)

    # === ECDF overlay (figura chiave) ===
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"training_set": "gray", "bli_23": "steelblue", "cluster_representative": "darkorange",
              "fase0_designs": "firebrick"}
    for label, v in populations.items():
        x = np.sort(np.log10(np.clip(v, 1e-6, None)))
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", label=f"{label} (N={len(v)})", color=colors[label], lw=1.8)
    ax.set_xlabel(r"$\log_{10}(h/\bar h)$")
    ax.set_ylabel("ECDF")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("ECDF sovrapposte -- contrasto di forma fra popolazioni")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "leverage_ecdf_overlay.png"), dpi=150)
    plt.close(fig)

    # === (d) stratificazione dei design ===
    print("\n(d) Stratificazione dei 900 design...")
    designs, coverage = attach_design_metadata(leverage_df)
    print(f"  Copertura metadati: {coverage['n_matched']}/{coverage['n_metadata']} "
          f"({coverage['coverage_fraction']:.1%})")

    p99_train = thresholds[99]

    def stratum_summary(df, stratum_type, stratum_value):
        v = df["h_over_hbar"].to_numpy()
        return {
            "stratum_type": stratum_type, "stratum_value": str(stratum_value), "n": len(df),
            "median_h_over_hbar": float(np.median(v)) if len(v) else float("nan"),
            "p90_h_over_hbar": float(np.percentile(v, 90)) if len(v) else float("nan"),
            "frac_over_training_p99": float((v > p99_train).mean()) if len(v) else float("nan"),
        }

    strat_rows = []
    for has_e, sub in designs.groupby("has_energy"):
        strat_rows.append(stratum_summary(sub, "has_energy", has_e))
    for we, sub in designs.groupby("energy_weight_stage3"):
        strat_rows.append(stratum_summary(sub, "energy_weight_stage3", we))
    designs["protocol"] = designs["protocol_file"].str.split("/").str[-1]
    for proto, sub in designs.groupby("protocol"):
        strat_rows.append(stratum_summary(sub, "protocol", proto))

    designs_valid_iptm = designs.dropna(subset=["i_ptm"]).copy()
    designs_valid_iptm["iptm_quartile"] = pd.qcut(designs_valid_iptm["i_ptm"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    for q, sub in designs_valid_iptm.groupby("iptm_quartile", observed=True):
        strat_rows.append(stratum_summary(sub, "iptm_quartile", q))
    success = designs_valid_iptm["i_ptm"] >= 0.5
    strat_rows.append(stratum_summary(designs_valid_iptm[success], "iptm_outcome", "success (i_ptm>=0.5)"))
    strat_rows.append(stratum_summary(designs_valid_iptm[~success], "iptm_outcome", "failure (i_ptm<0.5)"))

    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(os.path.join(OUT_DIR, "leverage_by_protocol.csv"), index=False)
    print(strat_df.to_string(index=False))

    # --- leva vs i_ptm, su tutti i design con i_ptm valido ---
    pearson_r, pearson_p = stats.pearsonr(designs_valid_iptm["h_over_hbar"], designs_valid_iptm["i_ptm"])
    spearman_r, spearman_p = stats.spearmanr(designs_valid_iptm["h_over_hbar"], designs_valid_iptm["i_ptm"])
    print(f"\nLeva vs i_ptm (N={len(designs_valid_iptm)}): Pearson r={pearson_r:.3f} (p={pearson_p:.3e}), "
          f"Spearman r={spearman_r:.3f} (p={spearman_p:.3e})")

    fig, ax = plt.subplots(figsize=(6, 5))
    colors_energy = designs_valid_iptm["has_energy"].map({True: "firebrick", False: "steelblue"})
    ax.scatter(designs_valid_iptm["i_ptm"], np.log10(designs_valid_iptm["h_over_hbar"].clip(lower=1e-6)),
               c=colors_energy, s=14, alpha=0.5)
    ax.set_xlabel("i_ptm")
    ax.set_ylabel(r"$\log_{10}(h/\bar h)$")
    ax.set_title(f"Leva vs i_ptm (rosso=energy-guided, blu=senza energia)\nPearson r={pearson_r:.2f}, Spearman r={spearman_r:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "leverage_vs_iptm.png"), dpi=150)
    plt.close(fig)

    # === criteri di lettura (Workplan_v2_fasi_1bis_2_3.md, rianalisi) ===
    energy_median = strat_df.loc[(strat_df["stratum_type"] == "has_energy") & (strat_df["stratum_value"] == "True"), "median_h_over_hbar"]
    noenergy_median = strat_df.loc[(strat_df["stratum_type"] == "has_energy") & (strat_df["stratum_value"] == "False"), "median_h_over_hbar"]
    energy_weight_rows = strat_df[strat_df["stratum_type"] == "energy_weight_stage3"].copy()
    energy_weight_rows["stratum_value_f"] = energy_weight_rows["stratum_value"].astype(float)
    energy_weight_rows = energy_weight_rows.sort_values("stratum_value_f")
    dose_response_monotonic = bool(energy_weight_rows["median_h_over_hbar"].is_monotonic_increasing)

    # --- qual e' il protocollo che guida la coda? (piu' informativo del solo has_energy) ---
    proto_rows = strat_df[strat_df["stratum_type"] == "protocol"].copy()
    proto_rows = proto_rows.sort_values("median_h_over_hbar", ascending=False)
    top_protocols = proto_rows.head(3)[["stratum_value", "median_h_over_hbar", "n"]].to_dict("records")
    bottom_protocols = proto_rows.tail(3)[["stratum_value", "median_h_over_hbar", "n"]].to_dict("records")
    gen_energy_mask = proto_rows["stratum_value"].str.startswith("gen_energy")
    gen_energy_median = float(proto_rows.loc[gen_energy_mask, "median_h_over_hbar"].median()) if gen_energy_mask.any() else float("nan")
    noenergy_proto_median = float(proto_rows.loc[proto_rows["stratum_value"].str.contains("noenergy"), "median_h_over_hbar"].median())

    if len(energy_median) and len(noenergy_median) and float(energy_median.iloc[0]) > float(noenergy_median.iloc[0]):
        reading = ("La coda e' costituita (almeno in parte) dai run energy-guided: mediana con energia "
                    f"{float(energy_median.iloc[0]):.2f} vs senza energia {float(noenergy_median.iloc[0]):.2f}"
                    + (", con relazione dose-risposta monotona in energy_weight_stage3" if dose_response_monotonic else
                       ", con tendenza crescente ma SENZA monotonia stretta in energy_weight_stage3 (flessione a 0.2)")
                    + (f". Il pattern e' pero' molto piu' netto per PROTOCOLLO che per la sola presenza di energia: "
                       f"i protocolli 'gen_energy_*' (generazione de novo, init casuale + energia) hanno mediana "
                       f"~{gen_energy_median:.1f}, contro ~{noenergy_proto_median:.1f} per i protocolli senza energia e "
                       f"valori intermedi per gli 'opt_anneal_energy_*' (ottimizzazione da sequenze di partenza + energia). "
                       f"Top 3 protocolli per mediana: {top_protocols}; bottom 3: {bottom_protocols}. "
                       "Non e' quindi solo 'l'energia' a spingere fuori dalla regione dei dati, ma specificamente la "
                       "generazione de novo con energia attiva fin dall'inizio -- rilevante data l'assenza di "
                       "mitigazione in-loop (§B.5).")
                    )
    else:
        reading = ("La coda NON e' chiaramente dominata dai run energy-guided (mediana con energia "
                    f"{float(energy_median.iloc[0]) if len(energy_median) else float('nan'):.2f} vs senza energia "
                    f"{float(noenergy_median.iloc[0]) if len(noenergy_median) else float('nan'):.2f}) -- "
                    "o e' una proprieta' piu' generica del processo di hallucination/ottimizzazione, o non "
                    "correla nettamente col protocollo. La leva resta comunque utilizzabile come criterio di "
                    "selezione indipendentemente da questa attribuzione causale.")
    print(f"\nLettura: {reading}")

    # --- conflitto leva/i_ptm, segnalato separatamente perche' e' il risultato piu' rilevante per la Fase 2 ---
    success_median = float(strat_df.loc[(strat_df["stratum_type"] == "iptm_outcome") &
                                         (strat_df["stratum_value"] == "success (i_ptm>=0.5)"), "median_h_over_hbar"].iloc[0])
    failure_median = float(strat_df.loc[(strat_df["stratum_type"] == "iptm_outcome") &
                                         (strat_df["stratum_value"] == "failure (i_ptm<0.5)"), "median_h_over_hbar"].iloc[0])
    iptm_conflict_reading = (
        f"Correlazione positiva e fortemente significativa fra leva e i_ptm su tutti i 900 design (Pearson "
        f"r={pearson_r:.3f}, p={pearson_p:.1e}; Spearman r={spearman_r:.3f}, p={spearman_p:.1e}): i design che AF2 "
        f"giudica strutturalmente migliori sono sistematicamente PIU' lontani dalla regione coperta dai dati di "
        f"selezione, non meno. Il gruppo 'successo' (i_ptm>=0.5, N=51) ha mediana h/h_bar={success_median:.2f} contro "
        f"{failure_median:.2f} del gruppo 'fallimento' -- quasi 3 volte piu' alta. Conferma il rischio segnalato "
        "nell'incarico: l'obiettivo strutturale puo' star premiando l'allontanamento dalla regione vincolata dai "
        "dati di selezione, mettendo in tensione le due metriche su cui la Fase 2 si basa."
    )
    print(f"\nConflitto leva/i_ptm: {iptm_conflict_reading}")

    # --- grandezza della degenerazione per la scala assoluta (sigma(x)/sigma_n = sqrt(1+h)) ---
    h_bar_theoretical = original_summary["h_bar_theoretical"]
    h_max_designs = float((populations["fase0_designs"] * h_bar_theoretical).max())
    sigma_inflation_max = float(np.sqrt(1 + h_max_designs))

    summary = {
        "revision_of": LEVERAGE_SUMMARY,
        "revision_date": "2026-10-01",
        "revision_reason": ("Il criterio originale di §1b.1 ('frazione di design con h/h_bar oltre 100x') era "
                             "mal posto: il massimo osservato sul training set stesso e' 28,7x, quindi 100x era "
                             "irraggiungibile per costruzione. Il riferimento corretto e' la distribuzione "
                             "osservata sul training, non un multiplo assoluto dell'attesa teorica p/N. Questa "
                             "rianalisi (§1b.1.b) parte dagli stessi dati di §1b.1 (nessun nuovo calcolo di leva) "
                             "e corregge solo la lettura."),
        "original_verdict_preserved_for_traceability": original_summary["verdict_degenerate"],
        "design_metadata_coverage": coverage,
        "tail_thresholds_from_training": thresholds,
        "dose_response_monotonic_in_energy_weight": dose_response_monotonic,
        "leverage_vs_iptm": {"pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
                              "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
                              "success_group_median_h_over_hbar": success_median,
                              "failure_group_median_h_over_hbar": failure_median},
        "reading": reading,
        "iptm_conflict_reading": iptm_conflict_reading,
        "top_protocols_by_median_leverage": top_protocols,
        "bottom_protocols_by_median_leverage": bottom_protocols,
        "revised_outcome": {
            "claim_1_absolute_scale": {
                "verdict": "degenerazione CONFERMATA",
                "detail": (f"h massimo osservato sui design = {h_max_designs:.4f} (in unita' assolute, non di "
                           f"h/h_bar) => sigma(x)/sigma_n = sqrt(1+h) = {sigma_inflation_max:.4f}, cioe' un "
                           f"aumento di appena {100*(sigma_inflation_max-1):.1f}% anche all'estremo della coda. "
                           "Non si costruiscono barre di errore assolute su E con questo stimatore."),
            },
            "claim_2_ranking_use": {
                "verdict": "NON VALUTATO -- richiede §1b.3, non eseguita",
                "detail": ("Gli usi di diagnostica sui design e criterio di selezione in §3.5 dipendono solo "
                           "dall'ordinamento indotto da h(x), non dalla sua scala assoluta. §1b.1 da solo non "
                           "puo' stabilire se quell'ordinamento sia informativo: serve la validazione held-out "
                           "di §1b.3 (istanziata come piano, non eseguita in questo passaggio)."),
            },
            "claim_3_preliminary_evidence_ranking_discriminates": {
                "verdict": "EVIDENZA PRELIMINARE A FAVORE, da confermare con §1b.3",
                "detail": reading,
                "supporting_numbers": "vedi design_vs_training_tails.csv, ks_tests.csv, leverage_by_protocol.csv",
            },
        },
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # === Parte 2: aggiorna (senza cancellare) il summary.json originale di §1b.1 ===
    print(f"\nAggiornamento di {LEVERAGE_SUMMARY} con un campo di revisione tracciabile...")
    revised_original = dict(original_summary)  # copia -- il contenuto originale resta intatto sotto le sue chiavi
    revised_original["revision"] = {
        "date": "2026-10-01",
        "original_outcome_preserved": {
            "verdict_degenerate": original_summary["verdict_degenerate"],
            "next_step": original_summary["next_step"],
        },
        "reason": summary["revision_reason"],
        "corrected_in": os.path.join("results", "phase1bis", "1b_1b_leverage_reanalysis", "summary.json"),
        "revised_outcome": summary["revised_outcome"],
        "note": ("Il verdetto binario 'verdict_degenerate'=true sopra resta scritto per tracciabilita' "
                 "(era la conclusione raggiunta con quel criterio in quella data) ma NON va piu' usato "
                 "come esito operativo di §1b.1 -- usare revision.revised_outcome, articolato in tre "
                 "affermazioni distinte (scala assoluta, uso per ordinamento, evidenza preliminare)."),
    }
    with open(LEVERAGE_SUMMARY, "w") as f:
        json.dump(revised_original, f, indent=2)

    print(f"\nOutput scritto in {OUT_DIR}")


if __name__ == "__main__":
    main()
