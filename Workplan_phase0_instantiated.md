# Fase 0 — Piano istanziato (0.1–0.5)

Istanziazione operativa di [`Workplan_peptidi_PA-PB1.md`](Workplan_peptidi_PA-PB1.md) §FASE 0, sottofasi 0.1–0.5. Per ciascuna: dati di input con path verificati nel repo, script da creare, output atteso, decisioni aperte.

**Perimetro:** nessuna delle sottofasi 0.1–0.5 esegue ColabDesign (nessun `AfDesign`, nessuna GPU). Il forward del modello di energia però **va importato dal fork ColabDesign modificato**, non reimplementato: `colabdesign.energy_model.model_3layer` (fork in `/home/guido/Projects/protein_design/methods/colabdesign_energy_guidance`, dipendenza editable già dichiarata in `pyproject.toml`, quindi `import colabdesign` funziona out-of-the-box con `uv run`) espone:

```python
from colabdesign.energy_model.model_3layer import load_energy_model, mlp_forward
```

- `load_energy_model(weights_path) -> params` — carica `data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json` (pesi già nella shape corretta per `x @ W`).
- `mlp_forward(params, x) -> scalar` — forward JAX puro (differenziabile, `jax.grad` diretto), `x` shape `(1, 15, 21)` in **ordine Julia** (`ACDEFGHIKLMNPQRSTVWY-`, 20 AA + gap).

Il modulo espone anche `make_energy_fn(weights_path, energy_weight, binder_len)`, ma è il wrapper pensato per l'integrazione con ColabDesign (si aspetta `seq_probs` in ordine AF2 per l'intera sequenza target+binder, applica una permutazione di colonne verso l'ordine Julia e appende la colonna gap a zero): per 0.1/0.4, dove gli input si costruiscono direttamente da stringhe, conviene chiamare `load_energy_model` + `mlp_forward` senza passare da quel wrapper. Manca solo un encoder `seq: str -> np.ndarray (15,21)` in ordine Julia (poche righe, va in `data_loading.py`) — non esiste ancora fuori da ColabDesign.

Nota sull'alfabeto: la 21ª colonna (gap) è sempre azzerata in produzione (`make_energy_fn` la appende come zero, "mai presente in input reale del binder design"). Per 0.1, campionare quindi il Dirichlet standard su 20 categorie con la colonna gap fissata a zero, non su 21 — replica esattamente cosa il modello vede in uso reale.

**Validazione consigliata prima di usare l'encoder in 0.1/0.4:** `results/colabdesign/custom/seqs_aff_alberto_darren_6best.json` contiene energie già calcolate dal vero modello (via `make_energy_fn`) per 6 sequenze, es. `MDFNPWLLFLKVPAQ → energy = -1.0169219970703125`. Ricalcolarle con `mlp_forward(load_energy_model(...), encode(seq))` e verificare accordo entro `1e-4` — è un buon test end-to-end che l'encoder diretto in ordine Julia sia equivalente al percorso via ColabDesign (permutazione AF→Julia compresa).

**Layout proposto (da confermare):**
```
src/phase0_diagnostics/
├── data_loading.py         # loader condivisi (training set NGS, BLI, design esistenti) + encoder one-hot ordine Julia
├── run_0_1_simplex_variance.py
├── run_0_2_clustering.py
├── run_0_3_bli_composition.py
├── run_0_4_gradient_direction.py
└── run_0_5_structural_context.py

results/phase0/
├── 0_1_simplex_variance/
├── 0_2_clustering/
├── 0_3_bli_composition/
├── 0_4_gradient_direction/
└── 0_5_structural_context/
```
Alternativa: notebook in `src/analysis/phase0/` invece di script in `src/`. Preferisco script perché 0.1/0.2/0.4 sono riusati come moduli da fasi successive (2.1, 2.4, 2.6), ma è una scelta reversibile.

---

## 0.1 — Varianza di E nell'interno del simplesso

**Input:**
- `colabdesign.energy_model.model_3layer.{load_energy_model, mlp_forward}` + encoder in `data_loading.py` (vedi sopra)
- Training set per stimare $\min_x E(x)$, $\max_x E(x)$, $\sigma_{\text{train}}$: sequenze osservate, **non** le sequenze disegnate. Vedi §0.2 sotto per l'esatta provenienza (`PD_energy_model/data/4_counts/{F,R}_{2,3}_count_protein.csv`).

**Script da creare: `src/phase0_diagnostics/run_0_1_simplex_variance.py`**
1. Carica i pesi (`load_energy_model`) e il training set one-hot (via `data_loading.py`, vedi 0.2).
2. Calcola `E_min, E_max, sigma_train` sul training set (one-hot, `mlp_forward` vettorizzato con `jax.vmap`).
3. Campiona $10^4$ punti Dirichlet$(\alpha \mathbf 1_{20})$ indipendenti per posizione, $\alpha$ log-spaziato $[0.05, 50]$, con la 21ª colonna (gap) fissata a zero — replica esattamente cosa vede il modello in produzione (vedi nota sull'alfabeto sopra).
4. `E(a)` via `mlp_forward` in batch.
5. Frazione di violazioni $E(a) \notin [E_{\min}, E_{\max}]$, ampiezza in unità di $\sigma_{\text{train}}$, scatter $E(a)$ vs $E(\text{one-hot}(\arg\max a))$ colorato per $\alpha$, curva violazione mediana vs $\alpha$.

**Output:** `results/phase0/0_1_simplex_variance/{metrics.json, scatter.png, violation_vs_alpha.png}`

**Stato: eseguita (2026-08-10).** Script scritti: `src/phase0_diagnostics/data_loading.py` (encoder + loader), `src/phase0_diagnostics/run_0_1_simplex_variance.py`. Eseguita con `JAX_PLATFORMS=cpu uv run python3 src/phase0_diagnostics/run_0_1_simplex_variance.py` — su questa macchina il driver NVIDIA è troppo vecchio per JAX-CUDA, forzare CPU è comunque coerente col perimetro CPU-only di Fase 0 (~15s di run).

**Risultati:**
- **Validazione encoder OK**: errore massimo 8×10⁻⁷ contro le energie già calcolate da ColabDesign sulle 6 sequenze BLI (soglia 10⁻⁴) — l'ordine Julia/flattening/trasposizione dei pesi è corretto.
- **Training set**: 27.342 sequenze valide (unione F2+F3+R2+R3), 64 scartate per residui ambigui (es. `_`). `E_min=-3.71`, `E_max=14.62`, `σ_train=2.09`.
- **Violazioni del bound sul simplesso, su 10.000 punti Dirichlet**: solo **0.02%** (2 su 10.000), e compaiono **esclusivamente all'α più piccola testata** (~0.065, punti quasi-vertice), ampiezza minima (~0.2σ, vedi `violation_vs_alpha.png`).
- **Questo è l'opposto di quanto ipotizzato dal piano originale** ("Attesa: violazioni massive a α grande"). Lo scatter (`scatter.png`) mostra `E(a)` **collassare** in una banda stretta (~6–8) man mano che α cresce — punti vicini al baricentro del simplesso restano ben **dentro** `[E_min, E_max]`, non esplodono.

**Interpretazione (da discutere, non conclusiva):** il campionamento Dirichlet uniforme non riproduce l'esplosione osservata nei run di design reali (da −5 a −189 in stage 2). L'esplosione è probabilmente **avversariale** — trovata dal gradiente lungo una direzione specifica ad alta curvatura (il meccanismo già ipotizzato in §0.7 del piano originale), non una proprietà generica dei punti interni del simplesso. Il campionamento casuale sotto-rileva il problema di calibrazione; la traiettoria reale dell'ottimizzatore probabilmente lo rileva. **Implicazione operativa**: valutare se anticipare 0.7 (replay delle traiettorie) rispetto all'ordine originale, o quantomeno arricchire il set di calibrazione di 2.1 con le traiettorie reali fin dal primo giro (il piano lo prevede già come opzione in §2.1 "Set di calibrazione", punto 2 — qui c'è evidenza diretta che serve, non è solo prudenza).

---

## 0.2 — Taglia effettiva del dataset e clustering

**Input — chiarito leggendo il notebook di training (`model_training_7_...ipynb`):** il modello finale è addestrato sull'unione di round 2 e round 3, forward+reverse:
```
PD_energy_model/data/4_counts/F_2_count_protein.csv
PD_energy_model/data/4_counts/F_3_count_protein.csv
PD_energy_model/data/4_counts/R_2_count_protein.csv
PD_energy_model/data/4_counts/R_3_count_protein.csv
```
(colonne `Sequence,Count`; il notebook fa `merge_counts` fra F2+F3 e fra R2+R3 — riprodurre in pandas come concat + `groupby("Sequence").sum()`). Le varianti `*_TAG_protein.csv` nella stessa cartella sono un'altra popolazione (letture con tag/stop codon) — da tenere separate, non incluse nel training set del modello attuale salvo verifica ulteriore.

`PD_energy_model/data/protein_counts_123/` contiene in più `F_1`/`R_1` (round 1) rispetto a `4_counts/`: **non fa parte del training set del modello a 3 layer**, ma è l'unico proxy disponibile per "libreria iniziale" in 0.3′ (con la caveat che è già post-round-1, non la libreria naive).

**"Sequenze osservate nei round tardi"** (per il clustering, non per il training set completo): round 3 (`F_3` + `R_3`) è il round più tardo disponibile; round 2 è quello immediatamente precedente. Decisione presa: usare **round 3 unito F+R** come "tardo" per 0.2, e produrre la stessa statistica anche su round 2 per confronto (il costo è marginale).

**Script da creare: `src/phase0_diagnostics/run_0_2_clustering.py`**
1. Carica e unisce i conteggi (funzione condivisa in `data_loading.py`, riusata anche da 0.1/0.3′/0.4).
2. Clustering al 70% identità. **MMseqs2/CD-HIT non sono installati su questa macchina** (verificato: nessuno dei due binari è in `PATH`) — da installare (modulo cluster CEA AAR, o `conda install -c bioconda mmseqs2`) oppure, dato che sono 15-meri molto corti, usare un fallback puro Python: distanza di Hamming/Levenshtein + `scipy.cluster.hierarchy` con soglia equivalente. **Decisione da confermare**: MMseqs2 se disponibile sul cluster, altrimenti il fallback — impatta la riproducibilità quindi va deciso prima di girare il conto definitivo, ma per una prima stima il fallback è sufficiente.
3. $N_{\text{eff}} = \sum_s 1/m_s$; istogramma dimensioni cluster; curva di Lorenz della massa di read; curva di copertura (cluster minimi per 50%/90% della massa nell'ultimo round).
4. **Salva l'assegnazione cluster→sequenza su disco** (`results/phase0/0_2_clustering/cluster_assignment.parquet` o `.csv`): è un prerequisito dichiarato di 2.3 e 2.5, non solo un output diagnostico di questa fase.

**Output:** `results/phase0/0_2_clustering/{n_eff.json, cluster_sizes.png, lorenz_curve.png, cluster_assignment.csv}`

---

## 0.3′ — Composizione amminoacidica dei leganti validati in BLI

**Input:**
- **Sequenze con $K_D$ misurato**: `data/affinity_measurements_PA-PB1_formatted.csv` — colonne `Name, Sequence, Kd_nM, Kd_Error, Kon_M-1s-1, Koff_s-1, ...` (copia identica anche in `PD_energy_model/data/`; usare quella in `data/` perché già tracciata in questo repo). Nota: le stesse 39 sequenze compaiono senza affinità in `data/seqs_aff_alberto_darren.txt`, e il sottoinsieme dei 6 migliori in `data/seqs_aff_alberto_darren_6best.txt`/`results/colabdesign/custom/seqs_aff_alberto_darren_6best.json` (quest'ultimo ha già `energy` calcolata dal vero modello — riusabile come cross-check, non da ricalcolare).
- **WT**: `MDVNPTLLFLKVPAQ` (già nel piano originale, nessun file — hardcoded).
- **Libreria iniziale**: round 1, `PD_energy_model/data/protein_counts_123/{F_1,R_1}_count_protein.csv` (vedi caveat in 0.2: non è la libreria naive, è post-round-1).
- **Training set stratificato per arricchimento**: stesso set di 0.2 (`4_counts/`), con selettività/arricchimento per sequenza calcolabile come già fatto nel notebook Julia (`selF`, `selR` — rapporto conteggi round3/round2 con pseudocount) oppure, più semplice in Python, direttamente il conteggio normalizzato in round 3 come proxy di arricchimento.

**Script da creare: `src/phase0_diagnostics/run_0_3_bli_composition.py`**
1. Carica le 4 popolazioni (BLI, WT, libreria iniziale, training set stratificato in decili di arricchimento).
2. Composizione amminoacidica per popolazione (frequenza per lettera, 20 categorie — qui niente gap, sono tutte sequenze reali).
3. Stratificare i BLI per fascia di $K_D$ (es. quartili) — occhio a `Kd_Error` mancante su quasi tutte le righe della tabella (formato "scouting", poche misurazioni): la stratificazione fine potrebbe non essere robusta con **N≈39** sequenze; dichiararlo nel risultato invece di sovrainterpretare quartili su campioni piccoli.
4. Test di arricchimento per amminoacido (es. binomiale o Fisher esatto contro la composizione della libreria iniziale).

**Output:** `results/phase0/0_3_bli_composition/{composition_table.csv, enrichment_test.csv}`

**Costo:** ~10 minuti come da piano originale — è lo script più semplice dei cinque, nessuna dipendenza da 0.0.

---

## 0.4 — Direzione del gradiente di E

**Input:**
- `colabdesign.energy_model.model_3layer.{load_energy_model, mlp_forward}` — `mlp_forward` è JAX puro, quindi `jax.grad` funziona direttamente senza altro lavoro.
- Test set: campione di sequenze one-hot dal training set (`4_counts/`, stesso loader di 0.2/0.3′). Il piano dice "test set" — dato che qui non esiste ancora uno split train/test formale (arriverà in 2.3/2.5), usare per ora un campione casuale del training set osservato, con nota esplicita che non è un vero held-out; il set BLI (39 sequenze) può fare da secondo controllo indipendente ma è troppo piccolo per una media stabile su 15×20 posizioni.

**Script da creare: `src/phase0_diagnostics/run_0_4_gradient_direction.py`**
1. Per ogni sequenza one-hot del campione, `jax.grad(mlp_forward, argnums=1)` rispetto a `x`, valutato nel punto one-hot (il gradiente della MLP in un vertice del simplesso, ben definito perché la rete è liscia — nessun problema di derivabilità qui, a differenza di 2.1).
2. Media su tutte le sequenze e posizioni → ranking dei 20 amminoacidi.
3. Media solo sulle posizioni, non sulle sequenze → mappa 15×20 non mediata.

**Output:** `results/phase0/0_4_gradient_direction/{aa_ranking.csv, position_map.png}`

**Dipendenza:** condivide pesi ed encoder con 0.1 — nessun dato nuovo da caricare oltre al training set già usato in 0.2.

---

## 0.5 — Composizione dei design esistenti, per contesto strutturale

**Input:**
- **Sequenze disegnate**: tutti i JSON in `results/colabdesign/{run1_protocols,run2_wide,run_multimer,run_multimer2}/*.json` — campo `results[].seq`, 15 caratteri, nessuna coordinata 3D salvata (verificato: i JSON contengono solo `seq, loss_*, energy, i_ptm, ptm, plddt`, mai un path a un PDB).
- **Classificazione interfaccia/esposto per posizione**: il piano originale dice "usando le strutture AF2 già generate" — **queste non esistono su disco** (nessuno script di design salva un PDB per seme). L'alternativa concreta trovata nel repo: `data/pdbs/2ZNL.pdb` contiene **sia** la catena target A (residui 257–716, include tutti i 27 hotspot 408–714) **sia** la catena B, che è il **frammento N-terminale nativo di PB1 legato a PA — esattamente i 15 residui del binder canonico** (`MDVNPTLLFLKVPAQ`, numerati 1–15 in catena B). È la struttura del complesso reale, non un modello.

  **Decisione proposta**: classificare le 15 posizioni come interfaccia/esposte **una sola volta**, dalla geometria catena B↔catena A in `2ZNL.pdb` (contatti sotto soglia di distanza per "interfaccia", SASA relativa per "esposta"), poi applicare questa classificazione posizionale a tutte le sequenze disegnate per indice (posizione 1 del design ↔ posizione 1 di catena B, ecc.), assumendo che ColabDesign mantenga il registro nativo (stesso `binder_len=15` e stesso target/hotspot). È una semplificazione rispetto al testo del piano (non cattura variazioni di registro *per-design*), ma è l'unica opzione senza rigenerare strutture — e per lo scopo di 0.5 (causa AF2 vs causa energia sul bias aromatico *in media*) è adeguata. Se si vuole la versione forte (registro per-design), serve rifoldare un campione con AF2/Boltz, il che richiede GPU e ricade fuori dal perimetro CPU-only di Fase 0 — da eventualmente spostare in Fase 4.

**Script da creare: `src/phase0_diagnostics/run_0_5_structural_context.py`**
1. Parsing di `2ZNL.pdb` (Biopython `PDBParser`, già in `pyproject.toml`), estrazione catena A (257–716) e catena B (1–15).
2. Per ciascuna delle 15 posizioni di catena B: (a) contatto = qualunque atomo entro soglia (es. 5 Å, da confermare) da un atomo di catena A → "interfaccia"; (b) SASA relativa (Shrake-Rupley, `Bio.PDB.SASA` o `freesasa` se disponibile) calcolata sul complesso vs sulla catena B isolata → "esposta" se alta.
3. Carica tutte le sequenze disegnate (`data_loading.py`, unione dei JSON in `results/colabdesign/`).
4. Frazione aromatica (F/W/Y) per classe di posizione (interfaccia vs esposta), confrontata con WT, libreria iniziale (round 1, come in 0.3′), training set.

**Output:** `results/phase0/0_5_structural_context/{position_classification.csv, aromatic_fraction_by_class.csv, summary.png}`

**Dipendenza aggiuntiva**: `freesasa` non è nelle dipendenze attuali di `pyproject.toml` (solo `biopython`, che include un calcolo SASA proprio via `Bio.PDB.SASA.ShrakeRupley` — sufficiente, non serve aggiungere `freesasa` come dipendenza esterna).

---

## Riepilogo dipendenze fra sottofasi

```
encoder + colabdesign.energy_model.model_3layer ──┬──> 0.1
                                                     └──> 0.4

0.2 (loader training set + clustering) ──> usato da 0.1, 0.3′, 0.4 (stesso data_loading.py)

0.3′  indipendente (solo CSV BLI + conteggi round 1)
0.5   indipendente dal modello di energia e da 0.2 (usa 2ZNL.pdb + i JSON dei design)
```

Ordine di implementazione consigliato: **encoder + validazione (§Perimetro) → 0.3′ (rapido, valida i dati BLI) → 0.2 (loader condiviso + clustering) → 0.1 → 0.4 → 0.5**. 0.3′ prima di 0.2 perché è il rapporto informazione/costo più alto del piano originale ed è del tutto disaccoppiato dal resto — buon primo risultato da vedere prima di investire nel loader condiviso.

## Decisioni aperte da confermare prima di scrivere codice

1. **MMseqs2 vs fallback Python** per 0.2 — dipende da cosa è installabile sul cluster AAR.
2. **Soglia di distanza per "contatto interfaccia"** in 0.5 (proposta: 5 Å heavy-atom, standard ma da confermare).
3. **Layout cartelle**: `src/phase0_diagnostics/` + `results/phase0/` come proposto sopra, oppure integrare nei notebook esistenti sotto `src/analysis/`.
