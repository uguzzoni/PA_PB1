# Fase 0 — Piano istanziato (0.1–0.8)

Istanziazione operativa di [`Workplan_peptidi_PA-PB1.md`](Workplan_peptidi_PA-PB1.md) §FASE 0 (versione 2, 12 agosto 2026), sottofasi 0.1–0.8. Per ciascuna: dati di input con path verificati nel repo, script da creare, output atteso, decisioni aperte.

**Nota di versione (aggiornata 2026-08-26).** §0.1–0.5 eseguite l'11 agosto; §0.6–0.8 (aggiunte nella versione 2 del piano madre, 12 agosto) eseguite fra il 12 e il 26 agosto — **l'intera Fase 0 è ora completa**, 0.8.d esclusa (§0.8.d: deciso di chiedere direttamente al gruppo sperimentale invece di inferire lo schema di codoni da dati post-selezione). La versione precedente di questo file (solo 0.1–0.5) è conservata in [`Workplan_phase0_instantiated_v1_old.md`](Workplan_phase0_instantiated_v1_old.md).

**Perimetro:** nessuna delle sottofasi 0.1–0.5 esegue ColabDesign (nessun `AfDesign`, nessuna GPU). Il forward del modello di energia però **va importato dal fork ColabDesign modificato**, non reimplementato: `colabdesign.energy_model.model_3layer` (fork in `/home/guido/Projects/protein_design/methods/colabdesign_energy_guidance`, dipendenza editable già dichiarata in `pyproject.toml`, quindi `import colabdesign` funziona out-of-the-box con `uv run`) espone:

```python
from colabdesign.energy_model.model_3layer import load_energy_model, mlp_forward
```

- `load_energy_model(weights_path) -> params` — carica `data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json` (pesi già nella shape corretta per `x @ W`).
- `mlp_forward(params, x) -> scalar` — forward JAX puro (differenziabile, `jax.grad` diretto), `x` shape `(1, 15, 21)` in **ordine Julia** (`ACDEFGHIKLMNPQRSTVWY-`, 20 AA + gap).

Il modulo espone anche `make_energy_fn(weights_path, energy_weight, binder_len)`, ma è il wrapper pensato per l'integrazione con ColabDesign (si aspetta `seq_probs` in ordine AF2 per l'intera sequenza target+binder, applica una permutazione di colonne verso l'ordine Julia e appende la colonna gap a zero): per 0.1/0.4, dove gli input si costruiscono direttamente da stringhe, conviene chiamare `load_energy_model` + `mlp_forward` senza passare da quel wrapper. Manca solo un encoder `seq: str -> np.ndarray (15,21)` in ordine Julia (poche righe, va in `data_loading.py`) — non esiste ancora fuori da ColabDesign.

Nota sull'alfabeto: la 21ª colonna (gap) è sempre azzerata in produzione (`make_energy_fn` la appende come zero, "mai presente in input reale del binder design"). Per 0.1, campionare quindi il Dirichlet standard su 20 categorie con la colonna gap fissata a zero, non su 21 — replica esattamente cosa il modello vede in uso reale.

**Validazione consigliata prima di usare l'encoder in 0.1/0.4:** `results/colabdesign/custom/seqs_aff_alberto_darren_6best.json` contiene energie già calcolate dal vero modello (via `make_energy_fn`) per 6 sequenze, es. `MDFNPWLLFLKVPAQ → energy = -1.0169219970703125`. Ricalcolarle con `mlp_forward(load_energy_model(...), encode(seq))` e verificare accordo entro `1e-4` — è un buon test end-to-end che l'encoder diretto in ordine Julia sia equivalente al percorso via ColabDesign (permutazione AF→Julia compresa).

**Layout (0.1–0.5 eseguite così, 0.6–0.8 proposti coerenti):**
```
src/phase0_diagnostics/
├── data_loading.py              # loader condivisi + encoder one-hot ordine Julia (esteso ad ogni sottofase)
├── run_0_1_simplex_variance.py
├── run_0_2_clustering.py
├── run_0_3_bli_composition.py
├── run_0_4_gradient_direction.py
├── run_0_5_structural_context.py
├── run_0_7_trajectory_gap.py    # da scrivere — vedi §0.7
├── run_0_8a_partial_correlation.py     # da scrivere — vedi §0.8.a
├── run_0_8b_design_stratification.py   # vedi §0.8.b (riusa la classificazione di 0.5)
├── run_0_8b_hotspot_proximity.py       # vedi §0.8.b — estensione, classificazione su HOTSPOT_RESIDUES fisse
├── run_0_8c_gap_composition.py         # da scrivere — vedi §0.8.c (dipende da 0.7)

results/phase0/
├── 0_1_simplex_variance/
├── 0_2_clustering/
├── 0_3_bli_composition/
├── 0_4_gradient_direction/
├── 0_5_structural_context/
├── 0_7_trajectory_gap/
└── 0_8_closing_checks/           # 0.8a/b/c/d condividono la cartella, file prefissati
```
§0.6 (screening TUP) **non produce uno script**: è una procedura manuale su web server esterno (SAROTUP), vedi sotto — l'unico artefatto generato in questo repo è l'elenco di sequenze da sottoporre, già estratto da `cluster_assignment.csv`.

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

**Stato: eseguita (2026-08-11).** Script scritti: `src/phase0_diagnostics/data_loading.py` (aggiunti `load_round2_counts`/`load_round3_counts`, entrambi via helper condiviso `_merge_counts` — anche `load_round1_counts`/`load_training_counts` refactored su di esso), `src/phase0_diagnostics/run_0_2_clustering.py`. Eseguita con `uv run python3 src/phase0_diagnostics/run_0_2_clustering.py` (~4s, CPU, nessun bisogno di JAX qui).

**Decisione aperta #1 risolta per questo run:** MMseqs2/CD-HIT confermati assenti (`which mmseqs cd-hit` vuoto). Implementato fallback greedy stile CD-HIT invece di hierarchical clustering scipy: per round 2 (N=23.677) una condensed distance matrix in float64 peserebbe ~2.2GB, al limite degli 8GB liberi su questa macchina (`free -h`: 15GB totali, ~8GB disponibili). Il greedy (ordina per abbondanza decrescente, assegna al rappresentante esistente più vicino in Hamming se a distanza ≤4/15 posizioni, altrimenti apre un nuovo cluster) evita la matrice N×N confrontando solo contro i rappresentanti già aperti: 0.38s per round 3, 3.5s per round 2. Soglia 70% identità → ≤4 mismatch su 15 posizioni (Hamming diretto, sequenze già a lunghezza fissa, nessun allineamento necessario).

**Nota sulla formula N_eff:** $N_{\text{eff}} = \sum_s 1/m_s$ con $s$ sulle sequenze uniche e $m_s$ la dimensione del cluster di $s$ collassa algebricamente al numero di cluster (ogni cluster di dimensione $m$ contribuisce $m \cdot (1/m) = 1$) — implementato come tale, con nota esplicita nel codice.

**Risultati:**
| | round 3 (F+R, primario) | round 2 (F+R, confronto) |
|---|---|---|
| sequenze uniche | 6.154 | 23.677 |
| reads totali | 32.873 | 81.068 |
| **N_eff (cluster @70%)** | **1.375** | **3.200** |
| cluster singleton | 769 (56%) | 1.513 (47%) |
| cluster più grande | 945 sequenze / 11.232 reads (34% della massa) | 2.229 sequenze |
| cluster per 50% massa | 2 | 3 |
| cluster per 90% massa | 19 | 177 |

- **Collasso di diversità forte da round 2 a round 3**: N_eff scende da 3.200 a 1.375 (-57%) nonostante meno sequenze uniche di partenza siano già un fattore (23.677→6.154, -74%) — il rapporto N_eff/N_unique cresce leggermente (13.5%→22.3%), cioè il collasso di round 3 non è solo "meno reads sequenziate" ma una vera contrazione post-selezione.
- **Massa di read fortemente concentrata in pochissimi cluster** in entrambi i round (curva di Lorenz molto lontana dalla diagonale): in round 3 un solo cluster (945 varianti quasi-identiche, verosimilmente attorno a un binder dominante) copre il 34% di tutti i reads; 2 cluster bastano per il 50%. Questo è coerente con una selezione di phage display convergente, non con un pool ancora diverso.
- **Implicazione operativa per 2.3/2.5** (che dichiarano `cluster_assignment.csv` come prerequisito): la "taglia effettiva" del dataset di training per il modello di energia è molto più piccola dei 27.342 conteggi grezzi usati in 0.1 — se il training/validation split in 2.3/2.5 non pesa o non deduplica per cluster, rischia di sovra-rappresentare le ~1-2 varianti dominanti di round 3 come se fossero centinaia di osservazioni indipendenti. Da tenere presente quando si disegna lo split.
- `cluster_assignment.csv` (round 3) e `cluster_assignment_round2.csv` (round 2, extra non richiesto esplicitamente ma a costo marginale) salvati con colonne `Sequence, Count, cluster_id, cluster_size, cluster_read_mass, is_representative`.

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

**Stato: eseguita (2026-08-11).** Script scritto: `src/phase0_diagnostics/run_0_3_bli_composition.py`. Eseguita con `uv run python3 src/phase0_diagnostics/run_0_3_bli_composition.py` (pochi secondi, CPU).

**Scoperta non ovvia dal piano originale**: il CSV BLI ha 39 **righe** ma solo **28 sequenze uniche** — molte righe sono ri-misurazioni della stessa sequenza in round sperimentali diversi (scouting preliminare → best-hits → side-by-side finale → confronto MPNN). Dopo dedup per sequenza (media geometrica del Kd sulle misure valide ripetute), esclusione delle 4 righe con Kd_nM=NA (P7, una riga di P10, MPNN_06, MPNN_07 — nessun segnale misurabile) ed esclusione delle righe "WT" (trattata come popolazione a sé): **N effettivo = 23 sequenze disegnate**, non 39. La stratificazione per quartile di Kd è quindi ancora più risicata di quanto stimato nel piano (~5-6 per quartile su N=23, non su N=39) — riportata ma esplicitamente etichettata come puramente descrittiva.

**Anomalia nei dati grezzi, non risolta silenziosamente**: il nome "WT" nel CSV è assegnato a **due sequenze diverse** — `MDVNPTLLFLKLPAQ` (2 righe) e `MDVNPTLLFLKVPAQ` (1 riga, posizione 12: L vs V), quest'ultima è quella hardcoded nel piano. Kd simili tra le 3 righe (13.6–33.8 nM) suggeriscono un typo di trascrizione, non due costrutti distinti — segnalato in `population_notes.json`, la composizione WT usa solo la sequenza del piano.

**Risultati:**
- **Bias aromatico crescente lungo tutta la catena di selezione**, non solo nel training set: frazione aromatica (F+W+Y) WT=6.7% → libreria round1=15.9% → decili di arricchimento (range 9–20%, decile 9 il più arricchito=20.0%) → BLI validati (tutti i quartili di Kd, 19–21%). Vedi `aromatic_fraction_by_population.png`. Conferma indipendente (via dati BLI reali, non solo il modello di energia) del bias già ipotizzato in 0.1 e nell'interpretazione del piano originale.
- **G e C completamente assenti nei 23 leganti BLI validati** (0/345 residui, tutti i quartili di Kd inclusi) — molto più forte della libreria iniziale (G=3.0%, C=0.08%) e coerente col decile 9 del training set (G=0.44%, C=0.03%, quasi-azzerato ma non zero). Plausibile: G/C sfavoriti sia dalla selezione di binding sia da vincoli sperimentali BLI (Cys libere → aggregazione/legami disolfuro spuri).
- **Test di arricchimento (Fisher esatto per amminoacido, BLI/decile-9 vs libreria round1, BH-corretto)**: 23/40 test significativi a p_BH<0.05. Y (tirosina) è l'arricchimento più forte e più robusto in entrambe le popolazioni (odds ratio 5.6 in BLI, 5.8 nel decile 9, p_BH≈0 per il decile, p_BH=3e-7 per BLI) — singolo amminoacido con il segnale più pulito.
- **La frazione aromatica non discrimina i quartili di Kd** (BLI_Q1_tightest=20.0% vs BLI_Q4_weakest=18.9%, differenza minima e nell'ordine del rumore per N≈6/quartile): il bias aromatico sembra una firma di "cosa sopravvive alla selezione/al filtro BLI" più che di "quanto stringe" tra i leganti già validati — coerente con l'ipotesi che l'energy model catturi un effetto di enrichment/druggability generico, non necessariamente l'affinità fine.
- Proxy di arricchimento scelto (alternativa "più semplice" indicata dal piano): abbondanza normalizzata in round 3 (0 se la sequenza non compare in round 3 — è la maggioranza: overlap round2∩round3 = 2.425 su 23.677+6.154 sequenze uniche). Decili forzati a numerosità uguale via rank invece di qcut diretto sul valore, perché ~78% delle sequenze ha proxy=0.
- File aggiuntivo `population_notes.json` (non nella lista output del piano, aggiunto per audit trail): righe escluse, discrepanza WT, sequenze scartate per stop-codon/ambiguità in libreria (18/4789) e training set (1076/27406) — qui le sequenze con stop codon sono escluse del tutto (a differenza di 0.1 dove il canale gap è input valido per il modello di energia), perché per una composizione amminoacidica un read troncato non è un 15-mero reale.

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

**Stato: eseguita (2026-08-11).** Script scritto: `src/phase0_diagnostics/run_0_4_gradient_direction.py`. Eseguita con `JAX_PLATFORMS=cpu uv run python3 src/phase0_diagnostics/run_0_4_gradient_direction.py` (~10s CPU). Refactor collaterale: `load_bli_population` (definita in 0.3') spostata in `data_loading.py` per essere riusata qui come secondo controllo indipendente, invece di duplicarla o importarla da uno script eseguibile.

**Metodo:** `jax.grad(mlp_forward, argnums=x)` valutato in ciascuno dei 27.342 vertici one-hot del training set (stesso set di 0.1/0.2) e, separatamente, nei 23 vertici BLI validati (0.3'). Nessuna proiezione sul simplesso: è la derivata parziale non vincolata, coerente con come ColabDesign perturba i logit soft durante l'ottimizzazione — non un gradiente "riprogettato" per rispettare il vincolo di somma-1.

**Risultati:**
- **W (triptofano) è l'unico amminoacido con gradiente medio negativo** su tutta la popolazione training (-0.17, cioè aumentarne la probabilità da un vertice osservato riduce E in media) — quasi tutti gli altri 19 hanno gradiente medio positivo (aumentarli aumenta E). Nel controllo BLI (N=23) anche S risulta marginalmente negativo, ma è entro il rumore (std comparabile alla media). **D (aspartato) è il più evitato** in entrambe le popolazioni.
- **Forte accordo fra le due popolazioni indipendenti**: correlazione di Spearman fra ranking training e ranking BLI = **0.86** — la direzione del gradiente non è un artefatto del training set, si riproduce sulle sequenze realmente validate in laboratorio.
- **Interpretazione strutturale, collegata a 0.1**: che quasi tutti gli amminoacidi abbiano gradiente positivo (aumentarli aumenta E) significa che i vertici one-hot osservati nel training set sono già vicini a un minimo locale di E lungo quasi tutte le direzioni — coerente con la scoperta di 0.1 che i punti interni del simplesso (verso il baricentro) collassano in una banda di E più bassa e stretta, non esplodono. L'unica "via di fuga" verso E più basso è la direzione W — che è anche l'amminoacido più arricchito nella composizione dei leganti validati (0.3'). I due risultati si rinforzano a vicenda: il modello di energia premia strutturalmente la direzione aromatica, e la selezione sperimentale sembra averla effettivamente seguita.
- **Attenzione a non sovrainterpretare il ranking pooled**: `position_map.png` mostra che l'effetto non è omogeneo per posizione — es. il segnale "C favorito" nel ranking aggregato è dominato quasi interamente dalla posizione 15 (gradiente ≈ -3σ lì, vicino a zero altrove), non un effetto generico di C su tutte le 15 posizioni. Il ranking in `aa_ranking.csv` è una media utile per un primo confronto ma va letto insieme alla mappa posizionale prima di trarre conclusioni per singolo amminoacido.
- **Caveat sui campioni rari**: C è solo l'1.06% dei residui osservati nel training set (H e T ancora più rari, 0.27%/0.64%) — il suo gradiente medio potrebbe riflettere in parte una stima meno affidabile in una regione poco campionata dello spazio, non necessariamente un vero minimo locale di E. Non c'è modo di distinguere i due casi senza dati aggiuntivi (es. sequenze sintetiche mirate), qui solo segnalato.

**Output:** `results/phase0/0_4_gradient_direction/{aa_ranking.csv, aa_ranking.png, position_map.png, metrics.json}` (`aa_ranking.png` e `metrics.json` non nella lista del piano originale, aggiunti per completezza — costo marginale).

---

## 0.5 — Composizione dei design esistenti, per contesto strutturale

**Aggiornamento (2026-08-11): il presupposto originale è superato.** Il piano diceva "usando le strutture AF2 già generate — queste non esistono su disco" e proponeva come unica alternativa la geometria statica di `2ZNL.pdb`, applicata per indice a tutti i design assumendo un registro nativo mai verificato. **Ora esistono strutture predette reali**: `af3_predictions/` (root del repo, git-ignored come `results/af3`/`results/boltz`, con tarball di backup `af3_predictions.tar.gz`) contiene **35 predizioni AF3 del complesso completo** (target + binder) per un set curato di "peptidi promettenti" — non l'intera popolazione disegnata, ma un sottoinsieme selezionato a valle dell'analisi (vedi sotto). Questo permette di sostituire l'assunzione di registro nativo con una verifica diretta, almeno per questo sottoinsieme.

**Input:**
- **Sequenze disegnate (popolazione completa)**: tutti i JSON in `results/colabdesign/{run1_protocols,run2_wide,run_multimer,run_multimer2}/*.json` — campo `results[].seq`, come nel piano originale. Resta la base per l'analisi di composizione su larga scala (step 4 sotto), perché `af3_predictions/` copre solo 34 sequenze uniche, una piccola frazione del totale disegnato.
- **Predizioni strutturali reali (sottoinsieme "promettente")**: `af3_predictions/fold_<data>_<seq-lowercase-o-uppercase>/`, 35 cartelle. Ciascuna contiene 5 modelli/seed AF3 (`*_model_{0..4}.cif`), le confidenze riassuntive per modello (`*_summary_confidences_{0..4}.json`: `iptm`, `ptm`, `chain_pair_iptm`, `chain_pair_pae_min`, `has_clash`, `fraction_disordered`, `ranking_score`), i dati completi per modello (`*_full_data_{0..4}.json`: `contact_probs` e `pae` come matrici NxN a livello di token/residuo, `token_chain_ids`, `token_res_ids`, `atom_plddts`, `atom_chain_ids` — **niente parsing di coordinate 3D necessario per la classificazione interfaccia/esposto**, `contact_probs` la dà già token-per-token) e `*_job_request.json` (sequenza reale per catena, come sottomessa ad AF3 — target chain A ~478 aa, binder chain B 15 aa).

  **Attenzione alla nomenclatura, verificata sui file**: il batch del 2026-04-08 (10 cartelle, nomi sequenza in MAIUSCOLO) usa nomi interni dei file basati su un timestamp generico (es. `fold_2026_04_08_14_28_summary_confidences_0.json`), **non** sul nome della cartella — uno script deve fare `glob("*_summary_confidences_*.json")` dentro ogni cartella, mai assumere il pattern `<nome_cartella>_summary_confidences_N.json`. Il batch del 2026-04-16 (25 cartelle, minuscolo) segue invece quel pattern. In entrambi i casi la sequenza va letta da `job_request.json` (`sequences[1].proteinChain.sequence`), mai dal nome della cartella (case-inconsistente, puramente cosmetico).

  **Provenienza già tracciata**: `src/analysis/results/summary_best_candidates.csv` (60 righe, 34 sequenze uniche, 35 `af3_folder` unici, **tutte e 35 presenti su disco** — verificato) è il manifest che collega ogni cartella `af3_predictions/` alla sequenza, al protocollo di design che l'ha generata (colonna `protocol`: soprattutto `opt_anneal_energy_C`/`gen_energy_C`/`opt_hard_energy_C`, le varianti energy-guided; 6 righe sono `seqs_aff_alberto_darren_6best`, cioè le stesse sequenze già validate in BLI e usate come validazione dell'encoder in 0.1/0.4), alle metriche ColabDesign-time (`colab_iptm/ptm/plddt`, `energy`) e alle metriche di confidenza AF3 (`af3_iptm`, `af3_ptm`, `af3_ranking_score`, `af3_has_clash`, `af3_frac_disordered`) — **non serve ricostruire questo join**, è già fatto.

  **Decisione proposta (sostituisce quella precedente)**: classificare interfaccia/esposto **direttamente dai contatti predetti da AF3** sulle 35 sequenze reali, invece che dalla sola geometria statica di `2ZNL.pdb`. Per ciascuna cartella: scegliere il modello con `ranking_score` più alto fra i 5 (scartare o segnalare se `has_clash>0`); dai token della catena binder in `full_data_<best>.json`, interfaccia = `contact_probs` verso almeno un token della catena target sopra soglia (es. 0.5, da confermare — vedi decisioni aperte); "esposta" per esclusione o da `atom_plddts`/pattern di contatto intra-catena. Il calcolo su `2ZNL.pdb` (geometria nativa, contatti a soglia di distanza + SASA) **resta nel piano ma come cross-check**, non più come unica fonte: se le due classificazioni concordano in gran parte delle 15 posizioni, valida retroattivamente l'assunzione "ColabDesign mantiene il registro nativo" usata finora (mai verificata direttamente); se divergono, è un risultato in sé (il registro non è stabile fra design). La classificazione via AF3 è per costruzione mediata solo sulle 35 sequenze promettenti (non un campione casuale — sono già le sequenze a punteggio più alto, quindi un possibile bias di selezione sulla classificazione posizionale va segnalato in output, non ignorato), quella via 2ZNL resta l'unica indipendente dal processo di design.

**Script da creare: `src/phase0_diagnostics/run_0_5_structural_context.py`**
1. Parsing di `2ZNL.pdb` (Biopython `PDBParser`) come nel piano originale: estrazione catena A (257–716) e catena B (1–15), contatto a soglia di distanza (es. 5 Å, da confermare) + SASA relativa (Shrake-Rupley via `Bio.PDB.SASA`, niente `freesasa` esterno) → classificazione posizionale "nativa".
2. Parsing di `af3_predictions/` via `summary_best_candidates.csv` come indice (evita di dover dedurre sequenza/provenienza dai nomi cartella): per ciascuna delle 35 cartelle, `glob` dei file `*_summary_confidences_*.json` (non assumere naming), selezione del modello a `ranking_score` massimo, lettura di `contact_probs`/`token_chain_ids` dal `full_data` corrispondente, classificazione binder-per-binder → aggregazione (frazione di sequenze in cui ciascuna delle 15 posizioni risulta interfaccia) → classificazione "AF3-consensus".
3. Confronto fra le due classificazioni (agreement per posizione, non solo aggregato) → `af3_vs_2znl_agreement.csv`/json.
4. Carica tutte le sequenze disegnate (`data_loading.py`, unione dei JSON in `results/colabdesign/`, popolazione completa — invariato dal piano originale).
5. Frazione aromatica (F/W/Y) per classe di posizione (interfaccia vs esposta, usando la classificazione scelta al punto 3 — AF3 se concorde/più affidabile, 2ZNL come fallback dove i 15mer AF3 non aiutano a disambiguare), confrontata con WT, libreria iniziale (round 1, come in 0.3′), training set (come in 0.2/0.3′).

**Output:** `results/phase0/0_5_structural_context/{position_classification.csv, af3_vs_2znl_agreement.csv, aromatic_fraction_by_class.csv, summary.png}` (`af3_vs_2znl_agreement.csv` nuovo rispetto al piano originale, motivato dalla disponibilità delle predizioni reali).

**Dipendenza aggiuntiva**: nessuna nuova libreria — `contact_probs` è già nei JSON di `af3_predictions/`, non serve parsing di coordinate né `freesasa` esterno (Biopython copre il cross-check SASA su 2ZNL, invariato dal piano originale).

**Limite dichiarato**: le 35 sequenze con struttura reale sono un sottoinsieme selezionato (i migliori candidati per protocollo, non un campione casuale della popolazione disegnata) — utili per validare/raffinare la classificazione posizionale interfaccia/esposto, non per rifare l'intera analisi di composizione punto-per-punto: quella resta sulla popolazione completa `results/colabdesign/*.json` come nel piano originale, con la classificazione (validata) applicata per indice.

**Stato: eseguita (2026-08-11).** Script scritto: `src/phase0_diagnostics/run_0_5_structural_context.py`. Aggiunto `load_designed_sequences()` a `data_loading.py` (unione dei JSON in `results/colabdesign/{run1_protocols,run2_wide,run_multimer,run_multimer2}/`, esclude `custom/`; 900 design, nessun dedup — si contano i design non le sequenze uniche). Eseguita con `uv run python3 src/phase0_diagnostics/run_0_5_structural_context.py` (pochi secondi, CPU).

**Il criterio di contatto a 5Å su 2ZNL è risultato degenere, e riportato come tale invece di forzare una soglia diversa per ottenere un risultato più "pulito"**: le distanze minime osservate catena B↔A vanno da 2.58 a 3.75 Å su **tutte e 15** le posizioni — il peptide di 15 residui è incassato per l'intera lunghezza nel solco del target, non c'è un sottoinsieme di posizioni "in contatto" e un altro "no" a nessuna soglia ragionevole. Il contatto binario non è quindi l'asse che discrimina; la **SASA relativa** (complesso vs catena B isolata) sì: 13/15 posizioni restano sepolte (SASA relativa <0.5) anche isolando la sola catena B, solo le posizioni 13 e 15 emergono come chiaramente esposte (SASA relativa 0.85 e 0.61).

**Confronto 2ZNL (sepoltura) vs AF3-consensus (35 sequenze reali)**: accordo su 6/15 posizioni — moderato, non una validazione netta. AF3 identifica un nucleo di interfaccia molto più ristretto e concentrato (posizioni 6-9, frazione di sequenze con contatto >0.5 su tutte le 35: 51-57%) rispetto alla sepoltura quasi totale (13/15) suggerita dalla sola struttura nativa 2ZNL. Le due letture sono conciliabili (sepolto nella struttura nativa non implica necessariamente "in contatto diretto con probabilità >50%" nella predizione medita su design reali, spesso non nativi), ma l'accordo parziale è esso stesso un risultato: **il registro di legame non è identico fra le 35 sequenze promettenti e il binder nativo WT** — l'assunzione "ColabDesign mantiene il registro nativo", usata implicitamente ovunque nel piano originale, non è confermata in modo netto. Le posizioni 13 e 15 (esposte per SASA) sono fra le poche in accordo (anche AF3 le classifica non-interfaccia), le posizioni 1-5 e 10-12 sono i casi di disaccordo (sepolte per 2ZNL, ma sotto soglia di contatto AF3 nel 35-49% delle sequenze).

**Classificazione finale usata per l'analisi di composizione (AF3-consensus, come da piano): interfaccia = posizioni 6-9, esposta = le altre 11.**

**Risultato più forte di tutta la Fase 0**: il bias aromatico (F+W+Y) è fortemente concentrato all'interfaccia AF3, e la concentrazione cresce esattamente lungo la stessa progressione di selezione già vista in 0.1/0.3'/0.4:

| popolazione | aromatica interfaccia (4 pos.) | aromatica esposta (11 pos.) | rapporto |
|---|---|---|---|
| WT | 25.0% | 0.0% | — |
| libreria round1 | 31.9% | 10.1% | 3.2× |
| training set (selezionato) | 41.6% | 9.4% | 4.4× |
| **BLI validati** | **47.8%** | **9.9%** | **4.8×** |
| design (tutte le campagne) | 31.5% | 13.9% | 2.3× |

- La frazione aromatica all'interfaccia cresce monotonicamente WT→libreria→training→BLI (25%→32%→42%→48%), mentre quella "esposta" resta piatta intorno al 9-10% per tutte le popolazioni selezionate — il bias aromatico non è un effetto generico su tutta la sequenza, è specificamente un effetto di interfaccia, e si rafforza esattamente nelle popolazioni più vicine alla validazione sperimentale reale (BLI).
- **La popolazione disegnata (ColabDesign, 900 sequenze da tutte le campagne) mostra il rapporto interfaccia/esposta più debole di tutti (2.3×)** — più debole non solo di BLI ma anche della sola libreria round1 pre-selezione per il denominatore (esposta 13.9% vs 10.1%, quindi il design "sparge" più residui aromatici anche in posizioni non di interfaccia rispetto a quanto farebbe la sola libreria naturale). Suggerisce un gap fra cosa il pipeline di design genera e cosa la selezione biologica reale (NGS + BLI) converge a preferire — il primo concentra meno strettamente il bias aromatico sulle posizioni che contano strutturalmente.
- File aggiuntivi rispetto al piano originale: `af3_vs_2znl_agreement.csv`/`af3_vs_2znl_agreement_summary.json` (confronto fra le due classificazioni, motivato dalla disponibilità delle predizioni reali — vedi sopra).

---

## 0.6 — Screening TUP

**Non produce codice**: è una procedura manuale su web server esterno (SAROTUP, `i.uestc.edu.cn/sarotup3`), non uno script CPU. L'unico lavoro fatto qui è estrarre le sequenze giuste da sottoporre — il resto (submission web, lettura dei risultati) resta fuori da questo repo.

**Input — già estratto da `results/phase0/0_2_clustering/cluster_assignment.csv`** (colonne `is_representative`, `cluster_read_mass`):

1. **19 rappresentanti dei cluster che coprono il 90,1% della massa di read in round 3** (`cluster_read_mass` decrescente):
   ```
   VDYNPWLLFLAQPWQ  MDFNPWLLFLKVPAQ  QDYNPYLLFLKKPKQ  YDVNPYLLFLSQPRQ
   QDLNPWLLFLRLPVQ  FDFNPYMLLLKLPAQ  GDRNPKGRRRKRPAQ  LDFNPYFVFLKFPAQ
   MDYNPLLLFLRRPLQ  YDMNPWLLFLQRPKQ  VDINPYLLFLQYPIQ  GGQESEEEEEERAGA
   IDINPWLLFLYKPQQ  FDWNPFLVLLKVPAQ  GDSNPRRSRRKGPAQ  GDGNPRGGGGKGRRR
   VDLNPWLVWLKLPAQ  VDGGPLLLFLSGPVQ  GDGNPREEGEERAGA
   ```
2. **Cluster dominante**, per lo screening separato richiesto dal piano (è già il primo della lista sopra): `VDYNPWLLFLAQPWQ` (440 sequenze, 11.232 reads, 34% della massa totale).
3. **23 sequenze validate in BLI** (via `load_bli_population()`, 0.3'):
   ```
   IDFNPYLLFLKVPAQ  VDFNPWLLFLKVPAQ  MDFNPYLLFLKKPKQ  MDFNPWLLFLKVPAQ
   VDYNPWLLFLRKPKQ  VDFNPWLLFLKLPAQ  MDFNPWMLFLKLPAQ  LDFNPYLIFLKMPAQ
   IDYNPYLLFLKQPKQ  MDFNPYLLFLRKPSQ  WDYNPWLLFLRQPQQ  IDYNPWLLFLTRPQQ
   QDYNPYLLFLKKPKQ  YDINPYLLFLKKPQQ  TDYNPWLLFLKQPEQ  QDMNPYLLFLWKPKQ
   FDFNPYMLLLKLPAQ  LDFNPWFVFLKVPAQ  QDINPYLLFLKRPTQ  MDWNPLLLHLKRPAQ
   FDQNPWLLFLKMPYQ  WDMNPWLLFLKLPRQ
   ```
   (`QDYNPYLLFLKKPKQ` compare in entrambe le liste — rappresentante di cluster arricchito **e** validato in BLI, Kd = 1,12 nM.)

   **Correzione (2026-08-12): questa lista, come trascritta in questa conversazione, era inizialmente incompleta di una sequenza** — mancava `LDFNPWLLFLKLPAQ` (P3, Kd = 1,20 nM), poi sottoposta separatamente (risultato incluso sotto).

**Osservazioni già disponibili senza SAROTUP, da tenere presenti nella lettura dei risultati:**
- **Il rappresentante del cluster dominante è `P7`**, l'unica delle 23 sequenze BLI **senza segnale di legame misurabile** (Kd_nM = NA in `data/affinity_measurements_PA-PB1_formatted.csv`). È l'indizio circostanziale più diretto possibile a favore dell'ipotesi TUP per questo cluster specifico — indipendente da qualunque database esterno.
- Il secondo cluster più abbondante è invece `MDFNPWLLFLKVPAQ` = P22, un legante vero (Kd = 0,65 nM): la contaminazione, se confermata, non riguarderebbe l'intero pool selezionato ma specificamente il cluster #1.
- **4 delle 19 sequenze si discostano nettamente dal motivo di consenso** (`XDXNPXLLFLXXPXQ` circa, comune a tutte le altre 15): `GGQESEEEEEERAGA`, `GDSNPRRSRRKGPAQ`, `GDGNPRGGGGKGRRR`, `GDGNPREEGEERAGA` — composizione ricca in G/R/E, nessuna somiglianza compositiva con gli altri leganti candidati. Buoni candidati a priori per TUP anche a occhio, indipendentemente dall'esito di SAROTUP.

**Strumenti da usare** (verificato: SAROTUP 3.1 è online e attivo, con anche una versione standalone GUI/CLI open source oltre al web server): `TUPScan`/`TUPredict` (motivi TUP noti / predizione ML), `PSBinder` (leganti al polistirene — il supporto di screening, quindi il più rilevante qui),`MimoSearch`/`MimoBlast` (ricerca di identità/similarità in BDB, il "Biopanning Data Bank").

**Criteri di lettura** (dal piano madre): se il cluster dominante risulta un TUP noto (o PSBinder-positivo), il modello di energia è ancorato in misura sostanziale su un artefatto sperimentale — da dichiarare nei limiti. Se le 23 sequenze BLI non risultano segnalate mentre la coda arricchita (i 19 rappresentanti) sì, il modello apprende una miscela di due segnali non separabile con i dati disponibili.

**Output:** nessun file in questo repo oltre a questa lista — i risultati SAROTUP vanno registrati manualmente (screenshot o CSV incollato in `results/phase0/0_6_tup_screening/sarotup_results.csv`, da creare a mano dopo la submission).

**Costo stimato: 10 minuti** (come da piano madre).

**Stato: parzialmente eseguita (2026-08-12).** TUPScan (motivi noti) sui 19 rappresentanti di cluster, e **PSBinder** (predittore ML per legame al polistirene, soglia 0,5) su entrambi i gruppi — fatto manualmente dall'utente sul web server SAROTUP. Risultati grezzi in `results/phase0/SAROTUP_results.txt`.

| Strumento | Gruppo | N testate | Positive |
|---|---|---|---|
| TUPScan | 19 rappresentanti di cluster | 19 | 1 |
| PSBinder | 19 rappresentanti di cluster | 19 | 3 |
| PSBinder | sequenze BLI | 22 (manca P3, vedi sopra) | 1 |

| Sequenza | Strumento | Punteggio | Esito |
|---|---|---|---|
| `VDYNPWLLFLAQPWQ` (**cluster dominante, = P7**) | PSBinder | **0,86** | **Yes** |
| `YDMNPWLLFLQRPKQ` | PSBinder | 0,62 | Yes |
| `VDLNPWLVWLKLPAQ` | TUPScan (`W-x(2)-W`) **e** PSBinder | — / 0,50 | Yes / Yes |
| `LDFNPWFVFLKVPAQ` (**P26, BLI, Kd = 1,59 nM**) | PSBinder | 0,63 | Yes |

**Letture, riviste rispetto alla prima bozza di questa sezione (che si basava sul solo TUPScan):**
- **Il cluster dominante è positivo a PSBinder con punteggio alto (0,86).** A differenza di TUPScan (motivi noti, non lo segnala), il predittore ML lo classifica come probabile legante del polistirene — questo **conferma**, non indebolisce, l'ipotesi formulata sopra: `VDYNPWLLFLAQPWQ`/P7, il 34% di tutta la massa di read in round 3, è un candidato TUP credibile secondo almeno uno dei due strumenti, coerentemente con l'assenza di segnale in BLI.
- **Un legante BLI reale, P26 (Kd = 1,59 nM), è anch'esso positivo a PSBinder.** Questo è il risultato più importante per l'interpretazione complessiva: PSBinder positivo **non implica** "non lega PA" — un peptide può legare il target reale e *anche* mostrare la firma composizionale (aromatica) che PSBinder associa al legame al polistirene. Non risolve la distinzione fra causa B (contaminazione) e causa A (bias strutturale AF2)/C: mostra che, con questo strumento, il segnale aromatico-plastico e il segnale di legame reale **non sono cleanly separabili** nemmeno a valle, riecheggiando esattamente l'ambiguità di fondo di tutto §B del piano madre.
- Delle 4 sequenze compositivamente anomale segnalate sopra (ricche in G/R/E), **nessuna è positiva a PSBinder** (0,11–0,36, tutte sotto soglia) — l'anomalia composizionale di quelle sequenze, se reale, non è (secondo questo strumento) legame al polistirene; resta senza spiegazione.
- **§0.6 resta aperta**:Verifica del sistema di phage display per capire se è stato usato polistirene.

---

## 0.7 — Traiettorie di design: divergenza fra rappresentazione soft e discreta

**Input verificato: la copertura è parziale e limitata a una sola campagna.** `src/slurm_subs/*.out` contiene **9 file**, da due job array SLURM (`pa_pb1_wide_292363_{0..5}.out`, array completo 0-5; `pa_pb1_wide_304814_{0,1,2}.out`, riesecuzione parziale di soli 3 protocolli) — entrambi della campagna `run2_wide` (`submit_wide_search.sh`). **Nessun `.out` esiste per `run1_protocols`, `run_multimer`, `run_multimer2` o `custom`**: la campagna principale (`run1_protocols`, quella più rappresentata in `results/colabdesign/`) non ha traiettorie salvate, quindi §0.7 può caratterizzare solo il sottoinsieme `run2_wide` — limite da dichiarare esplicitamente nei risultati, non silenziare.

Protocolli coperti, con relativo $w_E$ per stadio (da `PROTOCOL_CONFIGS` in `src/slurm_subs/parse_slurm_out.py`, già hardcoded lì — riusare quel dizionario invece di reinventarlo):

| Protocollo | job/array | blocchi (seed) | $w_E$ (stage 1a/1b/2/3) |
|---|---|---|---|
| `gen_energy_A` | 292363_0 | 29 (parziale, atteso 100) | 0,02 / 0,05 / 0,02 / 0,05 |
| `gen_energy_C` | 292363_1 + 304814_0 | 30 + 18 | 0,05 / 0,20 / 0,05 / 0,50 |
| `opt_hard_energy_C` | 292363_2 | 31 | solo stage 3: 0,50 |
| `opt_anneal_noenergy` | 292363_3 + 304814_1 | 66 + 42 | **0 / 0** — controllo essenziale |
| `opt_anneal_energy_B` | 292363_4 | 66 | stage 2/3: 0,02 / 0,20 |
| `opt_anneal_energy_C` | 292363_5 + 304814_2 | 66 + 42 | stage 2/3: 0,05 / 0,50 |

Il controllo senza energia (`opt_anneal_noenergy`) richiesto esplicitamente dal piano madre **è disponibile**, con buona numerosità (108 blocchi combinati). Tutti i file sono log di job **interrotti** (`parse_slurm_out.py`, già presente nel repo, è stato scritto proprio per recuperare risultati da job uccisi dal cluster prima del salvataggio) — i conteggi di blocchi sono quindi censurati rispetto al piano originale (es. `gen_energy_A` atteso a 100 semi, osservati 29), non un problema per l'analisi di traiettoria (che non richiede completezza) ma da segnalare per l'interpretazione della numerosità per protocollo.

**Formato verificato riga per riga** (non solo il formato SUMMARY che `parse_slurm_out.py` già estrae): ogni step di ogni blocco stampa una riga tipo
```
128 models [1] recycles 0 hard 0 soft 0 temp 0.46 loss 2.46 i_con 2.85 plddt 0.64 ptm 0.91 i_ptm 0.71 [energy -21.52]
```
cioè contatore di step **locale al blocco** (riparte da 1 a ogni nuovo `[N/M] seed=...`), flag `hard`/`soft`, `temp`, e tutte le metriche inclusa `energy` — quest'ultima è il valore **soft** durante gli stadi 1a/1b/2 (`hard 0`) e il valore **discreto** durante lo stadio 3 (`hard 1`). Esempio reale ispezionato (`pa_pb1_wide_292363_0.out`, blocco `gen_energy_A` seed=0): `energy` scende fino a **-25,08** nello stadio di annealing (step 147, `temp 0.08`) e si stabilizza a **-3,27** per tutti gli step dello stadio 3 (righe 151-155, valore identico a 3 cifre decimali) — stesso ordine di grandezza del gap preliminare (-50 → -5) descritto nel piano madre.

**Decisione aperta, da confermare prima di scrivere il parser: $a^{(t)}$ e i logit $L^{(t)}$ non sono salvati.** Solo gli scalari per-step lo sono (nessuna distribuzione per posizione, nessun accesso ai logit). Per i criteri stessi del piano madre ("se non sono salvati neppure i logit, il run è escluso"), una lettura stretta escluderebbe tutti e 9 i file. **Proposta**, da confermare: usare comunque questi dati definendo un proxy operativo,
```
Δ_proxy(t) = E_soft(t) − E_hard,finale
```
dove `E_hard,finale` è il valore di energia a cui lo stadio 3 converge (verificato costante sulle ~30 iterazioni hard nel blocco ispezionato — la sequenza ha smesso di cambiare prima che lo stadio hard inizi, quindi `E_hard,finale` è un riferimento stabile, non un singolo campione rumoroso). Questo **non** è identico alla definizione del piano madre — $E(\text{onehot}(\arg\max a^{(t)}))$ **al medesimo passo** $t$, che richiederebbe $a^{(t)}$ — ma è la quantità più vicina calcolabile da questi log senza rieseguire ColabDesign su GPU. La differenza è probabilmente trascurabile a fine annealing (quando $a$ è già quasi concentrato) e più significativa nello stadio 1a/1b iniziale (dove il vertice più prossimo ad $a$ può ancora cambiare) — da dichiarare come limite, non da correggere silenziosamente.

**Conseguenza sul controllo di correttezza del piano ("Δ_hard atteso nullo").** Sotto questo proxy, $\Delta_{\text{proxy}}$ all'ultimo step è **zero per costruzione** (è la definizione del riferimento), quindi non è più un test indipendente. Il controllo di correttezza analogo disponibile con questi dati è invece: **l'energia nello stadio hard deve essere costante entro la singola traiettoria** (verificato nell'esempio sopra) — se non lo è, la sequenza sta ancora cambiando durante lo stadio 3 e il protocollo o il parsing vanno rivisti.

**Script da creare: `src/phase0_diagnostics/run_0_7_trajectory_gap.py`**
1. Parser esteso rispetto a `parse_slurm_out.py` (che tiene solo la riga finale per blocco): per ciascuno dei 9 `.out`, estrarre **ogni riga di step** dentro ciascun blocco `[N/M] seed=...`/`[N/M] input=... seed=...`, con protocollo/stadio dedotto dal contatore locale e dai confini di `PROTOCOL_CONFIGS[proto]["protocol"]` (import diretto da `src/slurm_subs/parse_slurm_out.py` — non ridichiarare i pesi).
2. Calcolare $\Delta_{\text{proxy}}(t)$ normalizzato in unità di $\sigma_{\text{train}} = 2{,}0876$ (valore esatto da `results/phase0/0_1_simplex_variance/metrics.json`, non ricalcolare).
3. Scalari per run: $\Delta_{\max}$, $t^\ast$/stadio, $\int\Delta\,dt$ sullo stadio soft (1a+1b+2), $\Delta_{\text{fine-temp}}$ (ultimo step di stage 2), più $E_{\text{soft}}^{\min}$ e $E_{\text{hard,finale}}$.
4. Aggregazione per protocollo e per $w_E$ (mediana + IQR, non singoli valori).
5. Correlazione $\Delta_{\max}$/$\int\Delta\,dt$ vs `i_ptm` finale del blocco (dalla riga `seq=...` o dalla tabella SUMMARY se presente).
6. Regressione $\Delta_{\max}$ vs $w_E$ (stage 3, il valore attivo quando la sequenza discreta finale viene prodotta).
7. Salvare `trajectories_normalized.csv` con **tutte** le traiettorie normalizzate — dichiarato riusabile da §0.8.c, non ricalcolare lì.

**Output:** `results/phase0/0_7_trajectory_gap/{trajectories_normalized.csv, summary_scalars_by_run.csv, summary_by_protocol.csv, delta_curves_by_protocol.png, delta_max_vs_iptm.png, delta_max_vs_wE.png}`

**Costo stimato: mezza giornata, CPU** (parsing di 9 file + qualche migliaio di step totali — nessun ricalcolo del modello di energia necessario, i valori sono già nei log).

**Stato: eseguita (2026-08-12), con un limite scoperto in esecuzione non anticipabile dal solo controllo dei file.** Script scritto ed eseguito: `src/phase0_diagnostics/run_0_7_trajectory_gap.py`, con `RE_STEP` esteso da `parse_slurm_out.py` per estrarre ogni riga di step (non solo il blocco finale) e proxy $\Delta_{\text{proxy}}(t) = E_{\text{soft}}(t) - E_{\text{hard,finale}}$ come confermato dall'utente. Output in `results/phase0/0_7_trajectory_gap/` (`trajectories_normalized.csv`, `summary_scalars_by_run.csv`, `summary_by_protocol.csv`, 3 grafici). **Aggiornamento (2026-08-26)**: `summary_scalars_by_run.csv` esteso con la colonna `final_seq` (sequenza discreta finale per blocco), necessaria per 0.8.c e non presente nella prima esecuzione — nessuna modifica ai valori già calcolati.

**Il controllo essenziale (`opt_anneal_noenergy`) non è misurabile: `energy` non è stampata affatto per-step quando $w_E=0$ su tutti gli stadi** — solo nella riga finale `→ seq=...` (un valore per blocco, nessuna traccia). Verificato direttamente sui file (`grep` su `pa_pb1_wide_292363_3.out`): le righe di step per questo protocollo terminano a `i_ptm=...`, senza il campo `energy` che invece è sempre presente negli altri protocolli. Non è un bug del parser né una scelta di questo script: è come ColabDesign stampava il log quando $w_E=0$, la riga di stampa evidentemente non include il campo energia in quel ramo di codice. **Conseguenza**: l'obiettivo dichiarato nel piano madre per questo controllo ("misura l'ampiezza del gap di Jensen su una traiettoria che nessuno sta sfruttando") non è raggiungibile con questi log — servirebbe una riesecuzione con logging diverso, fuori dal perimetro CPU-su-dati-esistenti di Fase 0. Analogamente **`opt_hard_energy_C` non contribuisce**, ma per un motivo diverso e atteso: è hard fin dal primo step (nessuno stadio soft per costruzione del protocollo "hard-only").

**Risultati sui 4 protocolli con traccia soft valida** (272 blocchi):

| Protocollo | N blocchi | $w_E$ (stadio 3) | gap$_{\max}$ mediana ($\sigma_{\text{train}}$) | IQR | i_ptm finale mediana |
|---|---|---|---|---|---|
| `opt_anneal_energy_B` | 65 | 0,20 | 5,7 | 4,6–7,4 | 0,17 |
| `opt_anneal_energy_C` | 106 | 0,50 | 12,0 | 10,4–16,4 | 0,23 |
| `gen_energy_A` | 28 | 0,05 | 15,3 | 13,1–16,5 | 0,35 |
| `gen_energy_C` | 46 | 0,50 | 29,5 | 24,0–35,3 | 0,36 (picco singolo osservato: 0,71) |

- **Il gap è reale, sistematico e di ampiezza notevole**: mediane fra 5,7 e 29,5 $\sigma_{\text{train}}$, con un picco individuale a 47$\sigma$ (`gen_energy_C`) — quantifica su dati reali quanto solo stimato qualitativamente nel piano madre (-50 → -5). Le curve `gap(t)` (`delta_curves_by_protocol.png`) mostrano una crescita **progressiva e pressoché monotona** lungo l'intero stadio soft/anneal, non un picco isolato.
- **La dipendenza da $w_E$ non è pulita**: `opt_anneal_energy_B` ($w_E$=0,20) ha gap mediano *inferiore* a `gen_energy_A` ($w_E$=0,05) nonostante un peso 4× più alto (5,7 vs 15,3); e i due protocolli a $w_E$=0,50 (`gen_energy_C` vs `opt_anneal_energy_C`) differiscono di oltre 2× fra loro (29,5 vs 12,0). Il confondente più plausibile è il **numero di step soft**: i protocolli `gen_*` hanno ~150 step soft (stadi 1a+1b+2) contro i ~50 di `opt_anneal_*` (solo stadio 2) — dato che il gap cresce monotonicamente con gli step (vedi sopra), più iterazioni soft producono più tempo per accumularlo, indipendentemente da $w_E$. La regressione richiesta dal piano madre (§0.7 passo 6) andrebbe quindi condotta controllando per il numero di step soft, non su $w_E$ da solo — non fatto qui, segnalato come raffinamento per chi riprende l'analisi.
- **Correlazione gap$_{\max}$ vs i_ptm finale: dipende criticamente da come viene calcolata.** Aggregata su tutti i protocolli insieme, $r=+0{,}43$ (positiva — l'opposto del segno "conferma il meccanismo" ipotizzato dal piano madre). Ma è quasi interamente un **confondimento fra protocolli** (`delta_max_vs_iptm.png` mostra chiaramente 4 nuvole separate per colore): **entro ciascun protocollo la correlazione è debole e di segno incoerente** — `gen_energy_A`: $r=0{,}26$; `gen_energy_C`: $r=-0{,}27$; `opt_anneal_energy_B`: $r=0{,}02$; `opt_anneal_energy_C`: $r=-0{,}07$. Nessuna evidenza pulita, in nessuna direzione, che l'ampiezza del gap entro uno stesso protocollo predica l'esito strutturale finale.
- **Nessuno dei quattro protocolli raggiunge la soglia di accettabilità i_ptm > 0,5** dichiarata in §B del piano madre (mediane 0,17–0,36) — il confronto sopra riguarda gradi diversi di fallimento strutturale, non successo vs fallimento. Da tenere presente per non sovrainterpretare il segno della correlazione (positiva o negativa che sia) come "il gap aiuta/danneggia il design", quando in realtà nessun run di questo campione ha prodotto una struttura accettabile.
- **Nessuna delle tre letture proposte dal piano madre (§0.7, "Criteri di lettura") si applica pulitamente**: la prima e la terza richiedono il confronto con il controllo senza energia, non misurabile qui; la correlazione negativa attesa dalla prima non si osserva nemmeno nei protocolli con energia. La domanda "il gap è un effetto della guida energetica o una proprietà generica del rilassamento" **resta aperta** — non per mancanza di segnale nei dati disponibili, ma per l'assenza strutturale del controllo nei log salvati.

---

## 0.8 — Verifiche di chiusura

Tre sotto-analisi (0.8.d non eseguita, vedi sotto — risposta diretta più rapida dal gruppo sperimentale che uno script). 0.8.c dipende da 0.7. File condivisi in `results/phase0/0_8_closing_checks/`, prefissati `0_8a_`/`0_8b_`/`0_8c_`.

### 0.8.a — Correlazione parziale $K_D$ / energia / contenuto aromatico

**Input:** le 23 sequenze BLI (`load_bli_population()`), energia calcolata con `load_energy_model`+`mlp_forward` (stesso encoder validato in 0.1/0.4 — nessun nuovo calcolo del modello), conteggio di residui aromatici (F+W+Y) per sequenza.

**Script da creare: `src/phase0_diagnostics/run_0_8a_partial_correlation.py`**
1. $\log K_D$ (media geometrica già in `Kd_nM_geomean`, colonna esistente in `load_bli_population()`) vs conteggio aromatico: regressione OLS (`numpy.linalg.lstsq` o `scipy.stats.linregress`, non serve `statsmodels` come nuova dipendenza).
2. $\log K_D$ vs energia: stessa regressione.
3. **Correlazione parziale** energia vs $\log K_D$ controllando per conteggio aromatico: residualizzazione (regredire entrambe le variabili sul conteggio aromatico, correlare i residui — equivalente alla formula chiusa della correlazione parziale a 3 variabili, non serve una libreria dedicata).
4. Tutto in continuo su N=23, **non per quartili** (0.3' ha già mostrato che il potere è insufficiente a quel livello di stratificazione).

**Output:** `results/phase0/0_8_closing_checks/{0_8a_partial_correlation.json, 0_8a_scatter_logKd_vs_energy_and_aromatic.png}`

**Costo stimato: trascurabile** (N=23, nessun nuovo calcolo pesante).

**Stato: eseguita (2026-08-12).** Script scritto ed eseguito: `src/phase0_diagnostics/run_0_8a_partial_correlation.py` (`JAX_PLATFORMS=cpu`, come 0.1/0.4 — il driver NVIDIA di questa macchina resta troppo vecchio per JAX-CUDA).

| Regressione | r | R² | p |
|---|---|---|---|
| $\log K_D$ ~ conteggio aromatico | −0,05 | 0,00 | 0,83 |
| $\log K_D$ ~ energia (diretta) | **0,37** | 0,14 | 0,079 |
| $\log K_D$ ~ energia, **parziale** (controllato per aromatici) | **0,39** | — | 0,069 |

- **Il conteggio aromatico non correla con $K_D$** (r=−0,05, p=0,83) — replica in continuo, su N=23, lo stesso risultato che 0.3' aveva trovato per quartili: gli aromatici discriminano *selezionato* da *non selezionato*, non *forte* da *debole* entro i leganti già selezionati.
- **La correlazione parziale non collassa controllando per gli aromatici — anzi cresce leggermente** (da $r=0{,}37$ diretta a $r=0{,}39$ parziale). Questo è il segnale più diretto disponibile a favore della prima lettura proposta dal piano madre ("la correlazione parziale sopravvive: il modello ha segnale oltre gli aromatici"), anche se **nessuna delle due correlazioni raggiunge $p<0{,}05$** ($p=0{,}079$ e $p=0{,}069$ rispettivamente) — con N=23 il test resta sotto-potenziato, non è possibile scartare l'ipotesi nulla con sicurezza convenzionale, ma la direzione del risultato (il controllo per aromatici rinforza, non indebolisce, l'associazione) è quella attesa se il modello contenesse segnale reale.
- **Segno coerente con l'atteso**: energia più bassa → $K_D$ più basso (legame più forte) — il modello, addestrato per arricchimento, produce energie che vanno nella direzione giusta rispetto all'affinità reale misurata indipendentemente.
- **Limite non trascurabile**: il conteggio aromatico ha varianza molto bassa in questa popolazione (`0_8a_scatter_...png`, pannello destro) — quasi tutte le 23 sequenze hanno esattamente 3 residui aromatici su 15 (range osservato 1–4). Con così poca variabilità nella variabile di controllo, la correlazione parziale ha un potere limitato di per sé nel discriminare fra le due ipotesi del piano madre — un risultato più conclusivo richiederebbe sequenze validate con maggiore variazione aromatica, non disponibili in questo set.
- **Lettura**: il risultato pende verso "il modello ha segnale oltre gli aromatici" ma non lo dimostra con la significatività convenzionale — coerente con l'impostazione del piano madre di trattare questa domanda come aperta fino a nuovi dati (§2.4, formulazione della penalità composizionale sospesa), non come già risolta in un senso o nell'altro.

### 0.8.b — Stratificazione della popolazione di design

**Input:** le 900 sequenze disegnate (`load_designed_sequences()`, 0.5) — **da estendere**: la funzione attuale non porta il peso $w_E$ né il flag "con/senza energia" per sequenza, presenti però già nel campo `protocol` di ciascun JSON sorgente (es. `data["protocol"]["stage_3"]["energy_weight"]`, verificato presente in tutti i file), quindi va aggiunta come colonna senza dover incrociare `PROTOCOL_CONFIGS` per nome file. La classificazione posizionale interfaccia/esposta è quella AF3-consensus già prodotta in 0.5 (`results/phase0/0_5_structural_context/position_classification.csv`), non va rifatta da zero.

**Script da creare: `src/phase0_diagnostics/run_0_8b_design_stratification.py`**
1. Estendere `load_designed_sequences()` in `data_loading.py` con `energy_weight_stage3` (o l'ultimo stadio presente) e `has_energy = energy_weight_stage3 > 0`.
2. Frazione aromatica **aggregata** (nessuna classificazione posizionale) per: presenza/assenza di energia; per $w_E$; per riuscita/fallimento (`i_ptm` sopra/sotto 0,5, soglia già usata nel piano madre §B).
3. Ripetere la decomposizione interfaccia/esposta di 0.5 **separatamente per strato** (con/senza energia almeno), riusando `classify_af3_consensus()` da `run_0_5_structural_context.py` (importare, non duplicare) con **analisi di sensitività sulla soglia di `contact_probs`**: 0,3 / 0,5 / 0,7, per verificare quanto la conclusione di 0.5 dipenda dalla soglia arbitraria già segnalata come limite in quella sezione.

**Output:** `results/phase0/0_8_closing_checks/{0_8b_aromatic_by_stratum.csv, 0_8b_aromatic_by_class_and_stratum.csv, 0_8b_threshold_sensitivity.csv}`

**Costo stimato: ~1 ora, CPU** (riuso quasi completo di codice esistente).

**Stato: eseguita (2026-08-12).** Script scritto: `src/phase0_diagnostics/run_0_8b_design_stratification.py`. `load_designed_sequences()` esteso con `energy_weight_stage3`/`has_energy` (letti da `protocol.stage_3.energy_weight`, presente in tutti i 27 file sorgente). `classify_af3_consensus()` in `run_0_5_structural_context.py` reso parametrico su `threshold` (refactor minimo, non invasivo — 0.5 ri-eseguita per verifica, risultati identici a prima).

**900 design: 725 con energia (stage 3), 175 senza.**

**Frazione aromatica aggregata (nessuna classe posizionale):**

| Strato | N | Frazione aromatica |
|---|---|---|
| senza energia | 175 | 17,0% |
| con energia | 725 | 19,0% |
| $w_E$=0,05 | 73 | **9,7%** (anomalia, sotto anche il controllo) |
| $w_E$=0,20 | 110 | 17,3% |
| $w_E$=0,50 | 363 | 20,3% |
| $w_E$=0,70 | 68 | 21,7% |
| $w_E$=0,90 | 111 | 20,6% |
| esito: successo (i_ptm≥0,5, **N=51 soli su 900**) | 51 | 21,4% |
| esito: fallimento (i_ptm<0,5) | 849 | 18,4% |

- Con/senza energia: differenza modesta e nella direzione attesa (19,0% vs 17,0%), ma **non monotona in $w_E$** — il gruppo a $w_E$=0,05 (n=73) ha la frazione aromatica più bassa di tutti (9,7%), sotto perfino il controllo senza energia. Non spiegato da questi dati; segnalato, non interpretato oltre.
- Solo 51/900 design (5,7%) superano la soglia di accettabilità i_ptm≥0,5 — il confronto successo/fallimento ha quindi potenza statistica intrinsecamente limitata.

**Decomposizione interfaccia/esposta per strato (soglia contact_probs=0,5, come in 0.5) — risultato che ribalta la lettura tentativa di 0.5:**

| Strato | N | Aromatica interfaccia | Aromatica esposta | Rapporto |
|---|---|---|---|---|
| senza energia | 175 | **39,3%** | 8,8% | **4,5×** |
| con energia | 725 | 29,6% | 15,1% | 2,0× |
| successo (i_ptm≥0,5) | 51 | 21,1% | 21,6% | **1,0× (nessuna concentrazione)** |
| fallimento (i_ptm<0,5) | 849 | 32,1% | 13,4% | 2,4× |

- **I design senza energia mostrano una concentrazione interfaccia/esposta più forte (4,5×) di quelli con energia (2,0×)** — l'opposto di quanto atteso se il termine di energia fosse la causa della concentrazione posizionale del bias aromatico. Il puro obiettivo AF2 (nessun segnale sperimentale) produce da solo una concentrazione interfaccia più marcata: **evidenza a favore della causa A** (l'obiettivo strutturale, non il modello di energia, guida la concentrazione posizionale), non della causa B/C come la lettura tentativa di 0.5 lasciava aperto.
- **I design "di successo" (i_ptm≥0,5) non mostrano concentrazione posizionale alcuna** (21,1% vs 21,6%, rapporto ≈1) — il pattern "aromatico concentrato all'interfaccia" è una **firma dei design falliti** (rapporto 2,4×), non dei design strutturalmente validi. Coerente con l'ipotesi che il pattern via via emerso in 0.1/0.3'/0.4/0.5 sia associato alla patologia del processo di ottimizzazione, non a un tratto dei buoni leganti — ma il confronto poggia su soli 51 design "di successo", da leggere con cautela.

**Analisi di sensitività sulla soglia `contact_probs` — la conclusione di 0.5 è fragile alla soglia:**

| Soglia | N posizioni interfaccia | Posizioni | Aromatica interfaccia | Aromatica esposta | Rapporto |
|---|---|---|---|---|---|
| 0,3 | 14/15 | quasi tutte | 18,1% | 25,1% | **0,72× (invertito!)** |
| 0,5 (usata in 0.5) | 4/15 | 6,7,8,9 | 31,5% | 13,9% | 2,27× |
| 0,7 | **0/15** | nessuna | — | 18,6% | non definito |

- **A soglia 0,3 la direzione della conclusione di 0.5 si inverte** (gli aromatici risultano più concentrati nella classe "esposta" che in quella "interfaccia") — praticamente perché a soglia bassa quasi tutte le posizioni (14/15) finiscono classificate "interfaccia", rendendo la partizione poco informativa nella direzione opposta a quella vista per il criterio di contatto 2ZNL a 5Å (degenere tutto-interfaccia). A soglia 0,7 non sopravvive **nessuna** posizione "interfaccia": nessuna delle 35 sequenze promettenti ha una singola posizione con frequenza di contatto >70% sulle 35 — la soglia 0,5 usata in 0.5 è già vicina al limite superiore di soglie che producono una classificazione non vuota.
- **Conclusione**: la lettura posizionale di 0.5 (bias concentrato all'interfaccia 6-9) **dipende in modo sostanziale dalla scelta della soglia**, non è un risultato robusto — coerente con il limite già dichiarato esplicitamente in 0.5 ("non c'è discontinuità, c'è un gradiente tagliato a 0,5"), qui quantificato: basta scendere a 0,3 per ribaltare il segno del rapporto. Il risultato "senza energia > con energia" e "successo ≈ nessuna concentrazione" sopra (a soglia 0,5) va quindi trattato come indicativo, non come conclusivo, finché non si trova una definizione di interfaccia meno sensibile alla soglia arbitraria (es. basata su un margine statistico anziché su un singolo taglio percentuale).

**Estensione richiesta dall'utente (2026-08-26): classificazione posizionale basata sui 28 residui hotspot fissati nel design, non su AF3.** `src/generative_protocols/model_energy_guidance.py` (identico in `src/config.py`) definisce `HOTSPOT_RESIDUES` (28 residui di catena A, passati come `hotspot=` a `model.prep_inputs(...)`) — i residui che ColabDesign è esplicitamente istruito a contattare. A differenza della classificazione AF3-consensus (derivata da predizioni, con la soglia fragile sopra), questa è la classificazione più direttamente legata a **cosa il processo di design ottimizza per costruzione**. Script scritto ed eseguito: `src/phase0_diagnostics/run_0_8b_hotspot_proximity.py`.

**Anche qui la sola distanza minima è degenere** (tutte le 15 posizioni entro 3,75Å da almeno uno dei 28 hotspot, verificato su `2ZNL.pdb`) — stessa patologia di 0.5, ora confermata pure ristringendo il target ai soli hotspot invece che a tutta la catena A: il binder WT è incassato per l'intera lunghezza nel solco che gli hotspot delimitano, non solo in un sottoinsieme di posizioni. Usata quindi una metrica graduata: **numero di hotspot distinti (su 28) con un atomo entro 4,0Å** (soglia più conservativa dei 5Å usati altrove) da ciascuna posizione — range osservato 1–7, non degenere. Soglia di classificazione: **≥4 hotspot distinti → "near_hotspot"**, altrimenti "far_from_hotspot" (split bilanciato 7 vs 8 posizioni: near = {1,2,3,4,5,10,11}, far = {6,7,8,9,12,13,14,15}).

| Popolazione | Aromatica *near_hotspot* (7 pos.) | Aromatica *far_from_hotspot* (8 pos.) | Rapporto |
|---|---|---|---|
| Riferimento nativo (WT) | 0,0% | 12,5% | 0 |
| Libreria round 1 | 12,5% | 18,9% | 0,66× |
| Training set | 12,2% | 23,0% | 0,53× |
| Leganti validati in BLI | 14,9% | 24,5% | 0,61× |
| Design (900, aggregati) | 16,4% | 20,5% | 0,80× |
| Design senza energia | 10,2% | 22,9% | **0,45×** |
| Design con energia | 17,9% | 19,9% | 0,90× |
| Design "di successo" (i_ptm≥0,5, N=51) | **24,9%** | 18,4% | **1,36× (invertito)** |
| Design falliti (i_ptm<0,5) | 15,9% | 20,6% | 0,77× |

**Risultato di segno opposto a quello di 0.5/AF3-consensus, e più coerente al proprio interno:**
- **In ogni popolazione tranne una, gli aromatici sono più frequenti LONTANO dagli hotspot che vicino** — pattern presente già nella libreria pre-selezione (0,66×) e che **si rafforza con la selezione** (0,53× nel training set, invece di attenuarsi). L'ipotesi più semplice è sterica: le posizioni a contatto diretto con più hotspot contemporaneamente hanno meno margine geometrico per una catena laterale aromatica ingombrante senza clash, indipendentemente da cosa la selezione stia premiando altrove.
- **Coerente con 0.8.b**: i design senza energia mostrano di nuovo l'asimmetria più marcata (0,45×), quelli con energia sono più bilanciati (0,90×) — stessa direzione qualitativa del risultato AF3-based (causa A più che B/C nel determinare la collocazione), ma qui ottenuta da una classificazione indipendente e non soggetta alla stessa fragilità di soglia (il profilo di contatto a sinistra in `0_8b_hotspot_proximity_summary.png` mostra una separazione abbastanza netta fra i due gruppi di posizioni, non un continuo tagliato arbitrariamente).
- **L'unica inversione è nei design "di successo" (i_ptm≥0,5): qui gli aromatici sono più frequenti VICINO agli hotspot (1,36×), non lontano.** Letto insieme al risultato AF3-based di 0.8.b (i design di successo non mostravano concentrazione posizionale, rapporto ≈1 su quella classificazione): le due letture non sono in conflitto, sono complementari — rispetto alla classificazione strutturalmente più rilevante (i residui che il design doveva effettivamente contattare), i design che funzionano sembrano collocare gli aromatici *dove contano*, mentre il pattern generico (aromatici ovunque tranne dove servono) è una firma dei design falliti e di quelli generati senza segnale sperimentale. Base ridotta (N=51), da confermare con più dati prima di appoggiarcisi.

**Output:** `results/phase0/0_8_closing_checks/{0_8b_hotspot_contact_profile.csv, 0_8b_aromatic_by_hotspot_proximity.csv, 0_8b_hotspot_proximity_summary.png}`.

### 0.8.c — Composizione al minimo del gap

**Dipendenza:** richiede l'output di 0.7 (`trajectories_normalized.csv`) — non eseguibile prima.

**Limite ereditato da 0.7**: poiché $a^{(t)}$ non è salvato, "composizione amminoacidica media di $a^{(t^\ast)}$" nel senso letterale del piano madre non è calcolabile. L'unica composizione disponibile a $t^\ast$ (il passo di $\Delta_{\max}$) è quella della **sequenza discreta finale del blocco** (`seq=` di fine blocco, non la sequenza al passo $t^\ast$ stesso — altro scarto fra piano madre e dati disponibili, da dichiarare insieme a quello di 0.7). Il confronto resta comunque informativo: confronta la composizione dei design **il cui $t^\ast$ cade nello stadio soft con $\Delta_{\max}$ grande** contro quella dei run di controllo senza energia.

**Script da creare: `src/phase0_diagnostics/run_0_8c_gap_composition.py`**
1. Da `trajectories_normalized.csv`, per ciascun blocco: $t^\ast$, stadio di $t^\ast$, sequenza finale del blocco.
2. Frazione aromatica della sequenza finale, stratificata per: quartile di $\Delta_{\max}$ del blocco; protocollo (energy-guided vs `opt_anneal_noenergy`).
3. Confronto diretto: i blocchi nel quartile più alto di $\Delta_{\max}$ hanno frazione aromatica maggiore di quelli nel quartile più basso?

**Output:** `results/phase0/0_8_closing_checks/0_8c_aromatic_by_gap_quartile.csv`

**Costo stimato: trascurabile**, dopo 0.7.

**Stato: eseguita (2026-08-26).** Script scritto: `src/phase0_diagnostics/run_0_8c_gap_composition.py`. Estensione minima a 0.7 richiesta e fatta prima di poter procedere: `summarize_block()` in `run_0_7_trajectory_gap.py` non catturava la sequenza finale del blocco (solo la sua energia) — aggiunta colonna `final_seq` a `summary_scalars_by_run.csv`, 0.7 ri-eseguita (stessi numeri di prima, nessuna regressione).

**Quartili di gap$_{\max}$ pooled su tutti i protocolli insieme (245 blocchi con traccia soft valida) — nessun trend pulito:**

| Quartile | N | Frazione aromatica media | Range gap$_{\max}$ |
|---|---|---|---|
| Q1 (gap minimo) | 62 | 19,2% | 1,8–8,1 |
| Q2 | 61 | 17,3% | 8,2–11,9 |
| Q3 | 61 | 16,1% | 12,0–17,0 |
| Q4 (gap massimo) | 61 | 21,0% | 17,0–47,0 |

**Entro ciascun protocollo (per evitare lo stesso confondimento fra protocolli già trovato in 0.7) — segno del tutto incoerente:**

| Protocollo | Q1 | Q2 | Q3 | Q4 | Direzione |
|---|---|---|---|---|---|
| `gen_energy_A` | 2,9% | 7,6% | 7,6% | **10,5%** | crescente |
| `gen_energy_C` | **26,7%** | 23,0% | 17,6% | 13,9% | **decrescente** |
| `opt_anneal_energy_B` | 20,0% | 21,2% | 20,9% | 20,8% | piatta |
| `opt_anneal_energy_C` | 17,0% | 15,9% | 17,4% | 25,9% | quasi piatta poi salto in Q4 |

- **Nessun pattern comune ai quattro protocolli**: `gen_energy_A` mostra una relazione crescente pulita (gap grande → sequenza finale più aromatica), `gen_energy_C` mostra l'esatto opposto (gap grande → sequenza finale MENO aromatica), gli altri due sono sostanzialmente piatti. Il quartile pooled (che media questi segni opposti) è quindi privo di significato — un altro caso, dopo 0.7, in cui l'aggregazione fra protocolli nasconde più di quanto riveli.
- **I riferimenti senza traccia di gap non discriminano**: `opt_anneal_noenergy` (n=106, il controllo vero) ha frazione aromatica finale 19,4%, `opt_hard_energy_C` (n=31) 18,7% — entrambi cadono dentro il range già coperto dai quartili dei protocolli energy-guided (13,9–26,7%), non sotto né sopra in modo sistematico.
- **Nessuna delle due letture proposte dal piano madre si applica pulitamente**: non è vero che "la massa si concentra su W/F/Y" in modo generalizzato al crescere del gap (solo `gen_energy_A` lo mostra), ma non è nemmeno chiaramente il caso opposto ("composizione generica", visto che due protocolli su quattro mostrano un trend chiaro, solo di segno opposto). **Numerosità per cella bassa** (7–27 blocchi) rende comunque questi trend per-protocollo poco solidi singolarmente presi.
- **Lettura complessiva di 0.8.c**: coerente con quanto già visto in 0.7 per la correlazione gap$_{\max}$/i_ptm (debole, incoerente di segno entro protocollo) — l'ampiezza del gap di Jensen non sembra tradursi in un effetto compositivo sistematico e protocollo-indipendente sulla sequenza discreta finale. Non chiude la domanda causale del piano madre (§B), la lascia genuinamente aperta invece di forzarla verso una delle due letture proposte.

### 0.8.d — Attesa composizionale della libreria (schema di codoni)

**Deciso (2026-08-12): non eseguita come analisi bioinformatica.** Più rapido ed affidabile chiedere direttamente al gruppo sperimentale quale schema di codoni degenerati è stato effettivamente usato in sintesi, invece di inferirlo indirettamente da dati già post-selezione (nessun file nucleotidico di round 1/libreria naiva è disponibile in questo repo, vedi §0.2/§0.3 — l'inferenza sarebbe comunque stata indiretta anche eseguendo lo script).

**Ipotesi di lavoro dell'utente: NNK, da confermare con chi ha eseguito il phage display.** Osservazione preliminare raccolta durante l'istanziazione, utile come punto di partenza per quella conversazione ma non come sostituto: la distribuzione grezza della terza base del codone in `PD_energy_model/data/4_counts/{F,R}_{2,3}_count.csv` (round 2/3, non la libreria naiva) non è pulitamente compatibile con NNK preso alla lettera (G+T atteso ~100%, osservato 81-83% a seconda della popolazione) — ma è dato post-selezione, quindi non dirimente da solo; il conteggio degli stop codon (TAG fortemente dominante, TGA/TAA solo 1-2 ordini di grandezza sotto) è comunque coerente con uno schema a singolo stop, incluso NNK.

**Output:** nessuno — nessuno script, nessun file in `results/phase0/`. Il denominatore F+W+Y atteso per l'analisi di arricchimento di 0.3' resta da fissare una volta confermato lo schema con il gruppo sperimentale.

---

## Riepilogo dipendenze fra sottofasi

```
encoder + colabdesign.energy_model.model_3layer ──┬──> 0.1
                                                     └──> 0.4

0.2 (loader training set + clustering) ──> usato da 0.1, 0.3′, 0.4, 0.6 (cluster_assignment.csv)

0.3′  indipendente (solo CSV BLI + conteggi round 1) ──> usato da 0.6, 0.8.a
0.5   indipendente dal modello di energia e da 0.2 (usa 2ZNL.pdb + af3_predictions/ + i JSON dei design) ──> usato da 0.8.b (classify_af3_consensus)
0.6   manuale, nessuna dipendenza di codice (solo le liste di sequenze da 0.2/0.3′)
0.7   indipendente (solo i 9 .out di run2_wide + PROTOCOL_CONFIGS) ──> usato da 0.8.c
0.8.a indipendente (solo 0.3′)
0.8.b dipende da 0.5 (riusa classify_af3_consensus) e load_designed_sequences esteso
0.8.c dipende da 0.7 (trajectories_normalized.csv)
0.8.d indipendente (solo i conteggi nucleotidici round 2/3)
```

Ordine di implementazione consigliato per 0.1–0.5 (eseguito): **encoder + validazione (§Perimetro) → 0.3′ (rapido, valida i dati BLI) → 0.2 (loader condiviso + clustering) → 0.1 → 0.4 → 0.5**. 0.3′ prima di 0.2 perché è il rapporto informazione/costo più alto del piano originale ed è del tutto disaccoppiato dal resto — buon primo risultato da vedere prima di investire nel loader condiviso.

Per 0.6–0.8: **0.6 (manuale, indipendente, avviabile subito) → 0.8.a e 0.8.d in parallelo (entrambi rapidi e indipendenti) → 0.7 (mezza giornata) → 0.8.c (dipende da 0.7) → 0.8.b (indipendente da 0.7, ma a basso rapporto urgenza/costo essendo perlopiù un'analisi di sensitività su 0.5 già conclusa)**.

## Decisioni aperte da confermare prima di scrivere codice

1. **MMseqs2 vs fallback Python** per 0.2 — dipende da cosa è installabile sul cluster AAR. **Risolto**: fallback greedy Python (vedi §0.2, eseguita).
2. **Soglia di distanza per "contatto interfaccia"** in 0.5 su `2ZNL.pdb` (proposta: 5 Å heavy-atom, standard ma da confermare) **e** soglia di `contact_probs` per l'interfaccia AF3-based (proposta: 0.5, da confermare — le due soglie non sono direttamente comparabili, una è geometrica sulla struttura nativa, l'altra è una probabilità predetta da AF3 su 35 design reali).
3. **Layout cartelle**: `src/phase0_diagnostics/` + `results/phase0/` come proposto sopra, oppure integrare nei notebook esistenti sotto `src/analysis/`. **Risolto de facto**: script in `src/phase0_diagnostics/` (0.1–0.5 eseguite così).
4. **Proxy per $\Delta(t)$ in 0.7** — **Risolto**: confermato dall'utente, eseguito con $\Delta_{\text{proxy}}(t) = E_{\text{soft}}(t) - E_{\text{hard,finale}}$ (vedi §0.7, eseguita). Limite aggiuntivo scoperto in esecuzione, non solo teorico: il controllo `opt_anneal_noenergy` non ha traccia energetica per-step affatto (non solo logit assenti), quindi resta comunque non misurabile con questo o qualunque altro proxy calcolabile dai log esistenti.
5. **Copertura parziale di 0.7** — confermato come limite reale in esecuzione: solo `run2_wide` ha traiettorie salvate (9 file, 2 job array parziali), e anche entro questi il controllo senza energia non ha traccia per-step (vedi punto 4). Estendere a `run1_protocols` richiederebbe le sue `.out` originali (se reperibili sul cluster AAR) o una riesecuzione — fuori dal perimetro di Fase 0.
6. **Schema di codoni (0.8.d)** — **Non più una decisione di questo documento**: deciso di chiedere direttamente al gruppo sperimentale invece di inferirlo da dati post-selezione. Ipotesi di lavoro dell'utente: NNK, da confermare.
