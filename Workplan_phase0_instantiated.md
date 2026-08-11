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

## Riepilogo dipendenze fra sottofasi

```
encoder + colabdesign.energy_model.model_3layer ──┬──> 0.1
                                                     └──> 0.4

0.2 (loader training set + clustering) ──> usato da 0.1, 0.3′, 0.4 (stesso data_loading.py)

0.3′  indipendente (solo CSV BLI + conteggi round 1)
0.5   indipendente dal modello di energia e da 0.2 (usa 2ZNL.pdb + af3_predictions/ + i JSON dei design)
```

Ordine di implementazione consigliato: **encoder + validazione (§Perimetro) → 0.3′ (rapido, valida i dati BLI) → 0.2 (loader condiviso + clustering) → 0.1 → 0.4 → 0.5**. 0.3′ prima di 0.2 perché è il rapporto informazione/costo più alto del piano originale ed è del tutto disaccoppiato dal resto — buon primo risultato da vedere prima di investire nel loader condiviso.

## Decisioni aperte da confermare prima di scrivere codice

1. **MMseqs2 vs fallback Python** per 0.2 — dipende da cosa è installabile sul cluster AAR. **Risolto**: fallback greedy Python (vedi §0.2, eseguita).
2. **Soglia di distanza per "contatto interfaccia"** in 0.5 su `2ZNL.pdb` (proposta: 5 Å heavy-atom, standard ma da confermare) **e** soglia di `contact_probs` per l'interfaccia AF3-based (proposta: 0.5, da confermare — le due soglie non sono direttamente comparabili, una è geometrica sulla struttura nativa, l'altra è una probabilità predetta da AF3 su 35 design reali).
3. **Layout cartelle**: `src/phase0_diagnostics/` + `results/phase0/` come proposto sopra, oppure integrare nei notebook esistenti sotto `src/analysis/`. **Risolto de facto**: script in `src/phase0_diagnostics/` (0.1–0.4 eseguite così).
