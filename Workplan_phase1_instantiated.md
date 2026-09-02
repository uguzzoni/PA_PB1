# Fase 1 — Piano istanziato (1.1–1.6)

Istanziazione operativa di [`Workplan_peptidi_PA-PB1.md`](Workplan_peptidi_PA-PB1.md) §FASE 1 (versione 2, 12 agosto 2026), sullo stesso schema di [`Workplan_phase0_instantiated.md`](Workplan_phase0_instantiated.md): per ciascuna sottofase, dati/file verificati nel repo, script da scrivere, output atteso, decisioni aperte. **Nessuna sottofase è ancora eseguita** — questo documento è il piano, non il resoconto.

**Perimetro dichiarato dal piano madre: CPU, nessuna esecuzione di ColabDesign/AfDesign.** A differenza della Fase 0 (un solo repo, `PA_PB1`), la Fase 1 tocca **tre codebase distinte**, verificate in questa sessione:

1. **`PA_PB1`** (questo repo) — dati NGS/BLI, output di Fase 0, nuovi script di valutazione lato Python.
2. **`methods/colabdesign_energy_guidance`** — il fork di ColabDesign menzionato nel piano madre (§A.2), repo git a sé (`git remote`: `git@github.com:uguzzoni/colabdesign_energy_guidance.git`, branch corrente `energy_guidance`, working tree pulito). `colabdesign/energy_model/model_3layer.py` è l'unico file che espone `load_energy_model`/`mlp_forward`/`make_energy_fn` — **è qui che vanno implementate le correzioni di 1.1, 1.2 e il confezionamento di 1.6**, non in `PA_PB1`. Contiene anche `energy_guidance/test_energy_guidance.py`, un test già esistente dell'hook `energy_fn` (con un placeholder `test_energy_fn` in `colabdesign/af/loss.py` che penalizza la glicina — probabilmente il punto di partenza per il "test placeholder" di §1.6 criterio 7) — da estendere, non da ricreare da zero.
3. **`PD_energy_model`** (nested, git-ignored in `PA_PB1`, già usato in Fase 0 solo per i CSV) — qui vive il **training** Julia del modello di energia, e **il training resta in Julia** (nessuna reimplementazione in Python/JAX per 1.3/1.5). Ambiente: definito in `PD_energy_model/Project.toml`, si attiva con `Pkg.activate(".")` (o `".."` dalla cartella del notebook — vedi nota sui path in §1.3) — pacchetti già installati in locale, nessuna azione di setup necessaria. Verificato: `training/training_2rounds/model_training_7_PNB_negbinom_ll_3layers.ipynb` è il notebook che ha prodotto i pesi correnti (md5 identico fra `PD_energy_model/model_params/PNB_3lay_negbinom_energy_model_weights.json` e `data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json`). Il notebook costruisce una struttura ad albero a 2 livelli per ciascuna delle due repliche F/R (`root → child`, cioè round 2 → round 3), per modellare la relazione fra i campioni nella verosimiglianza — una struttura generale e utile del framework `PhageNegBinom`, qui semplicemente minima perché i dati hanno solo 2 round per replica: non entra in gioco a livello di singola sequenza/cluster, che restano un array separato (`seqs`/`cnts`) — l'esclusione di un cluster è quindi un filtro su quell'array, non una modifica alla struttura ad albero (dettaglio verificato leggendo il notebook, vedi §1.3).

**Hook contract verificato** (`colabdesign/af/model.py:85-90`): `energy_fn(seq["pseudo"])`, `seq["pseudo"]` shape `(1, L, 20)` — la stessa rappresentazione (soft durante gli stadi soft, one-hot durante lo stadio hard) usata per tutte le altre metriche di ColabDesign (`i_con`, `pLDDT`, l'`aatype` finale). **Nessun trattamento straight-through è già presente lato ColabDesign**: la correzione di 1.1 va implementata per intero dentro il modulo dell'energia, senza toccare `af/loss.py`, `af/model.py` o `af/design.py`. Firma attuale da preservare per retrocompatibilità (usata da `generate_sequences.py`, `optimize_sequences.py`, `model_energy_guidance.py` in questo repo, e dai loro analoghi in `energy_guidance/` nel fork): `make_energy_fn(weights_path: str, energy_weight: float = 0.1, binder_len: int = None)`.

**Convenzione per le modifiche nel fork (precisata dall'utente): mai editare in-place i file legacy.** `model_3layer.py` (e gli script in `energy_guidance/`) restano intoccati; ogni modifica di Fase 1 va fatta su una **copia** nuova, che è poi quella da commitare/pushare sul fork. Copia proposta per 1.1/1.2/1.6: `colabdesign/energy_model/model_3layer_v2.py` (nome da confermare/rinominare a piacere — dettaglio reversibile). `generate_sequences.py`/`optimize_sequences.py`/`model_energy_guidance.py` (in questo repo e nel fork) continuano a importare da `model_3layer` (legacy, invariato) finché non si decide esplicitamente di farli passare a `model_3layer_v2` — un cambio a parte, non implicito in questo piano.

**Layout proposto:**
```
PA_PB1/src/phase1_diagnostics/            # nuovo, in questo repo — valutazione lato Python, riusa src/phase0_diagnostics/data_loading.py
├── run_1_1_st_estimator_check.py         # verifica indipendente (oltre ai test nel fork) su dati di Fase 0
├── run_1_2_standardization.py
├── run_1_4_baseline_comparison.py        # baseline 1/2 in Python puro, si ferma dove serve il modello 3 riaddestrato
├── run_1_3_replica_ensemble_eval.py      # valutazione delle repliche, UNA VOLTA che esistono (dipende da risposta Julia)
├── run_1_5_heldout_family_eval.py        # idem
└── run_1_6_bundle_validation.py          # verifica l'artefatto .npz prodotto dal fork

methods/colabdesign_energy_guidance/       # repo separato — model_3layer.py legacy INTOCCATO, copia nuova da modificare
├── colabdesign/energy_model/model_3layer.py      # legacy, invariato — resta usato da tutti gli script esistenti
├── colabdesign/energy_model/model_3layer_v2.py   # NUOVO — mlp_forward_st, energy_aux, standardizzazione, bundle multi-replica
└── energy_guidance/test_energy_guidance.py       # esteso con i 7 criteri di accettazione di §1.6

PD_energy_model/training/training_2rounds/
├── model_training_7_PNB_negbinom_ll_3layers.ipynb                    # legacy, invariato
└── model_training_7_PNB_negbinom_ll_3layers_cluster_exclusion.ipynb  # NUOVO — creato, vedi §1.3

PA_PB1/results/phase1/
├── 1_1_st_estimator/
├── 1_2_standardization/
├── 1_3_replica_ensemble/
├── 1_4_baseline_comparison/
├── 1_5_heldout_family/
└── 1_6_bundle_packaging/
```

---

## 1.1 — Correzione del forward pass: straight-through estimator

**Input verificato:** vedi sopra (hook contract, firma da preservare). Nessun dato nuovo — riusa pesi e encoder già validati in 0.1 (`data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json`, errore max 8×10⁻⁷ contro le 6 sequenze di controllo).

**Modifiche da fare in `methods/colabdesign_energy_guidance/colabdesign/energy_model/model_3layer_v2.py`** (copia nuova, `model_3layer.py` legacy invariato — vedi convenzione sopra):
1. Nuova funzione (o branch interno) che implementa esattamente lo snippet del piano madre:
   ```python
   hard = jax.nn.one_hot(x.argmax(-1), x.shape[-1])
   x_st = x + jax.lax.stop_gradient(hard - x)
   return mlp_forward(params, x_st)
   ```
   Nota sulla shape: `x` qui è `(1, 15, 21)` (formato interno gap-padded di `mlp_forward`, non `(1, L, 20)` dell'hook AF2 — la conversione avviene già in `make_energy_fn`). L'argmax include la 21ª colonna (gap, sempre zero in produzione per costruzione, §0.1) — va verificato esplicitamente come primo test che il gap non venga mai selezionato, non assunto.
2. Estendere `make_energy_fn(weights_path, energy_weight=0.1, binder_len=None, mode="soft")` — `mode="soft"` default (comportamento invariato, retrocompatibile con ogni chiamante esistente che non passa `mode`), `mode="st"` attiva il nuovo branch.
3. Nuova funzione `make_energy_aux(weights_path, binder_len=None)` → dict con `e_fwd`, `e_hard`, `gap` (= `e_fwd - e_hard`), come da piano madre §1.6(e) — utile da subito per strumentare i run di Fase 2, non solo come parte del confezionamento finale.

**Test — estendere `energy_guidance/test_energy_guidance.py`** (già presente nel fork, testa l'hook con il placeholder `test_energy_fn` che penalizza la glicina):
1. `mlp_forward_st(params, a) == mlp_forward(params, one_hot(argmax(a)))` esatto (entro `1e-6`), per `a` sia ai vertici sia campionato dall'interno del simplesso — riusare come fixture i punti Dirichlet già generati in 0.1 se salvati, altrimenti ricampionare con lo stesso seed (`results/phase0/0_1_simplex_variance/`).
2. Gradiente `∂mlp_forward_st/∂a` non identicamente nullo (test esplicito — il piano madre segnala che uno `stop_gradient` avvolto sull'espressione sbagliata annulla il gradiente silenziosamente, senza errore).
3. Verifica visiva: rigenerare `delta_curves_by_protocol.png` (0.7) sostituendo `mode="soft"` con `mode="st"` su un run minimale (pochi step, CPU) — il gap deve essere identicamente zero per costruzione.

**Output:** `results/phase1/1_1_st_estimator/{test_report.json, gap_zero_verification.png}` (in `PA_PB1`, anche se il codice modificato vive nel fork — qui restano solo i risultati di verifica).

**Stato: eseguita (2026-08-27).** Creato `colabdesign/energy_model/model_3layer_v2.py` (copia di `model_3layer.py`, legacy invariato) con `mlp_forward_st`, `make_energy_fn(..., mode="soft"|"st")` (default `"soft"`, retrocompatibile), `make_energy_aux(..., mode=...)`. Test unitari in `energy_guidance/test_model_3layer_v2.py` (nuovo file, CPU-only, nessuna dipendenza da AfDesign/PDB — distinto da `test_energy_guidance.py` che è un test di integrazione con GPU/PDB reali). **Commit locale sul branch `energy_guidance`** (come richiesto), non pushato — `1d5277b "Add straight-through estimator for energy model (Fase 1 §1.1)"`.

**Risultati dei test (tutti superati):**
- `model_3layer_v2` in modalità `"soft"` riproduce **esattamente** (entro $10^{-6}$) sia `model_3layer` legacy sia le 6 energie di validazione già note da Fase 0 — nessuna regressione introdotta.
- $\texttt{mlp\_forward\_st}(a) = \texttt{mlp\_forward}(\text{one\_hot}(\arg\max a))$ esatto (errore $0{,}00\times10^{0}$) su 200 punti Dirichlet a concentrazione mista ($\alpha \in \{0{,}1, 1, 20\}$); colonna gap (21ª) mai selezionata dall'argmax, verificato esplicitamente e non assunto.
- Gradiente di `mlp_forward_st` non nullo su 20/20 punti testati.
- `make_energy_aux`: gap identicamente nullo (errore $0{,}00\times10^{0}$) in modalità `"st"` su 200 punti; non nullo (soglia $10^{-6}$) in modalità `"soft"` su 200/200 punti.

**Verifica visiva, con uno scarto dichiarato dal piano madre**: invece di un run AfDesign reale (richiederebbe PDB + parametri AF2, fuori dal perimetro CPU-only di questa fase), ho generato una **traiettoria sintetica** (interpolazione soft→hard verso un vertice fisso, temperatura 1→0,02, senza passare da ColabDesign) — dimostra correttamente il *meccanismo* (gap ST identicamente zero lungo l'intera traiettoria in tutti i casi testati, gap soft che si apre e si richiude verso zero), ma **non riproduce l'ampiezza reale** vista in 0.7 (qui gap soft massimo ~0,25σ, contro le mediane 5,7–29,5σ osservate sui run reali) — il percorso sintetico è una rampa liscia verso un bersaglio fisso, non la direzione trovata dal gradiente reale lungo l'asse ad alta curvatura del modello. Una verifica su traiettoria reale resta da fare in Fase 2 (§2.1), quando ColabDesign va comunque eseguito per il confronto soft/ST.

**Decisioni aperte:**
- Push del commit sul remote (`github.com/uguzzoni/colabdesign_energy_guidance`) — non ancora fatto, da confermare esplicitamente.
- *Moment propagation* (alternativa di riserva nel piano madre) non istanziata qui: si attiva solo se lo straight-through, in Fase 2, produce traiettorie che si arrestano o gradienti troppo sparsi.

---

## 1.2 — Standardizzazione

**Input verificato:** $\sigma_{\text{train}} = 2{,}0876$ già calcolato e salvato in `results/phase0/0_1_simplex_variance/metrics.json`. **$\mu_{\text{train}}$ non è stato salvato in 0.1** (quello script calcola `train_E` solo per `E_min`/`E_max`/`sigma_train`, non ne salva la media) — va ricalcolato, banale (poche righe, stesso array `train_E` già prodotto da `load_training_sequences_onehot()` + `mlp_forward` vettorizzato, nessun nuovo dato).

**Nota di sequenziamento:** 1.1 e 1.2 modificano lo stesso file (`model_3layer_v2.py`) e concettualmente si compongono (standardizzare *dopo* aver scelto la modalità di forward, sul valore scalare finale — non prima) — conviene implementarle nello stesso passaggio di modifica, pur restando due sottofasi distinte nel piano madre e qui.

**Modifiche:** $\tilde E = (E - \mu_{\text{train}})/\sigma_{\text{train}}$ applicata all'uscita scalare di `mlp_forward`/`mlp_forward_st`, prima di moltiplicare per `energy_weight`. Esposta come opzione della stessa factory di 1.1 (non una funzione separata) — la standardizzazione a regime vive nel bundle multi-replica di 1.6 (`mu`/`sd` per replica), ma il calcolo su singola replica va comunque validato qui prima.

**Script da creare: `src/phase1_diagnostics/run_1_2_standardization.py`** (in `PA_PB1`)
1. Calcola $\mu_{\text{train}}$ sul training set (stesso array di 0.1).
2. Verifica: $\tilde E$ ha media ≈0 e deviazione standard ≈1 sul training set stesso (sanity check banale ma da mettere a verbale).
3. Confronto qualitativo: range di $\tilde E$ sui run reali di 0.7 (traiettorie già in unità di $\sigma_{\text{train}}$, ricalcolabile da lì) per illustrare l'interpretazione "quante deviazioni standard sperimentali".

**Output:** `results/phase1/1_2_standardization/standardization_params.json` (`mu_train`, `sigma_train`, entrambi da riusare ovunque altrove serva, non ricalcolare di nuovo).

**Stato: eseguita (2026-08-27).** `make_energy_fn`/`make_energy_aux` estese in `model_3layer_v2.py` con `mu`/`sigma` opzionali (entrambi o nessuno — `ValueError` esplicito se solo uno è passato), applicati come $z=(E-\mu)/\sigma$ dopo la modalità di forward e prima di `energy_weight`. Script `run_1_2_standardization.py` eseguito. Commit locale su `energy_guidance` (`b8e2557`), non pushato.

**Risultati:**
- $\mu_{\text{train}} = 3{,}398753$ (nuovo), $\sigma_{\text{train}} = 2{,}087558$ — **ricalcolato e verificato identico** (entro $10^{-6}$) a quello già salvato in `results/phase0/0_1_simplex_variance/metrics.json`: nessun problema di riproducibilità fra le due esecuzioni.
- $z$ sul training set: media $-4\times10^{-8}$, deviazione standard $1{,}000000$ — la standardizzazione fa esattamente quello che deve.
- `make_energy_fn(..., mu=, sigma=)` coerente con $(E_{\text{raw}}-\mu)/\sigma$ entro $4\times10^{-7}$ (arrotondamento float32, non un errore).
- **Il gap resta identicamente nullo in modalità `st` anche con standardizzazione attiva** ($\max|{\rm gap}| < 2\times10^{-7}$) — atteso: $z_{\text{fwd}}-z_{\text{hard}} = (e_{\text{fwd}}-e_{\text{hard}})/\sigma$, $\mu$ si cancella nella differenza, e $e_{\text{fwd}}=e_{\text{hard}}$ esattamente in modalità `st` indipendentemente da qualunque riscalamento affine successivo — verificato, non solo atteso per costruzione.
- Test aggiuntivo: passare `mu` senza `sigma` (o viceversa) solleva `ValueError` esplicito — non fallisce in silenzio con una standardizzazione parziale/sbagliata.

---

## 1.3 — Repliche su split di cluster

**Input verificato:** `results/phase0/0_2_clustering/cluster_assignment.csv` (round 3 — Sequence, Count, cluster_id, cluster_size, cluster_read_mass, is_representative), già dichiarato prerequisito di 2.3/2.5 in Fase 0, riusabile qui per costruire gli split per cluster (non per sequenza — 0.2 ha già mostrato che 2 cluster coprono il 50% della massa di read, uno split per sequenza sovra-rappresenterebbe quelle famiglie).

**Ambiente Julia: risolto.** `Pkg.activate("..")` nel notebook attiva `PD_energy_model/Project.toml`; i pacchetti privati (`PhageNegBinom`, `PD_data_utils`, ecc.) sono già installati in locale — nessuna azione di setup necessaria.

**Meccanismo di esclusione cluster: più semplice del previsto, verificato leggendo il notebook riga per riga (non più solo dal titolo dei commenti).** La struttura ad albero (cella `nodes_F`/`nodes_R`/`roots`) codifica solo la relazione fra i **4 campioni sperimentali** (round 2 → round 3, per F e per R separatamente) — 2 nodi per ramo, un dettaglio dell'impianto sperimentale, non una struttura per-sequenza o per-cluster. Le sequenze vivono in un array separato (`seqs::Vector{String}`, `cnts::Matrix{Int64}` di shape `(N,4)`, colonne F2/F3/R2/R3 nell'ordine — confermato dall'output già eseguito nel notebook: `27406×4 Matrix{Int64}`, stesso conteggio di 0.1). **Escludere un cluster è quindi un filtro riga per riga su `seqs`/`cnts`, prima di `sample2hot` — nessun trattamento speciale della struttura ad albero.**

**Fatto:** copia del notebook creata — `PD_energy_model/training/training_2rounds/model_training_7_PNB_negbinom_ll_3layers_cluster_exclusion.ipynb` (il notebook originale resta invariato). Modifiche rispetto all'originale:
1. La cella di caricamento dati è trimmata per fermarsi a `seqs`/`cnts` (rimossi `sample2hot`/`global sequences`/`global counts`, spostati dopo il filtro).
2. Nuova cella: carica `results/phase0/0_2_clustering/cluster_assignment.csv` (path robusto via `@__DIR__`, non dipende dalla cwd di lancio del kernel — a differenza dei path relativi della cella originale, che ne assumono una specifica non verificata), costruisce l'insieme di sequenze da escludere a partire da `EXCLUDED_CLUSTER_IDS`.
3. Nuova cella: applica il filtro, poi ricostruisce `sequences`/`counts` (`sample2hot` sulle sole sequenze tenute).

**Segnalato ma non corretto nella copia** (non è compito di questo intervento, la cella non è stata toccata): la cella di caricamento originale referenzia `"../NGS_data_sharing/4_counts/..."`, una cartella che **non esiste** in questo checkout di `PD_energy_model` (esiste invece `data/4_counts/`) — se il notebook non gira più così com'è, è indipendente dalle modifiche di questa sessione, verificare separatamente.

**Da verificare a mano prima di eseguire (esplicitamente lasciato a te)**: l'assunzione sull'ordine delle colonne di `cnts` (F2/F3/R2/R3) e sul fatto che `seqs`/`cnts` siano allineati riga per riga — inferito dall'uso di `counts[:,1:2]`/`counts[:,3:4]` più sotto nel notebook originale e dalla shape già osservata, non confermabile senza leggere la firma di `merge_counts` (in `PD_data_utils`, non disponibile in questo ambiente).

### 1.3.a — Prima ablazione: escludere solo il cluster dominante

**Configurato come default nella copia del notebook.** `EXCLUDED_CLUSTER_IDS = [0]` — il cluster dominante (rappresentante `VDYNPWLLFLAQPWQ`, 440 sequenze, 34% della massa di read in round 3), l'unica delle 23 sequenze BLI senza segnale di legame misurabile (P7, Kd_nM=NA) e PSBinder-positivo (probabilità 0,86) — vedi `Workplan_phase0_instantiated.md` §0.6/§0.8.b. Prima di generalizzare a $M=5$–$10$ repliche su split casuali per cluster, questo run isolato risponde a una domanda più mirata: **il modello cambia in modo sostanziale escludendo il solo cluster più sospetto di essere un artefatto (TUP/legame al supporto)?** Se sì, è un argomento indipendente (non richiede le repliche complete) a favore della causa B già discussa in Fase 0. Esecuzione: manuale, a tuo carico (nessuna esecuzione Julia possibile da questo ambiente).

**Script preparabile lato valutazione (non training), una volta ottenuti i pesi di 1.3.a: `src/phase1_diagnostics/run_1_3_replica_ensemble_eval.py`.**
1. Confronta le predizioni del modello 1.3.a con quelle del modello attuale, sulle stesse popolazioni già usate in Fase 0 (training set, BLI, i 19 rappresentanti di cluster di §0.6) — variazione di ranking, non solo di valore assoluto (l'energia non standardizzata non è comparabile 1:1 fra due fit diversi).
2. Una volta disponibili $M\geq2$ repliche (split generali, non solo 1.3.a): calcola $\sigma_{\text{repliche}}(a)$ sui punti Dirichlet di 0.1 e sulle 23 sequenze BLI, verificando esplicitamente la distinzione del piano madre fra $\mathrm{Var}_{x\sim a}[E(x)]$ (aleatoria, deve annullarsi allo stadio hard) e $\mathrm{Var}_{\text{repliche}}$ (epistemica, quella che entra in $\kappa\sigma$).

**Output:** `results/phase1/1_3_replica_ensemble/{model_1_3a_vs_current_ranking.csv, sigma_repliche.csv, sigma_diagnostics.png}` (il primo file appena 1.3.a è addestrato manualmente, gli altri due dopo le repliche generali).

**Repliche generali (oltre 1.3.a): $M=5$–$10$ split casuali per cluster**, stesso meccanismo di esclusione della copia del notebook con `EXCLUDED_CLUSTER_IDS` diverso per ciascuna — esecuzione manuale, non ancora programmata in dettaglio (numero di repliche e criterio di split da confermare quando si arriva a questo punto).

---

## 1.4 — Scala di baseline

**Input verificato:** stesso training set (0.1/0.2), stesso `cluster_assignment.csv` per gli split.

**Percorso proposto: (1) e (2) interamente in Python, senza Julia — (3) resta subordinato alla stessa domanda di 1.3.** I baseline 1 (composizione, 20 feature) e 2 (additivo one-hot, 300 feature) sono modelli lineari semplici: non serve la verosimiglianza binomiale negativa completa di `PhageNegBinom` per confrontare *quanta struttura* catturano rispetto alla MLP, un fit lineare standard sulle stesse osservazioni è sufficiente per il confronto di Spearman richiesto dal piano madre.

**Script da creare: `src/phase1_diagnostics/run_1_4_baseline_comparison.py`**
1. Split 80/20 **per cluster** (non per read) usando `cluster_assignment.csv`, seed fisso.
2. Baseline 1 — 20 feature di composizione (frequenza per amminoacido, ignora posizione): regressione lineare (o Ridge leggero) su `log(Count+1)` o sull'arricchimento già usato altrove.
3. Baseline 2 — 300 feature one-hot (posizione × amminoacido): stessa regressione, con regolarizzazione (Ridge) vista la dimensionalità.
4. Modello 3 (MLP attuale) valutato sullo split di test **con i pesi già disponibili**, via `mlp_forward`.
5. Spearman fra predizione e osservato, sul set di test, per i tre modelli.

**Limite da dichiarare esplicitamente, non aggirabile senza Julia**: il modello 3 attuale è stato addestrato sull'**intero** training set, incluso quello che qui diventa "test" per (1)/(2) — il confronto a 3 vie ha quindi una asimmetria strutturale a favore di (3) (nessun vero held-out per lui). Confronto onesto a 3 vie richiede riaddestrare anche (3) sullo stesso split ridotto — stessa infrastruttura Julia di 1.3, stessa domanda aperta. **Fino a risposta, questo script produce un confronto onesto (1) vs (2), e un confronto (1)/(2) vs (3) esplicitamente etichettato come "non a parità di condizioni".**

**Output:** `results/phase1/1_4_baseline_comparison/{spearman_by_model.csv, scatter_baselines.png}`.

---

## 1.5 — Predizione su famiglia non osservata

**Stessa dipendenza infrastrutturale di 1.3** (richiede di riaddestrare escludendo interi cluster, non solo valutare) — nessuna parte è eseguibile prima di avere risposta alle domande di §1.3. Una volta disponibile un modello riaddestrato con $k$ cluster arricchiti esclusi:

**Script da creare: `src/phase1_diagnostics/run_1_5_heldout_family_eval.py`**
1. Predizione (energia) sulle sequenze dei cluster esclusi, confronto con il loro arricchimento osservato (Spearman).
2. Diagnostica di supporto: distribuzione dei residui dopo un fit additivo (riusa il baseline 2 di 1.4) — una coda pesante concentrata in poche famiglie è già suggerita dalla concentrazione di massa vista in 0.2.
3. Prestazione in funzione di $N_{\text{eff}}$ usato in training (0.2) — richiede più run a diversa taglia di training set, quindi più ripetizioni Julia.

**Output:** `results/phase1/1_5_heldout_family/{spearman_heldout.json, residual_distribution.png}`.

**Soglia numerica (Gate 1, Spearman su famiglia esclusa): da fissare dopo 1.3, prima di eseguire 1.5** (deciso con l'utente) — 1.3/1.3.a danno la prima evidenza concreta su quanto il modello sia sensibile all'esclusione di famiglie, informazione utile per calibrare una soglia sensata invece di sceglierla nel vuoto. Resta comunque fissata **prima** di eseguire 1.5 stessa, non dopo averne visto l'esito — l'ordine cambia solo rispetto a 1.3, non rispetto a 1.5.

---

## 1.6 — Confezionamento del modello di energia

**A differenza di 1.3/1.5, questa è interamente instantiabile lato Python/JAX fin da ora** — il meccanismo di confezionamento (formato artefatto, loader, factory con `mode`/`kappa`) non dipende dal numero di repliche realmente disponibili: si può implementare e testare con una sola replica come placeholder, poi estendere a $M$ reali non appena 1.3 le produce.

**Modifiche in `methods/colabdesign_energy_guidance/colabdesign/energy_model/model_3layer_v2.py`** (stessa copia di 1.1/1.2, non `model_3layer.py` legacy):
1. Formato artefatto: `.npz` (nessuna nuova dipendenza — `pyarrow`/`safetensors` non sono già fra le dipendenze di `PA_PB1`, verificato in 0.7; `.npz` usa solo `numpy`, già presente ovunque). Contenuto: pesi impilati (`dense1/2/3`, `W`+`b`, dimensione principale $M$), $\mu_m$/$\sigma_m$ per replica, metadata (versione formato, hash commit del fork, data, modalità).
2. `save_bundle(paths: list[str], out_path: str)`: prende $M$ file JSON (uno per replica, formato attuale) + le rispettive statistiche di standardizzazione, produce l'artefatto unico.
3. `load_bundle(path) -> dict` con validazione esplicita delle shape per replica (`(45,300)`, `(15,45)`, `(1,15)` — asserzioni immediate, non un traceback a valle dopo ore di calcolo, come richiesto dal piano madre).
4. `make_energy_fn(bundle, mode="st", kappa=1.0, eps=1e-6)` — factory esattamente come lo snippet del piano madre §1.6(a), `jax.vmap` sulle repliche (§1.6(b)), standardizzazione per-replica prima di aggregare (§1.6(c), già isolata in 1.2), guardia numerica su `jnp.sqrt` (§1.6(d)).
5. `make_energy_aux(bundle)` — la versione multi-replica di quanto già introdotto in 1.1 per singola replica, con l'aggiunta di `sigma` (dispersione fra repliche) e `arom_frac`.

**Retrocompatibilità esplicita da testare**: `make_energy_fn(weights_path, ...)` (firma attuale, singolo file JSON) deve continuare a funzionare invariata — i chiamanti esistenti (`generate_sequences.py`, `optimize_sequences.py`, `model_energy_guidance.py`, gli analoghi in `energy_guidance/`) non vanno rotti da questo refactor. Può restare come funzione separata che internamente costruisce un bundle "a una replica" e delega alla nuova factory, o restare del tutto indipendente — dettaglio implementativo, non un vincolo di design.

**Test — 7 criteri di accettazione del piano madre, ciascuno mappato a un test esplicito in `energy_guidance/test_energy_guidance.py`:**

| # | Criterio (piano madre) | Test |
|---|---|---|
| 1 | Coerenza ai vertici | `energy_fn(one_hot(x))` con `kappa=0` riproduce l'energia standardizzata entro $10^{-5}$ |
| 2 | Gap identicamente nullo in modalità `st` | `energy_aux(a)["gap"] == 0` per ogni `a`, inclusi punti vicino al baricentro (0.1 ha già mostrato che il campionamento casuale non li distingue — usare comunque come test di regressione, non come criterio di calibrazione) |
| 3 | Gradiente non nullo | `∂energy_fn/∂a ≠ 0` |
| 4 | Gradiente di $\kappa\sigma$ | finito e non nullo dove le repliche concordano |
| 5 | Retrocompatibilità | `energy_fn=None` riproduce la traiettoria di ColabDesign originale a parità di seed |
| 6 | Modalità legacy | `mode="soft", kappa=0` riproduce i run precedenti, gap incluso — è il braccio di controllo della Fase 2, non solo un test |
| 7 | Test placeholder esteso | `test_energy_fn` (già in `af/loss.py`) esteso a verificare che $\kappa>0$ modifichi il comportamento nella direzione attesa |

**Output:** artefatto `.npz` in `data/energy_model_params/` (in `PA_PB1`, stessa cartella dei pesi attuali) + `results/phase1/1_6_bundle_packaging/acceptance_test_report.json`.

**Script di verifica lato `PA_PB1`: `src/phase1_diagnostics/run_1_6_bundle_validation.py`** — carica l'artefatto prodotto dal fork e ne verifica le shape/statistiche indipendentemente dal codice del fork stesso (doppio controllo, stesso spirito della validazione encoder incrociata in 0.1).

**Stato: eseguita, con 2 dei 7 criteri non verificabili in questo ambiente (2026-08-27).** Implementato in `model_3layer_v2.py`: `save_bundle`/`load_bundle` (`.npz`, validazione shape esplicita — testata anche su un bundle deliberatamente corrotto, rilevato correttamente), `make_energy_fn`/`make_energy_aux` **unificate** per accettare sia un path singolo (comportamento legacy, delega a un bundle $M=1$ costruito al volo) sia un bundle multi-replica già caricato — un solo punto di codice per entrambi i casi, non due funzioni separate come inizialmente descritto sopra. Bundle placeholder ($M=1$, $\mu$/$\sigma$ reali di 1.2) scritto in `data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_bundle_v1.npz`. Commit locale su `energy_guidance` (`e1ab467`), non pushato.

**Deviazione dichiarata dal piano madre**: `kappa` di default a `0.0` (non `1.0` come nello pseudocodice di §1.6(a)) — scelto per garantire che una chiamata legacy (`make_energy_fn(weights_path, ...)` senza specificare `kappa`) sia **esattamente** equivalente al comportamento pre-1.6, invece di introdurre silenziosamente un termine di pessimismo per chi non lo richiede esplicitamente. Stesso ragionamento per `mode`: default `"soft"` anche per il bundle (il piano madre mostra `"st"` come default nello pseudocodice) — i punti di chiamata di Fase 2 sceglieranno `mode` esplicitamente quando confronteranno i due bracci.

**Criteri di accettazione (piano madre):**

| # | Criterio | Esito |
|---|---|---|
| 1 | Coerenza ai vertici (kappa=0) | ✅ verificato — media ensemble coerente con calcolo a mano |
| 2 | Gap identicamente nullo in `st` | ✅ verificato — singola replica, bundle M=3 di test, bundle M=1 reale su disco |
| 3 | Gradiente non nullo | ✅ verificato — 20/20 punti |
| 4 | Gradiente di $\kappa\sigma$ | ⚠️ parziale — finito e non-NaN verificato, non isolata esplicitamente la non-nullità del solo termine $\kappa\sigma$ |
| 5 | Retrocompatibilità (`energy_fn=None`, traiettoria AfDesign) | ❌ **non verificabile qui** — richiede un run AfDesign reale (PDB + parametri AF2), fuori dal perimetro CPU-only di Fase 1 |
| 6 | Modalità legacy riproduce i run precedenti | ✅ verificato — `make_energy_fn(path,...)` bit-identico prima/dopo il refactor a bundle |
| 7 | Test placeholder (glicina) esteso a $\kappa>0$ | ❌ **non fatto** — `energy_guidance/test_energy_guidance.py` (test di integrazione AfDesign/PDB) non toccato in questa sessione, richiederebbe comunque GPU/PDB per essere eseguito |

**4 su 7 pienamente verificati in questo ambiente CPU-only; i 2 che richiedono AfDesign reale (criteri 5 e 7) restano da fare quando si avrà accesso a GPU/PDB — naturale in Fase 2 §2.1, che deve comunque eseguire ColabDesign per il confronto soft/ST.** Non forzati né simulati qui.

**Output:** `results/phase1/1_6_bundle_packaging/acceptance_test_report.json`, `data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_bundle_v1.npz`.

---

## Riepilogo dipendenze e ordine

```
1.1, 1.2, 1.6 ── FATTE (2026-08-27), stesso file (model_3layer_v2.py nel fork), commit locali
                  su energy_guidance (1d5277b, b8e2557, e1ab467), non pushati
1.3.a ── indipendente, eseguibile subito (manualmente, Julia) — prossimo passo
1.3 (repliche generali), 1.5 ── dopo 1.3.a: stessa infrastruttura Julia, ambiente e meccanismo di
                                 esclusione ormai chiari, solo l'esecuzione manuale resta da fare
1.4 ── (1)/(2) indipendenti (solo Python) eseguibili subito, (3) a parità di condizioni dipende dal
        riaddestramento di 1.3
```

**Fatto**: 1.1 → 1.2 → 1.6 (packaging, con placeholder a una replica $M=1$) — un artefatto funzionante e testato, 4/7 criteri di accettazione verificati in CPU (i rimanenti 2 richiedono AfDesign/PDB/GPU, rimandati a Fase 2 §2.1), pronto per essere consumato da Fase 2 **senza aspettare le repliche**, che restano necessarie solo per §2.3 (lower confidence bound). **Prossimo**: 1.3.a (manuale, Julia) → valutazione via `run_1_3_replica_ensemble_eval.py` → repliche generali di 1.3 → 1.4 (confronto a 3 vie completo) → 1.5 (soglia fissata dopo aver visto 1.3).

## Decisioni/domande aperte (riepilogo)

1. ~~Ambiente Julia~~ — **risolto**: `PD_energy_model/Project.toml`, pacchetti già in locale.
2. ~~Trattamento della struttura ad albero nell'esclusione cluster~~ — **risolto**: è un filtro su `seqs`/`cnts`, la struttura ad albero (F/R, round2→round3) non è toccata (§1.3).
3. ~~Nome del file copia~~ — **fatto**: `model_3layer_v2.py`, creato e committato (§1.1).
4. **Push dei commit di 1.1/1.2/1.6** (`1d5277b`, `b8e2557`, `e1ab467`) sul remote `github.com/uguzzoni/colabdesign_energy_guidance` (branch `energy_guidance`, come richiesto per il branch — i commit sono locali, il push non ancora autorizzato) (§1.1/§1.2/§1.6).
8. **2 dei 7 criteri di accettazione di §1.6 richiedono AfDesign/PDB/GPU** (retrocompatibilità della traiettoria reale, test placeholder glicina con κ>0) — non verificabili nel perimetro CPU-only di Fase 1, da fare in Fase 2 §2.1.
5. **Esecuzione di 1.3.a**: manuale, a carico dell'utente (aprire `model_training_7_PNB_negbinom_ll_3layers_cluster_exclusion.ipynb`, verificare le assunzioni segnalate nella cella markdown, eseguire) — nessuna azione richiesta da parte mia finché non ci sono pesi da valutare.
6. **Numero e criterio delle repliche generali** oltre 1.3.a (§1.3) — non ancora specificato, da definire quando si arriva a quel punto.
7. **Soglia numerica di Gate 1** (Spearman su famiglia esclusa, §1.5) — da fissare dopo 1.3, prima di 1.5.
