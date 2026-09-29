# Fase 1 — Addendum (1.7–1.10)

Addendum a [`Workplan_phase1_instantiated.md`](Workplan_phase1_instantiated.md), aperto dopo la revisione dei risultati di §1.1–1.6 (2026-09-14). Raccoglie quattro verifiche di chiusura emerse da quella revisione: nessuna era prevista dal piano madre, tutte operano su materiale già esistente, nessuna richiede GPU.

**Perché esistono.** Tre riguardano conclusioni di Fase 1 che non reggono il peso loro attribuito nel riepilogo (§1.9), o che sono state lasciate implicite (§1.10); una (§1.7) è il prerequisito mancante di §2.3 del piano madre. Una quinta voce (§1.11) non è un'analisi ma un elenco di ricadute sul piano madre.

**Convenzioni ereditate, invariate:**
- I file legacy non si editano in-place. `model_3layer.py` resta intoccato; le modifiche vanno in `model_3layer_v2.py` (già creato in §1.1).
- I notebook Julia legacy non si editano. Ogni nuovo lavoro Julia va in `PD_energy_model/training/training_2rounds/workplan/`, in file nuovi.
- L'esecuzione Julia è **manuale, a carico dell'utente**. Claude Code scrive gli script e prepara le celle; non li esegue.
- I commit sul fork vanno su branch `energy_guidance`. Il push resta non autorizzato (decisione aperta #4 del piano istanziato) — non pushare senza conferma esplicita.

**Ordine consigliato:** 1.7 → 1.9 → 1.10 → 1.8. (1.7 sblocca 1.10 ed è il prerequisito di §2.3; 1.9 è il più veloce fra i restanti.)

**Layout aggiuntivo:**
```
PD_energy_model/training/training_2rounds/workplan/
├── export_replica_weights.jl              # NUOVO — §1.7 passo A
└── dump_baseline_test_predictions.jl      # NUOVO — §1.8 passo A

PA_PB1/src/phase1_diagnostics/
├── build_replica_bundle.py                # NUOVO — §1.7 passo B
├── run_1_8_baseline_bootstrap.py          # NUOVO — §1.8 passo B
└── run_1_10_variance_diagnostics.py       # NUOVO — §1.10

PA_PB1/results/phase1/
├── 1_7_replica_bundle/
├── 1_8_baseline_bootstrap/
├── 1_9_gradient_cluster0/
└── 1_10_variance_diagnostics/
```

---

## 1.7 — Export delle repliche e bundle multi-replica

**Motivazione.** Il bundle prodotto in §1.6 ha $M=1$ (placeholder). Il *lower confidence bound* $E + \kappa\sigma$ di §2.3 del piano madre richiede $M \geq 2$ e **non è eseguibile** allo stato attuale. Le repliche esistono già come `.jld2` Julia (§1.3); manca solo la conversione. È l'unico blocco duro fra Fase 1 e Fase 2.

### Decisione da confermare con l'utente prima di iniziare

Quale famiglia di repliche esportare:

| Opzione | Pro | Contro |
|---|---|---|
| **F-only / 10%** (`random_split_Fonly_10pct_*.jld2`, M=10) | revisione più recente; è la serie da cui provengono i numeri di generalizzazione di §1.3/1.5 | meno diversità fra repliche (esclusione più piccola) |
| **F+R / 20,8%** (serie originale, M=10) | maggiore diversità → $\sigma_{\text{repliche}}$ più informativo | usa anche R, che §1.3 ha poi giudicato meno adatta al test di generalizzazione |

**Raccomandazione**: F-only/10%, per coerenza con i numeri su cui poggia Gate 1. Da confermare.

**Seconda decisione**: il bundle contiene **solo i 10 split** (media = stima bagged) oppure **modello completo + 10 split**. Raccomandazione: solo i 10 split, perché media e dispersione dell'ensemble devono provenire dalla stessa popolazione di modelli; mescolare un modello addestrato su dati diversi introduce un outlier sistematico nella dispersione. Da confermare.

### Passo A — esportatore Julia (script da scrivere, esecuzione manuale)

**File: `PD_energy_model/training/training_2rounds/workplan/export_replica_weights.jl`**

Per ciascun `.jld2` di replica, produrre due artefatti in `workplan/exported_replicas/`:
1. `<nome>_weights.json` — pesi dello stato `selection`, **nello schema identico** al file di produzione `data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json` (stesse chiavi, stesso ordine di indici, stesse convenzioni di trasposizione).
2. `<nome>_kept_sequences.txt` — una sequenza per riga, le sequenze effettivamente usate nel training di quella replica (dopo il filtro dei cluster esclusi).

Il secondo file serve a calcolare $\mu_m/\sigma_m$ in Python sul training set **proprio** di ciascuna replica, come richiede §1.6(c) del piano madre. Non dedurlo dai `cluster_id` esclusi: il filtro effettivo dipende anche da quali sequenze sopravvivono ai filtri di conteggio del notebook, e ricostruirlo a posteriori è una fonte di disallineamento silenzioso.

**Passo A0 — test di regressione dell'esportatore, da fare per primo.** Esportare con lo stesso codice il modello **di produzione** dal suo `.jld2` e verificare che il JSON risultante riproduca quello esistente entro $10^{-6}$. Se questo test non passa, l'esportatore è sbagliato e tutto ciò che segue è contaminato. Non procedere finché non passa.

### Passo B — costruzione del bundle (Python)

**File: `PA_PB1/src/phase1_diagnostics/build_replica_bundle.py`**

1. Caricare gli $M$ JSON esportati con `load_energy_model` (già esistente, invariato).
2. Per ciascuna replica $m$: caricare `<nome>_kept_sequences.txt`, codificare con l'encoder di `src/phase0_diagnostics/data_loading.py` (quello validato in §0.1 a $8\times10^{-7}$), calcolare $E$ su tutte le sequenze tenute con `mlp_forward`, e da lì $\mu_m$ e $\sigma_m$.
   **Non riusare $\mu = 3{,}398753$ / $\sigma = 2{,}087558$ di §1.2**: quelle sono le statistiche globali del modello di produzione sul training set completo. Applicarle a tutte le repliche reintroduce esattamente l'offset di nuisance che la standardizzazione per replica esiste per rimuovere, e la dispersione fra repliche finirebbe a misurare differenze di intercetta invece di disaccordo sul ranking.
3. Chiamare `save_bundle` (già esistente da §1.6) con i pesi impilati e i $\mu_m/\sigma_m$ per replica.
4. Metadati obbligatori nel bundle: famiglia di repliche scelta, $M$, cluster esclusi per replica, hash del commit del fork, data.

**Output:**
- `data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_bundle_M10.npz`
- `results/phase1/1_7_replica_bundle/{replica_stats.csv, bundle_validation.json}`

### Verifiche di accettazione

| # | Test | Criterio |
|---|---|---|
| 1 | Round-trip dell'esportatore | JSON del modello di produzione riprodotto entro $10^{-6}$ (passo A0) |
| 2 | Shape | `load_bundle` valida tutte le repliche senza errori; dimensione principale $= M$ |
| 3 | Standardizzazione per replica | per ogni $m$, $z_m$ ha media $\approx 0$ e deviazione standard $\approx 1$ **sul training set di quella replica**, non su quello globale |
| 4 | Dispersione non degenere | $\sigma_{\text{repliche}}$ sulle 23 sequenze BLI è $> 0$ e non costante; se risultasse quasi costante, è il sintomo descritto in §1.6(c) e va indagato prima di procedere |
| 5 | Modalità `st` preservata | `energy_aux(a)["gap"] == 0` per ogni $a$ anche con $M=10$ |
| 6 | Retrocompatibilità | `make_energy_fn(weights_path, ...)` con path singolo continua a funzionare invariata |

**Costo stimato:** passo A ~1 h di scrittura + esecuzione manuale; passo B ~1 h. CPU.

---

## 1.8 — Robustezza statistica del confronto fra baseline (§1.4)

**Motivazione.** Il riepilogo di Fase 1 conclude che "solo l'MLP a piena capacità generalizza in modo non banale, le baseline lineari collassano/invertono segno". Questa conclusione poggia su **N = 19 sequenze di test**. Con N = 19, l'intervallo di confidenza al 95% di $r = 0{,}34$ è circa $[-0{,}14,\ +0{,}69]$: include lo zero, e non è distinguibile da $r = 0{,}00$ (one-hot) né da $r = -0{,}06$ (composizione).

È l'argomento che giustifica mantenere l'MLP a ~14.300 parametri invece di passare ai modelli semplici di Fase 4, con $N_{\text{eff}} \approx 1.400$. Merita più di 19 punti.

Nota di contrasto: §1.3/1.5 hanno la replicazione ($M=10$, segno positivo 10 volte su 10) e reggono. È specificamente §1.4 a esserne priva.

### Passo A — dump delle predizioni per sequenza (Julia, esecuzione manuale)

**File: `PD_energy_model/training/training_2rounds/workplan/dump_baseline_test_predictions.jl`**

I `.jld2` dei 5 modelli di §1.4 esistono già (`model_A_*_Fonly_90split.jld2`, `model_B_*_Fonly_90split.jld2`). Manca solo il dump **per sequenza** — il notebook ha scritto solo le correlazioni aggregate.

Per ciascun modello, e separatamente per train e test, emettere un CSV con: `sequence`, `energy_selection_pred`, `selectivity_pred`, `enrichment_obs`, `split` (train/test). Stesso filtro `counts > 10` in entrambi i round già usato nel notebook, per non cambiare la popolazione rispetto ai numeri pubblicati.

**Output:** `workplan/exported_predictions/1_4_per_sequence_{model}.csv`

### Passo B — bootstrap (Python)

**File: `PA_PB1/src/phase1_diagnostics/run_1_8_baseline_bootstrap.py`**

1. Bootstrap non parametrico (10.000 ricampionamenti con reimmissione sulle 19 sequenze di test) della correlazione, per ciascuno dei 5 modelli e per ciascuna delle metriche già riportate (energia, selectivity).
2. Intervalli di confidenza percentile al 95% per ciascun modello.
3. **Confronto appaiato**, che è il test che conta davvero: bootstrap della *differenza* $r_{\text{MLP}} - r_{\text{baseline}}$ ricampionando **le stesse sequenze** per entrambi i modelli. Riportare la frazione di ricampionamenti in cui la differenza ha il segno atteso. Il confronto appaiato è molto più potente di due intervalli separati, perché elimina la varianza dovuta a quali sequenze sono nel test set.
4. Stessa procedura sul train (N = 126) come controllo: lì le differenze devono risultare nette, e se non lo sono c'è un problema nell'implementazione del bootstrap, non nei dati.

**Output:** `results/phase1/1_8_baseline_bootstrap/{bootstrap_ci.csv, paired_differences.csv, forest_plot.png}`

### Criteri di lettura, da fissare prima di guardare i risultati

- **La differenza appaiata MLP − baseline ha segno atteso in ≥ 95% dei ricampionamenti** → la conclusione di §1.4 regge, l'MLP è giustificato, la Fase 4 resta rimandata.
- **La differenza appaiata è ambigua (< 95%)** → §1.4 non discrimina fra le architetture con i dati disponibili. Due esiti possibili, da decidere con l'utente: replicare §1.4 con $M=5$ seed di split (~4–5 h di training Julia, riusa l'infrastruttura di §1.3), oppure riformulare la conclusione come "non discriminante" e anticipare la valutazione delle architetture semplici.

In entrambi i casi, il riepilogo di Fase 1 va riscritto di conseguenza: la formulazione attuale non è sostenibile senza questo test.

**Costo stimato:** passo A ~30 min + esecuzione; passo B ~1 h. CPU.

---

## 1.9 — Direzione del gradiente sul modello con cluster dominante escluso

**Motivazione.** §0.4 ha stabilito che W è l'unico amminoacido con gradiente medio negativo, cioè che la direzione di discesa del modello di energia è quasi esclusivamente aromatica. §0.6 ha identificato il cluster dominante (`VDYNPWLLFLAQPWQ`, 34% della massa di read in round 3) come candidato TUP credibile: unica sequenza BLI senza segnale di legame misurabile, PSBinder-positiva a 0,86.

L'ipotesi implicita mai testata è che la direzione aromatica di §0.4 sia indotta da quel cluster. §1.3.a ha prodotto il modello che permette di verificarlo (`modelPNB_3layers_negbinom_ll_excluded_cluster0.jld2`), ma il confronto eseguito è stato modello-vs-modello (correlazione 0,645 sull'energia `selection`), **non** una ripetizione dell'analisi di gradiente.

Nota che i risultati di §1.3 danno già un'attesa: l'esclusione del cluster 0 perturba il modello **meno** di un'esclusione casuale di massa equivalente (0,645 contro 0,43 ± 0,05 sull'energia `selection`; 0,797 contro 0,35 ± 0,04 sulla selectivity F). Ci si attende quindi che la direzione aromatica **sopravviva**. Se sopravvive, l'ipotesi "artefatto sperimentale" non è più disponibile come spiegazione del gradiente aromatico, e la questione va riportata interamente sulla popolazione selezionata.

### Metodo

1. Esportare i pesi di `modelPNB_3layers_negbinom_ll_excluded_cluster0.jld2` in JSON con lo stesso esportatore di §1.7 passo A (già scritto a quel punto — riusare, non duplicare).
2. Parametrizzare `src/phase0_diagnostics/run_0_4_gradient_direction.py` sul path dei pesi e sulla cartella di output (argomento CLI o variabile d'ambiente). **Refactor minimo e non invasivo**: il comportamento di default deve restare identico, in modo che rieseguirlo senza argomenti riproduca esattamente i risultati di §0.4 — verificarlo come primo passo.
3. Rieseguire sul modello cluster0-escluso, valutando il gradiente sulle stesse popolazioni di §0.4: i vertici one-hot del training set e i 23 vertici BLI.
4. Produrre il confronto: ranking dei 20 amminoacidi affiancato (modello completo vs cluster0-escluso), correlazione di Spearman fra i due ranking, e differenza della mappa posizionale $15 \times 20$.

**Attenzione alla popolazione di valutazione**: valutare il modello cluster0-escluso anche sulle sequenze del cluster 0 (che non ha visto) è legittimo e informativo, ma va riportato separatamente dalla valutazione sulle sequenze tenute. Non mescolare i due insiemi in una media unica.

**Output:** `results/phase1/1_9_gradient_cluster0/{aa_ranking_comparison.csv, position_map_diff.png, summary.json}`

### Criteri di lettura

- **W resta l'unico (o fra i pochi) con gradiente medio negativo, ranking correlato con quello di §0.4** → la direzione aromatica è una proprietà della popolazione selezionata, non del contaminante. §0.6 si ridimensiona a limite dichiarato, e la questione aromatica resta aperta senza la spiegazione "artefatto".
- **La direzione aromatica scompare o si attenua nettamente** → §0.4 e §0.6 descrivono lo stesso fenomeno, il modello aveva appreso la firma del legame al supporto, e il modello cluster0-escluso diventa il candidato naturale per la Fase 2. Sarebbe un esito importante e va trattato come tale, non come dettaglio.

**Costo stimato:** ~30 min, CPU (l'export riusa §1.7, il refactor è minimo, il run è quello di §0.4).

---

## 1.10 — Diagnostica delle due varianze

**Motivazione.** Il punto 2 di §1.3 nel piano istanziato non è stato eseguito. Il piano madre (§1.3, "Distinzione da mantenere") richiede di verificare esplicitamente che le due varianze in gioco siano distinte e che quella usata in $\kappa\sigma$ sia quella giusta:

| Varianza | Natura | Comportamento atteso |
|---|---|---|
| $\mathrm{Var}_{x\sim a}[E(x)]$ | aleatoria rispetto al rilassamento | decresce durante l'annealing, si annulla allo stadio hard |
| $\mathrm{Var}_{\text{repliche}}[E(a)]$ | epistemica | cresce allontanandosi dai dati; è quella che entra in $\kappa\sigma$ |

È il controllo che verifica che il termine di pessimismo stia misurando la cosa giusta **prima** di metterlo in una loss. Diventa banale una volta esistente il bundle $M=10$.

**Dipendenza: richiede §1.7.**

### Metodo

**File: `PA_PB1/src/phase1_diagnostics/run_1_10_variance_diagnostics.py`**

1. Caricare il bundle $M=10$ di §1.7.
2. Su tre popolazioni di punti: (a) i punti Dirichlet di §0.1 (ricampionare con lo stesso seed se non salvati), (b) i 23 vertici BLI, (c) i 19 rappresentanti di cluster di §0.6.
3. Calcolare per ciascun punto:
   - $\sigma_{\text{repliche}}$, cioè la dispersione fra le $M$ repliche del valore standardizzato **dopo** la standardizzazione per replica;
   - $\mathrm{Var}_{x\sim a}[E(x)]$ stimata per Monte Carlo ($K = 64$ campioni one-hot da $a$), solo sui punti Dirichlet, dove $a$ non è degenere.
4. Verifiche:
   - $\sigma_{\text{repliche}}$ **cresce** allontanandosi dal training set. Proxy operativo di "distanza": distanza di Hamming minima dal training set, oppure appartenenza a un cluster osservato. Se $\sigma_{\text{repliche}}$ non cresce, il termine $\kappa\sigma$ non porta l'informazione per cui esiste, e va detto prima di §2.3, non dopo.
   - $\mathrm{Var}_{x\sim a} \to 0$ ai vertici e cresce verso il baricentro (comportamento atteso per costruzione — è un controllo di correttezza dell'implementazione).
   - Le due quantità sono **poco correlate** fra loro. Se lo fossero fortemente, il pessimismo starebbe penalizzando l'indecisione dell'annealing invece dell'ignoranza del modello.
5. Riportare la scala di $\kappa\sigma$ in unità di $\sigma_{\text{train}}$, per informare la scansione di $\kappa \in \{0,\ 0{,}5,\ 1,\ 2\}$ di §2.3: se $\sigma_{\text{repliche}}$ è di ordine 0,1 mentre l'energia standardizzata varia di ordine 1, allora $\kappa = 2$ è un termine trascurabile e la griglia va rivista.

**Output:** `results/phase1/1_10_variance_diagnostics/{variance_by_population.csv, sigma_vs_distance.png, aleatoric_vs_epistemic.png}`

**Costo stimato:** ~1–2 h, CPU. Dipende da §1.7.

---

## 1.11 — Ricadute sul piano madre

Non è un'analisi: è l'elenco delle modifiche da portare in `Workplan_peptidi_PA-PB1_v2.md` una volta chiuse le quattro sottofasi sopra. Da applicare in un unico passaggio, non frammentariamente.

1. **§0.6 (screening TUP), criteri di lettura.** Il risultato di §1.3.a (esclusione del cluster dominante meno perturbante di un'esclusione casuale di massa equivalente) contraddice l'attesa "il modello di energia è ancorato in misura sostanziale su un artefatto sperimentale". La contaminazione resta un limite da dichiarare, non un fattore che condiziona la Fase 2. Da riscrivere dopo §1.9.

2. **Gate 1.** La soglia numerica non è mai stata fissata prima dell'esecuzione, quindi fissarla ora equivarrebbe a fissarla sui risultati. Sostituirla con il criterio che i dati effettivamente sostengono:

   > Gate 1 superato sulla componente **relativa** (selectivity): segno positivo in 10 split su 10 (test dei segni, $p \approx 0{,}001$), media $0{,}45 \pm 0{,}08$ (errore standard della media su 10 split), replica indipendente in §1.4 con lo stesso ordine di grandezza. Non superato — e non applicabile — sulla componente **assoluta** (abbondanza), per le ragioni strutturali sotto.

   Aggiungere la motivazione: il conteggio in F2 è il prodotto della rappresentazione iniziale in libreria (estrazione casuale in sintesi, impredicibile dalla sequenza e dominante a conteggi bassi) per la sopravvivenza al round (dipendente dalla sequenza). La selectivity $\log(F3/F2)$ cancella il primo fattore. Il fallimento sull'abbondanza held-out è quindi strutturale e atteso, e riguarda una quantità che la Fase 2 non usa.

3. **Vincolo d'uso da propagare a §2.2 e §2.3.** Il termine di energia è validato come **confronto relativo fra candidati**, non come stima assoluta. Non blocca nulla (è l'uso previsto), ma determina come si interpreta $\kappa\sigma$ e come si tara $w_E$.

4. **§1.4 nel riepilogo di Fase 1.** Da riformulare secondo l'esito di §1.8.

5. **Nuova sottofase §2.0 in Fase 2.** I criteri 5 e 7 di §1.6 e la verifica dello straight-through su traiettoria reale sono tutti GPU-dipendenti e tutti prerequisiti del confronto di §2.1. La verifica sintetica di §1.1 ha prodotto un gap massimo di $0{,}25\,\sigma$ contro le mediane reali di $5{,}7$–$29{,}5\,\sigma$ misurate in §0.7: due ordini di grandezza sotto il regime che conta. Il meccanismo è dimostrato, l'efficacia no. Vanno scritti come blocco di apertura esplicito, altrimenti finiscono per essere saltati sotto la pressione di far partire le ablazioni. Il campo `gap` di `energy_aux` è già strumentato per farlo: basta un run.

---

## Riepilogo dipendenze

```
1.7 passo A (Julia, manuale) ──┬──> 1.7 passo B (bundle M=10) ──> 1.10
                               └──> 1.9 (riusa l'esportatore)

1.8 passo A (Julia, manuale) ──> 1.8 passo B (bootstrap)

1.7, 1.8, 1.9, 1.10 ──> 1.11 (ricadute sul piano madre, in un unico passaggio)
```

Nessuna delle quattro richiede GPU. Tutte operano su materiale esistente. Le due esecuzioni Julia (§1.7 A, §1.8 A) restano manuali.

## Decisioni aperte introdotte da questo addendum

1. **Famiglia di repliche da esportare** (F-only/10% vs F+R/20,8%) — §1.7. Raccomandazione: F-only/10%.
2. **Composizione del bundle** (solo i 10 split vs completo + 10 split) — §1.7. Raccomandazione: solo i 10 split.
3. **Esito di §1.8**: se la differenza appaiata è ambigua, replicare §1.4 con $M=5$ seed oppure riformulare la conclusione come non discriminante.
4. Resta aperta la decisione #4 del piano istanziato: **push dei commit** `1d5277b`, `b8e2557`, `e1ab467` (più quelli di questo addendum) sul remote.
