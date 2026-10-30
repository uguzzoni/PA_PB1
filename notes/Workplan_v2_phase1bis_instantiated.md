# Fase 1-bis — Piano istanziato (1b.1–1b.4)

Istanziazione operativa di [`Workplan_v2_fasi_1bis_2_3.md`](Workplan_v2_fasi_1bis_2_3.md) § FASE 1-bis, **versione 2.2** (revisione del 15 settembre 2026). Stesso schema di [`legacy_notes/Workplan1_phase1_instantiated.md`](legacy_notes/Workplan1_phase1_instantiated.md): per ciascuna sottofase, dati/file già verificati nel repo, script da scrivere/già scritti, output atteso o già prodotto, decisioni aperte.

**Stato di questo documento (2026-10-01): parzialmente eseguito.** §1b.1 e la sua rianalisi §1b.1.b sono eseguite, con un esito **rivisto durante l'esecuzione** (vedi sotto — il criterio originale di §1b.1 era mal posto, non il calcolo). §1b.2 (nella forma originale, per la stima assoluta di $\sigma$) è **cancellata**, non solo non eseguita — non serve più, per una ragione strutturale spiegata nella sua sezione. §1b.3 è **riaperta** in forma ridotta e invariante di scala, **istanziata come piano in questo passaggio, non ancora eseguita**. §1b.4 non è ancora decidibile: dipende dall'esecuzione di §1b.3 ridotta.

**Cosa cambia rispetto all'istanziazione precedente (v2.1 → v2.2 del piano madre) — semplificazione sostanziale:**
- **Rimossi interamente**: il riallineamento del gauge (vecchia §1b.1), la diagnostica delle due varianze sull'ensemble (vecchio riuso di `run_1_10_variance_diagnostics.py`), e il **termine di pessimismo $\kappa\sigma$ nella loss** — di conseguenza spariscono anche `run_1b_1_bundle_realigned.py`, `run_1b_1_dispersion_reduction.py`, e ogni bundle `..._realigned_*.npz` pianificato in precedenza. **Nessuno di questi va scritto.**
- Fase 1-bis produce ora **solo** uno stimatore di incertezza **a posteriori** (diagnostica, criterio di selezione in §3.5, annotazione) — mai un componente della loss di design. Il Gate 1-bis è quindi non bloccante per una ragione più forte di prima: lo stimatore non entra più nel loop.
- Il pre-controllo di degenerazione è riformulato sulla **leva** $h(x)$ (invariante di scala, calcolabile senza aver definito $\sigma_n^2$/$\lambda$), non più su $\phi^\top\Sigma\phi/\sigma_n^2$ con una $\Sigma$ provvisoria arbitraria.
- La selezione di $\lambda$ è ora per **massimizzazione dell'evidenza**, non più validazione incrociata sugli stessi split poi usati per validare (circolarità corretta).
- I **due esemplari** dello stimatore (dieci per replica, uno di produzione) hanno ora un'avvertenza esplicita di trasferibilità (la leva scala come $p/N$, le due popolazioni di training hanno numerosità diverse) e due modi alternativi per gestirla, da scegliere in istanziazione.
- La validazione (ora §1b.3, non più un passo dentro §1b.2) specifica gli stimatori (correlazione di Pearson **pesata**, bootstrap pesato, numerosità efficace $N_{\text{eff}}$) invece di lasciarli impliciti.
- Nuova sottofase **§1b.4** (decisione sull'uso) — l'artefatto di decisione alimenta direttamente §3.5, non più anche §2.5 (che non esiste più). **Ora a due esiti, non tre — vedi revisione 2026-10-01 sotto.**

**Revisione del 2026-10-01, dopo l'esecuzione di §1b.1**: il criterio originale di degenerazione era mal posto (soglia "100×" irraggiungibile per costruzione — il massimo osservato sul training stesso è 28,7×). La rianalisi distribuzionale (nuova §1b.1.b) mostra che la leva **non è degenere per l'ordinamento**, solo per la scala assoluta. Di conseguenza: §1b.2 (rifit per la scala assoluta di $\sigma$) è cancellata per una ragione strutturale, non solo perché il vecchio criterio diceva di fermarsi; §1b.3 è riaperta in forma ridotta, basata sulla leva $h(x)$ direttamente invece che su $\sigma$ calibrata; §1b.4 passa da tre a due esiti. Dettagli in ciascuna sezione.

**Perimetro, invariato**: CPU-only, quasi interamente in `PA_PB1`. Nessuna esecuzione Julia nuova (repliche F-only/10% già esportate, §1.7 dell'addendum di Fase 1). Nessuna modifica al fork: l'estrazione di $\phi(x)$ è pura algebra sui pesi già caricati con `load_energy_model`.

**Layout, aggiornato al 2026-10-01 (§1b.2 cancellata, §1b.3 ridotta e riprogettata):**
```
PA_PB1/src/phase1bis_diagnostics/
├── _features.py                                        # condiviso — SCRITTO: extract_phi_batch, compute_leverage
├── run_1b_1_leverage_check.py                           # §1b.1 — SCRITTO ED ESEGUITO (esito rivisto, vedi sotto)
├── run_1b_1b_leverage_reanalysis.py                     # §1b.1.b — SCRITTO ED ESEGUITO (rianalisi, nessun nuovo calcolo di leva)
├── run_1b_3_validation.py                               # §1b.3 (ridotta) — DA SCRIVERE, istanziata come piano qui
└── run_1b_4_decision.py                                 # §1b.4 — SCRITTO, da rieseguire dopo §1b.3 ridotta

PA_PB1/src/phase0_diagnostics/data_loading.py             # load_designed_sequences() GIÀ ESISTENTE, riusata da §1b.1/§1b.1.b
                                                            # load_training_counts_Fonly() ancora da aggiungere, serve a §1b.3

PA_PB1/results/phase1bis/
├── 1b_1_leverage_check/            # fatto — leverage_by_population.csv, summary.json (con campo "revision")
├── 1b_1b_leverage_reanalysis/      # fatto — percentili, code, KS, stratificazione, ECDF, leva-vs-i_ptm
├── 1b_3_validation/                # da fare — vedi §1b.3 ridotta
└── 1b_4_decision/                  # fatto ma da rigenerare dopo §1b.3 ridotta (decision.json attuale riflette
                                     # la lettura superata "degenerazione totale", non più valida)
```

**Nessun artefatto `laplace_last_layer/*.npz` va più prodotto**: §1b.2 (che li avrebbe generati) è cancellata — vedi la sua sezione per il perché. §1b.3 ridotta usa direttamente $h(x)$, non una $\sigma$ calibrata.

---

## Prerequisito comune — pool F-only completo, non ancora disponibile in Python

Le repliche F-only/10% (`PD_energy_model/training/training_2rounds/workplan_phase1/exported_replicas/random_split_Fonly_10pct_{1..10}_weights.json` + `_kept_sequences.txt`, **già su disco**, §1.7) sono addestrate su **F soltanto**, un pool diverso — per composizione **e per numerosità** — da quello F+R su cui è addestrato il modello di **produzione** (`load_training_sequences_onehot()`, esistente). Questa distinzione resta necessaria per §1b.3 ridotta (sotto): ciascuna $h_m(x)$ va costruita sul training set F-only **della propria replica**, non su quello di produzione — la leva scala come $p/N$, quindi usare il pool sbagliato cambierebbe la scala di $h_m$ in modo non uniforme fra repliche.

Funzione da aggiungere a `data_loading.py` (non esiste ancora):

```python
def load_training_counts_Fonly() -> pd.DataFrame:
    """Merge F_2+F_3 soltanto (nessuna R) — lo stesso identico pool usato da
    TRAINING_models_random_split_clusters.ipynb per le repliche F-only/10% di §1.7.
    Popolazione di training delle dieci repliche di validazione — NON quella del
    modello di produzione (F+R, load_training_sequences_onehot() già esistente)."""
```
seguita da un `load_training_sequences_onehot_Fonly()` analogo a quello esistente.

**Verifica esplicita, non solo dichiarazione**: prima di costruire $\Phi$ per le repliche, verificare che l'unione dei 10 `_kept_sequences.txt` più le rispettive sequenze escluse ricostruisca esattamente questo pool (stessa cardinalità, stesse sequenze) — non assumerlo. Registrare $N$ per entrambi i pool (F-only e F+R) negli output di §1b.1, perché $\bar h = p/N$ dipende direttamente da questo numero per ciascun esemplare.

`load_designed_sequences()` (in `data_loading.py`, **già esistente**, Fase 0 §0.5/§0.8.b) fornisce la quarta popolazione richiesta da §1b.1 — i design prodotti nelle campagne ColabDesign (`results/colabdesign/{run1_protocols,run2_wide,run_multimer,run_multimer2}`) — nessun nuovo loader necessario per questa.

---

## Metodo (definizioni, nessuno script)

Approssimazione di Laplace sull'ultimo strato: $\phi(x)\in\mathbb{R}^{15}$ (output dell'ultimo strato nascosto, `dense2` dopo ReLU) + bias costante, $p=16$. Estrazione pura algebra sui pesi già caricati con `load_energy_model` (dense1/dense2), nessuna modifica al fork:

```python
def extract_phi(params, x):  # x: (15,21) Julia-order, come encode_sequence
    h1 = relu(x.flatten() @ params["dense1"]["W"] + params["dense1"]["b"])
    phi = relu(h1 @ params["dense2"]["W"] + params["dense2"]["b"])  # (15,)
    return np.concatenate([phi, [1.0]])  # (16,), bias incluso
```

Usata identica in §1b.1 (leva) e §1b.2 (posterior) — un solo punto di codice, condiviso fra i tre script nuovi (es. come funzione importata da un modulo comune `phase1bis_diagnostics/_features.py`, non duplicata tre volte).

---

## 1b.1 — Pre-controllo di degenerazione

**Da eseguire per primo, prima di scrivere qualunque altro script**: costa minuti e può chiudere l'intera fase.

**Script da creare: `src/phase1bis_diagnostics/run_1b_1_leverage_check.py`**

1. `extract_phi` (sopra) applicata ai pesi del modello di **produzione soltanto** (`data/energy_model_params/PNB_2R_3lay_negbinom_energy_model_weights.json`) — questo pre-controllo non usa le repliche.
2. Costruisce $\Phi$ sul training set di produzione (F+R, `load_training_sequences_onehot()`, $N=27342$), calcola $h(x)=\phi(x)^\top(\Phi^\top\Phi+\lambda I)^{-1}\phi(x)$ con $\lambda$ piccolo (matrice $16\times16$ da invertire — istantaneo).
3. Calcola $h(x)$ su **quattro popolazioni**:
   - il training set stesso (distribuzione di riferimento);
   - le 23 sequenze BLI (`load_bli_population()`, esistente);
   - i rappresentanti di cluster (`results/phase0/0_2_clustering/cluster_assignment.csv`, colonna `is_representative` — stessa popolazione riusata in §1.10 di Fase 1);
   - **un campione dei design di Fase 0** (`load_designed_sequences()`, esistente, ~900 sequenze dalle 4 campagne — popolazione più informativa delle quattro, perché è l'unica che rappresenta il regime in cui lo stimatore verrebbe effettivamente usato).
4. Riporta la distribuzione di $h(x)/\bar h$ per popolazione, con $\bar h=p/N=16/27342\approx5{,}9\times10^{-4}$ (valore di riferimento teorico, verificato contro quello osservato sul training set stesso come controllo di correttezza dell'implementazione — devono coincidere entro l'errore numerico per $\lambda\to0$).

**Output:** `results/phase1bis/1b_1_leverage_check/{leverage_by_population.csv, leverage_ratio_histograms.png, summary.json}`.

**Criterio**:
- $h(x)/\bar h$ resta entro un ordine di grandezza su tutte e quattro le popolazioni, design inclusi → **lo stimatore non discrimina, la Fase 1-bis si chiude qui con esito negativo** (nessuno script successivo va scritto/eseguito — §1b.2/§1b.3 restano non istanziati, si passa direttamente a §1b.4 con l'esito "degenerazione"). La caratterizzazione dell'incertezza è rimandata alla Fase 4.
- $h(x)/\bar h$ supera due o tre ordini di grandezza su una frazione apprezzabile delle sequenze, in particolare sui design → procedere a §1b.2.

**Nota sui proxy di distanza (verifica, non nuovo calcolo)**: la leva sostituisce la distanza di Hamming come misura di lontananza dai dati, per una ragione dimostrabile a priori (attesa teorica di 0,1–17 vicini a distanza 8–10 su $27{.}342$ sequenze contro $20^{15}$ possibili — saturazione quasi certa). Verificarlo una volta con un istogramma **riusando i valori di distanza di Hamming già calcolati in Fase 1** (`results/phase1/1_10_variance_diagnostics/variance_by_population.csv`, colonna `hamming_dist_to_training`) — nessun nuovo calcolo pesante, un paio di righe nello stesso script del punto 4. Registrare esplicitamente che il fallimento del criterio corrispondente in Fase 1 (§1.10) era **non informativo**, non un esito negativo.

**Stato: eseguito (2026-10-01).** `run_1b_1_leverage_check.py` + modulo condiviso `_features.py` (`extract_phi_batch`, `compute_leverage`) scritti ed eseguiti. Controllo di correttezza: $\bar h$ osservato sul training set stesso vs atteso teorico $p/N$, errore relativo $0{,}00\%$ — implementazione validata.

**Nota sul criterio, corretta durante l'esecuzione**: "resta entro un ordine di grandezza" va letto sul comportamento **tipico** (mediana) e sulla **frazione** che supera 100× per i design, non sul singolo valore massimo per popolazione — un outlier isolato è normale nella distribuzione della leva anche quando lo stimatore non discrimina affatto (il primo tentativo di implementazione usava il massimo ed era quindi troppo permissivo). Soglia di "frazione apprezzabile" fissata a 5% (non data numericamente dal piano madre, scelta qui e registrata in `summary.json`).

**Risultato — mediane per popolazione (tabella originale, conservata):**

| Popolazione | N | $h/\bar h$ mediana | $h/\bar h$ massimo | frazione >100× |
|---|---|---|---|---|
| training set | 27.342 | 0,88 | 28,74 | 0,00% |
| 23 BLI | 23 | 0,82 | 1,44 | 0,00% |
| rappresentanti di cluster | 20 | 0,74 | 1,58 | 0,00% |
| design di Fase 0 (tutte le campagne) | 900 | 1,74 | 35,58 | 0,00% |

**Esito originale ("DEGENERE", frazione >100× = 0,00% ovunque) rivisto il 2026-10-01 — vedi §1b.1.b sotto per l'analisi completa.** La correzione introdotta durante l'esecuzione (giudicare sulla mediana/frazione invece che sul massimo per popolazione) era corretta nel principio — un singolo outlier non deve decidere il verdetto. **Ma la soglia stessa ("100×") era mal posta**: il massimo osservato sul training set stesso è 28,7×, quindi nessuna sequenza poteva superare 100× senza essere un'anomalia numerica — la soglia era irraggiungibile per costruzione, non un criterio informativo. Il riferimento corretto è la **distribuzione del training stesso**, non un multiplo assoluto dell'attesa teorica $p/N$.

**Esito corretto, in tre affermazioni distinte (non un verdetto binario):**

1. **Degenerazione confermata per la scala assoluta.** $h$ massimo osservato sui design $\approx 0{,}0208$ (unità assolute) $\Rightarrow \sigma(x)/\sigma_n=\sqrt{1+h}\approx1{,}010$, un aumento di appena l'1,0% anche all'estremo della coda. **Non si costruiscono barre di errore assolute su $E$ con questo stimatore** — questa parte della conclusione originale resta valida.
2. **Non valutato per l'ordinamento.** Gli usi di diagnostica e di selezione (§3.5) dipendono solo dall'ordinamento indotto da $h(x)$, non dalla sua scala assoluta. §1b.1 da solo non può stabilirne l'utilità: serve la validazione held-out, non eseguita perché il criterio originale ne ha impedito l'avvio.
3. **Evidenza preliminare che l'ordinamento discrimini, forte.** La rianalisi §1b.1.b mostra che la distribuzione dei 900 design è nettamente diversa da quella del training (Kolmogorov–Smirnov $D=0{,}47$, $p=6\times10^{-181}$), con il 48,8%/36,4%/10,8% dei design oltre il 95°/99°/99,9° percentile del training — contro lo 0% di BLI e rappresentanti di cluster, che restano dentro la distribuzione del training come atteso. Da confermare con §1b.3 ridotta (sotto).

Verifica della saturazione di Hamming (invariata): confermata (mediana 10, 51% dei punti di §1.10 in [8,10]) — il fallimento del criterio di distanza in Fase 1 §1.10 era non informativo, non un esito negativo.

**Conseguenza**: §1b.2 nella sua forma originale è **cancellata** (ragione strutturale, non solo "il vecchio criterio diceva di fermarsi" — vedi sua sezione). §1b.3 è **riaperta in forma ridotta** (sotto), basata direttamente su $h(x)$. `run_1b_4_decision.py` era stato eseguito sulla base dell'esito originale e **va rieseguito** dopo §1b.3 ridotta — il `decision.json` attualmente su disco riflette la lettura superata.

**Output:** `results/phase1bis/1b_1_leverage_check/{leverage_by_population.csv, leverage_ratio_histograms.png, hamming_saturation_check.png, summary.json}` (quest'ultimo ora con un campo `"revision"` che conserva l'esito originale, la data e la ragione della correzione, oltre al contenuto originale invariato).

---

## 1b.1.b — Rianalisi distribuzionale

**Nessun nuovo calcolo di leva**: parte interamente da `leverage_by_population.csv` già scritto da §1b.1.

**Script creato ed eseguito: `src/phase1bis_diagnostics/run_1b_1b_leverage_reanalysis.py`**

Metadati dei design (`protocol_file`, `energy_weight_stage3`, `has_energy`, `i_ptm`) da `load_designed_sequences()` (`data_loading.py`, **già esistente**, la stessa funzione usata in Fase 0 §0.8.b per la stratificazione composizionale — non riscritta). Attaccati a `leverage_by_population.csv` **posizionalmente**, non con un merge per valore: la popolazione `fase0_designs` del CSV è stata scritta iterando `load_designed_sequences()` in ordine, senza dedup e senza scarti (900/900 validi) — una nuova chiamata allo stesso loader riproduce deterministicamente la stessa lista nello stesso ordine (iterazione su file ordinati alfabeticamente). Verificato esplicitamente nello script (confronto riga-per-riga), non assunto — copertura 900/900 (100%).

**(a) Percentili per popolazione** (1/5/25/50/75/90/95/99/99,9, lineare e $\log_{10}$) — `percentiles_by_population.csv`. BLI (N=23) e rappresentanti di cluster (N=20) segnalati esplicitamente come a bassa numerosità per i percentili alti (soglia di cautela: N<50).

**(b) Code dei design contro le soglie del training** — `design_vs_training_tails.csv`. Controllo di sanità: sul training contro se stesso, le frazioni oltre i propri percentili 95/99/99,9 sono 5,00%/1,00%/0,10% esatte, come devono essere. BLI e rappresentanti di cluster: **0% oltre tutte e tre le soglie** (non solo "vicini alle attese nominali" come preventivato nell'incarico — nettamente *sotto*: questi leganti reali hanno leva tipica più bassa del training stesso). I 900 design: **48,8% / 36,4% / 10,8%** oltre le stesse tre soglie — una coda pesantissima rispetto al training.

**(c) Kolmogorov–Smirnov contro il training** — `ks_tests.csv`: design $D=0{,}4749$, $p=5{,}8\times10^{-181}$ (fortissimamente significativo); BLI $D=0{,}147$, $p=0{,}65$ (non significativo); rappresentanti $D=0{,}233$, $p=0{,}19$ (non significativo). Conferma (c) senza dipendere da alcuna soglia arbitraria: la forma della distribuzione dei design è diversa da quella del training, quella dei leganti reali e dei rappresentanti di cluster no.

**(d) Stratificazione dei 900 design** — `leverage_by_protocol.csv`:
- **has_energy**: mediana $h/\bar h$ 2,17 (con energia) vs 1,09 (senza) — quasi il doppio, ma non è il quadro completo.
- **energy_weight_stage3**: 0 → 1,09; 0,05 → 1,78; 0,2 → 1,24; 0,5 → 1,92; 0,7 → 5,29; 0,9 → 5,74 — tendenza complessivamente crescente ma **non monotona in senso stretto** (flessione a 0,2).
- **Per protocollo (il pattern più netto di tutti)**: `gen_energy_D/C/E` (generazione de novo, init casuale + energia fin dall'inizio) hanno mediana **9,9–14,7**, frazione oltre il 99° percentile del training **93–97%**; `opt_anneal_noenergy`/`opt_anneal_energy_B` (ottimizzazione da sequenza di partenza, con o senza energia) restano a mediana **1,08–1,17**, frazione 4–11%. **Non è "l'energia" in sé a spingere fuori dalla regione dei dati, ma specificamente la generazione de novo con energia attiva dall'inizio** — gli stessi pesi di energia applicati a un'ottimizzazione da seme restano vicini ai dati.
- **Per quartile di $i_{\text{ptm}}$**: mediana cresce monotonicamente Q1→Q4 (1,30 → 1,30 → 2,20 → 3,87).
- **Successo vs fallimento strutturale** ($i_{\text{ptm}}\geq0{,}5$, N=51 vs N=849): mediana **4,63** (successo) contro **1,64** (fallimento) — quasi 3 volte più alta.

**Incrocio leva–$i_{\text{ptm}}$, il risultato più rilevante per la Fase 2** (`leverage_vs_iptm.png`): correlazione **positiva e fortemente significativa** su tutti i 900 design (Pearson $r=0{,}244$, $p=1{,}3\times10^{-13}$; Spearman $r=0{,}325$, $p=1{,}2\times10^{-23}$). **I design che AF2 giudica strutturalmente migliori sono sistematicamente più lontani dalla regione coperta dai dati di selezione, non meno.** Conferma esattamente il rischio di conflitto fra le due metriche su cui la Fase 2 si basa: in assenza di mitigazione in-loop (§B.5 del piano madre), l'obiettivo strutturale può star premiando proprio l'allontanamento che rende il modello di energia meno affidabile.

**Output:** `results/phase1bis/1b_1b_leverage_reanalysis/{percentiles_by_population.csv, design_vs_training_tails.csv, ks_tests.csv, leverage_by_protocol.csv, leverage_vs_iptm.png, leverage_ecdf_overlay.png, summary.json}`.

---

## 1b.2 — Stima (CANCELLATA, 2026-10-01)

**Non eseguita, e non più da eseguire — ragione strutturale, non solo "il criterio di §1b.1 diceva di fermarsi".**

Il rifit ridge pesato sulla selectivity, la definizione del termine di rumore $\sigma_n^2$ e la selezione di $\lambda$ (passi 1-2 della versione precedente di questa sezione) servivano **esclusivamente a fissare la scala assoluta** di $\sigma(x)=\sigma_n\sqrt{1+h(x)}$ — cioè l'unico uso che §1b.1/§1b.1.b hanno confermato morto (claim 1: aumento dell'1% anche all'estremo della coda, inutilizzabile per barre di errore assolute).

Per gli usi che contano — diagnostica sui design e criterio di selezione in §3.5 — **non servono**: entrambi dipendono solo dall'**ordinamento** indotto da $\sigma(x)$, e poiché

$$\sigma^2(x) = \sigma_n^2\big(1+h(x)\big)$$

con $\sigma_n$ **costante** (non dipende da $x$), ordinare le sequenze per $\sigma$ è identico a ordinare per $h$, che è **già calcolata** da §1b.1 — nessun rifit, nessuna selezione di $\lambda$, nessun esemplare di produzione separato. È la ragione per cui la fase può proseguire direttamente a §1b.3 senza questa macchineria.

**Conseguenza sul layout**: nessun file `run_1b_2_fit_sigma_estimators.py`, nessun artefatto `data/energy_model_params/laplace_last_layer/*.npz` — rimossi dal piano.

---

## 1b.3 — Validazione, forma ridotta e invariante di scala (riaperta 2026-10-01)

**Istanziata come PIANO in questo passaggio — non eseguita.** Dipende da §1b.1 (fatto) e dalla verifica del pool F-only (prerequisito comune, sopra). Non dipende più da §1b.2 (cancellata): si lavora direttamente su $h(x)$, mai su una $\sigma$ calibrata — poiché $\sigma^2(x)=\sigma_n^2(1+h(x))$ con $\sigma_n$ costante, binnare per $h$ ed ordinare per $h$ sono identici a farlo per $\sigma$ (vedi §1b.2 cancellata).

**Domanda**: "$\sigma$/$h$ cresce con la distanza" non è misurabile con i proxy disponibili (saturazione di Hamming, §1b.1). Quella operativa, e misurabile, è se $h$ **predica l'errore** — esistono famiglie escluse con selectivity osservata.

**Script da scrivere: `src/phase1bis_diagnostics/run_1b_3_validation.py`**

### Protocollo

1. Per ciascuna delle 10 repliche F-only/10% (§1.7, pesi già esportati): calcolare $h_m(x) = \phi_m(x)^\top(\Phi_m^\top\Phi_m+\lambda I)^{-1}\phi_m(x)$ sul **proprio** training set (kept_sequences della replica $m$) — riusa `_features.py` (`extract_phi_batch`, `compute_leverage`) e `load_training_counts_Fonly()` (prerequisito comune, ancora da scrivere), nessuna nuova dipendenza.
2. Calcolare $h_m(x)$ sulle sequenze **escluse** da quella replica (held-out = pool F-only completo − kept_sequences della replica $m$, derivabile in Python, nessun nuovo export Julia).
3. Aggregare le dieci popolazioni held-out. **Usare i valori standardizzati $z_m$ dell'energia, non $E_m$ grezza** — le dieci repliche hanno scale diverse (Fase 1, $\mu_m$/$\sigma_m$ variabili) — per l'accordo con la selectivity osservata, mentre il binning usa $h_m$ direttamente (non necessita di standardizzazione: è già invariante di scala per costruzione).
4. Binnare per $h_m$; entro ciascun bin misurare l'accordo fra $z_m$ e $y(x)=\log(n_{F_3}/n_{F_2})$ (selectivity osservata) con intervalli di confidenza bootstrap, **pesati per la precisione dei conteggi** $w(x)\propto(1/n_{F_3}(x)+1/n_{F_2}(x))^{-1}$ (stessa definizione già usata altrove in questo piano — non richiede il rifit cancellato di §1b.2, è solo la precisione del dato osservato).

**Output:** `results/phase1bis/1b_3_validation/{n_eff_by_binning.csv, h_vs_selectivity_by_bin.csv, monotonicity_plot.png, comparator_results.csv, confound_check.csv, robustness_unweighted.csv}`.

### Criterio

Decrescita monotona dell'accordo held-out su almeno tre bin, intervalli di confidenza che non si sovrappongono fra primo e ultimo — invariato rispetto alla formulazione precedente.

### Tre controlli obbligatori, tutti nello stesso script (invariati nella sostanza)

- **Comparatore**: ripetere l'identico protocollo con $|E|$ standardizzata (cioè $|z_m|$) come predittore di incertezza banale. Se $h$ non lo batte, non porta informazione propria.
- **Confondimento $h$–energia**: riportare la distribuzione di $z_m$ per bin di $h_m$ (entrambi funzioni di $\phi(x)$); se sbilanciata, ripetere stratificando per $z_m$.
- **Robustezza alla ponderazione**: ripetere senza i pesi di precisione sui conteggi, verificare che il risultato non cambi qualitativamente.

**Costo stimato: una giornata, CPU.**

### Comparatore aggiuntivo — leva su feature fisse (one-hot)

Oltre alla leva sull'ultimo strato $h(x)$, calcolare anche la leva sulla codifica one-hot grezza:

$$h_{\text{oh}}(x) = \psi(x)^\top\big(\Psi^\top\Psi+\lambda I\big)^{-1}\psi(x), \qquad \psi(x)\in\mathbb{R}^{300}$$

**Motivazione**: i metodi basati sull'ultimo strato soffrono di *collasso delle feature* — la rete comprime 300 dimensioni di input in 15 attivazioni, quindi una sequenza radicalmente nuova in input potrebbe comunque produrre una $\phi(x)$ interna alla nuvola di training (falso negativo per $h$). La leva su feature fisse è puramente geometrica sui dati osservati (il one-hot grezzo, $A\times L=20\times15=300$, gap escluso), non ammette questo tipo di collasso, e ha $\bar h_{\text{oh}}=p_{\text{oh}}/N\approx300/27342\approx1{,}1\times10^{-2}$ — circa venti volte l'attesa dell'ultimo strato. L'inversione è una matrice $300\times300$ (non $16\times16$), comunque istantanea.

**Trattare come secondo candidato da validare accanto a $h$, non come sostituto**: ripetere l'intero protocollo di validazione sopra (binning, accordo, tre controlli) anche con $h_{\text{oh}}$ al posto di $h$, e confrontare i due criteri di monotonia fianco a fianco. La rianalisi di §1b.1.b suggerisce che $h$ abbia già un'escursione utilizzabile (coda pesante, KS fortemente significativo) — $h_{\text{oh}}$ serve a verificare che non sia un artefatto del collasso delle feature, non perché $h$ sia sospettato di essere insufficiente.

**Output aggiuntivo:** `results/phase1bis/1b_3_validation/{h_oh_vs_selectivity_by_bin.csv, h_vs_h_oh_comparison.png}`.

---

## 1b.4 — Decisione sull'uso

**Riformulata il 2026-10-01: due esiti, non tre.** La vecchia tripartizione distingueva "§1b.1 degenera" da "§1b.1 passa, poi §1b.3 passa/fallisce" — ma §1b.1/§1b.1.b hanno già stabilito che la degenerazione è **parziale e nota**: confermata per la scala assoluta (claim 1, sempre vera, indipendente da qualunque esecuzione successiva), non valutata per l'ordinamento (claim 2, dipende da §1b.3). Non ha più senso un ramo "degenerazione" allo stesso livello degli altri due: la domanda che resta aperta è una sola.

**L'uso con barre di errore assolute su $E$ è escluso in ogni caso** — non è più una riga della tabella, è una premessa che vale sempre, già registrata nella revisione di §1b.1.

| Esito di §1b.3 (ridotta) | Decisione |
|---|---|
| Criterio di monotonia soddisfatto (con $h$ e/o $h_{\text{oh}}$, controlli passati) | Esiste un ordinatore di incertezza utilizzabile. Adottato per diagnostica sui design e annotazione dei candidati finali. §3.5 include una soglia su $h$ e una quota riservata ad alta $h$ |
| Criterio di monotonia non soddisfatto, o comparatore non battuto | Nessun ordinatore di incertezza utilizzabile, oltre a quanto già stabilito in §1b.1 (scala assoluta comunque esclusa). Caratterizzazione rimandata a Fase 4. §3.5 non include soglie su $h$ |

**Script creato: `src/phase1bis_diagnostics/run_1b_4_decision.py`** — legge i `summary.json`/output delle sottofasi precedenti e scrive `results/phase1bis/1b_4_decision/decision.json`. **Già eseguito una volta (2026-10-01) sulla base dell'esito originale di §1b.1** ("degenerazione totale", tre esiti) — quel `decision.json` riflette una lettura ora superata e **va rigenerato** dopo l'esecuzione di §1b.3 ridotta, aggiornando anche la logica dello script ai due esiti sopra (attualmente legge solo `verdict_degenerate` da §1b.1, che non è più il campo da cui far dipendere la decisione — va fatto dipendere dall'esito di §1b.3).

**Stato**: §1b.1/§1b.1.b eseguite, **Fase 1-bis non è più chiusa** — riaperta in attesa di §1b.3 ridotta (istanziata come piano sopra, non eseguita in questo passaggio).

---

## GATE 1-bis

Non bloccante per costruzione, e ora per una ragione più forte che nelle versioni precedenti del piano: lo stimatore non entra mai nella loss di design, quindi nessun esito di Fase 1-bis può bloccare l'avvio della Fase 2. Cambia solo la formulazione dei criteri di selezione di §3.5 e l'esistenza dell'annotazione di incertezza sui candidati finali.

---

## Decisioni da fissare prima dell'esecuzione

**Aggiornate il 2026-10-01** rispetto a `Workplan_v2_fasi_1bis_2_3.md` § E, che non è stato modificato in questo passaggio (il compito riguardava solo questo documento istanziato) e quindi riporta ancora la decisione #1 originale, **resa superflua** dalla cancellazione di §1b.2:

1. ~~Gestione del disallineamento fra i due esemplari di §1b.2 passo 3~~ — **non più applicabile**: §1b.2 è cancellata, non esistono più "due esemplari" da riconciliare (vedi la sua sezione). Da segnalare come discrepanza se si rilegge `Workplan_v2_fasi_1bis_2_3.md` § E senza questo contesto.
2. **Numero di bin** per la validazione di §1b.3 (ridotta) — invariata: da fissare dopo aver misurato $N_{\text{eff}}$ sui dati reali, non prima.
3. **Soglia $\lambda$ per $h_{\text{oh}}$ (comparatore su feature fisse, nuovo)**: stesso principio di $h$ (piccolo, non calibrato — serve solo a rendere invertibile $\Psi^\top\Psi$), da confermare non essere sensibile alla scelta entro un ordine di grandezza prima di fidarsi del confronto fra $h$ e $h_{\text{oh}}$.
4. **Se validare sia $h$ sia $h_{\text{oh}}$ in parallelo o prima solo $h$** (che la rianalisi di §1b.1.b suggerisce già promettente) e aggiungere $h_{\text{oh}}$ solo come controllo successivo — non vincolante per l'esecuzione, ma incide sul costo stimato di una giornata.
5. *(Fuori perimetro di questo documento — riguarda Fase 3 §3.5.)*
