# Design computazionale di peptidi inibitori dell'interfaccia PA–PB1

## Piano di lavoro

**Versione 2 — 12 agosto 2026**
Revisione successiva all'esecuzione della Fase 0. Le fasi sono state rinumerate rispetto alla versione 1 (ex Fase 2 → Fase 1, e a seguire).

---

# A. Contesto e obiettivo

## A.1 Il sistema biologico

La polimerasi dell'influenza A è un eterotrimero (PA, PB1, PB2). L'assemblaggio del complesso richiede l'interazione fra il dominio C-terminale della subunità PA e l'estremità N-terminale di PB1. Questa interfaccia è un bersaglio antivirale consolidato: un peptide che occupi il sito di PA sulla subunità PB1 impedisce l'assemblaggio della polimerasi e quindi la replicazione virale.

- **Bersaglio strutturale**: 2ZNL, catena A, residui 408–714 (numerazione `auth`).
- **Binder**: peptide lineare di 15 residui, amminoacidi canonici.
- **Riferimento nativo**: `MDVNPTLLFLKVPAQ` (N-terminale di PB1), che nella struttura cristallografica occupa il solco di PA per l'intera lunghezza.

## A.2 L'approccio computazionale

Il progetto combina due sorgenti di informazione indipendenti e complementari:

**(a) Plausibilità strutturale.** ColabDesign/AfDesign esegue *hallucination*: la sequenza del binder viene ottimizzata per gradiente attraverso i pesi di AlphaFold2, massimizzando metriche di confidenza del complesso predetto. Il protocollo `design_3stage` procede in tre stadi:

| Stadio | Rappresentazione della sequenza | Note |
|---|---|---|
| **soft** | $a = \mathrm{softmax}(L/T)$, $T$ alto | $a$ vicino al baricentro del simplesso; dropout attivo |
| **temp** | $T$ decrescente (annealing) | $a$ migra verso i vertici; dropout attivo |
| **hard** | one-hot | sequenza discreta; dropout disattivato |

$L$ sono i logit ottimizzati, e $a \in \Delta^{20}$ per ciascuna delle 15 posizioni. Le metriche di confidenza usate come obiettivo sono `i_ptm` (interface predicted TM-score), `i_con` (numero di contatti all'interfaccia) e `pLDDT`.

**(b) Segnale sperimentale.** Un modello di energia appreso su dati NGS di phage display contro PA. La libreria è random (15-meri, nessuna sequenza di partenza). Il modello è una MLP `300 → 45 → 15 → 1` con attivazioni ReLU (~14.300 parametri), addestrata non per regressione sull'arricchimento ma **massimizzando una verosimiglianza binomiale negativa** derivata da un modello probabilistico del processo di selezione e amplificazione, applicata alle frequenze osservate nei round 2 e 3 (forward e reverse).

I due segnali sono combinati come termine additivo nella loss di design:

$$\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{AF}} + w_E \cdot E(a)$$

L'integrazione avviene tramite un hook `energy_fn(seq_probs) -> scalare` iniettabile dall'esterno in un fork mantenuto di ColabDesign (`af/loss.py`, `af/design.py`, `af/model.py`). Il modello di energia, originariamente in Julia/Flux, è stato riscritto in JAX puro per garantire la differenziabilità end-to-end.

**Motivazione della combinazione.** AlphaFold2 ottimizza la plausibilità strutturale ma è cieco rispetto a ciò per cui la selezione sperimentale ha effettivamente selezionato. Il modello di energia contiene il segnale sperimentale ma ignora la geometria dell'interfaccia. Nessuno dei due, da solo, è sufficiente.

## A.3 Validazione a valle

- **Boltz-2** (co-folding, cluster CEA AAR): filtro strutturale indipendente. Nota: il modulo di predizione di affinità di Boltz-2 è addestrato su ligandi small-molecule e **non è applicabile** alle interazioni peptide–proteina; `i_ptm` e `pLDDT` restano proxy di confidenza, non stime di affinità.
- **AlphaFold3**: validazione dei candidati migliori.
- **Sperimentale** (gruppo Hart, IBS/CEA-IRIG): BLI, TSA, eventualmente cristallografia.

Sono disponibili misure di $K_D$ via BLI su 23 peptidi selezionati, che costituiscono l'unico ancoraggio sperimentale diretto del modello di energia.

---

# B. Problema metodologico affrontato dal piano

Le campagne di design condotte prima di questo piano hanno mostrato due anomalie ricorrenti:

1. **Instabilità del termine di energia.** Con $w_E = 0{,}2$, il valore di $E$ diverge durante lo stadio di annealing e domina la loss, mentre `i_con` resta bloccato e `i_ptm` finale scende a ~0,19 (soglia di accettabilità: > 0,5).
2. **Composizione amminoacidica anomala.** Le sequenze prodotte appaiono arricchite in residui aromatici (F, W, Y) rispetto alle attese.

Tre cause candidate, non mutuamente esclusive:

| Causa | Meccanismo |
|---|---|
| **A — obiettivo AF2** | `i_con` conta contatti e i residui a catena laterale grande ne producono di più; `pLDDT` premia il packing rigido. Effetto documentato in letteratura per l'hallucination di binder con AfDesign |
| **B — dati sperimentali** | Contaminazione da *target-unrelated peptides* (TUP): peptidi che legano il supporto di screening (il polistirene lega preferenzialmente F/Y/W) o che si propagano più rapidamente, invece di legare PA |
| **C — rilassamento continuo** | Il modello di energia è addestrato su sequenze discrete (one-hot) ma durante gli stadi soft riceve distribuzioni continue: comportamento fuori distribuzione |

Il piano è organizzato per **discriminare fra queste cause prima di intervenire**, e per correggere la causa identificata invece di compensarne i sintomi.

## B.1 Il problema del rilassamento continuo, in dettaglio

La causa C merita una formalizzazione, perché determina la Fase 1.

Il forward pass durante gli stadi soft calcola $E(a)$. La quantità semanticamente corretta sarebbe invece l'energia **attesa** della distribuzione di sequenze che $a$ rappresenta:

$$\bar{E}(a) = \mathbb{E}_{x \sim a}\big[E(x)\big], \qquad x_i \sim \mathrm{Cat}(a_i) \text{ indipendenti per posizione}$$

Le due quantità differiscono già al primo layer. Il primo strato lineare è ben comportato ($\mathbb{E}[Wx+b] = Wa+b$), ma ReLU è convessa e per la disuguaglianza di Jensen

$$\mathrm{relu}(Wa+b) \;\leq\; \mathbb{E}\big[\mathrm{relu}(Wx+b)\big]$$

Il forward soft sottostima le post-attivazioni del primo layer; poi i pesi del secondo layer hanno segno arbitrario e la direzione dell'errore si perde. Dopo tre composizioni non esiste alcun controllo su segno né ampiezza della discrepanza.

**Proprietà di riferimento.** $\bar{E}$ è **multilineare** nelle variabili one-hot per posizione: nell'espansione

$$\bar{E}(a) = \sum_{x} \Big(\prod_{i=1}^{15} a_{i,x_i}\Big) E(x)$$

ogni termine contiene al più una potenza prima di ciascun $a_i$. Una funzione multilineare su un prodotto di simplessi attinge i suoi estremi sui vertici. Ne seguono tre proprietà desiderabili che $E(a)$ **non** possiede:

- $\min_x E(x) \le \bar{E}(a) \le \max_x E(x)$ per ogni $a$ — nessun minimo interno spurio;
- $\bar{E}(\text{one-hot}(x)) = E(x)$ — coerenza esatta ai vertici;
- $\partial\bar{E}/\partial a_{i\alpha} = \mathbb{E}[E(x) \mid x_i = \alpha]$ — il gradiente è l'energia media condizionata al residuo $\alpha$ in posizione $i$, quantità interpretabile sperimentalmente.

Una MLP con ReLU è invece lineare a tratti e non multilineare: i suoi minimi interni cadono nei *kink* (le frontiere di attivazione), il cui numero cresce esponenzialmente con profondità e larghezza. Le sequenze osservate ne vincolano una frazione trascurabile.

---

# C. Struttura del piano

**Principio d'ordine: prima la diagnostica, poi il modello di energia, poi il loop di design.** Tarare $w_E$ contro un'energia non calibrata equivale a ottimizzare accuratamente un obiettivo mal definito.

| Fase | Contenuto | Risorsa |
|---|---|---|
| **0** | Diagnostica su dati e modelli esistenti | CPU |
| **1** | Correzione e caratterizzazione del modello di energia | CPU |
| **2** | Loop di design e ablazioni | GPU |
| **3** | Validazione | GPU + CPU |
| **4** | Estensioni rimandate | — |

Le Fasi 0 e 1 producono la maggior parte dell'informazione decisionale a costo GPU quasi nullo.

## C.1 Limite noto della validazione esistente

Il modello di energia correla in modo soddisfacente con i $K_D$ misurati in BLI. Questo esclude il modo di fallimento "il modello non riconosce i leganti di PA", ma la validazione ha un **limite di copertura**: i peptidi misurati provengono dalla regione già arricchita dalla selezione, quindi certificano il ranking in regime di **interpolazione**.

Il loop di design opera invece in **estrapolazione**, e questo è un obiettivo, non un difetto: le energie finali dei run stanno fra $-2$ e $-9$ contro un minimo osservato nel training set di $-3{,}71$ ($\sigma_{\text{train}} = 2{,}09$), cioè un'estrapolazione dell'ordine di $2$–$2{,}5\,\sigma$. Se non ci fosse estrapolazione, il modello non starebbe guidando verso nulla di nuovo.

Nessun test della Fase 0 copre il regime di estrapolazione. Il controllo su target decoy (§3.1) è l'unico che lo fa.

---

# FASE 0 — Diagnostica

**Stato: §0.1–0.5 eseguite (10–11 agosto 2026). Restano §0.6, §0.7 e §0.8.**
Costo residuo stimato: ~1,5 giornate, CPU.

Nessuna sottofase richiede l'esecuzione di ColabDesign né GPU. Il forward del modello di energia è importato dal fork (`colabdesign.energy_model.model_3layer`), non reimplementato.

---

## 0.1 Varianza di E nell'interno del simplesso ✅

**Domanda.** L'instabilità osservata è un artefatto generico del rilassamento continuo, cioè si manifesta ovunque nell'interno del simplesso?

**Metodo.** $10^4$ punti campionati da $\mathrm{Dirichlet}(\alpha\mathbf{1}_{20})$ indipendentemente per ciascuna delle 15 posizioni, con $\alpha$ log-spaziato in $[0{,}05,\,50]$. Valori grandi di $\alpha$ producono distribuzioni quasi uniformi (inizio dello stadio soft), valori piccoli distribuzioni quasi one-hot (fine dell'annealing). Il canale gap è fissato a zero, come in produzione. Confronto con il range $[\min_x E, \max_x E]$ calcolato sul training set one-hot.

**Risultati.**

| Quantità | Valore |
|---|---|
| Sequenze di training valide | 27.342 (F2+F3+R2+R3) |
| $E_{\min}$ / $E_{\max}$ / $\sigma_{\text{train}}$ | $-3{,}71$ / $14{,}62$ / $2{,}09$ |
| Violazioni del bound | **0,02%** (2/10.000) |
| Localizzazione delle violazioni | solo a $\alpha \approx 0{,}065$ (quasi-vertice), ampiezza ~0,2σ |
| Comportamento verso il baricentro | $E(a)$ collassa in una banda stretta (~6–8) |

Validazione dell'encoder: errore massimo $8\times10^{-7}$ rispetto alle energie calcolate attraverso ColabDesign su 6 sequenze di controllo.

**Conclusione.** Non esiste miscalibrazione generica dell'interno del simplesso: i punti campionati casualmente restano nel range fisiologico. Poiché tuttavia i run reali sembrano raggiungere valori ben al di fuori di tale range — osservazione preliminare da quantificare in §0.7 — la regione problematica **esisterebbe ma con misura di Lebesgue trascurabile**. Con $15 \times 19 = 285$ gradi di libertà, $10^4$ campioni isotropi non la incontrano. È la fenomenologia standard degli *adversarial example*: invisibili sotto perturbazione casuale, densi lungo la direzione del gradiente.

**Implicazione.** Un intervento di calibrazione basato su campioni casuali dell'interno sarebbe inefficace per costruzione, perché su quei punti il modello è già corretto. Determina la scelta di §1.1.

---

## 0.2 Taglia effettiva del dataset e clustering ✅

**Domanda.** Quanta informazione indipendente contengono i dati NGS, e come costruire split di validazione privi di leakage fra famiglie di sequenze?

**Nota metodologica.** Il clustering **non** viene usato per ripesare la verosimiglianza di training. Nei dati di screening l'abbondanza *è* la misura: il modello di selezione–amplificazione è costruito per spiegare le frequenze osservate, e ripesare le sequenze corromperebbe l'osservabile. Il clustering serve unicamente a (i) stimare la complessità di modello ammissibile e (ii) costruire split di validazione.

**Metodo.** Clustering al 70% di identità (≤ 4 mismatch su 15 posizioni, distanza di Hamming diretta: le sequenze hanno lunghezza fissa e non richiedono allineamento). MMseqs2 e CD-HIT non erano disponibili; è stato implementato un algoritmo greedy in stile CD-HIT (ordinamento per abbondanza decrescente, assegnazione al rappresentante esistente più vicino entro soglia, altrimenti nuovo cluster), che evita la matrice $N\times N$. Tempi: 0,4 s (round 3), 3,5 s (round 2).

**Risultati.**

| | round 3 (F+R) | round 2 (F+R) |
|---|---|---|
| Sequenze uniche | 6.154 | 23.677 |
| Reads totali | 32.873 | 81.068 |
| **$N_{\text{eff}}$ (n. cluster @70%)** | **1.375** | **3.200** |
| Cluster singleton | 769 (56%) | 1.513 (47%) |
| Cluster più grande | 945 seq / 11.232 reads (**34% della massa**) | 2.229 seq |
| Cluster per il 50% della massa | **2** | 3 |
| Cluster per il 90% della massa | 19 | 177 |

Con $N_{\text{eff}} = \sum_s 1/m_s$ ($m_s$ = cardinalità del cluster di $s$), che si riduce algebricamente al numero di cluster.

**Conclusioni.**
- Il collasso di diversità fra round 2 e 3 ($-57\%$ in $N_{\text{eff}}$) non è imputabile alla sola profondità di sequenziamento: il rapporto $N_{\text{eff}}/N_{\text{unique}}$ *cresce* (13,5% → 22,3%), indicando una contrazione post-selezione reale.
- La verosimiglianza è **dominata da due cluster** (50% della massa di read in round 3). Poche famiglie molto arricchite determinano gran parte del fit.
- Il rapporto fra parametri (~14.300) e famiglie indipendenti (~1.400–3.200) richiede giustificazione esplicita → §1.4.

**Limite noto, da riportare in discussione.** Le varianti generate per errore di polimerasi all'interno di un clone già espanso non sono eventi di selezione indipendenti: sono co-amplificate con il parentale. Se il modello di verosimiglianza tratta esplicitamente la fase di amplificazione ne assorbe una parte; altrimenti resta un bias residuo.

---

## 0.3 Composizione amminoacidica dei leganti validati in BLI ✅

**Domanda.** Nell'esperimento in esame, l'arricchimento in residui aromatici è associato al legame reale o è un artefatto?

**Metodo.** Il file di misure contiene 39 righe ma solo 28 sequenze uniche: molte righe sono ri-misurazioni della stessa sequenza in campagne diverse. Dopo deduplicazione (media geometrica dei $K_D$ ripetuti), esclusione di 4 righe senza segnale misurabile e delle righe di riferimento, **N = 23 sequenze**. Test di arricchimento per amminoacido (Fisher esatto contro la libreria di round 1, correzione Benjamini–Hochberg).

**Anomalia registrata.** L'etichetta "WT" nel file designa due sequenze diverse (`MDVNPTLLFLKLPAQ` e `MDVNPTLLFLKVPAQ`, differenti in posizione 12) con $K_D$ simili (13,6–33,8 nM): verosimilmente un errore di trascrizione. L'analisi usa la sequenza cristallografica.

**Risultati.**

| Popolazione | Frazione aromatica (F+W+Y) |
|---|---|
| Riferimento nativo `MDVNPTLLFLKVPAQ` | 6,7% |
| Libreria round 1 | 15,9% |
| Training set, decile di arricchimento più alto | 20,0% |
| Leganti validati in BLI | 19–21% |

- **G e C completamente assenti** nei 23 leganti validati (0/345 residui).
- **Y è l'arricchimento più forte e robusto** in entrambe le popolazioni selezionate (odds ratio 5,6 in BLI, 5,8 nel decile 9; $p_{BH} = 3\times10^{-7}$ e $\approx 0$).
- La frazione aromatica **non discrimina i quartili di $K_D$** (Q1 20,0% vs Q4 18,9%). Con ~6 osservazioni per quartile il test è però troppo poco potente per rifiutare alcuna ipotesi → §0.8.a.

**Conclusione preliminare.** Gli aromatici sembrano discriminare *selezionato vs non selezionato*, non *forte vs debole* fra i leganti già validati. Combinata con §0.4 e con la correlazione nota fra energia e $K_D$, questa osservazione suggerisce una dissociazione: il potere predittivo del modello sull'affinità non passerebbe dagli aromatici, mentre la sua direzione di gradiente sì. L'ipotesi richiede il test di §0.8.a prima di essere assunta.

**Denominatore mancante.** La libreria di round 1 è già al 15,9%. Sotto uno schema di codoni NNK l'attesa teorica per F+W+Y è ~9–10% (un codone ciascuno su 32). Se lo schema è NNK, la maggior parte dell'arricchimento aromatico si è verificata **entro il primo round**, prima che la selezione target-specifica potesse accumularsi — profilo compatibile con una quota rilevante di legame non specifico al supporto. Lo schema effettivo va determinato (§0.8.d).

---

## 0.4 Direzione del gradiente del modello di energia ✅

**Domanda.** Quali sostituzioni il modello di energia premia, indipendentemente dal loop di design?

**Metodo.** $\partial E/\partial x$ calcolato con `jax.grad` nei 27.342 vertici one-hot del training set e, separatamente, nei 23 vertici corrispondenti ai leganti validati. Derivata parziale non vincolata (nessuna proiezione sul simplesso), coerente con il modo in cui l'ottimizzatore perturba i logit.

**Risultati.**
- **W è l'unico amminoacido con gradiente medio negativo** ($-0{,}17$): aumentarne la probabilità a partire da un vertice osservato *riduce* $E$. I restanti 19 hanno gradiente medio positivo. **D è il più sfavorito** in entrambe le popolazioni.
- **Correlazione di Spearman fra il ranking sul training set e quello sui leganti validati: 0,86.** La direzione non è un artefatto del training set: si riproduce sulle sequenze caratterizzate sperimentalmente.
- Il ranking aggregato va letto insieme alla mappa posizionale $15\times20$: il segnale apparente su C è dominato quasi interamente dalla posizione 15 ($\approx -3\sigma$ in quella posizione, trascurabile altrove).
- **Limite sui residui rari**: C rappresenta l'1,06% dei residui osservati (H 0,27%, T 0,64%). Il gradiente medio può riflettere una stima poco affidabile in una regione poco campionata piuttosto che una preferenza reale; i due casi non sono distinguibili senza dati mirati.

**Conclusioni.**
- La direzione di discesa dell'energia è quasi esclusivamente aromatica. Insieme a §0.3, questo costituisce l'indizio principale di *reward hacking*: l'ottimizzatore scende lungo l'asse che, per quanto misurabile, non predice l'affinità fra i leganti.
- Che 19 amminoacidi su 20 abbiano gradiente positivo ai vertici osservati significa che i punti di training sono minimi locali lungo quasi tutte le direzioni. È compatibile con un buon fit, ma con il rapporto parametri/$N_{\text{eff}}$ di §0.2 la memorizzazione resta un'ipotesi da testare → §1.5.

---

## 0.5 Composizione dei design per contesto strutturale ✅ *(conclusioni parziali)*

**Domanda.** Il bias aromatico dei design si concentra sulle posizioni di interfaccia (compatibile con la causa A) o è distribuito indifferentemente (compatibile con la causa B/C, poiché il modello di energia non ha nozione di geometria)?

**Metodo.** Sono disponibili 35 predizioni AlphaFold3 del complesso completo per un sottoinsieme curato di candidati. La classificazione delle 15 posizioni in "interfaccia" e "esposta" è stata derivata dalle `contact_probs` predette (soglia 0,5), usando come controllo indipendente la geometria della struttura cristallografica 2ZNL (contatti a soglia di distanza e SASA relativa via Shrake–Rupley). La composizione è stata poi calcolata sulla popolazione completa dei design (900 sequenze, tutte le campagne).

**Risultato metodologico: il criterio di contatto su 2ZNL è degenere.** Le distanze minime fra catena B e catena A vanno da 2,58 a 3,75 Å su **tutte e 15** le posizioni: il peptide è incassato nel solco per l'intera lunghezza, e nessuna soglia di distanza ragionevole produce una bipartizione. La SASA relativa discrimina: 13/15 posizioni restano sepolte, solo le posizioni 13 e 15 sono chiaramente esposte.

**Accordo fra le due classificazioni: 6/15 posizioni.** AlphaFold3 identifica un nucleo di interfaccia ristretto (posizioni 6–9, frequenza di contatto 51–57% sulle 35 sequenze) contro la sepoltura quasi totale suggerita dalla struttura nativa. Le posizioni esposte per SASA (13 e 15) sono fra quelle in accordo; il disaccordo riguarda le posizioni 1–5 e 10–12.

**Conseguenza di rilievo indipendente**: l'assunzione implicita che il design mantenga il registro di legame nativo **non è confermata**.

**Composizione per classe di posizione** (interfaccia = 6–9, esposta = le altre 11):

| Popolazione | Aromatica, interfaccia (4 pos.) | Aromatica, esposta (11 pos.) | Rapporto |
|---|---|---|---|
| Riferimento nativo | 25,0% | 0,0% | — |
| Libreria round 1 | 31,9% | 10,1% | 3,2× |
| Training set | 41,6% | 9,4% | 4,4× |
| Leganti validati in BLI | 47,8% | 9,9% | 4,8× |
| Design (900, aggregati) | 31,5% | 13,9% | 2,3× |

**Conclusione tenuta esplicitamente aperta.** La tabella suggerisce che il bias dei design sia di *collocazione* più che di *quantità*, e quindi che la causa A sia da escludere. Questa lettura **non è sostenuta con sufficiente robustezza**, per tre ragioni:

1. **La soglia è arbitraria e il segnale è continuo.** Le frequenze di contatto sono 51–57% sulle posizioni classificate interfaccia contro 35–49% sulle altre: non c'è discontinuità, c'è un gradiente tagliato a 0,5.
2. **Le due classificazioni disaccordano su 9/15 posizioni.** Sulla struttura nativa 13/15 sono sepolte; una partizione 4/11 contraddice metà dell'evidenza disponibile.
3. **I 900 design sono aggregati su campagne eterogenee**, che includono run senza termine di energia e run falliti (`i_ptm` ≈ 0,19). La statistica mescola condizioni sperimentali diverse.

**Quantità che sopravvive.** La frazione aromatica **aggregata** dei design non richiede alcuna classificazione posizionale ed è informativa indipendentemente da essa. Ricostruita dalla tabella dà $\approx 18{,}6\%$ contro $\approx 20{,}0\%$ dei leganti validati, ma eredita l'arbitrarietà della decomposizione: va ricalcolata direttamente (§0.8.b).

**Stato delle ipotesi.** Le cause A e B/C restano entrambe aperte. La formulazione della penalità composizionale (§2.4) resta sospesa.

---

## 0.6 Screening TUP ⏳ *(da eseguire)*

**Domanda.** Qual è il livello di contaminazione da *target-unrelated peptides* nei dati di training?

**Contesto.** I TUP *selection-related* legano componenti del sistema di screening invece del bersaglio; quelli *propagation-related* emergono perché alcuni cloni si replicano più rapidamente. La firma chimica dei leganti al polistirene è dominata da F, Y e W — lo stesso profilo osservato in §0.3.

Non essendo disponibili selezioni di controllo (biglie nude, libreria naive, bersaglio irrilevante), non è possibile inferire congiuntamente un'energia specifica e una non specifica e usarne la differenza. Lo screening bioinformatico è quindi **l'unica leva disponibile** su questa contaminazione.

**Metodo.** I risultati di §0.2 permettono di ridurre drasticamente l'ambito: 19 cluster coprono il 90% della massa di read in round 3, e 2 ne coprono il 50%. Lo screening si applica quindi ai rappresentanti, non alla popolazione.

1. Estrarre i rappresentanti dei 19 cluster che coprono il 90% della massa (colonna `is_representative` in `cluster_assignment.csv`).
2. Sottoporli a SAROTUP: motivi TUP noti, PSBinder (leganti al polistirene), PhD7Faster (propagazione rapida); ricerca in BDB/MimoDB.
3. Screening separato del **cluster dominante** (945 varianti, 34% della massa): è il singolo test più informativo della sottofase.
4. Screening delle 23 sequenze validate in BLI.

**Criteri di lettura.**
- Se il cluster dominante risulta un TUP noto, il modello di energia è ancorato in misura sostanziale su un artefatto sperimentale. Da dichiarare nei limiti del lavoro e da riflettere nel peso assegnato a $E$.
- Se i leganti validati non risultano segnalati mentre la coda arricchita sì, il modello apprende una miscela di due segnali, concettualmente separabili ma non separabili con i dati disponibili.

**Costo stimato: 10 minuti.**

---

## 0.7 Traiettorie di design: divergenza fra rappresentazione soft e discreta ⏳ *(da eseguire)*

**Domanda.** Di quanto il valore di energia effettivamente ottimizzato durante gli stadi soft diverge dall'energia della sequenza discreta corrispondente, e come tale divergenza dipende dal protocollo e dal peso $w_E$?

**Distinzione preliminare.** Due fenomeni distinti vanno tenuti separati, perché l'analisi delle sole energie finali non discrimina fra i due:

| Fenomeno | Definizione | Giudizio |
|---|---|---|
| **Estrapolazione nello spazio delle sequenze** | $E$ dei design finali inferiore a $E_{\min}$ del training set | **Atteso e desiderato** (endpoint osservati fra $-2$ e $-9$, cioè 2–2,5σ) |
| **Divergenza soft/discreto** | $E(a) \ll E(\arg\max a)$ allo stesso passo di ottimizzazione | **Patologia** |

Il diagnostico corretto è la traccia temporale

$$\Delta(t) = E\big(a^{(t)}\big) - E\big(\text{one-hot}(\arg\max a^{(t)})\big)$$

**non** la distribuzione delle energie finali.

**Osservazione preliminare, da quantificare.** Un'ispezione qualitativa di alcune traiettorie mostra un gap nello stadio soft, con $E(a)$ che scende a valori dell'ordine di $-50$ e risale a $\approx -5$ all'attivazione del vincolo di sequenza discreta. Il carattere sistematico di questo comportamento, la sua ampiezza tipica e la sua dipendenza dal protocollo **non sono ancora stabiliti**: l'analisi che segue serve a stabilirli.

### Metodo

**Passo 1 — inventario delle traiettorie.** Censire i run per cui i file di traiettoria sono disponibili, annotando per ciascuno: protocollo, $w_E$, seed, frequenza di campionamento dei passi, e quali campi sono registrati. Il caso favorevole è che $a^{(t)}$ (o i logit $L^{(t)}$ con la temperatura dello stadio) sia salvato; in tal caso $E(a^{(t)})$ ed $E(\text{one-hot}(\arg\max a^{(t)}))$ sono entrambi ricalcolabili offline con `mlp_forward` e l'encoder validato in §0.1. Se è salvata la sola energia soft, il termine discreto va comunque ricostruito dai logit; se non lo sono neppure quelli, il run è escluso e va segnalato come tale.

**Passo 2 — traccia per run.** Per ciascun run calcolare $\Delta(t)$ su tutti i passi disponibili, **normalizzata in unità di $\sigma_{\text{train}} = 2{,}09$** in modo da rendere confrontabili run con pesi diversi.

**Passo 3 — scalari riassuntivi per run.**

| Quantità | Definizione |
|---|---|
| $\Delta_{\max}$ | massimo di $\Delta(t)$ sull'intera traiettoria |
| $t^\ast$ | passo e stadio in cui $\Delta_{\max}$ si verifica |
| $\int \Delta\,dt$ | area sotto $\Delta(t)$ nello stadio soft: budget di gradiente speso sul guadagno illusorio |
| $\Delta_{\text{fine-temp}}$ | valore residuo alla fine dell'annealing |
| $\Delta_{\text{hard}}$ | valore nello stadio hard: **atteso nullo**, funge da controllo di correttezza dell'analisi |
| $E_{\text{soft}}^{\min}$, $E_{\text{hard}}$ al medesimo passo | separano il guadagno illusorio da quello reale |

**Passo 4 — stratificazione.** Aggregare gli scalari per **protocollo** (`opt_anneal_energy_C`, `gen_energy_C`, `opt_hard_energy_C`) e per **valore di $w_E$**, riportando mediana e intervallo interquartile anziché singoli valori.

I run **senza termine di energia** costituiscono il controllo essenziale: $\Delta(t)$ è calcolabile a posteriori anche su di essi, e misura l'ampiezza del gap di Jensen lungo una traiettoria che nessuno sta sfruttando. È il livello di riferimento rispetto a cui giudicare i run energy-guided.

**Passo 5 — relazione con l'esito del design.** Correlare $\Delta_{\max}$ e $\int \Delta\,dt$ con `i_ptm` e `i_con` finali. Questo converte in affermazione verificabile l'ipotesi secondo cui il gap è la causa del degrado strutturale osservato, invece di lasciarla come spiegazione plausibile.

**Passo 6 — dipendenza da $w_E$.** Regressione di $\Delta_{\max}$ su $w_E$. Una crescita super-lineare indicherebbe un meccanismo auto-rinforzante (energia più bassa → gradiente più forte → allontanamento dai vertici → gap maggiore) e identificherebbe la soglia sotto la quale i run precedenti restano interpretabili.

### Output

Tabella degli scalari per protocollo e $w_E$; curve $\Delta(t)$ per stadio, una per protocollo; dispersione di $\Delta_{\max}$ contro `i_ptm` finale; regressione $\Delta_{\max}$ vs $w_E$.

### Criteri di lettura

- **$\Delta_{\max}$ sistematicamente elevata nei run energy-guided e trascurabile nei controlli, con correlazione negativa fra $\Delta$ e `i_ptm`**: il meccanismo è confermato, la correzione di §1.1 è la leva principale e il beneficio atteso è quantificato.
- **$\Delta_{\max}$ comparabile fra run con e senza energia**: il gap è una proprietà del rilassamento, non un effetto della guida energetica. §1.1 resta corretta ma non spiega il degrado di `i_ptm`, che va attribuito altrove.
- **$\Delta_{\max}$ elevata solo per $w_E$ estremi**: il fenomeno è un caso limite di taratura. §1.1 rimane un intervento a costo trascurabile e va comunque adottata, ma perde il ruolo di correzione principale e la priorità si sposta sulla ritaratura di §2.2.

**Dipendenza.** Le stesse traiettorie servono a §0.8.c. Conviene produrre in questa sottofase un unico file di traiettorie normalizzate, riutilizzabile.

**Costo stimato: 1 giornata, CPU.** Nessun run nuovo.

---

## 0.8 Verifiche di chiusura ⏳ *(da eseguire)*

Quattro analisi su dati esistenti. Nessun run nuovo, nessuna GPU. Costo complessivo stimato: mezza giornata.

### 0.8.a — Correlazione parziale fra $K_D$, energia e contenuto aromatico

Verifica se il modello di energia contiene segnale predittivo oltre la composizione aromatica.

**Metodo.** Sulle 23 sequenze con $K_D$ misurato ed energia già calcolata:
1. regressione di $\log K_D$ sul solo conteggio aromatico;
2. regressione di $\log K_D$ sull'energia;
3. **correlazione parziale fra energia e $\log K_D$ controllando per il conteggio aromatico.**

Da condurre in continuo su N = 23, non per quartili: la stratificazione di §0.3 dispone di ~6 osservazioni per cella ed è priva di potenza.

**Criteri di lettura.**
- *La correlazione parziale sopravvive*: il modello ha segnale oltre gli aromatici. Comprimere l'asse aromatico non ne distrugge il potere predittivo, e una penalità composizionale (§2.4) è praticabile.
- *La correlazione parziale collassa*: il modello è essenzialmente un contatore di residui aromatici con buon fit. Una penalità composizionale entrerebbe in conflitto diretto con il termine di energia, e la Fase 2 va ridimensionata.

### 0.8.b — Stratificazione della popolazione di design

I metadati esistenti distinguono i protocolli con termine di energia (`opt_anneal_energy_C`, `gen_energy_C`, `opt_hard_energy_C`) da quelli senza. La popolazione di design costituisce quindi un esperimento controllato già disponibile.

**Metodo.** Frazione aromatica **aggregata** (senza classificazione posizionale) stratificata per: presenza/assenza del termine di energia; valore di $w_E$; `i_ptm` (separando run riusciti e falliti). Ripetere poi la decomposizione posizionale di §0.5 su ciascuno strato separatamente, con analisi di sensitività sulla soglia di `contact_probs` (0,3 / 0,5 / 0,7).

**Criterio di lettura.** Se la frazione aromatica cresce con l'attivazione o con il peso del termine di energia, la causa B/C è dimostrata anziché inferita. Se non varia, la causa A torna in primo piano.

### 0.8.c — Composizione al minimo del gap

Sulle traiettorie normalizzate prodotte in §0.7, nel passo $t^\ast$ in cui $\Delta(t)$ è massima: dove si concentra la massa di probabilità di $a$?

**Metodo.** Composizione amminoacidica media di $a^{(t^\ast)}$, per protocollo, confrontata con la composizione allo stesso passo nei run di controllo senza termine di energia.

**Criteri di lettura.**
- *Massa concentrata su W/F/Y*: il gap di Jensen si apre lungo la direzione aromatica, coerentemente con la direzione di discesa misurata in §0.4. Le due anomalie di §B sono manifestazioni dello stesso fenomeno e la correzione di §1.1 le riduce entrambe.
- *Composizione generica*: le due anomalie sono indipendenti; la causa A resta pienamente in gioco e la penalità composizionale di §2.4 va valutata separatamente dalla correzione del forward.

**Dipendenza:** richiede §0.7.

### 0.8.d — Attesa composizionale della libreria

Determinare lo schema di codoni effettivo e l'attesa teorica per F+W+Y. È il denominatore di ogni analisi di arricchimento e va fissato prima di interpretare i risultati di §0.3.

---

> ## GATE 0
>
> **Acquisito da §0.1–0.5:**
> 1. Non esiste miscalibrazione generica dell'interno del simplesso; la patologia è direzionale.
> 2. $N_{\text{eff}} \approx 1.400$ (round 3) contro ~14.300 parametri; due cluster coprono il 50% della massa di read.
> 3. La direzione di discesa dell'energia è quasi esclusivamente aromatica; gli aromatici discriminano selezionato da non selezionato, ma non necessariamente forte da debole.
> 4. Il registro di legame non è stabile fra i design e il riferimento nativo.
>
> **Da chiudere con §0.6, §0.7 e §0.8:** livello di contaminazione TUP; ampiezza e dipendenza dal protocollo della divergenza soft/discreto; attribuzione causale del bias aromatico; esistenza di segnale non aromatico nel modello di energia.
>
> **Condizione di riconsiderazione.** Se §0.8.a mostra il collasso della correlazione parziale **e** §0.6 identifica il cluster dominante come TUP, il modello di energia è prevalentemente un rilevatore di adesività non specifica. Ciò non blocca il progetto ma va dichiarato nei limiti e riduce la Fase 2 al confronto minimo.

---

# FASE 1 — Modello di energia

**Costo stimato: 1,5 settimane, CPU.**

---

## 1.1 Correzione del forward pass: straight-through estimator

**Obiettivo.** Eliminare la divergenza soft/discreto, la cui ampiezza è quantificata in §0.7.

**Scelta e sua motivazione.** Due strategie erano disponibili:

| Strategia | Principio | Valutazione alla luce della Fase 0 |
|---|---|---|
| Calibrazione per distillazione verso $\bar E$ | Addestrare una rete studente su target Monte Carlo $\frac{1}{K}\sum_k E(x_k)$, $x_k \sim a$ | **Scartata.** Su campioni casuali dell'interno $E(a)$ e $\bar E(a)$ già coincidono (§0.1): la distillazione produrrebbe una rete equivalente all'originale |
| Straight-through estimator | Valutare sempre al vertice, propagare il gradiente attraverso la discretizzazione | **Adottata** |

Una seconda motivazione, subordinata all'esito di §0.7, è che il gap osservato in via preliminare si apre e si richiude, lasciando endpoint fisiologici: se confermato, non esiste un ottimo interno da recuperare e ciò che il forward soft insegue è rumore, non segnale.

La correzione va comunque adottata a prescindere dall'esito di §0.7, essendo di costo trascurabile e teoricamente corretta; ciò che §0.7 determina è **l'entità del beneficio atteso**, e quindi se questa sia la leva principale sul problema di §B o un intervento marginale rispetto alla ritaratura di §2.2.

**Implementazione** (Bengio et al., 2013):

```python
hard = jax.nn.one_hot(a.argmax(-1), 20)      # vertice del simplesso
x    = a + jax.lax.stop_gradient(hard - a)
E    = mlp_forward(params, x)
```

- **Forward**: `stop_gradient` è l'identità sul valore, quindi $x = a + (\text{hard} - a) = \text{hard}$. La rete valuta sempre e solo un vertice, cioè una sequenza reale.
- **Backward**: `stop_gradient` ha derivata nulla, quindi $\partial x/\partial a = I$. Il gradiente $\partial E/\partial x$ calcolato in `hard` viene propagato ad $a$ come se la discretizzazione non fosse presente.

**Effetto.** La loss diventa per costruzione l'energia di una sequenza reale. Il gap non si riduce: cessa di esistere, perché il valore illusorio non viene mai calcolato. Valore registrato e valore ottimizzato coincidono.

**Costo dichiarato.** Il gradiente è distorto: è $\nabla E$ valutato al vertice, usato per aggiornare un punto $a$ che al vertice non si trova. È un'approssimazione del primo ordine, tanto migliore quanto più $a$ è concentrato. Ne segue che l'approssimazione è peggiore all'inizio dello stadio soft e migliora automaticamente durante l'annealing; e che il gradiente è più sparso, portando informazione solo sull'amminoacido attualmente dominante per posizione. Il compromesso è fra il gradiente esatto di una funzione errata (soft) e il gradiente approssimato della funzione corretta (straight-through); §0.7 quantifica l'errore della prima e permette quindi di giudicare il compromesso su base misurata.

**Alternativa di riserva: moment propagation.** Conserva un gradiente non distorto propagando media e varianza in forma chiusa attraverso i layer. La covarianza dell'input è block-diagonale (le posizioni sono campionate indipendentemente):

$$\mathrm{Cov}(x) = \mathrm{blkdiag}\big(\mathrm{diag}(a_i) - a_i a_i^\top\big)$$

e il ReLU ammette momenti analitici sotto approssimazione gaussiana della pre-attivazione:

$$\mathbb{E}[\mathrm{relu}(z)] = \mu\,\Phi(\mu/\sigma) + \sigma\,\phi(\mu/\sigma)$$

Richiede ~40 righe, nessun riaddestramento e nessun insieme di calibrazione. Il limite è che l'approssimazione gaussiana si indebolisce ai layer con 45 e 15 ingressi, dove il teorema centrale del limite è meno applicabile. **Da adottare solo se lo straight-through produce traiettorie che si arrestano o gradienti eccessivamente sparsi** (§2.1).

**Verifiche.**
1. `E_st(a) == E(one_hot(argmax(a)))` esattamente, per ogni $a$.
2. Il gradiente rispetto ad $a$ è **non nullo**. Se `stop_gradient` avvolge l'intera espressione invece della sola differenza, il gradiente si annulla e il termine di energia cessa silenziosamente di guidare l'ottimizzazione: la loss conserva un valore plausibile ma non ha più effetto. È l'errore di implementazione più comune e non è visibile nei log a meno di cercarlo esplicitamente.

---

## 1.2 Standardizzazione

**Obiettivo.** Rendere $w_E$ un iperparametro con significato stabile e trasferibile.

**Metodo.** $\tilde{E} = (E - \mu_{\text{train}})/\sigma_{\text{train}}$, con $\sigma_{\text{train}} = 2{,}09$ (§0.1).

**Effetto.** $w_E$ acquista l'interpretazione "quante deviazioni standard sperimentali valgono un'unità di loss strutturale", e la taratura diventa trasferibile a modelli di energia futuri — obiettivo per cui l'hook è stato reso iniettabile.

**Conseguenza.** Le tarature precedenti di $w_E$ sono state condotte contro un forward che raggiungeva $-50$ e non sono trasferibili. La ricerca del range utile va ripetuta (§2.2).

La standardizzazione va applicata **per replica**; vedi §1.6(c).

---

## 1.3 Repliche su split di cluster

**Obiettivo.** Ottenere una stima utilizzabile dell'incertezza epistemica del modello di energia.

**Metodo.** $M = 5$–$10$ repliche, ciascuna addestrata su uno **split di cluster diverso**, usando l'assegnazione prodotta in §0.2.

**Vincolo.** Repliche che differiscono solo per il seed di inizializzazione, sullo stesso split, sottostimano gravemente l'incertezza epistemica: convergono a soluzioni correlate. Sono necessari split di dati genuinamente diversi. Se più round sono disponibili, lo split per round costituisce un asse di variazione aggiuntivo.

**Precauzione derivante da §0.2.** Con due cluster che coprono il 50% della massa di read, uno split costruito per sequenza sovra-rappresenterebbe una o due varianti dominanti come se fossero centinaia di osservazioni indipendenti. Gli split vanno costruiti **per cluster**.

**Uso a valle.** $\sigma_{\text{repliche}}(a)$ entra nel *lower confidence bound* di §2.3:

$$\mathcal{L}_{\text{energia}} = E_{\text{ST}}(a) + \kappa\,\sigma(a)$$

con segno positivo perché la quantità è minimizzata: il disaccordo fra repliche viene penalizzato. Questo trasforma l'obiettivo da "cercare dove il modello promette di più" a "cercare dove il modello è confidente", che è la mitigazione principale contro l'ottimizzazione avversariale del surrogato.

**Distinzione da mantenere.** Due varianze diverse intervengono e solo una è appropriata per il termine di pessimismo:

| Varianza | Natura | Uso |
|---|---|---|
| $\mathrm{Var}_{x\sim a}[E(x)]$ | aleatoria rispetto al rilassamento: quanto le sequenze rappresentate da $a$ differiscono fra loro | diagnostica di convergenza — deve decrescere durante l'annealing e annullarsi allo stadio hard; se resta alta, l'annealing non sta convergendo a una sequenza |
| $\mathrm{Var}_{\text{repliche}}$ | epistemica: quanto il modello non sa | termine $\kappa\sigma$ |

---

## 1.4 Scala di baseline

**Obiettivo.** Misurare il guadagno effettivo dei ~14.300 parametri della MLP. Non è un cambio di architettura (rimandato alla Fase 4) ma una misurazione.

Priorità elevata dal risultato di §0.2: il rapporto fra parametri e famiglie indipendenti richiede giustificazione.

**Metodo.** Su split per cluster, confronto via correlazione di Spearman fra:
1. modello sulla sola composizione amminoacidica (20 feature, ignora le posizioni);
2. modello additivo su one-hot (300 parametri);
3. MLP attuale.

**Criteri di lettura.**
- (1) ≈ (3): il segnale è composizionale, non posizionale. Si sovrappone all'esito di §0.8.a; il modello sarebbe essenzialmente un contatore di amminoacidi, e una penalità composizionale entrerebbe in conflitto con il termine di energia.
- (2) ≈ (3): nessuna epistasi utilizzabile viene catturata; la capacità aggiuntiva non produce guadagno.
- (3) > (2) in misura netta: l'epistasi è presente ed è appresa. Giustifica la capacità e rende prioritaria la Fase 4.

---

## 1.5 Predizione su famiglia non osservata

**Obiettivo.** Distinguere generalizzazione da memorizzazione.

**Metodo.** Escludere dal training **interi cluster** arricchiti (non sequenze singole) e misurare la capacità di predire l'arricchimento della famiglia esclusa.

**Criteri di lettura.**
- Predizione accurata: esiste segnale generalizzante e la scelta della classe di modello è una questione di efficienza.
- Predizione nulla ma buona predizione *entro* famiglia: il modello memorizza le famiglie osservate. Usarlo per guidare il design significherebbe guidare verso i candidati già noti, non verso candidati nuovi.

**Contesto.** La validazione BLI stabilisce già che il modello ordina correttamente all'interno della regione selezionata. Questa sottofase misura una proprietà diversa: se tale ordinamento si estende a regioni non osservate.

**Diagnostiche di supporto.**
- Distribuzione dei residui dopo un fit additivo. Una coda destra pesante concentrata in poche famiglie indica una struttura "fondo debole più picchi discreti" piuttosto che un campo liscio — ipotesi coerente con la concentrazione di massa osservata in §0.2, e che se confermata rende inadeguato qualsiasi modello smooth indipendentemente dal grado.
- Prestazione in funzione di $N_{\text{eff}}$ usato in training: una saturazione immediata indica memorizzazione.

---

> ## GATE 1
>
> Procedere alla Fase 2 se:
> - le verifiche di §1.1 sono superate;
> - §1.4 mostra che la MLP supera i baseline, **oppure** si accetta esplicitamente di impiegare un modello equivalente a un additivo;
> - §1.5 non è catastrofico.
>
> Se §1.5 è nullo **e** §1.4 mostra (1) ≈ (3) **e** §0.8.a mostra il collasso della correlazione parziale, il modello di energia è un contatore di composizione adattato alle famiglie osservate. Resta utilizzabile come filtro a valle, non come guida per l'esplorazione di sequenze nuove; la Fase 2 si riduce al confronto minimo.
>
> **Soglia numerica da fissare prima di eseguire §1.5**: quale valore di Spearman su famiglia esclusa giustifica la Fase 2 a budget pieno, e quale la riduce.

---

## 1.6 Confezionamento del modello di energia

**Obiettivo.** Produrre un artefatto stabile e testato che ColabDesign possa consumare senza che alcuna configurazione della matrice di ablazione richieda di modificare il codice interno del fork.

**Motivazione.** La matrice di ablazione della Fase 2 fa variare tre assi: modalità di forward, schema di $w_E$, valore di $\kappa$. Se ogni combinazione richiede una modifica in `af/loss.py`, i confronti non sono riproducibili e gli errori di configurazione diventano indistinguibili dai risultati.

### (a) Preservare il contratto dell'hook tramite closure

L'hook ha firma `energy_fn(seq_probs: (1,L,20)) -> scalare JAX` e non va modificato. Lo stato aggiuntivo (repliche, $\kappa$, standardizzazione, modalità) entra come variabile catturata in una factory:

```python
import jax, jax.numpy as jnp

def make_energy_fn(bundle, mode="st", kappa=1.0, eps=1e-6):
    """
    bundle: artefatto serializzato (vedi (f))
      - "stacked_params": pytree, dimensione principale M = n. repliche
      - "mu", "sd": (M,) statistiche di standardizzazione per replica
    mode: "soft" (legacy, braccio di controllo) | "st" (default) | "moment" (riserva)
    kappa: peso del termine di pessimismo; 0 = media dell'ensemble
    """
    W, mu, sd = bundle["stacked_params"], bundle["mu"], bundle["sd"]

    def single(params, a):
        if mode == "soft":
            return mlp_forward(params, a)
        elif mode == "st":
            hard = jax.nn.one_hot(a.argmax(-1), a.shape[-1])
            x = a + jax.lax.stop_gradient(hard - a)   # stop_gradient sulla SOLA differenza
            return mlp_forward(params, x)
        elif mode == "moment":
            return mlp_moment(params, a)
        raise ValueError(mode)

    @jax.jit
    def energy_fn(a):
        e = jax.vmap(single, in_axes=(0, None))(W, a)   # (M,)
        z = (e - mu) / sd                                # standardizzazione per replica
        return z.mean() + kappa * jnp.sqrt(z.var() + eps)

    return energy_fn
```

### (b) Vettorizzazione sulle repliche

I pesi delle $M$ repliche vanno impilati con dimensione principale $M$ e valutati con `jax.vmap`. Con ~14.300 parametri e $M = 10$ il costo aggiuntivo è trascurabile rispetto al forward di AlphaFold2, mentre un ciclo Python forzerebbe $M$ tracciature separate e ricompilazioni a ogni cambio di configurazione.

### (c) Standardizzare per replica prima di aggregare

Repliche addestrate su split diversi presentano offset e scale di output diversi: differenze di *nuisance*, non di conoscenza. Se $\sigma$ viene calcolata sugli output grezzi, la dispersione è dominata da tali offset e il termine di pessimismo penalizza il disaccordo sbagliato — non "il modello non sa" ma "le repliche hanno intercette diverse" — producendo un $\kappa\sigma$ quasi costante e privo di effetto utile.

Ciascuna replica va standardizzata con i propri $\mu_m, \sigma_m$ calcolati sul rispettivo training set; l'aggregazione avviene dopo. La dispersione risultante misura il disaccordo sul ranking.

### (d) Guardie numeriche

`jnp.sqrt` ha derivata non limitata in zero, e $\sigma \to 0$ si verifica realmente: accade dove le repliche concordano, cioè nella regione ben coperta dai dati, in cui l'ottimizzatore trascorre gran parte del tempo. Il termine `eps` all'interno della radice è necessario, non cosmetico. Da verificare esplicitamente che il gradiente resti finito in un punto di accordo perfetto.

### (e) Diagnostica separata dalla loss

Le quantità di registrazione (§2.6) non appartengono alla loss. In particolare $E(\arg\max a)$ non è differenziabile e va calcolata sotto `stop_gradient` o fuori dal grafo. Va esportato un secondo callable:

```python
def make_energy_aux(bundle):
    @jax.jit
    def energy_aux(a):
        a_   = jax.lax.stop_gradient(a)
        hard = jax.nn.one_hot(a_.argmax(-1), a_.shape[-1])
        return {"e_fwd":  ...,   # valore effettivamente ottimizzato
                "e_hard": ...,   # E(one_hot(argmax a))
                "gap":    ...,   # e_fwd - e_hard
                "sigma":  ...,   # dispersione fra repliche
                "arom_frac": a_[..., AROMATIC_IDX].sum(-1).mean()}
    return energy_aux
```

Il campo `gap` costituisce la verifica continua che la correzione di §1.1 sia attiva: in modalità `st` deve essere identicamente nullo.

### (f) Artefatto serializzato e versionato

Il percorso attuale (JLD2 → script Julia → JSON → dizionario di array JAX) è adeguato per una singola rete ma diventa fragile con $M$ repliche. Serializzare un unico artefatto (`.npz` o safetensors) contenente: pesi impilati (dense1/2/3, W e b), $\mu_m$ e $\sigma_m$ per replica, specifica dell'architettura, metadati (hash del commit, versione del dataset, data, modalità).

Il loader deve validare le forme e fallire in modo esplicito. Forme attese per replica: `(45,300)`, `(15,45)`, `(1,15)`, con dimensione principale $M$. La convenzione di Flux `(out, in)` coincide con quella di JAX, ma è un punto di attrito noto: un assert costa nulla ed evita di individuare una trasposizione errata solo dopo decine di ore di calcolo. Il test di validazione dell'encoder di §0.1 (errore $8\times10^{-7}$ su 6 sequenze di controllo) va riutilizzato come test di regressione.

### Criteri di accettazione

| # | Test | Criterio |
|---|---|---|
| 1 | Coerenza ai vertici | `energy_fn(one_hot(x))` con `kappa=0` riproduce l'energia originale standardizzata entro $10^{-5}$ |
| 2 | Gap identicamente nullo | in modalità `st`, `energy_aux(a)["gap"] == 0` per ogni $a$, inclusi punti prossimi al baricentro |
| 3 | Gradiente non nullo | $\partial\,\texttt{energy\_fn}/\partial a \neq 0$ |
| 4 | Gradiente del termine $\kappa\sigma$ | finito e non nullo dove le repliche concordano |
| 5 | Retrocompatibilità | `energy_fn=None` riproduce la traiettoria di ColabDesign originale a parità di seed |
| 6 | Modalità legacy | `mode="soft", kappa=0` riproduce i run precedenti, gap incluso |
| 7 | Test placeholder | il test di penalizzazione della glicina, esteso a verificare che $\kappa>0$ modifichi il comportamento nella direzione attesa |

Il test 2 sostituisce un criterio basato sul campionamento casuale dell'interno del simplesso: §0.1 ha mostrato che tale campionamento non rileva le violazioni e certificherebbe qualunque implementazione.

Il test 6 non è retrocompatibilità di cortesia: la modalità legacy è il **braccio di controllo** dell'esperimento di Fase 2. In sua assenza i miglioramenti osservati non sono attribuibili.

---

# FASE 2 — Loop di design

**Costo stimato: 1,5 settimane, GPU.**

## 2.1 Confronto fra modalità di forward

Due bracci: soft (legacy) e straight-through. Il moment propagation costituisce un terzo braccio da attivare solo se lo straight-through produce traiettorie che si arrestano o gradienti eccessivamente sparsi.

Metrica primaria: `i_ptm` e `i_con` finali, a parità di seed e budget. Metrica di controllo: il campo `gap` deve essere identicamente nullo nel braccio straight-through e riprodurre nel braccio legacy il profilo caratterizzato in §0.7.

## 2.2 Ritaratura di $w_E$

Da ripetere integralmente: le tarature precedenti sono state condotte contro un forward divergente (§1.2). La ricerca va condotta in unità di $\sigma_{\text{train}}$.

**Bilanciamento sui gradienti** come opzione:

$$w_E^{(t)} = \rho \cdot \frac{\|\nabla\mathcal{L}_{\text{AF}}\|}{\|\nabla E\|}$$

con $\rho$ unico iperparametro. La motivazione è che i due gradienti non competono ad armi pari: quello strutturale attraversa l'intera rete di AlphaFold2 con dropout attivo (profondo, stocastico, ad alta varianza), quello dell'energia è un percorso diretto e deterministico attraverso tre layer. Su centinaia di passi il termine a bassa varianza prevale sistematicamente, perché il rumore dell'altro si media mentre il suo no.

Con lo straight-through il gradiente dell'energia diventa più sparso, quindi il rapporto delle norme cambia: va misurato, non assunto.

## 2.3 Lower confidence bound

Scansione di $\kappa \in \{0,\ 0{,}5,\ 1,\ 2\}$ sul termine $E_{\text{ST}}(a) + \kappa\,\sigma(a)$.

## 2.4 Penalità composizionale

**Formulazione sospesa in attesa di §0.8.a e §0.8.b.**

| Esito delle verifiche | Formulazione appropriata |
|---|---|
| Correlazione parziale conservata; bias attribuibile causalmente all'energia | Penalità posizionale (profilo composizionale per classe di posizione, calibrato sui leganti validati), subordinata a una classificazione posizionale più robusta di quella di §0.5 |
| Correlazione parziale conservata; bias non attribuibile all'energia | Penalità globale come vincolo di sviluppabilità (solubilità, propensione all'aggregazione), non come correzione di artefatto |
| Correlazione parziale collassata | Nessuna penalità: entrerebbe in conflitto diretto con il termine di energia. Il problema risiede nel modello, non nel loop |

In tutti i casi, una divergenza di Kullback–Leibler fra la composizione di $a$ e una distribuzione di riferimento è preferibile a una penalità sui soli aromatici: ha un solo iperparametro e vincola qualunque distorsione composizionale, non soltanto quella già identificata.

## 2.5 Matrice di ablazione

Seed fissi, stesso bersaglio, stesso budget. Fattori: modalità di forward (soft / straight-through) × schema di $w_E$ (fisso / bilanciamento sui gradienti) × $\kappa$ (0 / >0) × penalità (attiva / inattiva). Non è necessario il fattoriale completo: partire dai due bracci del confronto base e aggiungere un fattore per volta.

## 2.6 Schema di registrazione

Per ciascun passo: `e_fwd`, `e_hard`, `gap`, $\sigma(a)$, frazione aromatica di $a$, `i_con`, `i_ptm`, `pLDDT`, $\|\nabla\mathcal{L}_{\text{AF}}\|$, $\|\nabla E\|$.

In assenza di queste serie temporali le ablazioni non sono interpretabili a posteriori: la diagnosi di §0.7 è possibile solo perché le traiettorie dei run precedenti erano state conservate, e non lo sarebbe stata sui soli valori finali.

---

# FASE 3 — Validazione

**Costo stimato: 1–2 settimane.**

## 3.1 Controllo su bersaglio decoy

Il test più diagnostico dell'intero piano. Lo stesso loop, con lo stesso modello di energia, viene eseguito contro un bersaglio diverso o contro PA con gli hotspot permutati. Se le sequenze prodotte sono simili a quelle ottenute sul bersaglio reale, il termine di energia è indipendente dal bersaglio e sta guidando verso adesività generica.

È l'unico test che copre il regime di estrapolazione (§C.1), non coperto dalla validazione BLI.

## 3.2 Retrodizione

Il loop produce soluzioni simili al riferimento nativo? Inizializzato in prossimità del riferimento, vi permane o se ne allontana? Da leggere insieme a §0.5: poiché il registro di legame non risulta stabile fra design e riferimento, la somiglianza va definita anche compositionalmente e strutturalmente, non solo per identità di sequenza.

## 3.3 Diversità prima della selezione

I design vanno clusterizzati (stesso strumento di §0.2) e vanno selezionati rappresentanti diversi, non i primi $K$ per punteggio. In un'ottimizzazione contro un surrogato imperfetto, i primi $K$ sono tipicamente la stessa modalità replicata.

## 3.4 Ricalcolo delle correlazioni fra predittori

Le metriche di Boltz-2 e AlphaFold3 sono risultate leggermente anti-correlate ($-0{,}14$) sui candidati precedenti, mentre ColabDesign e AlphaFold3 correlano a $\approx 0{,}39$. Il risultato aveva motivato la strategia operativa di filtrare con le metriche di ColabDesign calibrate sul riferimento nativo (`i_ptm` = 0,965 per il complesso WT su Boltz-2) e validare con AlphaFold3 solo i candidati migliori.

Le correlazioni vanno ricalcolate sui design prodotti dopo la correzione di §1.1: il valore osservato potrebbe essere in parte un artefatto dell'aver ordinato design patologici.

## 3.5 Criteri di selezione per la validazione sperimentale

Da fissare **prima** di esaminare i risultati: soglie su `i_ptm`, su $\sigma(a)$, sulla frazione aromatica, sulla distanza dai cluster di training, e quota minima di diversità.

---

# FASE 4 — Estensioni rimandate

## 4.1 Architettura del modello di energia

I risultati di §0.2 (bassa taglia effettiva, dominanza di poche famiglie) e §0.5 (registro non stabile) indicano che i parametri posizione-specifici non sono l'inductive bias appropriato per una libreria random. In assenza di una sequenza di partenza non esiste un registro privilegiato: il segnale è verosimilmente costituito da motivi corti a posizione variabile, e un modello posizione-specifico deve apprendere lo stesso motivo separatamente in ciascuna posizione, frammentando dati già scarsi.

**Classe candidata: feature di k-mer gappati con pooling sulle posizioni.**

$$\phi_p(x) = \sum_{i} \prod_{j \in p} \mathbb{1}[x_{i+j} = \alpha_j]$$

per un dizionario di pattern del tipo `XX`, `X.X`, `X..X`, `XXX`, `XX.X`. Tre proprietà rilevanti: (i) invarianza posizionale, che elimina la frammentazione; (ii) grado effettivo elevato senza esplosione parametrica (`XX.X` è un termine di grado 4, ma il modello resta di dimensione contenuta); (iii) **multilinearità preservata**, poiché ogni termine coinvolge posizioni distinte — l'aspettazione si calcola in forma chiusa come $\sum_i \prod_j a_{i+j,\alpha_j}$, e il problema di §B.1 non si ripresenta.

Alternativa, se §1.5 indica una struttura "fondo debole più picchi discreti": processo gaussiano con kernel su stringhe, in cui un picco influenza le predizioni solo nel proprio intorno e non contamina i parametri globali, con incertezza predittiva in forma chiusa.

## 4.2 Altre direzioni

- Guidance sugli stati nascosti del Pairformer in modelli AF3-like.
- BoltzGen come generatore di sequenze iniziali diverse, in alternativa all'inizializzazione casuale.

---

# D. Budget

| Fase | Stato | Durata residua | Risorsa |
|---|---|---|---|
| 0 | §0.1–0.5 completate | ~1,5 giornate (§0.6, §0.7, §0.8) | CPU |
| 1 | da eseguire | ~1,5 settimane | CPU |
| 2 | da eseguire | ~1,5 settimane | GPU: ~7,5 h-GPU per run da 50 traiettorie (9 min/sequenza con `model_4`, `num_recycles=1`); matrice ridotta ≈ 40 h-GPU |
| 3 | da eseguire | 1–2 settimane | GPU (Boltz-2, cluster AAR) + CPU |

---

# E. Note sui gate

I due gate sono formulati in modo da poter fallire, e almeno uno ha probabilità non trascurabile di farlo. Il loro significato va stabilito prima che gli esiti siano noti: è la protezione contro la dinamica in cui ogni risultato negativo diventa un argomento per un ulteriore ciclo di tuning.

Il Gate 1 richiede una soglia numerica ancora da fissare (Spearman su famiglia esclusa), da stabilire prima di eseguire §1.5.

---

# Appendice I — Ipotesi di lavoro superate

Registro delle ipotesi formulate nella versione 1 del piano e abbandonate alla luce dei risultati di Fase 0. Conservato per tracciabilità metodologica.

| Ipotesi (versione 1) | Motivo dell'abbandono |
|---|---|
| Le violazioni del bound sull'interno del simplesso sono massive per $\alpha$ grande | Misurate allo 0,02%, e localizzate a $\alpha$ piccolo. La patologia è direzionale, non generica (§0.1) |
| L'instabilità raggiunge $-189$, ordine di decine di $\sigma$ sotto $E_{\min}$ | Valore relativo a un singolo run a $w_E$ = 0,2, non rappresentativo finché §0.7 non ne stabilisce la distribuzione. Il confronto con $E_{\min}$ mescolava inoltre scale soft e discrete, non commensurabili |
| Calibrazione per distillazione Monte Carlo come strategia principale | Inefficace su campioni casuali dell'interno (§0.1), che sono i soli costruibili senza le traiettorie reali. Sostituita dallo straight-through (§1.1) |
| Il bias aromatico dei design è di collocazione e non di quantità; la causa A è esclusa | Fondata su una classificazione posizionale con soglia arbitraria e accordo 6/15 fra criteri indipendenti, applicata a design aggregati su campagne eterogenee (§0.5) |
| Penalità composizionale posizionale calibrata sul profilo dei leganti validati | Sospesa insieme alla lettura di §0.5 su cui si fondava; la formulazione dipende da §0.8 |
| Riponderazione delle sequenze per cluster nella verosimiglianza di training | Nei dati di screening l'abbondanza costituisce la misura; riponderare corromperebbe l'osservabile che il modello è costruito per spiegare (§0.2) |

---

# Appendice II — Notazione

| Simbolo | Significato |
|---|---|
| $a$ | rappresentazione soft della sequenza, $a_i \in \Delta^{20}$ per ciascuna delle 15 posizioni |
| $L$, $T$ | logit ottimizzati e temperatura; $a = \mathrm{softmax}(L/T)$ |
| $E(x)$ | energia del modello appreso su phage display, valutata su sequenza discreta $x$ |
| $\bar E(a)$ | energia attesa, $\mathbb{E}_{x\sim a}[E(x)]$ |
| $w_E$ | peso del termine di energia nella loss totale |
| $\kappa$ | peso del termine di pessimismo $\sigma(a)$ |
| $N_{\text{eff}}$ | numero di cluster di sequenze al 70% di identità |
| `i_ptm`, `i_con`, `pLDDT` | metriche di confidenza di AlphaFold sull'interfaccia e sulla struttura |
| TUP | *target-unrelated peptide* |
