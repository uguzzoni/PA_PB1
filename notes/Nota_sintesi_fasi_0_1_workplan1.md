# Design computazionale di peptidi inibitori dell'interfaccia PA–PB1

## Nota di sintesi sulle Fasi 0 e 1

**14 settembre 2026**

---

## 1. Oggetto

L'assemblaggio della polimerasi dell'influenza A richiede l'interazione fra il dominio C-terminale della subunità PA e l'estremità N-terminale di PB1. Un peptide che occupi quel solco impedisce l'assemblaggio del complesso e quindi la replicazione virale. Il progetto sviluppa una pipeline computazionale per progettare tali peptidi.

**Bersaglio**: 2ZNL, catena A, residui 408–714 (numerazione `auth`). **Binder**: peptide lineare di 15 residui, amminoacidi canonici. **Riferimento nativo**: `MDVNPTLLFLKVPAQ`, l'N-terminale di PB1, che nella struttura cristallografica occupa il solco per l'intera lunghezza.

La pipeline combina due sorgenti di informazione indipendenti.

**(a) Plausibilità strutturale.** ColabDesign/AfDesign ottimizza per gradiente la sequenza del binder attraverso i pesi di AlphaFold2 (*hallucination*), massimizzando metriche di confidenza del complesso predetto (`i_ptm`, `i_con`, `pLDDT`). Il protocollo procede in tre stadi: **soft** (la sequenza è una distribuzione continua $a = \mathrm{softmax}(L/T)$ su ciascuna delle 15 posizioni), **temp** (annealing di $T$), **hard** (sequenza discreta).

**(b) Segnale sperimentale.** Un modello di energia $E$ appreso su dati NGS di phage display contro PA (libreria random di 15-meri, due round di selezione, repliche forward e reverse). Il modello è una MLP a tre strati (circa 14.300 parametri) addestrata massimizzando una verosimiglianza binomiale negativa derivata da un modello probabilistico del processo di selezione e amplificazione.

I due segnali sono combinati additivamente nella loss di design:

$$\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{AF}} + w_E \cdot E(a)$$

Sono disponibili misure di costante di dissociazione ($K_D$, interferometria a biostrato) su 23 peptidi selezionati: l'unico ancoraggio sperimentale diretto del modello di energia.

---

## 2. Motivo della diagnostica

Le campagne di design precedenti mostravano due anomalie ricorrenti:

1. **Instabilità del termine di energia.** Il valore di $E$ divergeva durante lo stadio soft e dominava la loss, mentre `i_ptm` finale scendeva a circa 0,19 (soglia di accettabilità: 0,5).
2. **Composizione amminoacidica anomala.** Le sequenze prodotte apparivano arricchite in residui aromatici (F, W, Y).

Tre cause candidate, non mutuamente esclusive: **(A)** l'obiettivo di AlphaFold2, che premia contatti e packing e quindi favorisce catene laterali voluminose; **(B)** contaminazione dei dati di selezione da *target-unrelated peptides*, cioè peptidi che legano il supporto di screening (il polistirene lega preferenzialmente F, Y, W) invece del bersaglio; **(C)** il rilassamento continuo, cioè il fatto che il modello di energia sia addestrato su sequenze discrete ma valutato su distribuzioni continue durante gli stadi soft.

La **Fase 0** discrimina fra queste cause. La **Fase 1** corregge quanto identificato e caratterizza il modello di energia.

---

## 3. Risultati della Fase 0 (diagnostica)

### 3.1 Struttura dei dati sperimentali

Il training set contiene 27.342 sequenze valide. Il clustering al 70% di identità dà una taglia effettiva $N_{\text{eff}}$ di 1.375 famiglie indipendenti in round 3 (3.200 in round 2), a fronte di circa 14.300 parametri. La massa di sequenziamento è fortemente concentrata: **due cluster coprono il 50% delle letture** in round 3, diciannove ne coprono il 90%, e il cluster dominante da solo raccoglie il 34% (945 sequenze).

Lo screening bioinformatico per *target-unrelated peptides* assegna al rappresentante del cluster dominante (`VDYNPWLLFLAQPWQ`) una probabilità di legame al polistirene di 0,86. Quella stessa sequenza è, fra le 23 caratterizzate sperimentalmente, **l'unica priva di segnale di legame misurabile**. Tre indicatori indipendenti convergono quindi sullo stesso candidato artefatto. Una seconda sequenza risulta positiva allo stesso predittore pur essendo un legante reale ($K_D$ = 1,59 nM), il che circoscrive l'affidabilità dello screening.

### 3.2 La questione aromatica

| Popolazione | Frazione aromatica (F+W+Y) |
|---|---|
| Riferimento nativo `MDVNPTLLFLKVPAQ` | 6,7% |
| Libreria dopo round 1 | 15,9% |
| Training set, decile di arricchimento più alto | 20,0% |
| Leganti caratterizzati in BLI | 19–21% |
| Design senza termine di energia | 17,0% |
| Design con termine di energia | 19,0% |

**L'anomalia di partenza non trova conferma a livello di popolazione.** I design si collocano fra la libreria pre-selezione e i leganti validati, e non li superano nemmeno negli strati a peso energetico massimo (20,6% a $w_E = 0{,}90$). L'impressione iniziale nasceva verosimilmente dal confronto con il riferimento nativo, che è il ligando naturale evoluto in un contesto proteico e non il punto d'arrivo di una selezione su libreria random: come denominatore è inappropriato.

Tre analisi complementari:

- **Relazione con l'affinità.** Il conteggio aromatico non correla con $\log K_D$ fra i 23 leganti ($r = -0{,}05$), ma il test ha potenza quasi nulla: il conteggio varia fra 1 e 4, con la grande maggioranza a 3. L'assenza di relazione **non è stabilita**, è non verificabile su questo insieme. Il modello di energia correla invece con $\log K_D$ ($r = 0{,}37$), e la correlazione parziale al netto del conteggio aromatico è $0{,}39$ ($p \approx 0{,}07$): il segnale predittivo del modello **non passa** dalla composizione aromatica.
- **Direzione preferita dal modello.** Il triptofano è il residuo più favorito nel ranking del gradiente (rango 1 su 20), e il ranking calcolato sul training set correla a 0,86 con quello calcolato sui leganti caratterizzati.
- **Distribuzione posizionale.** Due criteri strutturali indipendenti (frequenza di contatto predetta su 35 complessi AlphaFold3; geometria della struttura cristallografica) concordano su 6 posizioni su 15. La struttura nativa mostra il peptide incassato per l'intera lunghezza (distanze minime 2,58–3,75 Å su tutte le posizioni), con solo due posizioni chiaramente esposte al solvente. Le analisi composizionali per classe di posizione risultano **fragili rispetto alla soglia** adottata e non sostengono conclusioni causali.

**Stato della questione: aperta.** Non esiste un eccesso aromatico da correggere; esiste una differenza di distribuzione fra design e leganti selezionati, la cui caratterizzazione richiede una classificazione posizionale più robusta di quelle disponibili.

### 3.3 Il rilassamento continuo

Il campionamento casuale dell'interno del simplesso (10.000 punti Dirichlet a concentrazione variabile) mostra che $E(a)$ resta nel range dei valori osservati su sequenze reali nel 99,98% dei casi. **Non esiste una miscalibrazione generica dell'interno.**

Le traiettorie dei run reali mostrano però un comportamento diverso. Definendo

$$\Delta(t) = E\big(a^{(t)}\big) - E\big(\text{one-hot}(\arg\max a^{(t)})\big)$$

la divergenza fra il valore effettivamente ottimizzato e l'energia della sequenza discreta corrispondente, si misurano mediane di $\Delta_{\max}$ fra **5,7 e 29,5 deviazioni standard** a seconda del protocollo, con il gap che si apre durante lo stadio soft e si richiude all'imposizione del vincolo discreto.

I due risultati insieme identificano la natura del fenomeno: la regione problematica **esiste ma ha misura trascurabile**, e viene raggiunta solo seguendo il gradiente, non per campionamento isotropo. È la fenomenologia degli *adversarial example*. Il guadagno energetico inseguito durante lo stadio soft è quindi illusorio: non un ottimo perso nella conversione al discreto, ma un artefatto del rilassamento. Il costo non è un'energia finale errata (le energie finali stanno fra $-2$ e $-9$, cioè 2–2,5 deviazioni standard sotto il minimo osservato, che è il regime di estrapolazione desiderato) ma un budget di gradiente speso, per l'intera durata dello stadio soft, verso un guadagno inesistente.

L'ampiezza di $\Delta_{\max}$ risulta confondata dal numero di passi soft, che varia per protocollo (circa 150 contro circa 50), e non mostra correlazione pulita con l'esito strutturale del design.

### 3.4 Tasso di successo strutturale

Le mediane di `i_ptm` sui protocolli testati stanno fra 0,17 e 0,36; **51 design su 900 superano la soglia di 0,5**. Il tasso è analogo nei run con e senza termine di energia. Questa è, in termini quantitativi, la lacuna più ampia della pipeline, ed è indipendente dalla questione aromatica.

---

## 4. Risultati della Fase 1 (correzione e caratterizzazione)

### 4.1 Correzione del forward pass

Il forward soft calcola $E(a)$, mentre la quantità semanticamente corretta è l'energia attesa $\bar E(a) = \mathbb{E}_{x \sim a}[E(x)]$. Le due divergono già al primo strato non lineare per la disuguaglianza di Jensen.

Poiché il campionamento casuale dell'interno non raggiunge la regione problematica (§3.3), una calibrazione basata su campioni casuali sarebbe inefficace per costruzione. È stato adottato invece lo **straight-through estimator**: il forward valuta sempre al vertice, cioè su una sequenza reale, mentre il gradiente attraversa la discretizzazione.

```python
hard = jax.nn.one_hot(a.argmax(-1), 20)
x    = a + jax.lax.stop_gradient(hard - a)
```

**Verificato**: il gap è identicamente nullo (errore $0{,}00$ esatto su 200 punti di test), il gradiente resta non nullo, e la modalità legacy riproduce il comportamento precedente entro $10^{-6}$. La standardizzazione ($\mu = 3{,}3988$, $\sigma = 2{,}0876$) rende $w_E$ interpretabile come "deviazioni standard sperimentali per unità di loss strutturale" e non altera la nullità del gap.

**Limite**: la verifica è stata condotta su traiettoria sintetica, che raggiunge un gap massimo di 0,25 deviazioni standard contro le 5,7–29,5 dei run reali. Il meccanismo è dimostrato, l'efficacia nel regime rilevante no. La verifica su traiettoria reale richiede GPU ed è rimandata alla Fase 2.

### 4.2 Generalizzazione del modello di energia

Dieci repliche addestrate su altrettanti split per cluster (10% della massa esclusa ciascuna), valutate contro i dati veri sulle famiglie mai viste:

| Quantità | Training | Famiglie escluse |
|---|---|---|
| Abbondanza F2 | 0,59 ± 0,03 | 0,08 ± 0,18 |
| Abbondanza F3 | 0,70 ± 0,02 | −0,11 ± 0,23 |
| **Selectivity** (log F3/F2) | 0,85 ± 0,02 | **0,45 ± 0,13** |

Il controllo necessario è stato eseguito: il modello **completo**, sulle stesse sequenze, dà 0,53 e 0,71 sulle abbondanze. Le sequenze sono quindi fittabili, e il crollo è genuina perdita di generalizzazione, non rumorosità intrinseca.

**Interpretazione.** Il conteggio di una sequenza in F2 è il prodotto della sua rappresentazione iniziale in libreria (estrazione casuale in sintesi, impredicibile dalla sequenza e dominante a conteggi bassi) per la sua sopravvivenza al round (dipendente dalla sequenza). La selectivity $\log(F3/F2)$ cancella il primo fattore. Il fallimento sull'abbondanza è dunque **strutturale e atteso**, e riguarda una quantità che la Fase 2 non utilizza; la selectivity, che è la quantità pertinente, conserva circa metà del segnale in-sample, con segno positivo in 10 split su 10 (test dei segni, $p \approx 0{,}001$).

### 4.3 Confronto con architetture più semplici

Tre architetture per lo stato di selezione, stesso protocollo di training, stesso split (test: 19 sequenze):

| Modello | Selectivity, training | Selectivity, test |
|---|---|---|
| Composizione amminoacidica (21 parametri) | 0,24 | −0,06 |
| Additivo one-hot (300 parametri) | 0,66 | 0,00 |
| MLP (circa 14.300 parametri) | 0,85 | 0,34 |

L'ordinamento per capacità appare rispettato anche sul test. Un bootstrap appaiato a 10.000 ricampionamenti mostra però che **la superiorità dell'MLP non è statisticamente stabilita**: solo 1 confronto su 8 raggiunge la soglia del 95% di segno atteso (e al limite, 95,05%), mentre gli altri stanno fra il 46% e il 94%. Il controllo sul training set (126 sequenze) è pulito su tutti e 8 (≥ 99,99%), il che esclude un difetto della procedura di bootstrap.

**Conseguenza**: con 19 punti di test, questi dati non discriminano fra le architetture. La scelta di mantenere un modello a 14.300 parametri a fronte di $N_{\text{eff}} \approx 1.400$ resta non dimostrata.

### 4.4 Influenza del cluster dominante

Escludendo dal training il cluster identificato come candidato artefatto (§3.1), e confrontando il modello risultante con quello completo:

| Esclusione (massa comparabile, circa 20,8%) | Correlazione sull'energia di selezione |
|---|---|
| Cluster dominante | **0,645** |
| Split casuale, media su 10 repliche | **0,43 ± 0,05** |

**Escludere il cluster dominante perturba il modello meno di quanto faccia l'esclusione di una massa equivalente scelta a caso**, di oltre due deviazioni standard. Lo stesso ordinamento si ripete su una metrica indipendente (selectivity: 0,797 contro 0,35 ± 0,04). La spiegazione risiede nella taglia effettiva: 945 varianti quasi identiche portano l'informazione di una famiglia, non del 34% del dataset. La concentrazione riguarda la massa di letture, non l'influenza sul fit.

Il ranking dei residui preferiti dal gradiente si comporta però diversamente dalle predizioni: il triptofano resta al primo posto in entrambi i modelli, ma la correlazione fra i due ranking completi è solo **0,51**, con riassestamenti ampi (la fenilalanina passa dal rango 18 al 3, la leucina dal 10 al 19). **La derivata è meno stabile della funzione**, ed è la derivata che il loop di design consuma.

### 4.5 Quantificazione dell'incertezza

Un artefatto multi-replica ($M = 10$) è stato costruito per abilitare il termine di pessimismo $E + \kappa\sigma$ previsto per la Fase 2, dove $\sigma$ è la dispersione fra repliche e dovrebbe crescere allontanandosi dai dati di training.

Le statistiche di standardizzazione variano fortemente fra repliche ($\mu$ fra 5,1 e 9,3; $\sigma$ fra 2,8 e 6,5) benché le repliche condividano il 90% dei dati: la standardizzazione per replica, e non globale, era necessaria.

**La verifica di funzionamento fallisce.** La dispersione fra repliche **non cresce** allontanandosi dai dati: la correlazione con la distanza di Hamming minima dal training set è 0,003, cioè nulla. Inoltre la correlazione fra dispersione epistemica e dispersione aleatoria (dovuta al rilassamento) è −0,369, oltre la soglia di indipendenza desiderata.

La causa più plausibile è la sovrapposizione fra repliche: ciascuna esclude solo il 10% della massa, quindi le dieci condividono il 90% del training set e convergono a soluzioni correlate. È una limitazione nota degli ensemble con dati largamente condivisi.

---

## 5. Sintesi

**1. Il modello di energia determina bene le differenze di energia e male i livelli assoluti.** Un solo fenomeno spiega tre osservazioni indipendenti: l'abbondanza non generalizza mentre la selectivity sì (§4.2); le statistiche di standardizzazione variano di un fattore due fra repliche che condividono il 90% dei dati (§4.5); i ranking restano concordanti dopo standardizzazione. L'energia è identificata a meno di una trasformazione affine. Tutto ciò che è relativo è stabile, tutto ciò che è assoluto non lo è.

**2. Questo è compatibile con l'uso previsto, e solo con quello.** La guida energetica del design è un confronto relativo fra candidati. Il modello è validato per quell'uso (Gate 1 superato sulla selectivity) e non per la predizione di quantità assolute.

**3. L'anomalia del rilassamento è reale ed è stata eliminata per costruzione**, ma la correzione non è ancora stata verificata nel regime in cui il problema si manifesta.

**4. La contaminazione dei dati è un limite da dichiarare, non un fattore condizionante.** Il cluster più sospetto ha meno influenza sul fit di un campione casuale di massa equivalente.

**5. La questione aromatica non è chiusa, ma è stata circoscritta.** Non c'è un eccesso da correggere; il segnale predittivo del modello non passa dalla composizione aromatica; la relazione fra aromatici e affinità non è verificabile sui dati disponibili. Resta una differenza di distribuzione posizionale fra design e leganti selezionati, non caratterizzabile con le classificazioni strutturali attuali.

**6. Due elementi consumati dal loop di design risultano meno solidi delle quantità su cui sono stati validati.** Il gradiente del modello è meno stabile delle sue predizioni (§4.4); la dispersione fra repliche non misura la distanza dai dati (§4.5).

**7. Il tasso di successo strutturale resta il divario quantitativamente più ampio** e non dipende dal termine di energia.

---

## 6. Stato all'ingresso della Fase 2

**Validato e utilizzabile**
- Correzione straight-through del forward pass, con standardizzazione; gap nullo per costruzione.
- Modello di energia come stima della componente relativa del legame, incluse famiglie di sequenze non osservate.
- Artefatto multi-replica e strumentazione diagnostica per la registrazione per passo.

**Da verificare in apertura di Fase 2 (richiede GPU)**
- Efficacia dello straight-through su traiettoria reale, nel regime di 5,7–29,5 deviazioni standard.
- Due criteri di accettazione dell'artefatto che richiedono l'esecuzione di AlphaFold2 (retrocompatibilità su traiettoria reale; test placeholder con termine di pessimismo attivo).

**Aperto, con impatto sul disegno della Fase 2**
- Il termine di pessimismo $\kappa\sigma$ non misura ciò per cui è stato introdotto. Va valutato se escluderlo ($\kappa = 0$) o ricostruire l'ensemble con repliche meno sovrapposte.
- La taratura di $w_E$ va ripetuta interamente: le tarature precedenti sono state condotte contro un forward divergente e non sono trasferibili. La nuova ricerca va condotta in unità di deviazione standard sperimentale.
- La formulazione di un'eventuale penalità composizionale resta sospesa: i dati non sostengono né una penalità globale (non c'è eccesso) né una posizionale (le classificazioni disponibili sono fragili).
- La giustificazione della capacità del modello di energia non è dimostrata (§4.3). Rende prioritaria la valutazione di architetture alternative, in particolare rappresentazioni invarianti per posizione, più adatte a una libreria random priva di registro privilegiato.
