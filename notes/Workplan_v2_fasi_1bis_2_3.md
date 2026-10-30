# Design computazionale di peptidi inibitori dell'interfaccia PA–PB1

## Piano di lavoro, versione 2.2 — Fasi 1-bis, 2 e 3

**15 settembre 2026**

Sostituisce le Fasi 2, 3 e 4 della versione 1 del piano. Le Fasi 0 e 1 sono concluse; i loro risultati sono riassunti nella *Nota di sintesi sulle Fasi 0 e 1* e nei documenti operativi corrispondenti.

> **Revisione 2.2.** La Fase 1-bis è ridotta alla sola caratterizzazione dell'incertezza tramite approssimazione di Laplace. Sono rimossi: il riallineamento del gauge, la diagnostica delle due varianze sull'ensemble, e il **termine di pessimismo $\kappa\sigma$ nella loss di design**. Di conseguenza esce anche il blocco di ablazione corrispondente dalla Fase 2, e la mitigazione contro lo sfruttamento del surrogato passa interamente alla selezione (§3.3) e al controllo decoy (§3.1). Rispetto alla versione 2.1 sono inoltre corretti: la formulazione del pre-controllo di degenerazione (ora sulla leva, quantità invariante di scala), la selezione di $\lambda$ (ora per massimizzazione dell'evidenza, non per validazione incrociata sulle stesse popolazioni usate poi per validare), e la descrizione del rapporto fra i due esemplari dello stimatore.

---

# A. Contesto

L'assemblaggio della polimerasi dell'influenza A richiede l'interazione fra il dominio C-terminale della subunità PA e l'estremità N-terminale di PB1. Un peptide che occupi quel solco impedisce l'assemblaggio del complesso. Il progetto sviluppa una pipeline per progettare tali peptidi.

**Bersaglio**: 2ZNL, catena A, residui 408–714. **Binder**: peptide lineare di 15 residui. **Riferimento nativo**: `MDVNPTLLFLKVPAQ`.

La pipeline ottimizza per gradiente la sequenza del binder attraverso i pesi di AlphaFold2 (*hallucination*, protocollo a tre stadi soft / temp / hard), combinando due termini:

$$\mathcal{L}_{\text{tot}} = \mathcal{L}_{\text{AF}} + w_E \cdot E(a)$$

dove $\mathcal{L}_{\text{AF}}$ è costruita sulle metriche di confidenza di AlphaFold2 (`i_ptm`, `i_con`, `pLDDT`) ed $E$ è un modello di energia (MLP a tre strati, circa 14.300 parametri) appreso su dati NGS di phage display contro PA tramite una verosimiglianza binomiale negativa del processo di selezione e amplificazione.

Sono disponibili misure di $K_D$ (interferometria a biostrato) su 23 peptidi selezionati.

---

# B. Stato all'ingresso

## B.1 Acquisito

**Il modello di energia determina bene le differenze e male i livelli assoluti.** La selectivity $\log(F3/F2)$ generalizza a famiglie di sequenze mai viste (0,45 ± 0,08 su 10 split per cluster, segno positivo 10 volte su 10); l'abbondanza assoluta no, per ragioni proprie del disegno sperimentale e non del modello. Il termine di energia è validato come **confronto relativo fra candidati**, che è l'uso previsto.

**La divergenza fra rappresentazione soft e discreta è stata eliminata per costruzione.** Il correttore straight-through valuta sempre su sequenze reali; il gap è identicamente nullo in ambiente di test.

**La contaminazione dei dati di selezione non è un fattore condizionante.** Il cluster più sospetto, pur coprendo il 34% della massa di letture, perturba il modello meno di un'esclusione casuale di massa equivalente.

**Non esiste un eccesso aromatico da correggere.** I design si collocano fra la libreria pre-selezione e i leganti validati. Resta una differenza di distribuzione posizionale, non caratterizzabile con le classificazioni strutturali disponibili. **Nessuna penalità composizionale viene introdotta** in questo piano.

## B.2 Non acquisito, con impatto sul disegno

| Questione | Stato | Dove viene affrontata |
|---|---|---|
| Efficacia dello straight-through nel regime reale (5,7–29,5 σ) | verificata solo su traiettoria sintetica (0,25 σ) | §2.1.b |
| Calibrazione delle metriche strutturali su questa classe di interfaccia | mai verificata | §2.1.c |
| Incertezza epistemica del modello di energia | l'ensemble non fornisce uno stimatore utilizzabile | Fase 1-bis |
| Giustificazione della capacità del modello di energia | non dimostrata (bootstrap: 1 confronto su 8) | Fase 4 |
| Taratura di $w_E$ | le tarature precedenti sono state condotte contro un forward divergente e non sono trasferibili | §2.4 |

## B.3 Struttura del piano

**(1) Separazione fra verifica di comportamento e verifica di merito.** Poiché l'affidabilità delle metriche di confidenza strutturale per i complessi peptide–proteina corti non è stabilita per questo sistema, la **Fase 2 verifica che la macchina si comporti correttamente**, con endpoint interni che non presuppongono alcuna metrica di affinità, e la **Fase 3 verifica il merito e la calibrazione** con le metriche che l'audit di §2.1.c avrà caratterizzato.

**(2) L'incertezza è caratterizzata, non usata nella loss.** La Fase 1-bis produce uno stimatore di incertezza epistemica per il modello di energia, impiegato **a posteriori**: come diagnostica sui design prodotti e come criterio di selezione in Fase 3. Non entra in $\mathcal{L}_{\text{tot}}$.

**(3) Disegno sequenziale invece che fattoriale** per le ablazioni della Fase 2, che riduce il costo da oltre 300 a circa 70 ore GPU.

## B.4 Dichiarazione di circolarità

Le 23 sequenze con $K_D$ misurato sono l'unico ancoraggio sperimentale del progetto e vengono usate per tre scopi distinti:

1. validazione del modello di energia (Fase 0: correlazione 0,37; parziale al netto del contenuto aromatico 0,39);
2. audit di calibrazione delle metriche strutturali e scelta della metrica di merito (§2.1.c);
3. riferimento per la retrodizione (§3.2).

Gli usi 2 e 3 sono decisionali. **Ne consegue che la validazione della Fase 3 non è indipendente dai dati su cui le scelte di §2.1.c sono state fatte.** Uno split dell'insieme non è praticabile: con 23 punti, due sottoinsiemi da una decina non hanno potenza per nessuna delle due funzioni.

Questa circolarità viene **dichiarata e non risolta**. Conseguenze operative, vincolanti per tutta la durata delle Fasi 2 e 3:

- I risultati di §2.1.c sono riportati come **calibrazione**, mai come validazione.
- Ogni figura o tabella che usa le 23 sequenze ne riporta esplicitamente l'uso precedente.
- Gli intervalli di confidenza sono sempre riportati: con $N = 23$ sono ampi, e presentare correlazioni puntuali senza incertezza darebbe un'impressione di solidità che i dati non sostengono.
- La validazione realmente indipendente è rimandata al lotto sperimentale (§3.5), che è l'unico dato non ancora usato per alcuna decisione.

## B.5 Limitazione dichiarata: nessuna mitigazione in-loop

Il piano non prevede più un termine che penalizzi l'incertezza nella loss. Il loop di design esegue quindi discesa del gradiente contro un modello surrogato imperfetto **senza vincolo interno** che lo trattenga dalle regioni in cui il modello non è vincolato dai dati. È una limitazione nota e accettata, non un'omissione.

Le due mitigazioni residue sono entrambe a valle e vanno trattate come tali, cioè rafforzate rispetto a come erano formulate quando esisteva il termine in-loop:

- **selezione per diversità** (§3.3), che impedisce di spendere il lotto sperimentale su repliche della stessa modalità;
- **controllo su bersaglio decoy** (§3.1), che rileva se il protocollo stia producendo peptidi generici.

A queste si aggiunge la **caratterizzazione dell'incertezza** prodotta dalla Fase 1-bis, usata come filtro e come annotazione dei candidati finali.

---

# FASE 1-bis — Caratterizzazione dell'incertezza del modello di energia

**CPU. Stima: 3–4 giornate.** Nessuna dipendenza da GPU, da ColabDesign o dal fork: lo stimatore è un artefatto Python autonomo, non un componente della loss. Può procedere in parallelo alla preparazione della Fase 2.

## Scopo e usi previsti

Il modello di energia è validato come ordinatore relativo su famiglie di sequenze mai viste, ma non si dispone di alcuna misura di **dove** quell'ordinamento sia affidabile. Il loop di design opera per costruzione fuori dalla regione selezionata, quindi la domanda non è accademica.

Tre usi, tutti a posteriori:

1. **Diagnostica sui design di Fase 2**: quanto i candidati prodotti si collocano in regioni in cui il modello di energia è vincolato dai dati.
2. **Criterio di selezione in §3.5**: soglia sull'incertezza per i candidati destinati alla validazione sperimentale, e quota riservata ad alta incertezza per ottenere informazione su dove il modello sbaglia.
3. **Annotazione dei risultati**: ogni valore di energia riportato per un candidato finale è accompagnato dalla sua incertezza.

Nessuno di questi richiede differenziabilità, compatibilità con la compilazione JIT, o integrazione nell'hook `energy_fn`. Lo stimatore resta esterno.

## Metodo: approssimazione di Laplace sull'ultimo strato

L'ensemble di repliche non fornisce uno stimatore utilizzabile: con il 90% dei dati condivisi le repliche convergono a soluzioni correlate, e la dispersione misurata non ha relazione con la posizione della sequenza rispetto ai dati. È una limitazione nota degli ensemble con forte sovrapposizione.

L'approssimazione di Laplace tratta bayesianamente la sola testa della rete, dove il modello è lineare nelle feature apprese, e fornisce la varianza predittiva in forma chiusa. Non richiede riaddestramento, non dipende dalla scala arbitraria a cui ciascuna ottimizzazione si è fermata, e cresce per costruzione nelle direzioni dello spazio delle feature poco coperte dai dati.

La mappa di feature è l'ultimo strato nascosto, $\phi(x) \in \mathbb{R}^{15}$, con termine di bias, quindi $p = 16$.

---

## 1b.1 — Pre-controllo di degenerazione

**Da eseguire per primo.** Costa minuti e può chiudere l'intera fase.

### La quantità corretta è la leva

Per la regressione lineare bayesiana con rumore gaussiano, $\Sigma = \sigma_n^2(\Phi^\top W\Phi + \lambda I)^{-1}$, quindi il rapporto fra termine epistemico e termine di rumore

$$\frac{\phi(x)^\top \Sigma\, \phi(x)}{\sigma_n^2} \;=\; \phi(x)^\top\big(\Phi^\top W\Phi + \lambda I\big)^{-1}\phi(x) \;=\; h(x)$$

è la **leva**, che non dipende da $\sigma_n^2$ ed è quindi calcolabile prima di aver definito il termine di rumore. Questo è ciò che rende il pre-controllo eseguibile per primo.

> **Correzione rispetto alla versione 2.1.** La versione 2.1 chiedeva "l'escursione di $\phi^\top\Sigma\phi$ relativa a $\sigma_n^2$" con una $\Sigma$ provvisoria. Il criterio era mal posto: quel rapporto dipende interamente dalle costanti $\sigma_n^2$ e $\lambda$ fissate arbitrariamente. La leva è la formulazione invariante.

### L'ordine di grandezza è prevedibile

Per $\lambda \to 0$, la somma delle leve sui punti di training vale $p$. La leva media di un punto di training è dunque

$$\bar h \approx \frac{p}{N} = \frac{16}{N}$$

cioè circa $1{,}4 \times 10^{-3}$ sul pool di training delle repliche e circa $5{,}9 \times 10^{-4}$ sul pool completo. Con $\lambda > 0$ è ancora minore.

**Il termine epistemico è quindi dell'ordine di qualche decimo di punto percentuale del rumore per un punto tipico.** Perché lo stimatore sia utile, le sequenze di interesse devono avere leve di due o tre ordini di grandezza superiori alla media di training. È una richiesta forte, e la degenerazione è l'esito più probabile: ignorando l'incertezza sulle feature, il Laplace sull'ultimo strato sottostima sistematicamente.

### Metodo

1. Estrarre $\phi(x)$ dai pesi congelati del modello di produzione: pura algebra sugli array già caricati, nessuna modifica al fork.
2. Costruire $\Phi$ sul training set e calcolare $h(x) = \phi(x)^\top(\Phi^\top\Phi + \lambda I)^{-1}\phi(x)$ con $\lambda$ piccolo. La matrice da invertire è $16 \times 16$.
3. Calcolare $h(x)$ su quattro popolazioni: training set (per la distribuzione di riferimento), 23 sequenze caratterizzate, rappresentanti di cluster, e un campione dei design prodotti in Fase 0.
4. Riportare la distribuzione di $h(x)/\bar h$ per popolazione, con $\bar h = p/N$.

### Criterio

- **$h(x)/\bar h$ resta entro un ordine di grandezza** su tutte le popolazioni, design inclusi → lo stimatore non discrimina. **La Fase 1-bis si chiude qui con esito negativo**, e la caratterizzazione dell'incertezza viene rimandata alla Fase 4, dove il modello a forma multilineare la fornisce nativamente.
- **$h(x)/\bar h$ supera due o tre ordini di grandezza** su una frazione apprezzabile delle sequenze, in particolare sui design → procedere a §1b.2.

I design di Fase 0 sono la popolazione più informativa delle quattro: sono l'unica che rappresenta il regime in cui lo stimatore verrebbe effettivamente usato.

### Nota sui proxy di distanza

La leva sostituisce la distanza di Hamming come misura di quanto una sequenza sia lontana dai dati. Il motivo è dimostrabile a priori: in una libreria random di 15-meri, il numero di sequenze entro distanza $d$ è $\sum_{k \le d}\binom{15}{k}19^k$, e con circa 27.000 sequenze di training su $20^{15}$ possibili il numero atteso di vicini è dell'ordine di 0,1 a $d = 8$ e 17 a $d = 10$. La distanza minima è quindi concentrata fra 8 e 10 per quasi ogni sequenza, priva di escursione, e incapace di correlare con qualunque cosa.

Verificarlo una volta con l'istogramma dei valori già calcolati in Fase 1, e registrare che il fallimento del criterio corrispondente in Fase 1 era **non informativo**, non un esito negativo.

---

## 1b.2 — Stima

### Passo 1 — bersaglio e termine di rumore

Il modello è addestrato su una verosimiglianza binomiale negativa applicata a conteggi: non esiste una regressione ai minimi quadrati su $E$, quindi non esistono "residui dell'energia" da cui ricavare $\sigma_n^2$. La dispersione di $E$ sul training set non è una varianza di rumore.

**Formulazione adottata: rifit dell'ultimo strato sulla selectivity osservata.** Con $\phi(x)$ estratta dai pesi congelati, regressione ridge di

$$y(x) = \log\frac{n_{F_3}(x)}{n_{F_2}(x)}$$

su $\phi(x)$, pesata per la precisione dei conteggi:

$$w(x) \;\propto\; \Big(\tfrac{1}{n_{F_3}(x)} + \tfrac{1}{n_{F_2}(x)}\Big)^{-1}$$

che è l'inverso della varianza approssimata del logaritmo del rapporto. Qui i residui esistono e $\sigma_n^2$ è genuino. Il posterior è $\Sigma = (\Phi^\top W\Phi + \lambda I)^{-1}\sigma_n^2$, con $W$ diagonale dei pesi.

Questa formulazione ha una proprietà desiderabile oltre alla correttezza: l'incertezza risulta riferita **alla quantità per cui il modello è validato**, nello spazio di feature che il modello ha appreso.

**Dichiarazione obbligatoria negli output**: la stima puntuale usata ovunque resta $E$, prodotta dalla testa originale. Da questo rifit si prende soltanto $\sigma$.

### Passo 2 — selezione di $\lambda$

**Per massimizzazione dell'evidenza**, cioè della verosimiglianza marginale, disponibile in forma chiusa per la regressione lineare bayesiana e calcolabile sul solo insieme di stima. Volendo, ottimizzazione congiunta su $\lambda$ e $\sigma_n^2$.

> **Correzione rispetto alla versione 2.1.** La versione 2.1 prescriveva validazione incrociata sui dieci split per cluster. Quelle stesse popolazioni held-out sono ciò su cui $\sigma$ viene validata al passo 4: selezionare $\lambda$ lì e poi validare lì è circolare, e la circolarità riguarda proprio il parametro che governa la scala dell'incertezza.

La validazione incrociata resta utile come **controllo di sensibilità**: riportare come varia il risultato finale con $\lambda$ spostato di un ordine di grandezza in entrambe le direzioni rispetto al valore selezionato dall'evidenza.

### Passo 3 — due esemplari

| Esemplare | Costruito da | Scopo |
|---|---|---|
| **Dieci per replica** | $\Phi_m$ sul training set della replica $m$, con i pesi della replica $m$ | validazione di §1b.3: per ogni famiglia esclusa, lo stimatore usato è quello che non l'ha vista |
| **Uno di produzione** | $\Phi$ sul pool di training del modello di produzione, con i pesi di produzione | artefatto usato per gli scopi di §1b.4 |

**Avvertenza sulla trasferibilità, da riportare negli output.** Le due popolazioni di training differiscono per composizione e per numerosità, non del dieci per cento ma di un fattore rilevante. Poiché la leva scala come $p/N$, l'esemplare di produzione avrà un termine epistemico sistematicamente **più piccolo** di quelli validati. La conclusione tratta sui dieci non si trasferisce automaticamente.

Due modi per gestirlo, da scegliere in fase di istanziazione: produrre un esemplare aggiuntivo sul medesimo pool delle repliche, così che l'artefatto finale stia sulla stessa base di quelli validati; oppure riportare $p/N$ e la distribuzione di $h(x)$ per ciascun esemplare e riscalare esplicitamente l'inferenza. La prima è preferibile se il disallineamento fra pesi e popolazione è accettabile.

---

## 1b.3 — Validazione: $\sigma$ predice l'errore

La domanda "$\sigma$ cresce con la distanza" non è misurabile con i proxy disponibili. La domanda operativa è **se $\sigma$ predica l'errore**, ed è misurabile direttamente perché esistono famiglie escluse con selectivity osservata.

### Protocollo

1. Per ciascuna replica $m$, la popolazione held-out è costituita dalle sequenze escluse da quella replica. Calcolare $\sigma(x)$ con l'esemplare **della replica $m$**, mai con quello di produzione.
2. Aggregare le dieci popolazioni. **Usare i valori di energia standardizzati $z_m$, non $E_m$ grezza**: le dieci repliche hanno scale diverse, e aggregare i valori grezzi introdurrebbe una dispersione spuria.
3. Binnare per $\sigma$ predetta e misurare, entro ciascun bin, l'accordo fra energia predetta e selectivity osservata, con intervalli di confidenza bootstrap.

### Specificazione degli estimatori

Da fissare prima di implementare, perché non esistono scelte canoniche.

- **Misura di accordo**: correlazione di Pearson pesata fra $z_m$ e $y(x)$, con i pesi di precisione del passo 1. La correlazione di rango non pesata è riportata come controllo di robustezza, non come metrica primaria.
- **Bootstrap**: ricampionamento con probabilità proporzionali ai pesi, coerente con la misura di accordo.
- **Numerosità efficace**: con pesi che variano di ordini di grandezza, la potenza va calcolata su $N_{\text{eff}} = (\sum w)^2/\sum w^2$ e non sulla numerosità nominale. Il numero di bin va fissato dopo aver misurato $N_{\text{eff}}$, non prima.

### Criterio

Decrescita monotona dell'accordo held-out su almeno tre bin, con intervalli di confidenza che non si sovrappongono fra il primo e l'ultimo.

### Tre controlli obbligatori

Senza di essi il criterio è soddisfacibile da quantità prive di contenuto proprio.

- **Comparatore.** Ripetere l'identico protocollo con un predittore di incertezza banale, per esempio il valore assoluto dell'energia standardizzata. Se $\sigma$ non lo batte, non porta informazione propria.
- **Confondimento fra $\sigma$ ed energia.** Entrambi sono funzioni di $\phi(x)$: i bin di $\sigma$ possono risultare sistematicamente bin di energia estrema, dove la relazione con la selectivity differisce per ragioni indipendenti. Riportare la distribuzione dell'energia per bin e, se sbilanciata, stratificare.
- **Robustezza alla ponderazione.** Ripetere senza pesi di precisione e verificare che il risultato non cambi qualitativamente.

---

## 1b.4 — Decisione sull'uso

| Esito | Decisione |
|---|---|
| §1b.1 rileva degenerazione | Nessuno stimatore. La caratterizzazione dell'incertezza è rimandata alla Fase 4. §3.5 non usa soglie su $\sigma$ |
| §1b.1 passa, §1b.3 passa | Lo stimatore è adottato per i tre usi dichiarati in apertura. §3.5 include una soglia su $\sigma$ e una quota riservata ad alta $\sigma$ |
| §1b.1 passa, §1b.3 fallisce | Lo stimatore è riportato come diagnostica descrittiva sui design, senza ruolo decisionale in §3.5 |

La decisione, con i numeri che la motivano, va registrata in un artefatto versionato: è un input diretto per l'istanziazione di §3.5.

> ## GATE 1-bis
>
> Non bloccante per costruzione: si procede alla Fase 2 in ogni caso, poiché lo stimatore non entra nella loss. Cambia soltanto la formulazione dei criteri di selezione di §3.5 e l'esistenza o meno dell'annotazione di incertezza sui candidati finali.

---

# FASE 2 — Verifica di comportamento della pipeline

**GPU. Stima: 2 settimane, circa 70 ore GPU.**

## Che cosa questa fase stabilisce, e che cosa no

**Stabilisce** che il loop di design si comporti correttamente: che la correzione del forward funzioni nel regime reale, che nessuno dei due termini della loss schiacci l'altro, che esista un intervallo di $w_E$ utilizzabile, che l'ottimizzazione converga in modo riproducibile.

**Non stabilisce** che i design prodotti siano buoni. Gli endpoint sono interni e non presuppongono che alcuna metrica misuri l'affinità. Il merito è oggetto della Fase 3.

Questa separazione rende la Fase 2 interpretabile anche nell'eventualità che l'audit di §2.1.c trovi le metriche strutturali non calibrate su questo sistema.

## 2.1 — Blocco di apertura

Tre verifiche, tutte prerequisiti. Nessuna configurazione dell'ablazione viene lanciata prima che questo blocco sia chiuso.

### 2.1.a — Criteri di accettazione residui dell'artefatto

Rimasti in sospeso alla fine della Fase 1 perché richiedono l'esecuzione di AlphaFold2 reale: retrocompatibilità su traiettoria reale, cioè con il termine di energia disattivato la traiettoria deve essere identica a quella del ColabDesign originale a parità di seed.

### 2.1.b — Efficacia dello straight-through nel regime reale

La verifica di Fase 1 è stata condotta su traiettoria sintetica, che raggiunge un gap massimo di 0,25 deviazioni standard contro le 5,7–29,5 misurate nei run reali: due ordini di grandezza sotto il regime rilevante. Il meccanismo è dimostrato, l'efficacia no.

**Metodo.** Eseguire la configurazione di protocollo che in Fase 0 aveva prodotto i gap massimi, con registrazione per passo, in due bracci: forward legacy e forward straight-through.

**Criterio.** Nel braccio straight-through, gap identicamente nullo per l'intera traiettoria; nel braccio legacy, riproduzione del profilo misurato in Fase 0. Il secondo è il controllo positivo: se il braccio legacy non riproduce il gap, l'ambiente non è comparabile a quello dei run originali e il confronto non è valido.

### 2.1.c — Audit di calibrazione dei termini della loss

**Motivazione.** Il modello di energia è stato sottoposto ad audit sistematico nelle Fasi 0 e 1. Il termine strutturale, che domina la loss, non lo è mai stato, benché l'affidabilità delle metriche di confidenza per i complessi peptide–proteina corti sia una questione aperta per questa classe di sistemi.

**Metodo.** Predire i complessi delle 23 sequenze con $K_D$ misurato, con i tre predittori disponibili (AlphaFold2 nella configurazione usata in design, AlphaFold3, Boltz-2). Correlare ciascuna metrica (`i_ptm`, `i_con`, `pLDDT`, e le corrispondenti di Boltz-2) con $\log K_D$.

**Analisi complementare, che testa la premessa della pipeline.** Correlazione parziale di ciascuna metrica strutturale con $\log K_D$ **controllando per $E$**, e viceversa. I due termini sono stati combinati assumendo che portino informazione complementare; questa assunzione non è mai stata verificata. Se le correlazioni parziali collassano, i due termini sono ridondanti; se si conservano, sono complementari e la combinazione è giustificata.

**Obbligo di riportare l'incertezza.** Con $N = 23$ gli intervalli di confidenza bootstrap sono ampi e vanno riportati per tutte le metriche. Si applica §B.4: questi risultati sono calibrazione, non validazione.

**Criteri di lettura.**

| Esito | Conseguenza |
|---|---|
| Metriche strutturali meno calibrate di $E$ | Il bilanciamento attuale è invertito rispetto all'affidabilità dei due segnali. $w_E$ va cercato in un intervallo molto più alto di quanto tentato finora |
| Calibrazione comparabile, correlazioni parziali conservate | I due termini sono complementari. La combinazione è giustificata e la taratura procede come previsto |
| Metriche strutturali non calibrate | `i_ptm` è un criterio di plausibilità, non di affinità. Va usata come **vincolo** (soglia) anziché come obiettivo da massimizzare. È una modifica alla forma di $\mathcal{L}_{\text{AF}}$, non ai pesi, e va decisa prima di §2.2 |

La metrica risultata meglio calibrata diventa la metrica di merito della Fase 3.

**Costo: poche ore di GPU.**

## 2.2 — Modalità di forward

Primo blocco dell'ablazione sequenziale. Tre bracci:

| Braccio | Forward | Note |
|---|---|---|
| Legacy | $E(a)$ | controllo, riproduce i run precedenti |
| ST-argmax | $E(\text{one-hot}(\arg\max a))$ | corrente |
| ST-$K$ campioni | $\frac{1}{K}\sum_k E(x_k)$, $x_k \sim a$, $K = 32$–$64$ | vedi sotto |

**Lo stimatore a $K$ campioni** attenua i due difetti dello straight-through con argmax singolo: il gradiente valutato in un solo vertice, e la sparsità. Con $K$ campioni indipendenti da $a$, ciascuno discretizzato e trattato con straight-through, il forward è la stima Monte Carlo dell'energia attesa $\bar E(a) = \mathbb{E}_{x\sim a}[E(x)]$, calcolata esclusivamente su sequenze reali, e il gradiente è la somma di $K$ contributi valutati in vertici diversi.

Il costo è trascurabile: con 14.300 parametri, $K = 64$ forward restano irrilevanti accanto a un forward di AlphaFold2. La varianza introdotta è inferiore a quella già presente per via del dropout.

**Endpoint del blocco**: gap nullo, energia finale nel range fisiologico, densità del gradiente dell'energia (frazione di componenti non nulle, che lo stimatore a $K$ campioni deve aumentare), riproducibilità fra seed.

Il forward vincente viene fissato per i blocchi successivi.

## 2.3 — Schedulazione di $w_E$

Secondo blocco, a forward fissato. Tre livelli:

| Livello | Forma |
|---|---|
| Costante | $w_E$ fisso per tutti gli stadi |
| Accoppiato alla concentrazione | $w_E(t) = w_E^{\max}\big(1 - H(a^{(t)})/H_{\max}\big)$, con $H$ l'entropia media per posizione |
| Solo stadio hard | $w_E = 0$ negli stadi soft e temp |

**Motivazione.** All'inizio dello stadio soft, $a$ è prossimo al baricentro: l'argmax (o il campione) è quasi casuale e il gradiente dell'energia è rumore con una direzione preferenziale, cioè bias e non informazione. Il secondo livello fa parlare l'energia quando $a$ è abbastanza concentrata perché il suo argmax significhi qualcosa, con un solo iperparametro che sostituisce l'attuale tabella di pesi per stadio.

Il terzo livello ha un argomento distinto: nello stadio hard $a$ è già discreta, quindi lo straight-through è **esatto**, senza bias. L'energia smette di essere un termine di esplorazione e diventa un termine di raffinamento fra candidati già strutturalmente plausibili, coerentemente con ciò per cui il modello è validato.

## 2.4 — Magnitudine di $w_E$

Terzo blocco. Scansione in unità di deviazione standard sperimentale, resa possibile dalla standardizzazione: $w_E$ ha il significato di "quante deviazioni standard sperimentali valgono un'unità di loss strutturale".

**Le tarature precedenti non sono trasferibili**, essendo state condotte contro un forward divergente. La ricerca riparte da zero, e l'intervallo da esplorare dipende dall'esito di §2.1.c.

**Criterio di bilanciamento, misurato e non assunto.** Registrare $\|\nabla\mathcal{L}_{\text{AF}}\|$ e $\|\nabla E\|$ separatamente e riportarne il rapporto lungo la traiettoria. I due gradienti non competono ad armi pari: quello strutturale attraversa l'intera rete con dropout attivo, quindi è profondo, stocastico e ad alta varianza; quello dell'energia è un percorso diretto e deterministico attraverso tre strati. Su centinaia di passi il termine a bassa varianza prevale sistematicamente, perché il rumore dell'altro si media e il suo no.

**Opzione da testare**: bilanciamento automatico $w_E^{(t)} = \rho\,\|\nabla\mathcal{L}_{\text{AF}}\|/\|\nabla E\|$, con $\rho$ unico iperparametro. Con lo straight-through il gradiente dell'energia è più sparso che nel forward soft, quindi il rapporto delle norme cambia: va misurato prima di scegliere.

## 2.5 — Registrazione

Per ciascun passo di ogni run: `e_fwd`, `e_hard`, `gap`, frazione aromatica di $a$, entropia media di $a$, `i_con`, `i_ptm`, `pLDDT`, $\|\nabla\mathcal{L}_{\text{AF}}\|$, $\|\nabla E\|$.

In assenza di queste serie temporali le ablazioni non sono interpretabili a posteriori. La diagnosi del gap in Fase 0 è stata possibile soltanto perché le traiettorie dei run precedenti erano state conservate, e non lo sarebbe stata sui soli valori finali.

## 2.6 — Disegno sperimentale

**Sequenziale, non fattoriale.** Ordine: §2.2 (forward) → §2.3 (schedulazione) → §2.4 (magnitudine). A ciascun blocco si fissa il livello vincente e si procede.

Circa 50 traiettorie per configurazione, 3–4 configurazioni per blocco. Seed fissi e condivisi fra le configurazioni dello stesso blocco, per rendere i confronti appaiati.

Costo: circa 70 ore GPU. Il prezzo del disegno sequenziale è l'impossibilità di stimare le interazioni fra fattori, accettabile perché nessuna interazione forte è attesa fra correzione del forward e schedulazione del peso.

> ## GATE 2
>
> **Superato se esiste almeno una configurazione per cui:**
> 1. il gap soft/discreto è identicamente nullo lungo tutta la traiettoria;
> 2. l'energia finale resta nel range fisiologico (entro circa 2–3 deviazioni standard sotto il minimo osservato), senza divergenza;
> 3. il rapporto fra le norme dei gradienti resta in un intervallo in cui entrambi i termini contribuiscono, senza che uno domini per la maggior parte della traiettoria;
> 4. l'annealing converge a una sequenza definita;
> 5. i risultati sono riproducibili fra seed.
>
> **Nessuno di questi criteri riguarda la qualità dei design.** Il Gate 2 stabilisce che la macchina funziona, non che produca buoni candidati.
>
> **Se non superato**: la configurazione va diagnosticata prima di procedere. Il criterio 3 in particolare può richiedere di tornare a §2.4 con un intervallo diverso, oppure di rivedere la forma di $\mathcal{L}_{\text{AF}}$ secondo l'esito di §2.1.c.

---

# FASE 3 — Verifica di merito e calibrazione

**GPU e CPU. Stima: 1–2 settimane.**

Usa la configurazione vincente della Fase 2 e la metrica di merito selezionata in §2.1.c. Tutti i risultati ricadono sotto la dichiarazione di circolarità di §B.4.

## 3.1 — Controllo su bersaglio decoy

Il test più diagnostico del piano, ed è l'unico che copre il regime di estrapolazione: la validazione del modello di energia certifica il ranking in interpolazione, mentre il loop di design opera fuori dalla regione selezionata per costruzione. In assenza di mitigazione in-loop (§B.5), la sua importanza aumenta.

**Metodo.** Eseguire la configurazione vincente contro un bersaglio decoy: una proteina diversa di dimensione comparabile, oppure PA con le posizioni di hotspot permutate. Confrontare le sequenze prodotte con quelle ottenute sul bersaglio reale.

**Applicato a entrambi i termini**, non al solo termine di energia:
- con il solo termine strutturale attivo, per verificare che $\mathcal{L}_{\text{AF}}$ discrimini fra bersagli;
- con il solo termine di energia attivo, che per costruzione non ha nozione del bersaglio e deve quindi produrre sequenze indistinguibili nei due casi (controllo negativo atteso, utile come verifica della sensibilità del confronto);
- con la configurazione completa.

**Criterio.** Le sequenze prodotte sui due bersagli devono essere distinguibili nella configurazione completa. Confronto per distribuzione (composizione, distanza fra i cluster ottenuti), non per identità di sequenza.

**Se non lo sono**, il protocollo sta producendo peptidi generici e nessun risultato a valle è interpretabile come specifico per PA–PB1.

## 3.2 — Retrodizione

Il bersaglio non è il riferimento nativo: la Fase 0 ha mostrato che il registro di legame non è stabile fra design e sequenza nativa, quindi la somiglianza al riferimento non è il criterio giusto.

**Domanda.** Il loop, eseguito liberamente, produce sequenze con energia e profilo compositionale comparabili a quelli dei leganti caratterizzati sperimentalmente?

**Metodo.** Confrontare la distribuzione dei design con quella delle 23 sequenze validate su: energia standardizzata, composizione amminoacidica aggregata, distanza dai cluster arricchiti del training set. Riportare anche il confronto con la libreria pre-selezione come riferimento inferiore.

**Vincolo derivante da §B.4.** Le 23 sequenze sono già servite a calibrare la metrica di merito in §2.1.c. Questo confronto è quindi **descrittivo, non confermativo**, e va riportato come tale.

## 3.3 — Diversità nella selezione

**Mitigazione primaria** contro lo sfruttamento del surrogato, in assenza del termine in-loop (§B.5).

I candidati finali vanno selezionati per **rappresentanti di cluster**, non per punteggio decrescente. In un'ottimizzazione contro un surrogato imperfetto, i primi $N$ per punteggio sono tipicamente la stessa modalità replicata, e un lotto sperimentale speso su $N$ varianti della stessa soluzione non distingue nulla.

Usare lo stesso strumento di clustering della Fase 0 (soglia 70% di identità), e riportare quanti cluster distinti siano rappresentati fra i primi $N$ per punteggio: è la misura diretta di quanto l'ottimizzazione sia collassata, e va riportata come risultato, non solo usata come filtro.

## 3.4 — Coerenza fra predittori

In gran parte assorbita dall'audit di §2.1.c. Resta una domanda distinta: l'anticorrelazione osservata in precedenza fra le metriche di Boltz-2 e AlphaFold3 sui candidati era un artefatto dell'aver ordinato design patologici?

**Metodo.** Ricalcolare le correlazioni fra predittori sui design prodotti con la configurazione vincente della Fase 2, e confrontarle con i valori precedenti.

## 3.5 — Criteri di selezione per la validazione sperimentale

**Da fissare prima di esaminare i risultati**, e da registrare in questo documento al momento della fissazione:

- soglia sulla metrica di merito selezionata in §2.1.c;
- distanza minima dai cluster arricchiti del training set, per garantire che i candidati siano nuovi;
- numero minimo di cluster distinti rappresentati nel lotto;
- **se §1b.4 ha adottato lo stimatore**: soglia su $\sigma$ per la quota principale del lotto, e quota riservata a candidati ad alta $\sigma$.

**Il lotto sperimentale è l'unico dato del progetto non ancora utilizzato per alcuna decisione**, ed è quindi l'unica validazione genuinamente indipendente disponibile. La composizione del lotto va decisa di conseguenza: una quota va riservata a candidati che i criteri automatici non selezionerebbero, perché un lotto composto solo dai candidati migliori secondo il modello non produce informazione su dove il modello sbaglia.

---

# FASE 4 — Rimandato

**Modello di energia a forma multilineare (Potts).** Un modello di Potts, eventualmente a rango ridotto sugli accoppiamenti, è multilineare per costruzione: l'energia attesa $\bar E(a)$ si calcola in forma chiusa e il problema del rilassamento continuo non si ripresenta, né per approssimazione né per campionamento. Il rango fornisce inoltre un asse continuo di capacità lungo il quale scandire la complessità con lo stesso protocollo di split per cluster, che è la determinazione principiata che il confronto a tre architetture della Fase 1 non ha potuto dare. Essendo lineare nelle feature fisse, ammette anche incertezza in forma chiusa senza il regime di degenerazione discusso in §1b.1, perché la dimensione dello spazio delle feature è confrontabile con la numerosità dei dati anziché di tre ordini di grandezza inferiore.

**Termine di pessimismo nella loss.** Rimandato insieme allo stimatore che lo renderebbe possibile.

**Condizionamento continuo verso un riferimento.** Un termine che tira la sequenza verso un profilo di riferimento con peso regolabile, come parametrizzazione esplicita del compromesso fra esplorazione e sfruttamento.

**Confronto con un outer loop.** Generazione senza termine di energia, seguita da filtro strutturale e ordinamento su $E$, come braccio di controllo per stabilire se la guida in-loop aggiunga qualcosa rispetto all'ordinamento a posteriori.

**Guidance su modelli AF3-like.** Steering del modulo di diffusione tramite potenziali, o guidance sugli stati nascosti del Pairformer.

---

# C. Budget

| Fase | Durata | Risorsa |
|---|---|---|
| 1-bis | 3–4 giornate | CPU |
| 2 | ~2 settimane | GPU, circa 70 ore |
| 3 | 1–2 settimane | GPU + CPU |

La Fase 1-bis può procedere in parallelo alla preparazione della Fase 2 e non è bloccante per essa.

# D. Nota sui cancelli

I cancelli sono formulati in modo da poter fallire. Il Gate 2 in particolare non presuppone che i design siano buoni: il suo fallimento indicherebbe un problema della macchina, il suo superamento non garantisce alcun merito. Separare le due affermazioni è ciò che permette di diagnosticare l'una indipendentemente dall'altra.

# E. Decisioni da fissare prima dell'esecuzione

1. **Gestione del disallineamento fra i due esemplari** di §1b.2 passo 3: esemplare aggiuntivo sul pool delle repliche, oppure riscalatura esplicita dell'inferenza.
2. **Numero di bin** per la validazione di §1b.3, da fissare dopo aver misurato la numerosità efficace $N_{\text{eff}}$.
3. **Soglie di §3.5**, da fissare prima di esaminare i risultati della Fase 3.
