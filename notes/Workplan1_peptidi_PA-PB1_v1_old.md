# Piano di lavoro — Pipeline di design peptidi anti-influenza (interfaccia PA–PB1)

**Target**: interfaccia PA–PB1 della polimerasi di influenza A (2ZNL, catena A, residui 408–714 numerazione auth)
**Binder**: 15-mero lineare canonico; riferimento nativo `MDVNPTLLFLKVPAQ` (PB1 N-term)
**Pipeline**: ColabDesign/AfDesign (`design_3stage`, `model_4`) + modello di energia esterno appreso su NGS di phage display (libreria random, likelihood di selezione–amplificazione)
**Validazione a valle**: Boltz-2 su cluster CEA AAR; AF3 sui top candidate

---

## Principio d'ordine

**Prima la diagnostica, poi il modello di energia, poi il loop di design.**

Tarare `energy_weight` su un'energia non calibrata significa ottimizzare bene un obiettivo mal definito. Le Fasi 0 e 2 sono quasi interamente CPU e producono la maggior parte dell'informazione decisionale; la Fase 3 è l'unica costosa in GPU.

**Contesto ereditato da chiarire una volta per tutte.** Il modello di energia è già stato validato esternamente: correlazione buona con $K_D$ misurati in BLI su peptidi selezionati. Questo esclude il modo di fallimento "il modello non riconosce i leganti di PA" e indebolisce il gate di Fase 2 rispetto a versioni precedenti di questo piano. Resta però un limite di **copertura**: i peptidi misurati provengono dalla regione arricchita, quindi la validazione certifica il ranking in *interpolazione*. Il loop di design opera in *estrapolazione* per costruzione. Il controllo su target decoy (4.1) è l'unico test che copre quel regime.

---

## FASE 0 — Diagnostica su ciò che esiste già

Nessun riaddestramento, nessun run di design nuovo. **Costo stimato: 2–3 giornate, quasi tutto CPU.**

---

### 0.1 — Varianza di E nell'interno del simplesso

**Obiettivo.** Quantificare di quanto il forward soft `E(a)` diverge dai valori che il modello di energia su PD assegna a sequenze reali, e stabilire se l'esplosione osservata (da −5 a −189 in stage 2) è un artefatto del rilassamento continuo.

**Metodo.**
1. Campionare $10^4$ punti $a \sim \text{Dirichlet}(\alpha\mathbf{1})$ indipendentemente per ciascuna delle 15 posizioni, con $\alpha$ log-spaziato in $[0.05,\,50]$ (α grande ≈ baricentro uniforme, inizio stage 1; α piccolo ≈ quasi one-hot, fine stage 2).
2. Calcolare `E(a)` con la MLP attuale.
3. Calcolare $\min_x E(x)$ e $\max_x E(x)$ sul training set (one-hot).

**Output.**
- Frazione di punti con $E(a) \notin [\min_x E, \max_x E]$.
- Ampiezza massima della violazione, in unità di $\sigma_{\text{train}}$.
- Scatter $E(a)$ vs $E(\text{one-hot}(\arg\max a))$, colorato per $\alpha$.
- Curva: violazione mediana in funzione di $\alpha$.

**Criterio di lettura.** Un modello ben calibrato sull'interno avrebbe violazioni ≈ 0. Attesa: violazioni massive a α grande, decrescenti verso i vertici. Questa figura giustifica l'intera Fase 2.1 ed è materiale da metodi.

---

### 0.2 — Taglia effettiva del dataset e clustering

**Obiettivo.** Stabilire quanta informazione indipendente contengono i dati, e costruire l'infrastruttura di clustering necessaria per gli split di Fase 2.

**Nota importante sull'uso.** Il clustering **non** viene usato per ripesare la likelihood di training. Nei dati di screening l'abbondanza *è* la misura: il modello di selezione–amplificazione è costruito per spiegare le frequenze osservate, e ripesare le sequenze corromperebbe l'osservabile. Il clustering serve a due scopi diversi: (i) stimare la complessità ammissibile del modello, (ii) costruire split non contaminati da leakage di famiglia.

**Metodo.**
1. MMseqs2 o CD-HIT al 70% di identità sulle sequenze osservate nei round tardi.
2. $N_{\text{eff}} = \sum_s 1/m_s$, con $m_s$ = cardinalità del cluster di $s$.
3. Distribuzione delle dimensioni dei cluster; numero di cluster che coprono il 50% e il 90% della massa di read nell'ultimo round.

**Output.** $N_{\text{eff}}$, istogramma delle dimensioni di cluster, curva di Lorenz della massa di read, e l'assegnazione cluster→sequenza salvata su disco per riuso in Fase 2.3 e 2.5.

**Criterio di lettura.** $N_{\text{eff}}$, non il numero di read né di sequenze uniche, vincola la complessità del modello. Se $N_{\text{eff}}$ è nell'ordine delle centinaia di famiglie indipendenti, i 14k parametri della MLP attuale sono da giustificare esplicitamente (→ 2.4).

**Nota residua da tenere in discussione, non da correggere ora.** Le varianti generate per errore di polimerasi dentro un clone già espanso non sono eventi di selezione indipendenti: hanno fatto hitchhiking sull'espansione del parentale. Se la likelihood modella esplicitamente l'amplificazione ne assorbe una parte; altrimenti è un bias residuo da menzionare nei limiti.

---

### 0.3′ — Composizione amminoacidica dei leganti validati in BLI

**Obiettivo.** Stabilire se, nel sistema in esame, l'arricchimento in aromatici sia associato al legame reale o sia un artefatto.

**Metodo.** Composizione amminoacidica dei peptidi con $K_D$ misurato, stratificata per affinità. Confronto con: WT `MDVNPTLLFLKVPAQ` (una F, nessun W, nessuna Y), composizione della libreria iniziale, composizione del training set stratificata per livello di arricchimento.

**Output.** Tabella di composizione + test di arricchimento per amminoacido.

**Criterio di lettura — questo determina l'interpretazione dell'intera questione aromatica.**
- Se i binder confermati in BLI sono **essi stessi** arricchiti in aromatici → nel sistema gli aromatici sono genuinamente associati al legame. Il problema dei design non è la loro presenza ma il loro **eccesso**, e la penalità composizionale (3.4) va riformulata come vincolo di *sviluppabilità* (solubilità, aggregazione) e non come correzione di artefatto. Cambia anche come la si tara.
- Se i binder confermati **non** sono aromatici → l'arricchimento aromatico nei design è spurio, e la penalità è una correzione.

**Costo: ~10 minuti.** Rapporto informazione/costo più alto della Fase 0.

---


### 0.4 — Direzione del gradiente di E

**Obiettivo.** Ispezionare cosa il modello di energia premia, in modo indipendente dal loop di design.

**Metodo.** $\partial E/\partial a_{i\alpha}$ valutato su one-hot del test set. Produrre:
1. Ranking dei 20 amminoacidi, mediato sulle posizioni.
2. Mappa completa 15×20, non mediata.

**Criterio di lettura.** W/F/Y in testa **uniformemente su tutte le posizioni** = il modello ha appreso appiccicosità generica. Struttura posizionale marcata = il modello ha appreso preferenze specifiche di registro, che è più promettente (anche se, in libreria random, l'esistenza di un registro privilegiato va essa stessa interpretata con cautela).

---

### 0.5 — Composizione dei design esistenti, per contesto strutturale

**Obiettivo.** Discriminare fra causa AF2 e causa energia come sorgente del bias aromatico.

**Metodo.** Sui design già prodotti, e usando le strutture AF2 già generate:
1. Classificare le 15 posizioni in "interfaccia" (contatto con PA sotto soglia di distanza) vs "esposta al solvente" (SASA relativa alta).
2. Frazione aromatica in ciascuna classe.
3. Confronto con: WT, libreria iniziale, training set.

**Criterio di lettura.**
- Aromatici **concentrati all'interfaccia** → causa A (obiettivo AF2: `i_con` conta contatti, e i residui grandi ne fanno di più; pLDDT premia il packing). Coerente con la letteratura su AfDesign.
- Aromatici **uniformi su tutte le posizioni**, incluse quelle esposte → causa B/C (il modello di energia non ha nozione di geometria e non distingue una posizione di interfaccia da una esposta).

---

### 0.6 — Screening TUP sui dati di training

**Obiettivo.** Quantificare la contaminazione da target-unrelated peptides. Senza selezioni di controllo (beads nude, libreria naive, target irrilevante) questa è l'unica leva rimasta sulla contaminazione non-specifica: la separazione dei modi $E_{\text{spec}} - E_{\text{ns}}$ non è implementabile.

**Contesto.** I TUP *selection-related* legano componenti del sistema di screening invece del target; i *propagation-related* emergono perché alcuni cloni crescono più rapidamente. La firma chimica dei leganti al polistirene è dominata da residui aromatici (F, Y, W) — esattamente la patologia osservata nei design.

**Metodo.**
1. SAROTUP sul set arricchito: motivi TUP, PSBinder (leganti al polistirene), PhD7Faster (cloni a propagazione rapida).
2. Ricerca nel BDB/MimoDB.
3. Stratificare la frazione flaggata per livello di arricchimento.

**Output.** Frazione flaggata totale e per decile di arricchimento; overlap fra sequenze flaggate e peptidi validati in BLI.

**Criterio di lettura.** Se i cloni più arricchiti sono anche i più flaggati, la contaminazione è nel segnale principale e va dichiarata esplicitamente nei limiti del lavoro. L'overlap con i peptidi BLI-validati è particolarmente informativo: se i binder confermati **non** sono flaggati mentre la coda arricchita lo è, il modello sta apprendendo un mix di due segnali separabili almeno concettualmente.

---
### 0.7 — Replay delle traiettorie di design

**Obiettivo.** Verificare l'ipotesi che l'esplosione dell'energia nello stage 2 e il bias aromatico siano lo **stesso** fenomeno.

**Meccanismo sotto test.** Nell'interno del simplesso l'errore di estrapolazione di una MLP ReLU non è isotropo: cresce lungo le direzioni in cui i pesi del primo layer hanno magnitudine maggiore, cioè lungo le feature con il segnale appreso più forte. Se quel segnale è "aromatico → arricchimento", la direzione lungo cui $E(a)$ crolla senza limiti è la direzione aromatica.

**Metodo.** Su run in cui $E$ è esplosa, plottare sugli stessi step: massa di probabilità aromatica di $a$, $E(a)$, `i_con`, `i_ptm`. Se le traiettorie non sono state salvate, 2–3 run brevi con logging (~30 min GPU).

**Output aggiuntivo.** Salvare le traiettorie $a$ visitate: costituiscono parte del set di calibrazione in 2.1.

---


> ### GATE 0
> Al termine della Fase 0 devono essere note:
> 1. l'entità del problema di calibrazione sul simplesso (0.1);
> 2. $N_{\text{eff}}$ e quindi la complessità ammissibile (0.2);
> 3. quale causa domina il bias aromatico (0.3′, 0.4, 0.5, 0.6);
> 4. il livello di contaminazione TUP (0.7).
>
> **Condizione di riconsiderazione**: se 0.7 mostra che i cloni dominanti sono TUP *e* 0.3′ mostra che i binder BLI-validati non sono aromatici, il modello di energia contiene due segnali sovrapposti non separabili con i dati disponibili. Non blocca il progetto, ma va scritto nei limiti e cambia il peso da dare a $E$ nel loop.

---

## FASE 2 — Modello di energia

**Costo stimato: ~2 settimane, CPU.** Ordine interno: calibrazione e standardizzazione per prime (sono prerequisito di tutto il resto), poi split e baseline, poi packaging.

---

### 2.1 — Calibrazione sull'interno del simplesso

**Obiettivo.** Sostituire $E(a)$ con $\bar{E}(a) = \mathbb{E}_{x\sim a}[E(x)]$.

**Motivazione formale.** Il forward soft calcola $E(a)$; la quantità semanticamente corretta è l'energia attesa della distribuzione di sequenze che $a$ rappresenta. Le due differiscono già al primo ReLU: per Jensen, $\text{relu}(Wa+b) \leq \mathbb{E}[\text{relu}(Wx+b)]$, e dopo due composizioni con pesi di segno arbitrario non c'è più controllo su segno né ampiezza dell'errore.

$\bar{E}$ è **multilineare** nelle one-hot per posizione (ogni termine dell'espansione contiene al più una potenza prima di ciascun $a_i$), e una funzione multilineare su un prodotto di simplessi attinge gli estremi sui vertici. Conseguenze:
- $\min_x E(x) \le \bar{E}(a) \le \max_x E(x)$ per ogni $a$ — nessun minimo interno spurio può esistere;
- $\bar{E}(\text{one-hot}(x)) = E(x)$ — retrocompatibilità esatta;
- $\partial\bar{E}/\partial a_{i\alpha} = \mathbb{E}[E(x) \mid x_i = \alpha]$ — il gradiente è l'energia media condizionata al residuo $\alpha$ in posizione $i$, cioè una quantità sperimentalmente interpretabile.

**Metodo — distillazione Monte Carlo** (scelta preferita: indipendente dall'architettura, quindi sopravvive a un futuro cambio del modello di energia).

```
per ogni punto di calibrazione a:
    campiona K = 64 sequenze x_1..x_K ~ a  (indipendenti per posizione)
    target(a) = (1/K) Σ_k E_originale(one_hot(x_k))
addestra Ē_θ(a) su queste coppie
```

Rete studente: stessa architettura, inizializzata dai pesi originali. Con 14k parametri il training è questione di minuti.

**Perché distillare e non fare il Monte Carlo a runtime**: il campionamento non è differenziabile. A runtime servirebbe Gumbel-softmax (che riporta a input soft, cioè al problema di partenza) o REINFORCE (varianza troppo alta su una loss già rumorosa per il dropout di AF2). La distillazione paga il costo MC offline, una volta.

**Set di calibrazione.** Principio: far coincidere la distribuzione di calibrazione con la distribuzione di deployment.
1. Dirichlet log-spaziato come in 0.1 — copre l'annealing di `design_3stage`.
2. **Le traiettorie reali salvate in 0.6** — sono per definizione i punti dove l'ottimizzatore vuole andare, cioè dove l'errore è massimamente sfruttabile. Versione a costo zero dell'adversarial training: ogni giro di design arricchisce il set nella direzione di maggiore fragilità.

**Dipendenza.** 2.1 richiede 0.6. Se si vuole partire prima, iniziare con il solo Dirichlet e aggiungere le traiettorie in un secondo passaggio: $\bar{E}$ si riaddestra in minuti, quindi rifarlo costa nulla.

**Alternativa più economica da tenere in riserva.** *Moment propagation*: riscrivere il forward propagando media e varianza in forma chiusa attraverso i layer (covarianza dell'input block-diagonale perché le posizioni sono indipendenti; ReLU con momenti analitici sotto approssimazione gaussiana). ~40 righe, nessun riaddestramento. Svantaggio: l'approssimazione gaussiana si indebolisce ai layer 2 e 3 (45 e 15 input).

**Verifica.** Ripetere 0.1 su $\bar{E}$. La frazione di violazioni deve crollare.

---

### 2.2 — Standardizzazione

**Obiettivo.** Rendere `energy_weight` un iperparametro con significato stabile e trasferibile.

**Metodo.** $\tilde{E} = (E - \mu_{\text{train}})/\sigma_{\text{train}}$, con $\mu,\sigma$ calcolati sul training set.

**Effetto.** `energy_weight` smette di essere un numero magico e diventa "quante sigma sperimentali valgono un'unità di loss AF". Rende la taratura trasferibile a modelli di energia futuri, che è lo scopo per cui l'hook `energy_fn` è stato reso iniettabile.

**Nota critica per 2.6.** La standardizzazione va fatta **per replica**, con i $\mu_m,\sigma_m$ di ciascuna. Vedi 2.6(c).

---

### 2.3 — Repliche su split di cluster

**Obiettivo.** Ottenere una stima utilizzabile dell'incertezza epistemica.

**Metodo.** $M = 5$–$10$ repliche, ciascuna addestrata su uno **split di cluster diverso** (dall'assegnazione salvata in 0.2).

**Vincolo.** Seed diversi sullo stesso split **non** funzionano: sottostimano gravemente l'incertezza epistemica, perché le repliche convergono a soluzioni correlate. Servono split di dati genuinamente diversi. Se sono disponibili più round di selezione, considerare anche split per round come asse aggiuntivo di variazione.

**Uso a valle.** $\sigma_{\text{repliche}}(a)$ entra nel lower confidence bound di 3.3: $\mathcal{L}_{\text{energia}} = \bar{E}(a) + \kappa\,\sigma(a)$ (segno positivo perché si minimizza: si penalizza il disaccordo). Trasforma "spingi dove il modello promette di più" in "spingi dove il modello è confidente".

**Distinzione da tenere ferma.** Esistono due varianze diverse e solo una è quella giusta per il pessimismo:
- $\text{Var}_{x\sim a}[E(x)]$ — **aleatoria rispetto al rilassamento**: quanto le sequenze rappresentate da $a$ differiscono fra loro. Utile come *diagnostica di convergenza* (deve decrescere durante l'annealing e annullarsi in stage hard; se resta alta a fine stage 2, l'annealing non sta committendo).
- $\text{Var}_{\text{repliche}}[\bar{E}(a)]$ — **epistemica**: quanto il modello non sa. È questa che va nel LCB.

---

### 2.4 — Scala di baseline

**Obiettivo.** Misurare quanto valgono i 14k parametri della MLP. Non è un cambio di architettura (rimandato a Fase 5): è una misurazione.

**Metodo.** Su split per cluster, confrontare via Spearman:
1. **Sola composizione amminoacidica** (20 feature, ignora le posizioni);
2. **Additivo su one-hot** (300 parametri);
3. **MLP attuale** (~14.300 parametri).

**Criteri di lettura.**
- (1) ≈ (3) → il segnale è composizionale, non posizionale. Cambia radicalmente l'interpretazione del bias aromatico: il modello di energia sta essenzialmente contando amminoacidi, e la penalità composizionale di 3.4 si sovrappone quasi interamente a esso.
- (2) ≈ (3) → nessuna epistasi utilizzabile catturata; la MLP sta usando capacità senza guadagno.
- (3) > (2) in modo netto → l'epistasi c'è ed è appresa; giustifica la capacità e rende più interessante la Fase 5 (feature k-mer gappate posizione-invarianti).

---

### 2.5 — Predizione su famiglia mai vista

**Obiettivo.** Stabilire se il modello generalizza a famiglie di sequenze nuove o memorizza quelle arricchite.

**Metodo.** Tenere fuori **interi cluster** arricchiti (non sequenze singole). Addestrare. Misurare la capacità di predire l'arricchimento della famiglia held-out.

**Criteri di lettura.**
- Predizione buona → esiste segnale generalizzante; la scelta della classe di modello è questione di efficienza.
- Predizione ≈ 0 ma buona predizione *entro*-famiglia → il modello sta memorizzando le famiglie. Guidare AF2 con esso significa guidare verso hit già noti, non verso hit nuovi.

**Contesto attenuante.** La validazione BLI esistente riduce la severità di questo gate rispetto a formulazioni precedenti: sappiamo già che il modello ordina correttamente dentro la regione selezionata. 2.5 misura una cosa diversa — se quell'ordinamento si estende a regioni non viste.

**Diagnostiche di supporto.**
- Distribuzione dei residui dopo fit additivo: coda destra pesante concentrata in poche famiglie = struttura "fondo + picchi" più che campo liscio.
- Curva di performance in funzione di $N_{\text{eff}}$ usato in training: saturazione immediata = memorizzazione.

---

> ### GATE 2
> Procedere alla Fase 3 se:
> - 2.1 verificato (violazioni sul simplesso crollate);
> - 2.4 mostra che la MLP batte i baseline, oppure si accetta esplicitamente di usare un modello equivalente a un additivo;
> - 2.5 non è catastrofico.
>
> Se 2.5 è ≈ 0 **e** 2.4 mostra (1) ≈ (3), il modello di energia è un contatore di composizione amminoacidica adattato alle famiglie osservate. Rimane utilizzabile come filtro, ma non come guida per esplorare sequenze nuove — e la Fase 3 va ridimensionata di conseguenza.

---

### 2.6 — Packaging del modello di energia

**Obiettivo.** Trasformare il modello di energia in un artefatto stabile e testato che ColabDesign possa consumare senza che nessuna configurazione della matrice di ablazione di Fase 3 richieda di modificare `af/loss.py`, `af/design.py` o `af/model.py`.

**Motivazione.** La matrice di ablazione fa variare tre assi: forward mode (soft / straight-through / $\bar{E}$), $w_E$ (fisso / GradNorm), $\kappa$ (0 / >0). Se ogni combinazione richiede un edit nel codice interno del fork, i confronti non sono riproducibili e i bug di configurazione diventano indistinguibili dai risultati. **Mezza giornata di lavoro che ne risparmia molte in debugging.**

---

#### (a) Preservare il contratto dell'hook via closure

L'hook è già progettato come `energy_fn(seq_probs: (1,L,20)) -> scalare JAX`. **Non modificarlo.** Tutto lo stato aggiuntivo — repliche, $\kappa$, standardizzazione, forward mode — entra come variabile catturata in una factory:

```python
import jax, jax.numpy as jnp
from functools import partial

def make_energy_fn(bundle, mode="expect", kappa=1.0, eps=1e-6):
    """
    bundle: dict caricato da disco (vedi (f))
      - "stacked_params": pytree, dim principale M = n. repliche
      - "mu", "sd": (M,) statistiche di standardizzazione per replica
    mode: "soft" (legacy) | "st" (straight-through) | "expect" (Ē calibrata)
    kappa: peso del termine di pessimismo. 0 = media dell'ensemble.
    """
    W  = bundle["stacked_params"]
    mu = bundle["mu"]
    sd = bundle["sd"]

    def single(params, a):
        if mode == "soft":
            return mlp_forward(params, a)
        elif mode == "st":
            hard = jax.nn.one_hot(a.argmax(-1), a.shape[-1])
            x = a + jax.lax.stop_gradient(hard - a)
            return mlp_forward(params, x)
        elif mode == "expect":
            return mlp_expect(params, a)      # rete studente distillata in 2.1
        raise ValueError(mode)

    @jax.jit
    def energy_fn(a):
        e = jax.vmap(single, in_axes=(0, None))(W, a)   # (M,)
        z = (e - mu) / sd                                # standardizzazione per replica
        m = z.mean()
        s = jnp.sqrt(z.var() + eps)                      # eps: vedi (d)
        return m + kappa * s                             # + perché si minimizza

    return energy_fn
```

Le configurazioni dell'ablazione diventano argomenti della factory; ColabDesign resta invariato.

---

#### (b) Vmap sulle repliche, non un loop Python

Impilare i pesi delle $M$ repliche in array con dimensione principale $M$ e usare `jax.vmap`. Con 14k parametri e $M=10$ il costo aggiuntivo è trascurabile rispetto al forward di AF2, ma un loop Python forzerebbe $M$ tracce separate e ricompilazioni a ogni cambio di configurazione.

---

#### (c) Standardizzare per replica, **prima** di aggregare

**Il dettaglio più facile da sbagliare.** Repliche addestrate su split di cluster diversi hanno offset e scale di output diversi: sono differenze di *nuisance*, non di conoscenza.

Se $\sigma$ viene calcolata sugli output grezzi, la deviazione fra repliche è dominata da quegli offset, e il termine di pessimismo penalizza il disaccordo sbagliato — non "il modello non sa", ma "le repliche hanno intercette diverse". Il risultato è un $\kappa\sigma$ approssimativamente costante, che non fa nulla di utile.

Standardizzare **ciascuna replica con i suoi $\mu_m,\sigma_m$** calcolati sul suo training set, poi aggregare. Il $\sigma$ risultante misura disaccordo sul ranking, che è la quantità di interesse.

---

#### (d) Guardie numeriche

`jnp.sqrt` ha gradiente infinito in zero, e $\sigma \to 0$ accade realmente: succede quando le repliche concordano, cioè nella regione ben coperta dai dati, dove l'ottimizzatore passa molto tempo. L'`eps` dentro la radice non è cosmetico.

Da verificare esplicitamente: gradiente finito su un punto dove tutte le repliche restituiscono lo stesso valore.

---

#### (e) Diagnostica separata dalla loss

Le quantità di logging (3.6) non appartengono alla loss. In particolare $E(\arg\max a)$ è non differenziabile e va calcolata sotto `stop_gradient` o fuori dal grafo.

Esportare un secondo callable:

```python
def make_energy_aux(bundle):
    @jax.jit
    def energy_aux(a):
        a_ = jax.lax.stop_gradient(a)
        hard = jax.nn.one_hot(a_.argmax(-1), a_.shape[-1])
        e_soft = ...   # Ē(a)
        e_hard = ...   # E(one_hot(argmax a))
        sigma  = ...   # dispersione fra repliche
        arom   = a_[..., AROMATIC_IDX].sum(-1).mean()
        return {"e_soft": e_soft, "e_hard": e_hard,
                "sigma": sigma, "arom_frac": arom}
    return energy_aux
```

da agganciare all'aux dict di ColabDesign. Mettere queste quantità dentro `energy_fn` produce gradienti spuri o ricompilazioni.

---

#### (f) Bundle serializzato e versionato

Il percorso attuale — JLD2 → script Julia → JSON → dict di array JAX — è accettabile per una rete singola; con $M$ repliche più il modello calibrato diventa fragile.

Serializzare **un solo artefatto** (`.npz` o safetensors) contenente:
- pesi impilati delle $M$ repliche (dense1/2/3, W e b);
- $\mu_m$, $\sigma_m$ per replica;
- specifica dell'architettura;
- metadati: hash del commit, versione del dataset, versione del set di calibrazione, data, modalità (originale vs calibrato).

Il loader **valida le shape e fallisce rumorosamente**. Shape attese per replica: `(45,300)`, `(15,45)`, `(1,15)`, con dimensione principale $M$. La convenzione Flux `(out,in)` coincide con JAX, ma è un punto di attrito già noto: un assert lì costa nulla ed evita di scoprire una trasposizione dopo trenta ore di GPU.

---

#### Checklist di accettazione per 2.6

| # | Test | Criterio |
|---|---|---|
| 1 | Coerenza ai vertici | `energy_fn(one_hot(x))` con `kappa=0, mode="expect"` riproduce l'energia originale standardizzata entro $10^{-5}$ |
| 2 | Bound sul simplesso | 0.1 ripetuto su $\bar{E}$: violazioni crollate rispetto al forward soft |
| 3 | Gradiente | Differenze finite vs autodiff su $\partial/\partial a$, su punti interni **e** vicini ai vertici |
| 4 | Gradiente del termine $\kappa\sigma$ | Finito e non nullo dove le repliche concordano |
| 5 | Retrocompatibilità | `energy_fn=None` → traiettoria identica al ColabDesign originale, stesso seed |
| 6 | **Modalità legacy** | `mode="soft", kappa=0` riproduce i run attuali |
| 7 | Test placeholder | Il test glicina esistente, esteso: con $\kappa>0$ il comportamento cambia nella direzione attesa |

Il **punto 6** merita enfasi: la modalità legacy non è cortesia verso il codice vecchio, è il **braccio di controllo dell'esperimento**. Senza, i miglioramenti osservati in Fase 3 non sono attribuibili.

---

## FASI SUCCESSIVE — da dettagliare dopo il Gate 2

### Fase 3 — Loop di design (~2 settimane, GPU)
- **3.1** Confronto forward: soft (legacy) / straight-through / $\bar{E}$ calibrata.
- **3.2** Bilanciamento sui gradienti: $w_E^{(t)} = \rho\,\|\nabla\mathcal{L}_{AF}\|/\|\nabla E\|$, con $\rho$ unico iperparametro. Motivazione: i due gradienti non competono ad armi pari — quello di AF2 attraversa l'intera rete con dropout attivo (profondo, stocastico, alta varianza), quello dell'energia è una scorciatoia diretta e deterministica. Su centinaia di step il termine a bassa varianza vince sistematicamente.
- **3.3** Lower confidence bound: scansione $\kappa \in \{0, 0.5, 1, 2\}$.
- **3.4** Penalità composizionale: KL fra composizione di $a$ e distribuzione di riferimento. Formulazione dipendente dall'esito di 0.3′.
- **3.5** Matrice di ablazione: seed fissi, stesso target, stesso budget.
- **3.6** Schema di logging per step: $\bar{E}(a)$, $E(\arg\max a)$, $\sigma(a)$, frazione aromatica, `i_con`, `i_ptm`, `plddt`, $\|\nabla\mathcal{L}_{AF}\|$, $\|\nabla E\|$.

### Fase 4 — Validazione (~1–2 settimane)
- **4.1 Controllo su target decoy** — il più diagnostico. Stesso loop, stesso $\bar{E}$, target diverso o hotspot di PA scramblati. Se escono sequenze simili, il termine di energia è target-indipendente. **È l'unico test che copre il regime di estrapolazione**, non coperto dalla validazione BLI.
- **4.2** Retrodizione: il loop produce soluzioni WT-like? Inizializzato vicino al WT, ci resta o scappa verso gli aromatici?
- **4.3** Diversità prima della selezione: clusterizzare i design e scegliere rappresentanti diversi, non i top-K (che sono tipicamente lo stesso modo avversario replicato).
- **4.4** Ricalcolo delle correlazioni Boltz-2 / AF3 / ColabDesign sui design post-fix. Il $-0.14$ osservato potrebbe essere in parte artefatto di aver ordinato design patologici.
- **4.5** Criteri di selezione per il wetlab, **fissati prima** di guardare i risultati: soglie su `i_ptm`, $\sigma(a)$, frazione aromatica, distanza dai cluster di training, quota di diversità.

### Fase 5 — Rimandata
Architettura: feature k-mer gappate posizione-invarianti (inductive bias corretto per libreria random con segnale motif-shaped e register-free; multilinearità preservata, quindi compatibile con 2.1) oppure GP con kernel su stringhe. Guidance NOS sugli hidden state del Pairformer. BoltzGen come generatore di seed diversi.

---

## Budget

| Fase | Durata | Risorsa |
|---|---|---|
| 0 | 2–3 giorni | CPU (+ ~30 min GPU per 0.6 se le traiettorie non sono salvate) |
| 2 | ~2 settimane | CPU |
| 3 | ~2 settimane | GPU: ~7.5 h-GPU per run da 50 traiettorie (9 min/seq, `model_4`, `num_recycles=1`); matrice minima 8 config ≈ 60 h-GPU |
| 4 | 1–2 settimane | GPU (Boltz-2 su AAR) + CPU |

Le Fasi 0 e 2 producono la maggior parte dell'informazione decisionale a costo GPU quasi nullo.

---

## Nota sui gate

I due gate sono formulati in modo che possano **fallire**, e almeno uno ha probabilità non trascurabile di farlo. Il loro significato va deciso adesso, mentre gli esiti non sono noti: è la protezione contro la dinamica in cui ogni risultato negativo diventa un argomento per un altro giro di tuning.
