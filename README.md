

# Sign Language Recognition (PHOENIX-2014-T)

Pipeline di Deep Learning per il riconoscimento della lingua dei segni (CSLR). Il progetto implementa un'architettura **PoseNetworkCTC** (basata su DenseNet121, TCN, Attention e BiLSTM) con l'integrazione di tecniche di ottimizzazione avanzate e data augmentation basata su **TSSI** (Time-Shifted Skeleton Interpolation).

##  Setup e Installazione

Il progetto utilizza **[uv](https://github.com/astral-sh/uv)** per una gestione deterministica delle dipendenze.

1. **Clona la repository**:
```bash
git clone [https://github.com/Eldoardo26/Sign-Language-Recognition.git]
cd Sign-Language-Recognition

```


2. **Sincronizza l'ambiente**:
```bash
uv sync

```


*(Questo comando crea automaticamente l'ambiente virtuale con tutte le versioni esatte delle librerie necessarie).*

##  Architettura della Pipeline

Il progetto è modulare. Il notebook `csrl/main.ipynb` funge da orchestratore per i seguenti moduli Python:

| Modulo | Contenuto |
| --- | --- |
| `config.py` | Gestione percorsi e iperparametri di training |
| `skeleton.py` | Logica DFS order e generazione TSSI |
| `model.py` | Definizione PoseNetworkCTC (TCN + Attention + BiLSTM) |
| `decoding.py` | Beam search, greedy decoding e bigram integration |
| `training.py` | Loop di training (two-phase), gestione checkpoint |
| `ensemble.py` | Ensemble decode per ottimizzazione WER |

##  Esecuzione esperimenti

L'intero flusso di lavoro è gestito tramite il notebook Jupyter `csrl/main.ipynb`.

1. **Apri VS Code** nella cartella del progetto.
2. **Seleziona il Kernel**: Apri `csrl/main.ipynb`, clicca su "Select Kernel" (in alto a destra) e seleziona l'ambiente `Python (UV-Env)` che punta alla cartella `.venv` locale.
3. **Esecuzione**: Esegui le celle in ordine. Il notebook importa automaticamente tutti i moduli dalla cartella `csrl/`.

*Nota: Se aggiungi nuove dipendenze, utilizza `uv add <nome_libreria>` nel terminale e riavvia il kernel del notebook.*

##  Note tecniche

* **Merge sul Token Unico**: La pipeline integra i feature estratti dai frame e i dati temporali attraverso un meccanismo di *token merging* che riduce la dimensionalità preservando le informazioni spazio-temporali necessarie per la sequenza di gloss.
* **Ottimizzazione**: Il modello integra tecniche di `CTCLossWithEntropy` e `Ensemble Decoding` per minimizzare il Word Error Rate (WER).

##  Checkpoint

I pesi del modello (`.pth`) non sono inclusi nel repository.

* **Download**: Scarica i checkpoint(https://unibari-my.sharepoint.com/:f:/g/personal/e_bufi5_studenti_uniba_it/IgCVGx3DNvfIRImcOzTHF0_KAYepcnkJaVrHIrufkML-awA?e=1uk882).
* **Posizionamento**: Salva i file nella cartella `results/`.


