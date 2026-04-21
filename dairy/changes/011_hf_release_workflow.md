# [011] Repo training su HuggingFace reso privato + workflow di release separato

**Data**: Aprile 2026
**File modificati**: `utils/checkpoint.py` (1 riga), `core/config.py` (1 riga)
**File aggiunti**: `scripts/publish_release.py`
**Tag**: release (nessun tag `[v0.7-XN]` nuovo)

## Perché

Due problemi separati, risolti insieme perché sono di workflow, non di architettura.

**Problema 1 — il repo HF usato da `checkpoint.py` per il training era potenzialmente
pubblico.** `api.create_repo(...)` ha `private=False` di default. Se il repo non
esisteva già (es. prima iterazione di training per una nuova versione) veniva creato
pubblico, e i checkpoint intermedi — con optimizer state, loss history, config interno,
timestamp delle iterazioni — finivano visibili a tutti. Non grave per la sicurezza ma
poco professionale e crea rumore per chi cerca il modello ufficiale: tra `harold-v0.7`
(release) e `harold-v0.7-training` (intermedio) un utente casuale non sa distinguere.

**Problema 2 — nessun processo strutturato per il release pubblico.** `checkpoint.py`
uploada `.pt` + `.safetensors` ogni N iterazioni allo stesso filename, sovrascrivendo
il precedente. Questo è giusto per il workflow di training (resume da Vast.ai, "l'ultimo
è sempre quello buono") ma sbagliato per un release: mancano README, tokenizer, LICENSE,
config.json, generation_config.json, e non c'è versioning atomico ("tutti i file o
nessuno"). Se volevo pubblicare Harold v0.7 dovevo farlo tutto a mano, con rischio alto
di dimenticare pezzi o sbagliare metadata.

## Prima

`utils/checkpoint.py`:

```python
api.create_repo(
    repo_id=HF_REPO_ID, repo_type="model",
    exist_ok=True, token=hf_token,
)
```

`core/config.py`:

```python
HF_REPO_ID  = "JHN-MACHINE/harold-v0.6"
HF_FILENAME = "harold-v0.6.pt"
```

Niente script di release.

## Dopo

`utils/checkpoint.py`:

```python
api.create_repo(
    repo_id=HF_REPO_ID, repo_type="model",
    exist_ok=True, token=hf_token,
    private=True,
)
```

`core/config.py`:

```python
HF_REPO_ID  = "JHN-MACHINE/harold-v0.7-training"
HF_FILENAME = "harold-v0.7-training.pt"
```

Nuovo script `scripts/publish_release.py` che:

1. Legge un checkpoint `.pt` prodotto da `utils/checkpoint.py`
2. Estrae `model_state` e `model_cfg`, scartando tutto il resto (optimizer, scaler, loss history)
3. Casta i pesi a bfloat16 (eccetto tensori piccoli tipo `scale_x`, `mscale`, che restano fp32)
4. Salva come singolo file `.safetensors` (no sharding — 3.2B in bf16 = 6.4 GB, sotto la soglia di 5 GB/shard di HF)
5. Produce `config.json` con `model_type: "harold"`, `architectures: ["Harold"]`
6. Snapshotta il tokenizer da `NousResearch/Llama-2-7b-hf` (solo `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` — niente `.model` SentencePiece)
7. Aggiunge `generation_config.json` con i default del sampler diffusion
8. Copia `README.md`, `LICENSE` (Apache 2.0 per i pesi), `LICENSE_TOKENIZER` (nota LLaMA 2)
9. Scrive `.gitattributes` con le regole LFS
10. Esegue un audit automatico (file mancanti, README con placeholder residui, safetensors troppo grosso)

Lo script **non fa il push automaticamente**. Produce una directory pronta per un `git push` manuale. Questo è voluto: pubblicare un modello è una decisione discreta, non deve succedere "per sbaglio".

## Struttura dei due workflow

```
TRAINING (continuo, automatico)
────────────────────────────────
vast.ai training loop
     │
     │ ogni N iter
     ▼
utils/checkpoint.py::save_checkpoint(push_hf=True)
     │
     ▼
JHN-MACHINE/harold-v0.7-training  (PRIVATO)
     │
     │ file: harold-v0.7-training.pt (~18 GB con optimizer)
     │       harold-v0.7-training.safetensors (~6.4 GB, ridondante ma ok)


RELEASE PUBBLICO (una volta, manuale)
────────────────────────────────────
(decidi che v0.7 è pronto)
     │
     ▼
huggingface-cli download JHN-MACHINE/harold-v0.7-training ...
     │
     ▼
python scripts/publish_release.py --checkpoint ... --output ./release-v0.7
     │
     │ produce una directory con:
     │   - README.md
     │   - config.json
     │   - harold-v0.7.safetensors (~6.4 GB)
     │   - tokenizer.json, tokenizer_config.json, special_tokens_map.json
     │   - generation_config.json
     │   - LICENSE, LICENSE_TOKENIZER
     │   - .gitattributes
     │
     ▼
(inspect manuale + sanity-check caricamento pesi)
     │
     ▼
git push origin main
     │
     ▼
JHN-MACHINE/harold-v0.7  (PUBBLICO — il release vero)
```

## Pro

- **Separazione netta** tra training workflow (continuo, privato) e release workflow (discreto, pubblico). Cambio minimo al codice di training, tutta la logica di release è isolata in un file.
- **Zero rischio di release accidentali** — lo script non fa push automatico.
- **Training repo privato** protegge optimizer state, config interni, loss history. Riduce superficie di leak per chiunque esplori il profilo HF.
- **Release bundle completo** con audit automatico — se manca un file o c'è un placeholder "TBD" nel README, lo script lo segnala prima del push.
- **Gestione licenze esplicita** — Apache 2.0 per i pesi, LLaMA 2 Community License per il tokenizer, file separati. Utente enterprise capisce subito senza leggere il README.
- **Niente `.bin`** — solo `.safetensors`, standard moderno, parsing sicuro (no pickle).
- **Niente wrapper `transformers`** — config dichiara esplicitamente `model_type: "harold"`, utente usa il reference loader, niente `AutoModel` confuso.

## Contro / Trade-off

- **Repo esistenti non diventano privati retroattivamente** con `exist_ok=True, private=True`. Se `JHN-MACHINE/harold-v0.7-training` esiste già pubblico, va reso privato manualmente via UI o `huggingface-cli repo-visibility ... --private`.
- **HF_TOKEN necessita `write access` su repo privati.** Un token read-only fallirà con un errore non ovvio al primo upload dopo la modifica.
- **`publish_release.py` richiede che il checkpoint `.pt` sia localmente accessibile.** Non legge direttamente da HF — è una scelta (più prevedibile, meno magic) ma significa un passaggio manuale di `huggingface-cli download` prima di pubblicare.
- **Non c'è automazione per arXiv, GitHub release, social post.** Lo script si ferma al "directory pronta da pushare". Il resto del release (tweet, paper update, email a design partner) resta manuale. Per v0.7 va bene così, da v1.0 in poi vale automatizzare.

## Risultati attesi

- Dopo la modifica di `checkpoint.py` + cambio `HF_REPO_ID`, il prossimo training run uploada su un repo privato. Verifico con:

  ```bash
  huggingface-cli repo info JHN-MACHINE/harold-v0.7-training
  # → private: true
  ```

- `python scripts/publish_release.py --checkpoint ckpt.pt --output ./release-v0.7 --version v0.7` produce una directory con esattamente 9 file (README, config, safetensors, 3 tokenizer, generation_config, 2 license + .gitattributes).

- L'audit automatico non segnala problemi se il README è stato scritto correttamente (niente "MachineLab", "TBD", "<your-email>").

- `git push` verso `JHN-MACHINE/harold-v0.7` produce un repo HF che assomiglia a `mistralai/Ministral-3-14B-Base-2512` nella struttura (tokenizer incluso, config esplicito, README con benchmark table), pur con l'avvertenza che il modello NON è caricabile via `transformers.AutoModel` ma richiede il reference loader dal repo GitHub.

## Risultati reali

> *Da aggiornare quando viene pubblicato il primo release con questo workflow.*

## Checklist operativa

- [ ] Modificare `utils/checkpoint.py` aggiungendo `private=True` a `create_repo`
- [ ] Aggiornare `HF_REPO_ID` e `HF_FILENAME` in `core/config.py`
- [ ] Verificare che `HF_TOKEN` abbia write access sui repo privati
- [ ] Se `JHN-MACHINE/harold-v0.7-training` esiste già pubblico, renderlo privato
- [ ] Pubblicare `scripts/publish_release.py` nel repo
- [ ] Verificare `README.md` (zero riferimenti "MachineLab" / "axiolab" / email fake)
- [ ] (Al momento del release v0.7) dry-run di `publish_release.py` su un checkpoint v0.6 esistente per sanity-check