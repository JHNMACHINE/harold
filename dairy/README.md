# Harold — Training Diary

Diario di bordo pubblico del training di Harold, un diffusion language model ibrido sviluppato indipendentemente.

## Struttura

```
diary/
  README.md          — questo file
  v0.7/
    00_architecture.md   — decisioni architetturali v0.7
    01_setup.md          — setup ambiente e problemi risolti
    02_training_log.md   — log del training in corso (aggiornato manualmente)
```

## Filosofia

Ogni errore risolto è documentato. Ogni decisione architetturale ha una motivazione.
Il diario è pensato per essere utile a chiunque voglia addestrare un diffusion LM da zero.

## Harold in breve

Harold è un diffusion language model ibrido che combina:
- **Jamba** (Mamba3 + Attention) come backbone
- **Flow Matching** con x0-prediction come processo di diffusione
- **DeepSeek MoE** per efficienza computazionale
- **Iterative decoding** per generazione di qualità

Versione corrente: **v0.7** — 3.2B parametri, training in corso.