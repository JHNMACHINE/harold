import os
import queue
import threading
import time
import json

# ─────────────────────────────────────────────────────────────────────────────
# AsyncLogger — logging su file in background, zero impatto sul training
# ─────────────────────────────────────────────────────────────────────────────
 
class AsyncLogger:
    """
    Logger asincrono che scrive su file JSONL in un thread di background.
 
    Il training scrive nella queue senza aspettare (non-blocking).
    Il thread di background svuota la queue e appende al file di log.
 
    Formato JSONL: una riga JSON per evento, facile da parsare con pandas:
        import pandas as pd
        df = pd.read_json("training.log", lines=True)
 
    Uso:
        logger = AsyncLogger("checkpoints_v3/training.log")
        logger.log({"iter": 100, "loss": 0.45, "ce": 7.1})
        logger.close()  # sempre chiamare alla fine
    """
 
    def __init__(self, log_path: str, flush_every: int = 10):
        """
        Args:
            log_path:    percorso del file di log (viene appeso, non sovrascritto)
            flush_every: scrivi su disco ogni N eventi (default 10)
                         più alto = meno I/O, più basso = più sicuro contro crash
        """
        self.log_path   = log_path
        self.flush_every = flush_every
        self._queue     = queue.Queue()
        self._stop      = threading.Event()
 
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
 
        # Thread daemon: si chiude automaticamente se il processo principale muore
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
 
    def log(self, data: dict) -> None:
        """
        Accoda un evento da loggare. Non-blocking — ritorna immediatamente.
        Aggiunge automaticamente il timestamp Unix.
        """
        data["timestamp"] = time.time()
        self._queue.put(data)
 
    def _worker(self) -> None:
        """Thread di background: svuota la queue e scrive su file."""
        buffer = []
        while not self._stop.is_set() or not self._queue.empty():
            # Raccoglie eventi dalla queue con timeout
            try:
                item = self._queue.get(timeout=0.5)
                buffer.append(json.dumps(item))
                self._queue.task_done()
            except queue.Empty:
                pass
 
            # Flush su disco ogni flush_every eventi o quando la queue è vuota
            if len(buffer) >= self.flush_every or (buffer and self._queue.empty()):
                with open(self.log_path, "a") as f:
                    f.write("\n".join(buffer) + "\n")
                buffer.clear()
 
        # Flush finale di tutto quello che rimane
        if buffer:
            with open(self.log_path, "a") as f:
                f.write("\n".join(buffer) + "\n")
 
    def close(self) -> None:
        """
        Segnala al thread di fermarsi e aspetta che finisca di scrivere.
        Chiamare sempre alla fine del training.
        """
        self._stop.set()
        self._thread.join(timeout=10)
 
    def __enter__(self):
        return self
 
    def __exit__(self, *args):
        self.close()