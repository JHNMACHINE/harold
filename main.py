"""
Harold v0.6 — main.py
======================
Entry point unificato per pretraining e SFT.

Avvio single-GPU:
    torchrun --nproc_per_node=1 main.py --mode pretrain
    torchrun --nproc_per_node=1 main.py --mode sft
    torchrun --nproc_per_node=1 main.py --mode full

Avvio multi-GPU (es. 4 GPU):
    torchrun --nproc_per_node=4 main.py --mode full

Modalità:
  pretrain — solo pretraining (20k iter default)
  sft      — solo SFT, riprende dal best checkpoint pretraining
  full     — pretrain → sft in sequenza automatica
"""

import argparse
import os
import sys
import warnings

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/.torch_cache")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harold v0.6 — Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  torchrun --nproc_per_node=1 main.py --mode pretrain
  torchrun --nproc_per_node=1 main.py --mode sft
  torchrun --nproc_per_node=1 main.py --mode full
  torchrun --nproc_per_node=4 main.py --mode full --compile_mode reduce-overhead
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["pretrain", "sft", "full"],
        default="full",
        help="Modalità di training (default: full)",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=None,
        help="Override max iterazioni pretraining",
    )
    parser.add_argument(
        "--sft_max_iters",
        type=int,
        default=None,
        help="Override max iterazioni SFT",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default=None,
        help="Override torch.compile mode (es. reduce-overhead, max-autotune)",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disabilita torch.compile",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override directory checkpoint pretraining",
    )
    parser.add_argument(
        "--sft_checkpoint_dir",
        type=str,
        default=None,
        help="Override directory checkpoint SFT",
    )
    return parser.parse_args()


def run_pretrain(args: argparse.Namespace) -> dict:
    from core.config import get_model_config, get_train_config
    from training.train import run_training
    from utils.ddp import is_main

    model_cfg = get_model_config()
    train_cfg = get_train_config()

    # Override da argomenti CLI
    if args.max_iters is not None:
        train_cfg.max_iters = args.max_iters
    if args.compile_mode is not None:
        train_cfg.compile_mode = args.compile_mode
    if args.no_compile:
        train_cfg.use_compile = False
    if args.checkpoint_dir is not None:
        train_cfg.checkpoint_dir = args.checkpoint_dir

    if is_main():
        print("=" * 60)
        print(f"  PRETRAINING — {train_cfg.max_iters} iterazioni")
        print(f"  Checkpoint: {train_cfg.checkpoint_dir}")
        print("=" * 60)

    results = run_training(model_cfg, train_cfg)

    if is_main():
        print(f"\nPretraining completato.")
        print(f"  Best val loss: {results['best_val_loss']:.4f}")
        print(f"  Checkpoint:    {results['checkpoint_path']}")

    return results


def run_sft(args: argparse.Namespace, pretrain_ckpt: str | None = None) -> dict:
    from core.config import SFTConfig
    from training.train_sft import run_sft
    from utils.ddp import is_main

    sft_cfg = SFTConfig()

    # Se viene da una run full, usa il checkpoint del pretraining
    if pretrain_ckpt is not None:
        sft_cfg.pretrain_ckpt = pretrain_ckpt

    # Override da argomenti CLI
    if args.sft_max_iters is not None:
        sft_cfg.max_iters = args.sft_max_iters
    if args.sft_checkpoint_dir is not None:
        sft_cfg.checkpoint_dir = args.sft_checkpoint_dir

    if is_main():
        print("=" * 60)
        print(f"  SFT — {sft_cfg.max_iters} iterazioni")
        print(f"  Pretrain ckpt: {sft_cfg.pretrain_ckpt}")
        print(f"  Checkpoint:    {sft_cfg.checkpoint_dir}")
        print("=" * 60)

    results = run_sft(sft_cfg)

    if is_main():
        print(f"\nSFT completato.")
        print(f"  Best val loss: {results['best_val_loss']:.4f}")
        print(f"  Checkpoint:    {results['checkpoint_path']}")

    return results


def main() -> None:
    warnings.filterwarnings("ignore")
    args = parse_args()

    if args.mode == "pretrain":
        run_pretrain(args)

    elif args.mode == "sft":
        run_sft(args)

    elif args.mode == "full":
        from utils.ddp import is_main

        # 1. Pretraining
        pretrain_results = run_pretrain(args)

        # 2. SFT — usa il best checkpoint del pretraining
        pretrain_ckpt = pretrain_results.get("checkpoint_path")

        if is_main():
            print("\n" + "=" * 60)
            print("  Pretraining completato — avvio SFT")
            print("=" * 60 + "\n")

        run_sft(args, pretrain_ckpt=pretrain_ckpt)

    else:
        print(f"Modalità non riconosciuta: {args.mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()