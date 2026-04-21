#!/usr/bin/env python3
"""Prepare a Harold public release repository for HuggingFace Hub.

This script is intentionally separate from ``utils/checkpoint.py``:

* ``utils/checkpoint.py`` handles continuous upload of training checkpoints
  (full ``.pt`` with optimizer state) to a *private* training repo
  (e.g. ``JHN-MACHINE/harold-v0.7-training``).

* This script runs ONCE when you decide the model is ready for public
  release. It produces a clean *public* repo bundle with only what a user
  needs for inference: weights (safetensors), config, tokenizer, README,
  licenses.

The script does NOT push to the Hub automatically. It produces a directory
ready for a manual ``git push`` so you can inspect the bundle before release.

Typical workflow
----------------

1. Training runs on Vast.ai, checkpoints uploaded to
   ``JHN-MACHINE/harold-v0.7-training`` (private) by ``checkpoint.py``.

2. When v0.7 is ready for release, download the final ``.pt`` locally::

       huggingface-cli download JHN-MACHINE/harold-v0.7-training \\
           harold-v0.7-training.pt --local-dir ./checkpoints

3. Run this script to produce the release bundle::

       python scripts/publish_release.py \\
           --checkpoint ./checkpoints/harold-v0.7-training.pt \\
           --output ./release-v0.7 \\
           --version v0.7

4. Inspect ``./release-v0.7/``. Run a sanity check load of the safetensors
   to make sure it matches your reference model.

5. Push to public repo::

       cd release-v0.7
       huggingface-cli repo create JHN-MACHINE/harold-v0.7 --type model
       git init && git lfs install
       git remote add origin https://huggingface.co/JHN-MACHINE/harold-v0.7
       git add . && git commit -m "Harold v0.7 initial release"
       git push origin main
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APACHE_LICENSE_NOTICE = """\
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

The full Apache License 2.0 text applies to the Harold model weights distributed
in this repository. The canonical text is available at:

    https://www.apache.org/licenses/LICENSE-2.0.txt

Copyright 2026 Jonathan (JHN-MACHINE).

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

TOKENIZER_LICENSE_NOTICE = """\
Tokenizer License Notice
========================

The tokenizer files distributed in this repository (tokenizer.json,
tokenizer_config.json, special_tokens_map.json) are derived from Meta's
LLaMA 2 tokenizer via the non-gated mirror NousResearch/Llama-2-7b-hf
on the HuggingFace Hub.

These files are subject to the LLaMA 2 Community License Agreement,
available at:

    https://ai.meta.com/llama/license/

The LLaMA 2 Community License imposes restrictions on commercial use
that do NOT apply to the Harold model weights (which are Apache 2.0).

Downstream users combining the Harold weights with this tokenizer must
comply with BOTH licenses:

  * Apache 2.0 — for the model weights
  * LLaMA 2 Community License — for the tokenizer

If your use case is incompatible with the LLaMA 2 Community License,
replace the tokenizer with an Apache/MIT-compatible alternative. A
tokenizer swap requires re-embedding the vocabulary and is not a
drop-in operation for a pre-trained model; a future Harold release
may ship with an Apache-compatible tokenizer from the start.

--
Source repository: https://huggingface.co/NousResearch/Llama-2-7b-hf
LLaMA 2 License:   https://ai.meta.com/llama/license/
"""

GITATTRIBUTES = """\
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
tokenizer.json filter=lfs diff=lfs merge=lfs -text
"""

DEFAULT_GENERATION_CONFIG: Dict[str, Any] = {
    # Sampler hyperparameters for the Harold reference sampler.
    # NOT used by ``transformers.generate()`` — Harold requires ``harold.sampler``.
    "num_denoising_steps": 32,
    "guidance_scale": 1.5,
    "self_conditioning": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "_reference_sampler": "harold.sampler.sample",
    "_note": (
        "Harold is a diffusion language model and is NOT sampled autoregressively. "
        "These values are inputs to the reference sampler, not to HF generate()."
    ),
}

# State dict keys that must NOT appear in a public release — they leak
# information about the training run (loss history, optimizer moments, data
# ordering) and are useless for inference.
TRAINING_ONLY_KEYS = frozenset({
    "optimizer_state",
    "scaler_state",
    "train_losses",
    "val_losses",
    "train_cfg",
    "sft_cfg",
    "stage",
    "iter_num",
    "val_loss",
})


# ---------------------------------------------------------------------------
# Checkpoint parsing — aligned with utils/checkpoint.py conventions
# ---------------------------------------------------------------------------

def load_training_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a checkpoint produced by ``utils/checkpoint.py::save_checkpoint``.

    The expected top-level keys are:
        ``iter_num``, ``model_state``, ``val_loss``, ``model_cfg``,
        ``train_losses``, ``val_losses``, and optionally
        ``optimizer_state``, ``scaler_state``, ``train_cfg`` | ``sft_cfg``, ``stage``.

    We only need ``model_state`` and ``model_cfg`` for a public release.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"Unexpected checkpoint format: got {type(ckpt).__name__}, expected dict. "
            "This script expects checkpoints from utils/checkpoint.py::save_checkpoint."
        )

    if "model_state" not in ckpt:
        raise RuntimeError(
            "Checkpoint missing 'model_state' key. "
            f"Available keys: {sorted(ckpt.keys())}. "
            "This script expects checkpoints from utils/checkpoint.py."
        )

    if "model_cfg" not in ckpt:
        raise RuntimeError(
            "Checkpoint missing 'model_cfg' key. "
            f"Available keys: {sorted(ckpt.keys())}."
        )

    # Audit what's in the checkpoint and warn if anything unexpected.
    print(f"  checkpoint keys: {sorted(ckpt.keys())}")
    print(f"  iter_num: {ckpt.get('iter_num', '<unset>')}")
    print(f"  val_loss: {ckpt.get('val_loss', '<unset>')}")
    stage = ckpt.get("stage")
    if stage is not None:
        print(f"  stage:    {stage} (SFT checkpoint — confirm this is what you want to release)")
    return ckpt


def extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Extract the model state dict, stripping DDP/FSDP/compile prefixes."""
    sd = ckpt["model_state"]
    cleaned: Dict[str, torch.Tensor] = {}
    n_stripped = 0

    for key, tensor in sd.items():
        new_key = key
        for prefix in ("module.", "_orig_mod.", "_fsdp_wrapped_module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                n_stripped += 1
        if not isinstance(tensor, torch.Tensor):
            print(f"  WARNING: skipping non-tensor '{key}' ({type(tensor).__name__})")
            continue
        cleaned[new_key] = tensor.detach().contiguous()

    if n_stripped > 0:
        print(f"  stripped DDP/FSDP/compile prefix from {n_stripped} keys")
    print(f"  extracted {len(cleaned)} tensors")

    total_params = sum(t.numel() for t in cleaned.values() if t.dtype.is_floating_point)
    print(f"  total float params: {total_params / 1e9:.3f}B")

    # Safety: make sure no training-only keys slipped in.
    leaked = sorted(set(cleaned) & TRAINING_ONLY_KEYS)
    if leaked:
        raise RuntimeError(
            f"Training-only keys detected in model_state: {leaked}. "
            "This should not happen — state dict is malformed."
        )

    return cleaned


def extract_config(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ``model_cfg`` to a plain dict."""
    cfg = ckpt["model_cfg"]
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    raise RuntimeError(
        f"Cannot convert model_cfg to dict (type: {type(cfg).__name__})"
    )


def sanitize_config_for_json(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Strip non-JSON-serializable fields. Adds ``model_type`` and ``architectures``."""
    out: Dict[str, Any] = {}
    skipped = []

    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            kept = [x for x in v if isinstance(x, (int, float, bool, str))]
            if len(kept) == len(v):
                out[k] = kept
            else:
                skipped.append(f"{k} (mixed types in list)")
        elif isinstance(v, torch.dtype):
            out[k] = str(v).replace("torch.", "")
        elif isinstance(v, Path):
            out[k] = str(v)
        else:
            skipped.append(f"{k} ({type(v).__name__})")

    if skipped:
        print(f"  [config] skipped non-serializable fields: {', '.join(skipped)}")

    # Mark this as a custom architecture not available in `transformers`.
    out["model_type"] = "harold"
    out["architectures"] = ["Harold"]
    return out


# ---------------------------------------------------------------------------
# Weight processing
# ---------------------------------------------------------------------------

def cast_to_bf16(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Cast fp32/fp16 parameters to bfloat16 for the release.

    Small tensors (≤16 elements) are left untouched — those are typically
    scalar buffers that need fp32 precision (scales, YaRN mscale, etc.).
    """
    out: Dict[str, torch.Tensor] = {}
    n_cast, n_skip = 0, 0
    for k, v in state_dict.items():
        if v.dtype in (torch.float16, torch.float32) and v.numel() > 16:
            out[k] = v.to(torch.bfloat16)
            n_cast += 1
        else:
            out[k] = v
            n_skip += 1
    print(f"  [weights] cast to bf16: {n_cast} tensors; preserved: {n_skip} small/non-float tensors")
    return out


def save_weights_safetensors(
    state_dict: Dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    """Save as a single safetensors file.

    For Harold v0.7 (3.2B params in bf16 ≈ 6.4 GB) a single file is fine.
    If we ever exceed ~5 GB per file, we can add sharding later — HuggingFace
    tools handle both transparently.
    """
    save_file(state_dict, str(output_path))
    size_gb = output_path.stat().st_size / 1e9
    print(f"  wrote {output_path.name} ({size_gb:.2f} GB)")


# ---------------------------------------------------------------------------
# Tokenizer snapshot
# ---------------------------------------------------------------------------

def snapshot_tokenizer(source_id: str, output_dir: Path) -> None:
    """Save a self-contained snapshot of the LLaMA-2 tokenizer.

    Produces ``tokenizer.json``, ``tokenizer_config.json``, ``special_tokens_map.json``.
    Does NOT save ``tokenizer.model`` (the raw SentencePiece binary) — the fast
    tokenizer.json is self-contained.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  ERROR: transformers not installed")
        print("         pip install transformers")
        raise

    print(f"  loading {source_id} ...")
    tok = AutoTokenizer.from_pretrained(source_id, use_fast=True)

    # save_pretrained writes several files; we prune afterwards.
    tok.save_pretrained(str(output_dir))

    keep = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    for candidate in output_dir.iterdir():
        if candidate.is_dir() or candidate.name in keep:
            continue
        # Only prune tokenizer-looking files; leave README, LICENSE, etc. alone.
        if any(candidate.name.startswith(p) for p in ("tokenizer", "special", "added")):
            candidate.unlink()
            print(f"  pruned {candidate.name}")

    # Sanity check.
    missing = [fname for fname in keep if not (output_dir / fname).exists()]
    if missing:
        raise RuntimeError(f"Expected tokenizer files not written: {missing}")


# ---------------------------------------------------------------------------
# Pre-release audit — catch mistakes BEFORE pushing
# ---------------------------------------------------------------------------

def audit_release(output_dir: Path) -> None:
    """Run final sanity checks on the release bundle."""
    print("\nPre-release audit:")
    problems = []

    required = [
        "README.md",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "LICENSE",
        "LICENSE_TOKENIZER",
        ".gitattributes",
    ]
    for name in required:
        path = output_dir / name
        if not path.exists():
            problems.append(f"MISSING: {name}")

    # At least one safetensors file.
    safetensors_files = list(output_dir.glob("*.safetensors"))
    if not safetensors_files:
        problems.append("MISSING: no *.safetensors file found")
    else:
        for sf in safetensors_files:
            size_gb = sf.stat().st_size / 1e9
            if size_gb > 50.0:
                problems.append(
                    f"WARNING: {sf.name} is {size_gb:.1f} GB — consider sharding"
                )

    # README must not have leaked MachineLab/axiolab placeholders.
    readme = output_dir / "README.md"
    if readme.exists():
        content = readme.read_text()
        for leak in ("machinelab", "axiolab", "TBD", "<your-"):
            if leak.lower() in content.lower():
                problems.append(
                    f"WARNING: README contains '{leak}' — check if this is intentional"
                )

    if problems:
        print("  ISSUES FOUND:")
        for p in problems:
            print(f"    - {p}")
        print()
        print("  Fix these before pushing to avoid shipping a broken release.")
    else:
        print("  all checks passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Harold public release bundle from a training checkpoint. "
            "Produces a directory ready for manual 'git push' to a public HF repo."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the training checkpoint (.pt from utils/checkpoint.py).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for the release bundle (will be created).",
    )
    parser.add_argument(
        "--version", type=str, required=True,
        help="Release version tag used in filenames, e.g. 'v0.7'.",
    )
    parser.add_argument(
        "--readme", type=Path, default=None,
        help=(
            "Path to README.md to include in the release. "
            "Defaults to README.md next to this script."
        ),
    )
    parser.add_argument(
        "--tokenizer-source", type=str, default="JHN-MACHINE/harold",
        help="HuggingFace repo ID to snapshot the tokenizer from.",
    )
    parser.add_argument(
        "--skip-bf16-cast", action="store_true",
        help="Do not cast fp32 parameters to bf16 (keep original precision).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite output directory if it already has files.",
    )
    args = parser.parse_args()

    # --- pre-checks
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()) and not args.force:
        print(
            f"ERROR: output directory {args.output} is not empty. "
            "Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    print(f"Release version: {args.version}")
    print(f"Output:          {args.output.resolve()}")
    print()

    # --- 1. load and parse checkpoint
    print("[1/6] Loading and parsing checkpoint")
    ckpt = load_training_checkpoint(args.checkpoint)

    # --- 2. write config.json
    print("\n[2/6] Writing config.json")
    cfg_raw = extract_config(ckpt)
    cfg_clean = sanitize_config_for_json(cfg_raw)
    config_path = args.output / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg_clean, f, indent=2, sort_keys=True)
    print(f"  wrote {config_path.name} ({len(cfg_clean)} fields)")

    # --- 3. write weights as safetensors
    print("\n[3/6] Extracting and saving weights")
    state_dict = extract_state_dict(ckpt)
    if not args.skip_bf16_cast:
        state_dict = cast_to_bf16(state_dict)
    weights_path = args.output / f"harold-{args.version}.safetensors"
    save_weights_safetensors(state_dict, weights_path)

    # Free memory — checkpoint can be several GB.
    del ckpt, state_dict

    # --- 4. generation_config.json
    print("\n[4/6] Writing generation_config.json")
    gen_cfg_path = args.output / "generation_config.json"
    with open(gen_cfg_path, "w") as f:
        json.dump(DEFAULT_GENERATION_CONFIG, f, indent=2)
    print(f"  wrote {gen_cfg_path.name}")

    # --- 5. tokenizer
    print("\n[5/6] Snapshotting tokenizer")
    snapshot_tokenizer(args.tokenizer_source, args.output)

    # --- 6. README + licenses + .gitattributes
    print("\n[6/6] Writing README, licenses, .gitattributes")

    readme_src = args.readme or (Path(__file__).parent.parent / "README.md")
    if readme_src.exists():
        shutil.copy(readme_src, args.output / "README.md")
        print(f"  copied README.md from {readme_src}")
    else:
        print(f"  WARNING: README.md not found at {readme_src}")
        print("           Place README.md next to this script or pass --readme")

    (args.output / "LICENSE").write_text(APACHE_LICENSE_NOTICE)
    print("  wrote LICENSE")

    (args.output / "LICENSE_TOKENIZER").write_text(TOKENIZER_LICENSE_NOTICE)
    print("  wrote LICENSE_TOKENIZER")

    (args.output / ".gitattributes").write_text(GITATTRIBUTES)
    print("  wrote .gitattributes")

    # --- audit + summary
    audit_release(args.output)

    print("\n" + "=" * 72)
    print(f"Release bundle: {args.output.resolve()}")
    print("=" * 72)
    for item in sorted(args.output.iterdir()):
        size = item.stat().st_size
        if size > 1e9:
            size_str = f"{size / 1e9:.2f} GB"
        elif size > 1e6:
            size_str = f"{size / 1e6:.1f} MB"
        else:
            size_str = f"{size / 1e3:.1f} kB"
        print(f"  {item.name:<40s}  {size_str:>10s}")

    print()
    print("Next steps:")
    print()
    print("  # 1. Inspect the bundle manually, especially README.md and config.json")
    print(f"  ls -la {args.output}")
    print(f"  cat {args.output}/README.md | head -50")
    print(f"  cat {args.output}/config.json")
    print()
    print("  # 2. Sanity-check the weights load into a fresh Harold instance")
    print("  python -c 'from harold import Harold, ModelConfig; import json; \\")
    print(f"       from safetensors.torch import load_file; \\")
    print(f"       cfg = ModelConfig(**json.load(open(\"{args.output}/config.json\"))); \\")
    print("       m = Harold(cfg); \\")
    print(f"       m.load_state_dict(load_file(\"{args.output}/harold-{args.version}.safetensors\")); \\")
    print("       print(\"OK\")'")
    print()
    print("  # 3. Create the public HF repo and push")
    print(f"  cd {args.output}")
    print(f"  huggingface-cli repo create JHN-MACHINE/harold-{args.version} --type model")
    print("  git init && git lfs install")
    print(f"  git remote add origin https://huggingface.co/JHN-MACHINE/harold-{args.version}")
    print("  git add .")
    print(f'  git commit -m "Harold {args.version} initial release"')
    print("  git push origin main")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())