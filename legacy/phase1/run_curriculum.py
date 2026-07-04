"""Full BitThought curriculum: STSB→SNLI→C4→Minipile→C4 comp→Minipile comp→(K-predictor).

Phases:
  1-4: Uncompressed training (K = num_thoughts, all vectors)
  5-6: Compression curriculum (scheduler advances ratio, ±noise)
  7:   K-predictor training (encoder/decoder frozen)
  8:   Fine-tune all with K-predictor (via --unfreeze)

Run without args for phases 1-6. Then:
  --train-k-predictor   Phase 7: train K-predictor with frozen model
  --unfreeze            Phase 8: fine-tune all with K-predictor
"""
import sys, datetime, builtins, os, platform
from pathlib import Path

# Speed / memory
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── Platform auto-detect ──────────────────────────────────────────────
def _detect_gpu_vendor():
    rocm_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if "rocm" in p.lower()]
    if rocm_paths:
        return "amd"
    if platform.system() == "Linux":
        try:
            out = os.popen("lspci 2>/dev/null | grep -i 'vga.*amd\\|vga.*radeon' || true").read()
            if out:
                return "amd"
        except Exception:
            pass
    return "cuda"

_vendor = _detect_gpu_vendor()
if _vendor == "amd":
    if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    if "TORCH_BLAS_PREFER_HIPBLASLT" not in os.environ:
        os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
    if "ROCBLAS_LAYER" not in os.environ:
        os.environ["ROCBLAS_LAYER"] = "1"

# ── Log setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log_path = ROOT / "logs" / f"curriculum_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
log_path.parent.mkdir(parents=True, exist_ok=True)
_log_file = open(log_path, "w", encoding="utf-8")
_orig_print = builtins.print
def _log_print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    kwargs.pop("file", None)
    _orig_print(*args, file=_log_file, **kwargs)
    _log_file.flush()
builtins.print = _log_print

import torch
if _vendor == "amd":
    torch.backends.opt_einsum.enabled = True
    torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available() and _vendor == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from bitthought.config import BitThoughtConfig, ModelPresets
from bitthought.model import BitThoughtModel
from bitthought.tokenization import ThoughtTokenizer
from bitthought.data import load_groups, load_pairs, tokenize_dataset
from bitthought.train import train_model, CompressionScheduler

config = ModelPresets.get("medium768")
tok = ThoughtTokenizer.from_preset("llama2")
config.vocab_size = tok.vocab_size
config.max_seq_len = 384
config.num_thoughts = 128
# torch.compile handles memory optimization — gradient checkpointing causes
# graph breaks that fight torch.compile and create GPU idle bursts.
# Disabling both at same time — compile alone is sufficient for 12GB card.
config.use_gradient_checkpointing = False

# K-predictor config
config.use_k_predictor = True
config.k_temperature = 10.0
config.k_noise = 0.1

# Compression curriculum (starts at 0.7 tokens/vector = no real compression)
config.target_ratio_start = 0.7
config.target_ratio_inc = 0.1
config.target_ratio_max = 20.0
config.acc_threshold = 0.95
config.acc_window = 1000

print(f"Logging to {log_path}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ckpt_dir = ROOT / "checkpoints" / "model_sts_768"
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Parse command-line flags
train_k_predictor_flag = "--train-k-predictor" in sys.argv
unfreeze_flag = "--unfreeze" in sys.argv
resume_from = None
for arg in sys.argv[1:]:
    if arg.startswith("--resume="):
        resume_from = arg.split("=", 1)[1]

def save_state(history):
    torch.save({"model_state": model.state_dict(), "config": config,
                "tokenizer_name": "llama2", "history": history},
               ckpt_dir / "latest.pt")

def _compile(model):
    # torch.compile disabled: on ROCm + 12GB VRAM, first-iteration tracing
    # of the 136M param model is slow and may OOM. FP16 handled manually.
    return model

def load_checkpoint(name="latest.pt"):
    p = ckpt_dir / name
    if p.exists():
        with torch.serialization.safe_globals([BitThoughtConfig]):
            ckpt = torch.load(p, map_location="cpu")
        model = BitThoughtModel(config)
        # Handle both "model" (from train_model) and "model_state" (from save_state)
        state_key = "model_state" if "model_state" in ckpt else "model"
        try:
            model.load_state_dict(ckpt[state_key], strict=True)
        except Exception:
            print("[load] strict load failed, trying non-strict")
            model.load_state_dict(ckpt[state_key], strict=False)
        return _compile(model), ckpt.get("history", [])
    return None, []

# ────────────────────────────────────────────────────────────────────
# Phase Selector
# ────────────────────────────────────────────────────────────────────

if train_k_predictor_flag:
    # ── Phase 7: K-predictor training ──────────────────────────
    print(f"\n{'='*60}\nPhase 7: K-predictor training (encoder/decoder frozen)\n{'='*60}")
    model, history = load_checkpoint(resume_from or "minipile_comp.pt")
    if model is None:
        print("No checkpoint found! Run phases 1-6 first.")
        sys.exit(1)

    # Compression scheduler at its final ratio
    sched = CompressionScheduler(config, save_dir=ckpt_dir)
    # Load the final ratio from the checkpoint if available
    last_ratio_path = ckpt_dir / "minipile_comp.pt"
    if last_ratio_path.exists():
        ckpt = torch.load(last_ratio_path, map_location="cpu", weights_only=False)
        if "ratio" in ckpt:
            sched.ratio = ckpt["ratio"]
            print(f"[resume] compression ratio={sched.ratio:.1f}")

    groups = load_groups(ROOT / "datasets" / "minipile.csv", preprocess=True)
    half = len(groups) // 2
    groups = groups[half:]
    tokenized = tokenize_dataset(
        groups, tok.encode, config.max_seq_len,
        cache_path=ckpt_dir / "cache_minipile_comp.pt",
    )
    print(f"Samples: {len(tokenized)}")

    h = train_model(
        model, config, groups=groups,
        tokenizer_encode=tok.encode, pad_token_id=tok.pad_token_id,
        tokenized_data=tokenized,
        device=device,
        epochs=1, batch_size=16, learning_rate=0,  # lr=0 for main params, predictor_lr used for K-predictor
        predictor_lr=5e-3,
        warmup_steps=20,
        log_every=500, sample_every=2000,
        tokenizer_decode=tok.decode, save_every=5000,
        save_path=ckpt_dir / "k_predictor.pt",
        compression_scheduler=sched,
        train_k_predictor=True,
        freeze_encoder_decoder=True,
        k_noise=0.0,
    )
    history = history + h
    print(f"K-predictor training done: final K-acc={h[-1]:.4f}" if history else "")
    save_state(history)
    print("\nPhase 7 complete. Run with --unfreeze for Phase 8 fine-tuning.")

elif unfreeze_flag:
    # ── Phase 8: Fine-tune all with K-predictor ────────────────
    print(f"\n{'='*60}\nPhase 8: Fine-tune all with K-predictor\n{'='*60}")
    model, history = load_checkpoint(resume_from or "k_predictor.pt")
    if model is None:
        print("No checkpoint found! Run --train-k-predictor first.")
        sys.exit(1)

    sched = CompressionScheduler(config, save_dir=ckpt_dir)
    kp_path = ckpt_dir / "k_predictor.pt"
    if kp_path.exists():
        ckpt = torch.load(kp_path, map_location="cpu", weights_only=False)
        if "ratio" in ckpt:
            sched.ratio = ckpt["ratio"]

    # Unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

    groups = load_groups(ROOT / "datasets" / "C4subset-2.csv", preprocess=True)
    tokenized = tokenize_dataset(
        groups, tok.encode, config.max_seq_len,
        cache_path=ckpt_dir / "cache_c4_comp.pt",
    )[:800000]
    print(f"Samples: {len(tokenized)}")

    h = train_model(
        model, config, groups=groups,
        tokenizer_encode=tok.encode, pad_token_id=tok.pad_token_id,
        tokenized_data=tokenized,
        device=device,
        epochs=1, batch_size=16, learning_rate=1e-5,
        predictor_lr=5e-4,
        weight_decay=1e-6,
        length_penalty=0.002,
        warmup_steps=50,
        log_every=500, sample_every=2000,
        tokenizer_decode=tok.decode, save_every=5000,
        save_path=ckpt_dir / "finetuned.pt",
        compression_scheduler=sched,
        train_k_predictor=True,   # keep K-predictor loss active
        freeze_encoder_decoder=False,
        k_noise=0.05,
    )
    history = history + h
    save_state(history)
    print(f"\nFine-tuning complete!")

else:
    # ── Phases 1-6: Full curriculum (continuous) ───────────────
    def load_if_exists(name="latest.pt"):
        return load_checkpoint(name)

    # ── Stage 1: STSB ──────────────────────────────────────
    model, history = load_if_exists("stsb.pt")
    if model is None:
        print(f"\n{'='*60}\nStage 1: STSB (15 epochs, contrastive)\n{'='*60}")
        model = _compile(BitThoughtModel(config))
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {total:,} params")
        groups = load_groups(ROOT / "datasets" / "STSB_train.csv", preprocess=True)
        pairs = load_pairs(ROOT / "datasets" / "STSB_train.csv", preprocess=True)
        tokenized = tokenize_dataset(groups, tok.encode, config.max_seq_len,
                                     cache_path=ckpt_dir / "cache_stsb.pt")
        print(f"Groups: {len(groups)}, Pairs: {len(pairs)}")
        h = train_model(model, config, groups=groups,
                        tokenizer_encode=tok.encode, pad_token_id=tok.pad_token_id,
                        tokenized_data=tokenized, device=device,
                        epochs=15, batch_size=16, learning_rate=5e-4, weight_decay=1e-5,
                        length_penalty=0.005,
                        contrastive_weight=0.5, paired_pairs=pairs,
                        repeat_penalty=0.03, predictor_lr=2e-3,
                        warmup_steps=50, log_every=500, sample_every=500, batch_pause_ms=0,
                        tokenizer_decode=tok.decode, save_every=3000,
                        save_path=ckpt_dir / "stsb.pt")
        history = h
        print(f"STSB done: loss={history[-1]:.4f}")
    else:
        print(f"\nStage 1 STSB: found checkpoint, loss={history[-1]:.4f}, skipping")

    # ── Stage 2: SNLI ──────────────────────────────────────
    model2, h2 = load_if_exists("snli.pt")
    if model2 is None:
        print(f"\n{'='*60}\nStage 2: SNLI (2 epochs)\n{'='*60}")
        groups = load_groups(ROOT / "datasets" / "SNLI_train.csv", preprocess=True)
        tokenized = tokenize_dataset(groups, tok.encode, config.max_seq_len,
                                     cache_path=ckpt_dir / "cache_snli.pt")
        print(f"Groups: {len(groups)}")
        h = train_model(model, config, groups=groups,
                        tokenizer_encode=tok.encode, pad_token_id=tok.pad_token_id,
                        tokenized_data=tokenized, device=device,
                        epochs=2, batch_size=16, learning_rate=3e-4, weight_decay=1e-5,
                        length_penalty=0.005,
                        repeat_penalty=0.03, predictor_lr=1e-3,
                        warmup_steps=100, log_every=1000, sample_every=0, batch_pause_ms=0,
                        tokenizer_decode=tok.decode, save_every=5000,
                        save_path=ckpt_dir / "snli.pt")
        history = history + h
        print(f"SNLI done: loss={h[-1]:.4f}")
    else:
        model = model2
        history = h2
        print(f"Stage 2 SNLI: found checkpoint, loss={history[-1]:.4f}, skipping")

    # ── Stage 3: C4 (100K batches) ─────────────────────────
    if not (ckpt_dir / "c4.pt").exists():
        print(f"\n{'='*60}\nStage 3: C4 (1 epoch, full C4subset-1)\n{'='*60}")
        groups = load_groups(ROOT / "datasets" / "C4subset-1.csv", preprocess=True)
        tokenized = tokenize_dataset(groups, tok.encode, config.max_seq_len,
                                     cache_path=ckpt_dir / "cache_c4.pt")
        print(f"Samples: {len(tokenized)} ({len(tokenized)//16} batches)")
        h = train_model(model, config, groups=groups,
                        tokenizer_encode=tok.encode, pad_token_id=tok.pad_token_id,
                        tokenized_data=tokenized, device=device,
                        epochs=1, batch_size=16, learning_rate=2e-4, weight_decay=1e-5,
                        length_penalty=0.005,
                        repeat_penalty=0.03, predictor_lr=8e-4,
                        warmup_steps=100, log_every=1000, sample_every=0, batch_pause_ms=0,
                        tokenizer_decode=tok.decode, save_every=10000,
                        save_path=ckpt_dir / "c4.pt")
        history = history + h
        print(f"C4 done: loss={h[-1]:.4f}")

    # ── Stage 4: Minipile (200K batches) ───────────────────
    if not (ckpt_dir / "minipile.pt").exists():
        print(f"\n{'='*60}\nStage 4: Minipile (500K batches)\n{'='*60}")
        groups = load_groups(ROOT / "datasets" / "minipile.csv", preprocess=True)
        tokenized = tokenize_dataset(groups, tok.encode, config.max_seq_len,
                                     cache_path=ckpt_dir / "cache_minipile.pt")[:8000000]
        print(f"Samples: {len(tokenized)} ({len(tokenized)//16} batches)")
        h = train_model(model, config, groups=groups,
                        tokenizer_encode=tok.encode, pad_token_id=tok.pad_token_id,
                        tokenized_data=tokenized, device=device,
                        epochs=1, batch_size=16, learning_rate=1.5e-4, weight_decay=1e-5,
                        length_penalty=0.005,
                        repeat_penalty=0.03, predictor_lr=6e-4,
                        warmup_steps=100, log_every=1000, sample_every=0, batch_pause_ms=0,
                        tokenizer_decode=tok.decode, save_every=10000,
                        save_path=ckpt_dir / "minipile.pt")
        history = history + h
        print(f"Minipile done: loss={h[-1]:.4f}")

    # ── Stage 5: C4 compression (scheduler starts at 0.7) ──
    if not (ckpt_dir / "c4_comp.pt").exists():
        print(f"\n{'='*60}\nStage 5: C4 compression (same C4subset-1)\n{'='*60}")
        sched = CompressionScheduler(config, save_dir=ckpt_dir)
        groups = load_groups(ROOT / "datasets" / "C4subset-1.csv", preprocess=True)
        tokenized = tokenize_dataset(groups, tok.encode, config.max_seq_len,
                                     cache_path=ckpt_dir / "cache_c4.pt")
        print(f"Samples: {len(tokenized)} ({len(tokenized)//16} batches)")
        print(f"Compression: ratio={sched.ratio:.1f}, threshold={config.acc_threshold}, "
              f"window={config.acc_window}, noise={config.k_noise}")
        h = train_model(model, config, groups=groups,
                        tokenizer_encode=tok.encode, pad_token_id=tok.pad_token_id,
                        tokenized_data=tokenized, device=device,
                        epochs=1, batch_size=16, learning_rate=6e-5, weight_decay=1e-5,
                        length_penalty=0.005,
                        repeat_penalty=0.03, predictor_lr=4e-4,
                        warmup_steps=50, log_every=1000, sample_every=0, batch_pause_ms=0,
                        tokenizer_decode=tok.decode, save_every=5000,
                        save_path=ckpt_dir / "c4_comp.pt",
                        compression_scheduler=sched, k_noise=config.k_noise)
        history = history + h
        print(f"C4 compression done: loss={h[-1]:.4f}, final ratio={sched.ratio:.1f}")

    # ── Stage 6: Minipile compression (continue scheduler) ─
    if not (ckpt_dir / "minipile_comp.pt").exists():
        print(f"\n{'='*60}\nStage 6: Minipile compression (rest, ratio continues)\n{'='*60}")
        sched = CompressionScheduler(config, save_dir=ckpt_dir)
        sched.ratio = max(config.target_ratio_start, sched.ratio - 2 * config.target_ratio_inc)
        print(f"[compression] stepped ratio back by 2: now {sched.ratio:.1f}")
        groups = load_groups(ROOT / "datasets" / "minipile.csv", preprocess=True)
        tokenized = tokenize_dataset(groups, tok.encode, config.max_seq_len,
                                     cache_path=ckpt_dir / "cache_minipile_comp.pt")
        print(f"Samples: {len(tokenized)} ({len(tokenized)//16} batches)")
        print(f"Continuing from ratio={sched.ratio:.1f}")
        h = train_model(model, config, groups=groups,
                        tokenizer_encode=tok.encode, pad_token_id=tok.pad_token_id,
                        tokenized_data=tokenized, device=device,
                        epochs=1, batch_size=16, learning_rate=5e-5, weight_decay=1e-5,
                        length_penalty=0.005,
                        repeat_penalty=0.03, predictor_lr=3e-4,
                        warmup_steps=50, log_every=1000, sample_every=0, batch_pause_ms=0,
                        tokenizer_decode=tok.decode, save_every=5000,
                        save_path=ckpt_dir / "minipile_comp.pt",
                        compression_scheduler=sched, k_noise=config.k_noise)
        history = history + h
        print(f"Minipile compression done: loss={h[-1]:.4f}, final ratio={sched.ratio:.1f}")

    print(f"\n{'='*60}\nPhases 1-6 complete! Final loss: {history[-1]:.4f}")
    save_state(history)
    print("\nNext steps:")
    print("  python run_curriculum.py --train-k-predictor")
    print("  python run_curriculum.py --train-k-predictor --unfreeze")
