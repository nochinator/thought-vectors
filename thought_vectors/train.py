from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from thought_vectors.data import GroupTextDataset, collate_group_batch
from thought_vectors.inference import decode_greedy, find_minimum_vectors_for_target
from thought_vectors.model import ThoughtVectorModel


def _count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def compute_dynamic_loss_target(
    *,
    step_index: int,
    total_steps: int,
    mean_sequence_len: float,
    max_sequence_len: int,
    start_target: float = 1.8,
    end_target: float = 0.45,
    length_weight: float = 0.65,
    noise_std: float = 0.07,
    extreme_prob: float = 0.12,
) -> float:
    """Curriculum target: high target early, stricter later, length-aware, with random variance."""
    progress = 0.0 if total_steps <= 1 else step_index / float(total_steps - 1)
    base_target = start_target + (end_target - start_target) * progress

    normalized_len = min(1.0, max(0.0, mean_sequence_len / max(1.0, float(max_sequence_len))))
    length_adjustment = length_weight * normalized_len

    target = base_target + length_adjustment

    # random variance around base behavior
    target += random.gauss(0.0, noise_std)

    # occasional extreme compression demand
    if random.random() < extreme_prob:
        target *= random.uniform(0.45, 0.8)

    return max(0.05, target)


def training_step(
    model: ThoughtVectorModel,
    input_ids: torch.Tensor,
    pad_token_id: int,
    length_penalty: float = 0.01,
    loss_target: float | None = None,
    max_vectors: int | None = None,
    selection_stride: int = 2,
) -> tuple[torch.Tensor, dict[str, float]]:
    padding_mask = input_ids.eq(pad_token_id)

    thoughts = model.encoder(input_ids, padding_mask)

    selected_vectors = thoughts
    selected_vector_count = thoughts.size(1)

    if loss_target is not None:
        with torch.no_grad():
            selected_vector_count, _ = find_minimum_vectors_for_target(
                model,
                thoughts=thoughts,
                input_ids=input_ids,
                loss_target=loss_target,
                pad_token_id=pad_token_id,
                stride=selection_stride,
                max_vectors=max_vectors,
            )
        selected_vectors = thoughts[:, :selected_vector_count, :]

    logits = model.decoder(selected_vectors, input_ids[:, :-1], padding_mask[:, :-1])

    target = input_ids[:, 1:]
    recon = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=pad_token_id,
    )
    penalty = recon.new_tensor(length_penalty * selected_vector_count)
    total = recon + penalty
    stats = {
        "reconstruction_loss": float(recon.detach().cpu()),
        "length_penalty": float(penalty.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
        "selected_vectors": float(selected_vector_count),
        "loss_target": float(loss_target) if loss_target is not None else -1.0,
    }
    return total, stats


def train_model(
    model: ThoughtVectorModel,
    groups: list[list[str]],
    tokenizer_encode,
    pad_token_id: int,
    *,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    length_penalty: float = 0.01,
    shuffle: bool = True,
    seed: int = 0,
    use_dynamic_loss_target: bool = True,
    target_start: float = 1.8,
    target_end: float = 0.45,
    target_length_weight: float = 0.65,
    target_noise_std: float = 0.07,
    target_extreme_prob: float = 0.12,
    max_vectors: int | None = None,
    selection_stride: int = 2,
    log_every: int = 10,
    sample_every_batches: int = 8,
    tokenizer_decode=None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    sample_max_generate_length: int = 32,
) -> list[float]:
    """Train thought-vector model on grouped text data and return epoch losses."""
    random.seed(seed)
    torch.manual_seed(seed)

    dataset = GroupTextDataset(groups)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_group_batch(batch, tokenizer_encode, pad_token_id),
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    history: list[float] = []

    total_steps = max(1, epochs * len(loader))
    global_step = 0

    encoder_params = _count_trainable_parameters(model.encoder)
    decoder_params = _count_trainable_parameters(model.decoder)
    print(
        "[train] parameters "
        f"encoder={encoder_params:,} decoder={decoder_params:,} total={(encoder_params + decoder_params):,}"
    )

    batches = 0
    epoch_total = 0.0
    try:
        for epoch in range(epochs):
            model.train()
            epoch_total = 0.0
            batches = 0
            t0 = time.time()

            print(f"[train] epoch {epoch + 1}/{epochs} starting...")

            for batch_idx, input_ids in enumerate(loader, start=1):
                input_ids = input_ids.to(device)

                loss_target: float | None = None
                if use_dynamic_loss_target:
                    mean_seq_len = float((~input_ids.eq(pad_token_id)).sum(dim=1).float().mean().item())
                    loss_target = compute_dynamic_loss_target(
                        step_index=global_step,
                        total_steps=total_steps,
                        mean_sequence_len=mean_seq_len,
                        max_sequence_len=input_ids.size(1),
                        start_target=target_start,
                        end_target=target_end,
                        length_weight=target_length_weight,
                        noise_std=target_noise_std,
                        extreme_prob=target_extreme_prob,
                    )

                optimizer.zero_grad(set_to_none=True)
                loss, stats = training_step(
                    model,
                    input_ids,
                    pad_token_id=pad_token_id,
                    length_penalty=length_penalty,
                    loss_target=loss_target,
                    max_vectors=max_vectors,
                    selection_stride=selection_stride,
                )
                loss.backward()
                optimizer.step()

                epoch_total += float(loss.detach().cpu())
                batches += 1
                global_step += 1

                if (batch_idx % max(1, log_every) == 0) or (batch_idx == len(loader)):
                    elapsed = time.time() - t0
                    avg = epoch_total / max(1, batches)
                    print(
                        "[train] "
                        f"epoch={epoch + 1}/{epochs} "
                        f"batch={batch_idx}/{len(loader)} "
                        f"loss={stats['total_loss']:.4f} "
                        f"recon={stats['reconstruction_loss']:.4f} "
                        f"vectors={int(stats['selected_vectors'])} "
                        f"target={stats['loss_target']:.4f} "
                        f"epoch_avg={avg:.4f} "
                        f"elapsed={elapsed:.1f}s"
                    )

                if (batch_idx % max(1, sample_every_batches) == 0) and tokenizer_decode is not None and bos_token_id is not None and eos_token_id is not None:
                    with torch.no_grad():
                        sample_count = int(stats["selected_vectors"])
                        sample_vectors = thoughts[:1, :sample_count, :]
                        sample_generated = decode_greedy(
                            model,
                            sample_vectors,
                            bos_token_id=bos_token_id,
                            eos_token_id=eos_token_id,
                            max_length=sample_max_generate_length,
                        )
                    input_text = tokenizer_decode(input_ids[0].detach().cpu().tolist())
                    recon_text = tokenizer_decode(sample_generated[0].detach().cpu().tolist())
                    print(f"[sample] batch={batch_idx} input={input_text!r}")
                    print(f"[sample] batch={batch_idx} recon={recon_text!r}")

            epoch_avg = epoch_total / max(1, batches)
            history.append(epoch_avg)
            print(f"[train] epoch {epoch + 1}/{epochs} done: avg_loss={epoch_avg:.4f}")
    except KeyboardInterrupt:
        if batches > 0:
            epoch_avg = epoch_total / max(1, batches)
            history.append(epoch_avg)
            print(f"[train] interrupted: saved partial epoch avg_loss={epoch_avg:.4f}")
        raise

    return history
