from __future__ import annotations

import random
import time
from functools import partial

import torch
import torch.nn.functional as F
from torch import optim
from torch.amp import GradScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from thought_vectors.data import GroupTextDataset, collate_contrastive_batch, collate_group_batch
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


def contrastive_loss(
    thoughts_a: torch.Tensor,
    thoughts_b: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """InfoNCE loss between paired thought vector sets from the same group.

    Uses the standard SimCLR formulation: within a batch of paired views,
    each view's positive is its paired counterpart; all other views are negatives.
    """
    B = thoughts_a.size(0)
    # Mean-pool thought vectors → single representation per example
    z_a = thoughts_a.mean(dim=1)
    z_b = thoughts_b.mean(dim=1)
    z = torch.cat([z_a, z_b], dim=0)
    z = F.normalize(z, dim=-1)

    sim = z @ z.T  # (2B, 2B)
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = (torch.arange(2 * B, device=sim.device) + B) % (2 * B)

    sim = sim / temperature
    # Remove self-similarity from denominator
    self_mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -float("inf"))

    return F.cross_entropy(sim, labels)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    r"""KL[ N(mu, sigma^2) || N(0, 1) ] averaged over batch and thought slots.

    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) over d_model,
    then mean over (batch × num_thoughts).
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1).mean()


def training_step(
    model: ThoughtVectorModel,
    input_ids: torch.Tensor,
    pad_token_id: int,
    length_penalty: float = 0.01,
    loss_target: float | None = None,
    max_vectors: int | None = None,
    selection_stride: int = 2,
    diversity_weight: float = 0.1,
    input_ids_b: torch.Tensor | None = None,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.5,
    repetition_penalty_weight: float = 0.0,
    kl_beta: float = 0.0,
    noise_std: float = 0.0,
    mixup_alpha: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    padding_mask = input_ids.eq(pad_token_id)

    # VAE sampling: if kl_beta > 0, sample from posterior and compute KL
    if kl_beta > 0:
        z, mu, logvar = model.encoder.encode_with_kl(input_ids, padding_mask)
        thoughts = z
        kl = kl_divergence(mu, logvar)
    else:
        thoughts = model.encoder(input_ids, padding_mask)
        kl = thoughts.new_tensor(0.0)

    # Noise perturbation: add Gaussian noise to thought vectors during training
    if noise_std > 0 and model.training:
        thoughts = thoughts + torch.randn_like(thoughts) * noise_std

    # Mixup: interpolate between thought vectors of different batch items
    if mixup_alpha > 0 and model.training and random.random() < mixup_alpha:
        perm = torch.randperm(thoughts.size(0), device=thoughts.device)
        lam = torch.rand(1, device=thoughts.device).item()
        thoughts = lam * thoughts + (1 - lam) * thoughts[perm]
        # Targets stay the same — decoder learns blended → coherent mapping

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
    stats: dict[str, float] = {
        "reconstruction_loss": float(recon.detach().cpu()),
        "length_penalty": float(penalty.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
        "selected_vectors": float(selected_vector_count),
        "loss_target": float(loss_target) if loss_target is not None else -1.0,
    }

    if kl_beta > 0:
        kl_val = kl
        total = total + kl_beta * kl_val
        stats["kl_loss"] = float(kl_val.detach().cpu())
        stats["kl_beta"] = float(kl_beta)

    if total.isnan().any() or total.isinf().any():
        total = torch.nan_to_num(total, nan=100.0, posinf=100.0, neginf=100.0)

    # Cosine diversity loss: push off-diagonal pairwise cosine similarity toward zero
    if diversity_weight > 0 and thoughts.size(1) > 1:
        normed = F.normalize(thoughts, dim=-1)
        sim = normed @ normed.transpose(-2, -1)
        n = thoughts.size(1)
        eye = torch.eye(n, device=thoughts.device, dtype=torch.bool)
        diversity = (sim[:, ~eye] ** 2).mean()
        total = total + diversity_weight * diversity
        stats["diversity"] = float(diversity.detach().cpu())

    # Count prediction loss: teach the predictor to match the search result
    if loss_target is not None:
        pooled = thoughts.mean(dim=1)
        pred = model.count_predictor(pooled).squeeze(-1)
        target_count = torch.full_like(pred, float(selected_vector_count))
        count_pred_loss = F.mse_loss(pred, target_count)
        total = total + count_pred_loss
        stats["count_pred_loss"] = float(count_pred_loss.detach().cpu())

    # Repetition penalty: penalize predicting the same token as the immediate context
    if repetition_penalty_weight > 0:
        # logits[b, i, :] predicts target[b, i] = input_ids[b, i+1]
        # context_tokens[b, i] = input_ids[b, i] is the token the model just saw
        # If P(logits[b,i] == context_tokens[b,i]) is high, the model is repeating
        context_tokens = input_ids[:, :-1]  # (B, T-1)
        pred_probs = F.softmax(logits, dim=-1)
        rep_probs = pred_probs.gather(-1, context_tokens.unsqueeze(-1)).squeeze(-1)
        rep_loss = rep_probs.mean()
        total = total + repetition_penalty_weight * rep_loss
        stats["repetition_penalty"] = float(rep_loss.detach().cpu())

    # Contrastive loss: align thought vectors for paired group samples
    if input_ids_b is not None and contrastive_weight > 0:
        padding_mask_b = input_ids_b.eq(pad_token_id)
        thoughts_b = model.encoder(input_ids_b, padding_mask_b)
        c_loss = contrastive_loss(thoughts, thoughts_b, temperature=contrastive_temperature)
        total = total + contrastive_weight * c_loss
        stats["contrastive_loss"] = float(c_loss.detach().cpu())

    return total, stats, thoughts


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
    diversity_weight: float = 0.1,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.5,
    repetition_penalty_weight: float = 0.0,
    kl_beta: float = 0.0,
    noise_std: float = 0.0,
    mixup_alpha: float = 0.0,
    log_every: int = 10,
    sample_every_batches: int = 8,
    checkpoint_path: str | None = None,
    tokenizer_decode=None,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    sample_max_generate_length: int = 32,
) -> list[float]:
    """Train thought-vector model on grouped text data and return epoch losses."""
    random.seed(seed)
    torch.manual_seed(seed)

    use_contrastive = contrastive_weight > 0
    dataset = GroupTextDataset(groups)
    collate_fn = partial(
        collate_contrastive_batch if use_contrastive else collate_group_batch,
        tokenizer=tokenizer_encode,
        pad_token_id=pad_token_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    # Deduplicate parameters to avoid double-counting tied weights
    seen_ptrs: set[int] = set()
    params = []
    for p in model.parameters():
        if p.data_ptr() not in seen_ptrs:
            seen_ptrs.add(p.data_ptr())
            params.append(p)
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    model.to(device)

    # Enable TF32 on supported GPUs
    torch.set_float32_matmul_precision("high")

    # torch.compile disabled at this model scale — compilation overhead
    # dwarfs any compute gain.  Re-enable with mode="max-autotune" when
    # moving to d_model >= 768 where matmul dominates.

    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type) if use_amp else None
    history: list[float] = []

    total_steps = max(1, epochs * len(loader))

    # Anneal thought dropout from high (force encoder use) to low (fine-tune) over first half of training
    base_dropout = model.decoder.thought_dropout.p
    max_thought_dropout = 0.5
    thought_dropout_steps = total_steps // 2
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=learning_rate * 0.1
    )
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

            for batch_idx, batch in enumerate(loader, start=1):
                if use_contrastive:
                    input_ids, input_ids_b = batch
                    input_ids = input_ids.to(device)
                    input_ids_b = input_ids_b.to(device)
                else:
                    input_ids = batch.to(device)
                    input_ids_b = None

                # Truncate batch to model's max sequence length to avoid positional encoding mismatches
                max_seq = model.encoder.positional_encoding.pe.size(1)
                if input_ids.size(1) > max_seq:
                    input_ids = input_ids[:, :max_seq]

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

                # Anneal thought dropout: high early to force encoder use, low late for fine-tuning
                if global_step < thought_dropout_steps:
                    progress = global_step / max(1, thought_dropout_steps - 1)
                    model.decoder.thought_dropout.p = (
                        max_thought_dropout + (base_dropout - max_thought_dropout) * progress
                    )

                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                        loss, stats, thoughts = training_step(
                            model,
                            input_ids,
                            pad_token_id=pad_token_id,
                            length_penalty=length_penalty,
                            loss_target=loss_target,
                            max_vectors=max_vectors,
                            selection_stride=selection_stride,
                            diversity_weight=diversity_weight,
                            input_ids_b=input_ids_b,
                            contrastive_weight=contrastive_weight,
                            contrastive_temperature=contrastive_temperature,
                            repetition_penalty_weight=repetition_penalty_weight,
                            kl_beta=kl_beta,
                            noise_std=noise_std,
                            mixup_alpha=mixup_alpha,
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss, stats, thoughts = training_step(
                        model,
                        input_ids,
                        pad_token_id=pad_token_id,
                        length_penalty=length_penalty,
                        loss_target=loss_target,
                        max_vectors=max_vectors,
                        selection_stride=selection_stride,
                        diversity_weight=diversity_weight,
                        input_ids_b=input_ids_b,
                        contrastive_weight=contrastive_weight,
                        contrastive_temperature=contrastive_temperature,
                        repetition_penalty_weight=repetition_penalty_weight,
                        kl_beta=kl_beta,
                        noise_std=noise_std,
                        mixup_alpha=mixup_alpha,
                    )
                    loss.backward()
                    optimizer.step()

                scheduler.step()

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

                should_sample = (
                    batch_idx % max(1, sample_every_batches) == 0
                    and tokenizer_decode is not None
                    and bos_token_id is not None
                    and eos_token_id is not None
                )
                if should_sample:
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
                    sample_vectors = int(stats.get("selected_vectors", 0))
                    print(f"[sample] batch={batch_idx} vectors={sample_vectors} input={input_text!r}")
                    print(f"[sample] batch={batch_idx} vectors={sample_vectors} recon={recon_text!r}")

            epoch_avg = epoch_total / max(1, batches)
            history.append(epoch_avg)
            print(f"[train] epoch {epoch + 1}/{epochs} done: avg_loss={epoch_avg:.4f}")
            if checkpoint_path:
                torch.save(model.state_dict(), f"{checkpoint_path}.ep{epoch + 1}.pt")
    except KeyboardInterrupt:
        if batches > 0:
            epoch_avg = epoch_total / max(1, batches)
            history.append(epoch_avg)
            print(f"[train] interrupted: saved partial epoch avg_loss={epoch_avg:.4f}")
        raise

    return history

