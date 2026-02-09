#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thought_vectors import SimpleTokenizer, ThoughtDecoder, ThoughtEncoder, ThoughtVectorModel
from thought_vectors.inference import decode_greedy, find_minimum_vectors_for_target


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[ThoughtVectorModel, SimpleTokenizer, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload["config"]

    encoder = ThoughtEncoder(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["heads"],
        num_layers=config["layers"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
        num_thoughts=config["num_thoughts"],
    )
    decoder = ThoughtDecoder(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["heads"],
        num_layers=config["layers"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
    )
    model = ThoughtVectorModel(encoder, decoder)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    tokenizer = SimpleTokenizer.from_token_to_id(payload["token_to_id"])
    return model, tokenizer, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive reconstruction loop for a trained Thought Vector model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--loss-target", type=float, default=0.6)
    parser.add_argument("--max-vectors", type=int, default=None)
    parser.add_argument("--selection-stride", type=int, default=2)
    parser.add_argument("--max-generate-length", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_model(args.checkpoint, device)

    print(f"Loaded model on {device}. num_thoughts={config['num_thoughts']} loss_target={args.loss_target}")
    print("Type text and press enter. Type /quit to exit.\n")

    while True:
        user_text = input("> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"/quit", "/exit"}:
            break

        input_ids = torch.tensor([tokenizer.encode(user_text, add_special_tokens=True)], device=device)

        with torch.no_grad():
            all_thoughts = model.encoder(input_ids)
            chosen_count, losses = find_minimum_vectors_for_target(
                model,
                all_thoughts,
                input_ids,
                loss_target=args.loss_target,
                pad_token_id=tokenizer.pad_token_id,
                stride=args.selection_stride,
                max_vectors=args.max_vectors,
            )
            chosen = all_thoughts[:, :chosen_count, :]
            generated_ids = decode_greedy(
                model,
                chosen,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_length=args.max_generate_length,
            )
            logits = model.decoder(chosen, input_ids[:, :-1])
            recon_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_token_id,
            )

        reconstruction = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        print(f"reconstruction: {reconstruction}")
        print(
            f"vectors: {chosen_count}/{all_thoughts.size(1)} "
            f"loss_target: {args.loss_target:.4f} "
            f"reconstruction_loss: {float(recon_loss):.4f}"
        )
        print(f"loss_curve: {[round(x, 4) for x in losses]}\n")


if __name__ == "__main__":
    main()
