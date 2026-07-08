# Paper

`main.tex` + `references.bib` → `main.pdf` (12 pages).
No open TODOs — all citations verified against the actual papers 2026-07-04.

## Build

```bash
make            # needs a TeX distribution (sudo apt install texlive-full, or texlive-latex-extra + texlive-bibtex-extra + pdflatex/bibtex)
```

Produces `main.pdf`. The architecture figure is TikZ, compiled in-document —
no external assets.

## Publication plan (pseudonymous)

arXiv is out: its identity policy requires real names, and submitting under
a pseudonym risks the account. TMLR/OpenReview have the same requirement.
The paper publishes as:

1. **Zenodo** (primary) — Publication → Preprint record holding the PDF;
   mints the citable DOI. Author name is not identity-verified, so "nochi"
   works. Paper license: **CC BY 4.0**. New paper versions go to the same
   record (DOI persists, versioned).
2. **Hugging Face** — model repo `nochi/thought-vectors` (checkpoints +
   tokenizer + model card linking the DOI) and a free-CPU Gradio Space
   running the chat demo.
3. **GitHub** — the repo stays the project home; Release v1.0.0 carries the
   checkpoints; README gets the Zenodo DOI badge.

Staged upload materials and step-by-step instructions: `publish/` at the
repo root (untracked; delete after publishing).

## Pre-publication checklist

- [x] GitHub Release v1.0.0 with the flagship checkpoints; repo URL in the
      paper resolves.
- [ ] Final read-through of the whole PDF by the author.
- [ ] Zenodo record published; DOI badge added to the repo README and the
      HF model card.
- [ ] HF model repo + Space live.
