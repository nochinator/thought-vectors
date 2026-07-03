# Paper

Structural draft for the arXiv submission. `main.tex` + `references.bib`.

## Build

```bash
make            # needs a TeX distribution (sudo apt install texlive-full, or texlive-latex-extra + texlive-bibtex-extra)
```

Produces `main.pdf`.

## What's yours to fill in (search for `TODO(nochi)` / `\todo{`)

- **Title** — three candidate titles are commented above `\title`.
- **Author block** — name, "Independent Researcher" affiliation, public email.
- **"Origin of this project"** — your story, your voice.
- **Acknowledgments.**
- Everything red in the PDF (`\todo{...}`) is either yours or awaiting
  FINAL2_12H numbers.

## arXiv submission checklist (first paper)

1. Create an account at arxiv.org with the email you'll publish under.
2. Category: **cs.CL** (primary), cross-list **cs.LG**.
3. Endorsement: arXiv decides at submission time whether you need one. If
   asked, it gives you a code + URL; email authors of closely related papers
   (the LCM / Coconut / SONAR authors we cite are the natural pool) with the
   code, one short paragraph, and the finished PDF attached.
4. Upload the **LaTeX source** (main.tex, references.bib, any figures), not
   just the PDF — arXiv compiles it themselves.
5. License: **CC BY 4.0** is the usual pick if you want maximal reuse;
   arXiv's default non-exclusive license is fine too and keeps more options.
6. Abstract field on the form: plain text, no LaTeX macros beyond $math$.
7. Announcement happens on the next weekday cycle; you can replace (v2, v3…)
   freely afterwards — typo fixes are normal.

## Fallback / complements

- **Zenodo**: link the GitHub repo → automatic DOI per release. Do this
  regardless; it's free citability.
- **TMLR** (OpenReview) later if peer review is ever wanted.
