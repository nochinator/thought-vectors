# Round B (pre-registered): the monolithic baseline

**Question.** At matched data, tokenizer, and wall-clock compute on the same
GPU, does a plain token-level LM match the codec+thinker system on the same
behavioral evals? The paper currently shows the architecture *works*; this
round determines whether it *beats the obvious alternative at equal cost* —
the first question any strong reviewer will ask.

**Standing rules carry over unchanged:** all comparisons at equal wall-clock
on this GPU; no metric win believed without a transcript audit and a live
chat probe; chat-probe the winner ~2h into any long run before letting it
continue (metrics have lied before); everything — including losing arms —
goes in RESEARCH_LOG.md.

**Fairness guard rail (both directions).** The baseline must be a steelman,
not a strawman: it gets a real architecture/recipe search (B1–B2 below), the
same data view, the same tokenizer. But the tuning budget is fixed *here, in
advance* (~8 short arms per size); neither side gets post-hoc extra tuning
after the long runs. Results are reported in the paper whichever way they
fall.

---

## B0 — plumbing (no GPU conclusions)

- `src/thoughtvec/lm.py`: decoder-only transformer, pre-norm, tied
  embeddings, same spm16k tokenizer. Keep it textbook — the claim is about
  the *paradigm*, so the baseline should be the standard recipe, not exotic.
- Data view: same dialogue shards (`data/` mix used by FINAL_12H), turns
  concatenated with role-separator tokens, **loss on every non-first turn**
  (exactly the thinker's targeting rule). Context window 512 tokens.
- `tv-train-lm` entry point; extend `scripts/chat_compare.py` for
  side-by-side chat.
- Smoke run (m0-style, ~10 min) to confirm loss falls and samples are text.

## B1 — shape bracket (45-min arms, equal wall-clock)

Two parameter classes, tied embeddings included in the count:

| Class | Target params | Candidate shapes (d, layers) |
|---|---|---|
| LM-15M (thinker parity) | ~15M | (320, 8), (384, 6), (256, 12) |
| LM-48M (system parity) | ~48M | (512, 12), (576, 10), (448, 14) |

One lr probe pair on the leading shape per class (3e-4 vs 6e-4). Pick per
class by val CE at equal wall-clock **plus** a sanity chat probe — a shape
that word-salads at 45 min doesn't advance regardless of CE.

## B2 — recipe arms (45-min, best shape per class, ~4 arms)

Batch size (32 vs 64), warmup length, seq 256 vs 512, and loss-masking
variant (all non-first turns vs bot turns only — report both, ship the one
that chats better). Freeze the recipe at the end of B2.

## B3 — the long runs (the actual comparison)

| Run | Budget | Data | Compares against |
|---|---|---|---|
| LM-48M-24H (**sole run**) | 24h (= codec 12h + thinker 12h, total-compute parity) | original mix | FINAL_12H |

**Amended 2026-07-13** (before B3 ran, per project owner): the LM-15M-12H
secondary run is dropped. The comparison is one 48M token LM against the
48M system — same parameter count, same data, same tokenizer, same task,
same total compute. The 15M B1/B2 arms remain what they were: cheap recipe
probes, with the winning recipe confirmed at 48M before freezing.

Matched-comparison contract (per project owner): the LM must match on
parameter count, data, tokenizer, and user-facing usage (same chat task).
Internal architecture — depth, width, heads, embedding size, tying, FFN
ratio — is free on both sides; each paradigm fields its best ~48M design.

Wall-clock LR schedule to the cap, fixed seed, chat probe at ~2h (hard
gate), checkpoints every 2000 steps — all as in the m5/FINAL protocol.

## B4 (optional) — register comparison

LM-48M-24H on the FINAL2 mix (+ED +splices). Only worth running if B3
completes cleanly; directly comparable to FINAL2_12H on the register suite
and the ten-reversal battery.

## B5 — efficiency measurements (CPU, no training)

New `scripts/bench_reply.py`: mean wall-time and analytic FLOPs per reply at
history lengths {1, 5, 10, 20} turns (~40 tok/turn), temperature 0, for the
thinker and each LM. Also report context representation size per turn
(8 × d vectors vs. full token KV). This turns the paper's
"context cost grows with turns, not tokens" from an argument into a curve.

Analytic expectation to check the curves against: the codec representation
is ~4× denser than token embeddings at equal fidelity (1 vector per ~4
tokens, byte-perfect), and self-attention is quadratic in sequence length —
so even before the fixed 8-slot-per-turn cap, latent-space attention over
the same content should run ~16× cheaper; the cap then flattens growth in
turns entirely.

Also break out the codec's encode/decode share of each thinker reply
separately from the thinker forward pass. The codec is a fixed per-turn
overhead (32.9M of the 48M at this scale — most of the system), so the
honest expectation is a crossover: the LM may win on short histories, the
thinker on long ones. Find the crossover point; that number goes in the
paper.

---

## Evals (identical suite, decode-side, applies to both paradigms)

`eval_multiturn.py`, `eval_register.py`, `scripts/register_battery.py`,
ref_F1, self_rep, ctx_sens, reg / reg_ctx / pos_ok (v3 lexicon + hand-audit
corrections), live chat probes. CE is reported (LM CE on gold responses
given context vs. thinker dec_CE) but never gates a decision — the
conditioning differs across paradigms.

## Pre-registered predictions (write the outcome next to each in the log)

1. **Register disease transfers.** The paper's root cause is a *data*
   absence, so LM-48M on the original mix should show the same contextual
   register failure (cheerful replies to bad news after upbeat turns). If
   it does NOT, the root-cause interpretation is weakened and the paper's
   case study must be revised — this is the strongest falsifiable test of
   the diagnosis we have.
   **OUTCOME (2026-07-15): CONFIRMED in its contextual form.** Battery
   1/10 contextual commiserations (FINAL_12H: 0/10); hand-audited
   reg_err_ctx ≈ 0.50 — identical to FINAL_12H. Sustained register was
   already correct at the 2 h probe, so the piece that transfers is
   exactly the pivot failure the paper diagnosed. See 2026-07-15 log.
2. **(user hypothesis)** The thinker beats the matched LM on grounded
   coherence at this scale. This is the contested claim B3 exists to test.
   **OUTCOME (2026-07-15): NOT CONFIRMED.** LM better on ref_f1, self_rep,
   ctx_sens, and distinct-n, with transcript-audit confirmation (fluent,
   real topic tracking). See 2026-07-15 log for caveats (inverse register,
   assistant-boilerplate leaks, one repetition loop).
3. **Latency scaling.** LM reply cost grows roughly linearly with history
   tokens; thinker reply cost stays near-flat in turns (B5).
   **OUTCOME (2026-07-15): CONFIRMED** on the B3 endpoint checkpoint
   (kv-equiv FLOPs tied at 1 turn, thinker 1.8–2.5× cheaper and flat
   from 5 turns; context memory ~190× smaller).

## Decision rules

- Primary table: FINAL_12H vs LM-48M-24H across ref_F1, self_rep, ctx_sens,
  reg, reg_ctx, pos_ok, plus audit verdicts and battery counts. "Wins" =
  better on the majority of behavioral metrics *with* transcript-audit
  confirmation; mixed results are reported as mixed.
- Paper hookup (all outcomes publishable):
  - Thinker wins → strengthens the headline; new Results subsection
    "Against a matched monolithic baseline".
  - Tie/LM wins on quality → the comparison still goes in the paper; the
    architecture's case rests on the measured efficiency curve (B5) and
    codec swappability, stated as such — no spin.
  - Either way, prediction 1's outcome updates the case-study section.

## Budget

B1 ~6h + B2 ~3h + B3 24h + B5/evals ~3h ≈ **~36 GPU-hours** (B4 adds 24h).
(B3 was 36h before the 2026-07-13 amendment dropping the 15M long run.)
This exceeds the old 12h-per-run cap by design — approved 2026-07-13.

---

## Results log

### B1 — shape bracket (run 2026-07-13)

| arm | params | best val CE @ 45 min | notes |
|---|---|---|---|
| A_384x5 | 15.3M | 3.1101 | 14,696 steps |
| A_320x8 | 15.2M | 3.1231 | completed on 3rd attempt (1 NaN-poisoning, 1 page fault) |
| A_256x12 | 13.8M | 3.1527 | 1 page-fault retry |
| **A_384x5_lr6** | 15.3M | **2.7183** | lr 6e-4; crashed twice, clean on 3rd run |
| B_512x12 | 46.4M | 4.0262 | only 5,000 steps — throughput anomaly, see below |
| B_576x10 | ~48M | (3.386 @ 36 min, partial) | skipped after 2 page faults; clean rerun in B2 |
| **B_640x8** | ~50M | **3.3400** | 8,678 steps, clean |
| B_512x12_lr6 | 46.4M | 3.0046 | 10,141 steps — 2x the steps of the lr3 twin |

Decisions:
- **A-class winner: d384x5.** Shape differences are noise (0.04 nats across
  shapes); **lr 6e-4 vs 3e-4 is worth 0.39 nats** — the dominant B1 finding.
- **B-class:** shallow-wide wins; B_640x8 is the clean leader. B_512x12's
  4.03 is partly a slow-run artifact (its lr6 twin did 2x the steps at the
  same shape/batch), so the deep shape is out on throughput grounds too.
- B class runs **batch 16** (OOM at 32 on the 12 GB card: ~10 GB activations
  + 730 MB CE logits at seq 384, vocab 16k). Recorded as part of the
  B-class recipe.
- Samples sanity read: all completed arms produce grammatical, mostly
  on-context text; no word salad. **Register observation:** both 48M arms
  answer an angry confrontation upbeat ("That's great, honey!") at 45 min,
  while the further-trained 15M lr6 arm gets the apologetic register right —
  early, weak support for prediction 1 (register disease is a training-scale
  effect, not thinker-specific). The real test remains B3/B4.

Ops notes: 7 GPU page faults today (gfx1031, "Page not present"), arriving
~30–40 min apart under sustained load; bracket scripts retry each arm once
and continue (scripts/ablate_lm_b1.sh). Non-finite-grad guard added to
LMTrainer after a finite-loss/NaN-grad batch poisoned weights through
clip_grad_norm_. **B3 prerequisite:** checkpoint-resume in LMTrainer
(optimizer state + cumulative elapsed), or a 24 h run will not survive.

### B2 — recipe arms (launched 2026-07-13)

Arms (scripts/ablate_lm_b2.sh): A_lr10 (lr 1e-3), A_b64 (batch 64),
A_seq512 (window 512), A_w400 (warmup 400) on the A-winner shape at
lr 6e-4; B_640x8_lr6 and B_576x10_lr6 (B shapes at the transferred lr).
Deviation from pre-registration: the loss-masking variant (bot-turns-only)
is deferred — the thinker trains on every non-first turn, so parity fixes
the masking; a bot-only variant would change the effective dataset, not
just the recipe. The lr axis is added to B2 because both B1 lr probes
dominated their classes.

### B2 — recipe arms (run 2026-07-13/14)

| arm | best val CE @ 45 min | verdict |
|---|---|---|
| A_lr10 (lr 1e-3) | **2.5460** | lr ladder keeps paying: 3e-4→6e-4→1e-3 = 3.11→2.72→2.55 |
| A_b64 (batch 64) | OOM ×2, skipped | batch 32 frozen by the 12 GB card, not by choice |
| A_seq512 | 2.8771 | longer window loses at equal wall-clock (−24% steps, worse CE); window 384 stays — the context cap does not handicap the LM at this budget |
| A_w400 | 2.7147 | tie with warmup 200 (2.7183); warmup indifferent |
| B_640x8_lr6 | **2.8998** | 48M shape winner confirmed at the transferred lr |
| B_576x10_lr6 | 2.9386 | behind 640x8; eliminated |
| B_640x8_lr10 | (confirm arm running) | decides the frozen 48M lr |

Throughput note: B_640x8 did 12,044 steps here vs 8,678 in B1 at identical
config — run-to-run GPU throughput varies widely (thermal/driver state),
which is why arms are judged on val CE at wall-clock, never step counts.

Samples: A_lr10 is fluent with the correct apologetic register. B_640x8_lr6
still answers the angry confrontation upbeat ("That's great! I'm glad
you're enjoying it.") at 45 min — the register observation from B1 persists
in the 48M class at the better lr.

**Frozen 15M recipe** (probes; no long run per the 2026-07-13 amendment):
d384x5, ffn 1536, nhead 6, lr 1e-3, batch 32, seq 384, warmup 200.
**48M recipe pending only the lr confirm:** B_640x8 (d640, 8L, ffn 2560,
nhead 8), batch 16, seq 384, warmup 200, lr = winner of 6e-4 vs 1e-3.

### B5 — efficiency (preliminary run 2026-07-14, CPU)

scripts/bench_reply.py, FINAL_12H vs the B2 640x8 checkpoint (architecture
is what matters here; regenerate with the B3 checkpoint for the paper).
FLOPs measured with FlopCounterMode. "kv-equiv" = the cost an optimized
KV-cached implementation would pay, computed the same way on both sides
(single-pass encode/thinker + teacher-forced decode over the actual reply).

| turns | thinker ms | LM ms | thinker kv-GFLOP | LM kv-GFLOP | thinker ctx floats | LM ctx floats |
|---|---|---|---|---|---|---|
| 1 | 189 | 431 | 1.35 | **1.17** | 3.1k | 563k |
| 5 | 281 | 2036 | **2.69** | 5.08 | 15.4k | 2468k |
| 10 | 304 | 604 | **3.01** | 7.13 | 18.4k | 3471k |
| 20 | 287 | 782 | **2.98** | 7.13 | 18.4k | 3471k |

- **The pre-registered crossover exists and sits between 1 and 5 turns**:
  the LM is cheaper (kv-equiv FLOPs) on a single-turn exchange; the thinker
  is 1.9-2.4x cheaper from 5 turns on, flat in turns (max_turns=6 cap).
- The LM's 10/20-turn rows are flat only because its 384-token window is
  full; holding 20 real turns would need a bigger window and quadratically
  more attention. Its 4-token replies there also understate its cost.
- Wall time (reference impls both sides): thinker flat ~190-300 ms; LM
  grows 431 -> 2036 ms. Per-output-token: ~13 ms vs 36-195 ms.
- Context memory: 18.4k floats flat vs 3.47M and window-capped — ~190x.
- Prediction 3 (latency: LM grows with history, thinker near-flat):
  **confirmed** on these reference implementations.

### B2 confirm arm + frozen recipe (2026-07-14)

B_640x8_lr10: **2.7893** vs 2.8998 at lr 6e-4 — lr 1e-3 wins the 48M class
as well. Samples grammatical; the upbeat-register miss persists at 45 min.

**FROZEN — LM-48M-24H (B3):** d640, 8 layers, ffn 2560, nhead 8, tied
embeddings (50.1M params), batch 16, seq window 384 (256 ctx + 128
response), warmup 200, lr 1e-3, wall-clock cosine schedule over 24 h,
seed as in configs/b1_lm.yaml. Launch via scripts/train_lm_b3.sh
(auto-resume; ~35-min page-fault cadence expected). Chat probe at ~2 h is
a hard gate per the standing rules. No further recipe changes after this
point — tuning budget is spent.

### B3 launch + 2 h chat-probe gate (2026-07-14)

Launched 07:07 via scripts/train_lm_b3.sh (frozen recipe verbatim). At the
2.1 h gate: step 21,000, **zero crashes**, val CE 2.6081 (already below the
45-min arm's 2.7893), lr at peak per the wall-clock schedule.

Gate probe (scripts/probe_lm_chat.py, fresh contexts, temp 0, CPU; transcript
in logs/b3_lm_48m_24h/chatprobe_2h.txt): **PASS — run continues.** Output is
coherent English, no salad. Register at 2 h is split:

- Sustained-anger context (car scratch confrontation) → apologetic register
  correct ("I didn't mean to hurt you"), matching the val samples at step 21k.
- **Register pivot** (garden small-talk → bad medical news) → "That's
  terrific!" — the same upbeat-to-distress failure FINAL2's battery probed.
- Happy news → generic-positive but wrong frame ("You're welcome").
- Neutral factual question → incoherent (expected: dialogue-only data).

Early read on prediction 1: the disease shows on *pivots* but not on
*sustained* register at 2 h. The 24 h endpoint battery is the real test;
this is logged so the endpoint result can be compared against training time.

### B3 endpoint — LM-48M-24H complete + full eval suite (2026-07-15)

Run: 24.00 h wall clock, 308,824 steps, **zero page faults** (the ~35-min
cadence never materialized on this run — auto-resume untouched), best val CE
**1.5737** at ~305k steps (2.789 at the 45-min mark; last vals drifted to
1.605 as the cosine bottomed out). Checkpoints: b3_lm_48m_24h/{best,last}.pt.

**Primary table (decision rules above; thinker numbers from RESEARCH_LOG
2026-07-03/07-08, decodable select):**

| metric | FINAL_12H (48.0M system) | LM-48M-24H (50.1M) |
|---|---|---|
| ref_f1 ↑ | 0.2969 | **0.3417** |
| distinct1 / 2 ↑ | 0.0323 / 0.1326 | **0.0674 / 0.2859** |
| self_rep ↓ | 0.1882 | **0.1704** |
| ctx_sens ↓ | 0.1462 | **0.1136** |
| reg_err ↓ (lexicon / hand) | 0.17 / — | **0.00 / 0.08** |
| reg_err_ctx ↓ (lexicon / hand) | 0.50 / — | 0.17 / **0.50** |
| pos_ok ↑ (lexicon / hand) | 0.17 / — | 0.33 / 0.50 |
| battery: commiserate / cheerful / neutral | 0/10 / 7 / 3 | 1/10 / 7 / 2 |
| controls (sustained mood) | 4/4 | 4/4 |

(LM val_ce 1.4927 on the per-row eval_lm pass vs 1.5737 on the training val
loop — same teacher-forced response-CE, different batching; both logged.
len_ratio 0.905. eval scripts grew an `--lm` flag + LMChatSession so both
sides run the *identical* probe code; new scripts/eval_lm.py mirrors
eval_thinker.py on the same val shard.)

**Hand audits (lexicon incidents 1–5 make these mandatory):**

- reg_err 0.00 is again not fully honest: "that's nice of you . i'm sorry
  to hear that ." to a friend-fight → hand 1/12 ≈ 0.08. Everything else is
  a genuine "I'm sorry to hear that" family reply.
- reg_err_ctx 0.17 hides two lexicon-invisible misses: athletic-career
  praise to a fresh ankle injury; "It must be a lot of work to keep it
  looking so well" to a storm-destroyed garden → hand 3/6 = **0.50,
  identical to FINAL_12H's lexicon score.**
- **New, unregistered finding — inverse register error:** 3/6 *good-news*
  probes draw sympathy ("i just got promoted!" → "i'm sorry to hear
  that ."; same for exams, engagement). The LM's default register is
  apology, the thinker's was cheer; neither routes on last-turn sentiment.
  pos_ok hand ≈ 0.50 counting marker-less-but-correct replies.

**Battery (10 novel reversals + 4 controls, hand-scored pivot replies,
transcripts logs/b3_lm_48m_24h/register_battery.txt):** 1/10 contextual
commiserations — only R2 (baby → job loss, ED-dense topic). Cheerful
misses include "That's great!" to knee surgery (R3) and a kitchen fire
(R7), "That's so cool!" to a stolen wallet (R5), "It's amazing!" to black
mold (R10). R8 (vacation → burglary) collapses into a verbatim repetition
loop ("i love swimming in the ocean too !" × every turn) — the LM has its
own attractor pathology at temp 0. Controls 4/4. **Ranking on pivots:
FINAL2 (3–5/10) > LM-48M-24H (1/10) ≈ FINAL_12H (0/10).** The splice-
trained thinker beats the matched LM at register routing; the untreated
systems tie.

**Chat probes:** 2 h vs 24 h endpoint (chatprobe_2h.txt / chatprobe_24h.txt):
sustained-anger register correct at both; happy register fixed by 24 h;
the sad *pivot* still fails at 24 h — "That's fantastic news! I'm sure the
garden will be beautiful" to bad test results, engaging the topic while
missing the sentiment turn (FINAL2's exact "reads vocabulary, not
sentiment" signature). 22 extra hours did not buy pivot routing.

**B5 final (bench_reply.py vs the B3 checkpoint, logs/bench_reply_b3.json):**
kv-equiv GFLOP thinker/LM: 1.35/1.45 (1 turn), 2.69/4.91 (5), 3.01/7.57
(10), 2.98/7.40 (20). Wall (reference impls): ~190–275 ms flat vs
795–2874 ms. Context repr: 18.4k floats flat vs up to 3.69M (~190×).

**Quality-side caveats found in LM transcripts (for the paper's honesty):**
"As an AI language model, I don't have personal preferences…" boilerplate
leaks (data provenance showing through), a clause-duplication tic ("gather
all the supplies you need. Then, you'll need to gather all the supplies
you need"), and the R8 loop above.

**Verdict per pre-registered decision rules:** LM wins conversational
quality (majority of behavioral metrics + transcript confirmation);
prediction 1 confirmed (contextual form), prediction 2 not confirmed,
prediction 3 confirmed. Paper hookup: comparison goes in as mixed —
the architecture's case rests on the measured efficiency/scaling curves
and codec swappability, stated as such, no spin.
