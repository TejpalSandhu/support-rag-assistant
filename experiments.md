# Experiment log

Each entry: what changed, the numbers before and after, and the decision.
Retrieval is measured with `python evaluate.py` on the 28 answerable
questions in `eval_questions.json` (hit@k = right page in the top k;
MRR = average of 1/rank of the first right page).

> With 28 questions, one question is worth about 3.6 percentage points.
> Differences of one or two questions are within noise.

---

## 1. Chunk size and overlap sweep (2026-10-16)

Embedding model `gemini-embedding-001`, Chroma with cosine distance.

| chunk size | overlap | chunks | hit@1 | hit@3 | hit@5 | hit@12 | MRR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0 | 358 | 79% | 89% | 100% | 100% | 0.87 |
| **100** | **20** | **409** | **86%** | **93%** | **100%** | **100%** | **0.91** |
| 100 | 50 | 557 | 82% | 93% | 93% | 100% | 0.88 |
| 200 | 40 | 219 | 86% | 93% | 100% | 100% | 0.90 |
| 300 | 60 | 159 | 79% | 96% | 100% | 100% | 0.87 |

**Findings**

- **Overlap 0 -> 20 helped:** hit@1 79% -> 86%, MRR 0.87 -> 0.91. Three
  questions moved up to rank 1 (PLA temperatures, MBL failing, XL fuse) and one
  moved down (XL factory reset). Facts split across a chunk border appear
  whole in one chunk once neighbours overlap.
- **Overlap 50 did not help:** hit@5 dropped to 93%. With 557 heavily
  overlapping chunks, near-duplicates of the same wrong page fill several
  top-5 slots and push the right page out.
- **Chunk size 200/40 ties 100/20** (MRR 0.90 vs 0.91) with about half the
  chunks, so it's a valid cheaper option. 300/60 is weaker at rank 1.
- **Main failure pattern is product confusion.** For "Nothing comes out of
  my MINI's nozzle", the XL's almost identical nozzle-cleaning page (XL p53)
  ranked first, and the right MINI page came 4th. The manuals share a lot of
  wording.

**Decision:** keep 100 words / 20 overlap (already the default).

**Ideas to test later**
- Filter by product when the question names one (Chroma `where={"source": ...}`).
- Put the product name in front of each chunk before embedding.

---

## 2. TOP_K 12 -> 5 (2026-10-16)

| | TOP_K | chunks sent to Gemini | words of context (approx.) | right page included (eval set) |
|---|---:|---:|---:|---:|
| Before | 12 | 12 | ~1,200 | 100% (hit@12) |
| After | 5 | 5 | ~500 | 100% (hit@5) |

**Why:** the right page was always in the top 5, so chunks 6-12 only added
length and distractors (often the other printer's pages).

**Risk:** for "How do I take the filament out of my MINI+?" the right page
sits exactly at rank 5, so there's no margin. Watch it in the Phase 5 traces.
(Entry 4 shows what really happened.)

---

## 3. "I don't know" threshold (2026-10-16)

`python choose_threshold.py` takes the best retrieval score of every eval
question (28 answerable, 8 not covered by the manuals) for the 100/20 setting.

| | best score |
|---|---:|
| Lowest answerable question (XL factory reset) | 0.645 |
| Highest not-covered question **below** that (Tesla warranty) | 0.617 |
| Not-covered questions **above** every cutoff: resin 0.733, XL price 0.726, refund 0.667 | - |
| Fully off-topic (cat food, sourdough, World Cup) | 0.48-0.52 |

| threshold | answerable answered | not-covered refused before the model |
|---:|---:|---:|
| none | 100% | 0% |
| **0.631 (chosen)** | **100%** | **62% (5 of 8)** |
| 0.673 (best balanced accuracy) | 93% | 75% |

**Decision:** `MIN_SCORE = 0.631`, the highest cutoff that still answers every
real question. 0.673 would refuse one more off-topic question but also two
real customer questions. For a support bot, refusing a real question is the
worse mistake.

**Key finding:** near-miss questions (they sound like the product, but the
answer isn't in the manual) can score **higher** than real questions, so no
score cutoff can catch them. They're handled by the second layer, the prompt
rule "answer only from the context, otherwise say you don't have enough
information".

**Caveat:** the safety margin is small (0.645 vs 0.617), and there are only 8
not-covered questions. More real traffic would be needed to tune it well.

---

## 4. End-to-end check of both refusal layers (2026-10-16)

Real run of `rag_assistant.py` (TOP_K 5, MIN_SCORE 0.631):

| Question | Best score | Layer that handled it | Result |
|---|---:|---|---|
| What should I feed a cat? | 0.519 | 1. threshold | refused, model **not called** |
| Can the Prusa XL print with resin? | 0.733 | 2. prompt | model refused ("not enough information") |
| How much does the Prusa XL cost in the US? | 0.726 | 2. prompt | model refused |
| How do I return my printer and get a refund? | 0.667 | 2. prompt | model refused |
| How do I take the filament out of my MINI+? | 0.749 | - | partial: no invented steps, but it missed the real answer |

No hallucinated answers in this run.

**Limitation found: page-level evaluation is too generous.** For the MINI
unloading question, `evaluate.py` counted a hit because a chunk from page 19
ranked 5th. But that chunk (c1) is about *loading*. The chunk with the
*unloading* steps (c4) isn't even in the top 12. It mixes two topics
(unloading + "First Layer Calibration"), and the customer's words ("take the
filament out") don't match the manual's ("Unload filament"). The model
correctly said it didn't have the MINI steps, so the answer was safe but not
useful.

**Next steps**
- Score at chunk level (does the retrieved text contain the answer?), or
  judge the final answers, not just the page.
- Chunk by headings, so a section like "Unloading filament" stays in one chunk.
- Hybrid search (keyword + vector), so exact terms like "unload" count more.
- Filter by product when the question names one (fixes the MINI/XL mix-ups).
