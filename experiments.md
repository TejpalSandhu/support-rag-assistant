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
