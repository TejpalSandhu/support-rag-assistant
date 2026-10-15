# Retrieval evaluation

Run: 2026-10-16 00:46 | 28 answerable questions | embedding model: gemini-embedding-001

hit@k = share of questions whose expected page is in the top k results. MRR = average of 1/rank of the first correct result (1.0 = always first).

| chunk size | overlap | chunks | hit@1 | hit@3 | hit@5 | hit@12 | MRR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0 | 358 | 79% | 89% | 100% | 100% | 0.87 |
| 100 | 20 | 409 | 86% | 93% | 100% | 100% | 0.91 |
| 100 | 50 | 557 | 82% | 93% | 93% | 100% | 0.88 |
| 200 | 40 | 219 | 86% | 93% | 100% | 100% | 0.90 |
| 300 | 60 | 159 | 79% | 96% | 100% | 100% | 0.87 |
