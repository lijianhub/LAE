Here is a **dual-layer SHAPES.md** — the style many research labs use internally:

✅ **Research Layer** → precise, paper-ready definitions
✅ **Beginner Layer** → short intuitive explanation right under it

This lets:

* researchers verify correctness quickly
* new contributors understand immediately

You can drop this directly into your repo.

---

# SHAPES.md — LAE + OOD Inference Shape Guide

## Dual-Layer Edition (Research View + Beginner View)

> This document defines tensor shapes used in LAE inference with OOD-aware gating.
> Each item contains:
>
> * **Research View** — exact technical meaning
> * **Beginner View** — intuitive explanation

---

# 0) Notation

| Symbol    | Meaning                                       |
| --------- | --------------------------------------------- |
| **B**     | batch size                                    |
| **D'**    | feature dimension used by classifier head     |
| **C_sum** | total global classes accumulated across tasks |
| **C_t**   | number of classes in task *t*                 |
| **E**     | number of experts (here: 2 = online + EMA)    |

### Research View

All experts operate in a **shared global label space**. Their outputs must be shape-compatible for probabilistic ensembling.

### Beginner View

Every model speaks the same “language” of classes, so their answers can be combined safely.

---

# 1) Inference Overview

Pipeline:

```
inputs → features → logits → probabilities
           ↓            ↓
        online       EMA expert
              ↓
            gating
              ↓
         mixed prediction
```

### Research View

Inference performs a sample-wise convex combination of expert probability distributions using an OOD-aware gating function.

### Beginner View

Two model versions make predictions, and a smart switch decides how much to trust each one.

---

# 2) Inputs

| Tensor     | Shape      |
| ---------- | ---------- |
| `inputs`   | `(B, ...)` |
| `targets`  | `(B,)`     |
| `task_ids` | `(B,)`     |

### Research View

* `targets` are global class indices.
* `task_ids` specify task membership for evaluation slicing.

### Beginner View

We process many samples together. Each sample has:

* a correct answer
* a task label telling where it came from.

---

# 3) Feature Encoding

| Tensor     | Shape     |
| ---------- | --------- |
| `feat_on`  | `(B, D')` |
| `feat_off` | `(B, D')` |

### Research View

Features are embeddings produced by identical backbones with different PET adapters (online vs EMA).

Feature dimensionality must remain invariant.

### Beginner View

The model converts each input into a long list of numbers describing it.
Both experts describe things using the same number of numbers.

---

# 4) Logits (Raw Predictions)

| Tensor       | Shape        |
| ------------ | ------------ |
| `logits_on`  | `(B, C_sum)` |
| `logits_off` | `(B, C_sum)` |

### Research View

Classifier head maps features into global class scores. These logits live in a unified continual-learning label space.

### Beginner View

Each expert gives a score for every possible class.

Higher score = stronger belief.

---

# 5) Probabilities

| Tensor  | Shape        | Operation        |
| ------- | ------------ | ---------------- |
| `p_on`  | `(B, C_sum)` | `softmax(dim=1)` |
| `p_off` | `(B, C_sum)` | `softmax(dim=1)` |

### Research View

Softmax normalizes logits into categorical probability distributions over classes.

### Beginner View

Scores are turned into percentages that add up to 100%.

---

# 6) Energy (Confidence Signal)

| Tensor  | Shape  |
| ------- | ------ |
| `E_on`  | `(B,)` |
| `E_off` | `(B,)` |

Formula:

```
E = -T * logsumexp(logits / T, dim=1)
```

### Research View

Energy approximates log partition confidence. Lower energy indicates higher in-distribution likelihood.

Temperature **T must be identical across experts**.

### Beginner View

Energy measures how familiar the input looks to the model.

Low value → “I’ve seen this before.”
High value → “This looks strange.”

---

# 7) Gating Weight

| Tensor | Shape    |
| ------ | -------- |
| `w`    | `(B, 1)` |

### Research View

`w ∈ [0,1]` is a sample-level mixing coefficient produced by an OOD-aware gating function.

Broadcasting over class dimension enables convex combination.

### Beginner View

`w` decides how much we trust the online expert.

* 1 → fully trust online
* 0 → fully trust EMA
* between → listen to both

One decision per sample.

---

# 8) Final Ensemble Prediction

| Tensor | Shape        |
| ------ | ------------ |
| `p`    | `(B, C_sum)` |

Formula:

```
p = w*p_on + (1-w)*p_off
```

### Research View

Probability-space convex mixture preserves normalization and improves calibration compared to logit mixing or per-class selection.

### Beginner View

We average both experts’ answers based on trust level.

The final prediction is still a valid probability.

---

# 9) Final Class Decisions

| Tensor     | Shape  |
| ---------- | ------ |
| `pred_on`  | `(B,)` |
| `pred_off` | `(B,)` |
| `pred_ens` | `(B,)` |

Operation:

```
argmax(dim=1)
```

### Research View

Top-1 prediction extracted along class dimension.

### Beginner View

Pick the class with the biggest probability.

---

# 10) Task-Specific Output Slicing

| Tensor         | Shape          |
| -------------- | -------------- |
| `output_t`     | `(b_t, C_sum)` |
| `output_local` | `(b_t, C_t)`   |
| `target_local` | `(b_t,)`       |

### Research View

Predictions are restricted to task-specific class ranges for evaluation consistency in continual learning benchmarks.

### Beginner View

Each task only cares about its own classes, so we keep only the relevant answers.

---

# 11) Gating Strategies

---

## Energy Gating

### Research View

Uses energy difference:

```
w = sigmoid((E_off - E_on)/scale)
```

No backward pass required.

### Beginner View

Trust the expert that looks more confident.

---

## GradNorm Gating

### Research View

Computes gradient norm of KL divergence w.r.t. features to estimate uncertainty.

Gradients are taken only on features, not parameters.

### Beginner View

Measures how hard the model must think to understand the input.

Harder thinking → less confidence.

---

## Hybrid Gating

### Research View

Linear interpolation:

```
w = α*w_energy + (1-α)*w_grad
```

where `α ∈ [0,1]`.

### Beginner View

Combine fast intuition (energy) and deeper reasoning (gradient).

---

# 12) Dimension Rules

| dim      | Meaning                        |
| -------- | ------------------------------ |
| `dim=0`  | across batch samples           |
| `dim=1`  | across classes                 |
| `dim=-1` | last dimension (often experts) |

### Beginner Memory Trick

`dim=1` = choosing between classes.

---

# 13) Shape Sanity Check

```python
assert p.shape == (B, C_sum)
assert torch.allclose(
    p.sum(dim=1),
    torch.ones(B, device=p.device),
    atol=1e-5
)
```

### Research View

Ensures convexity and normalization invariants.

### Beginner View

Every prediction must still add up to 100%.

---

# 14) Mental Model Summary

### Research View

LAE inference implements OOD-aware mixture-of-experts over adaptation states within a unified continual-learning classifier.

### Beginner View

Two versions of the model answer each question, and a smart confidence checker decides how much to trust each one.

---

✅ This version works simultaneously as:

* collaborator onboarding doc
* reproducibility reference
* appendix-ready technical explanation

