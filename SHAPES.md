
# SHAPES.md — LAE + OOD Inference Shape Guide[SHAPES.md](SHAPES.md)

> **Purpose**
> This file explains the tensor shapes used in the LAE (Learning–Accumulation–Ensemble) inference pipeline with OOD-aware gating.
> Every item includes a short explanation in simple language so readers can understand *what the tensor means*, not just its size.

---

## 0) Symbols (What the letters mean)

| Symbol    | Meaning           | Simple explanation                                            |
| --------- | ----------------- | ------------------------------------------------------------- |
| **B**     | Batch size        | How many samples we process at the same time.                 |
| **D'**    | Feature dimension | How many numbers describe one sample after encoding.          |
| **C_sum** | Total classes     | All categories learned across every task so far.              |
| **C_t**   | Task classes      | Number of classes belonging to one task only.                 |
| **E**     | Number of experts | Different model versions giving predictions (here **E = 2**). |

### Experts in this repo

* **Online expert** → learns quickly (plastic).
* **EMA expert** → slow moving average (stable).

Both experts must always output predictions over the **same class space (`C_sum`)**.

---

## 1) High-Level Idea

Inference works like this:

1. Convert inputs into features.
2. Two experts make predictions.
3. A gate decides how much to trust each expert.
4. Their predictions are combined.
5. Outputs are sliced per task.

Simple analogy:

> Two students answer a question, and a teacher decides whose answer to trust more.

---

## 2) Tensor Shape Checklist

---

### Inputs

| Name       | Shape      | Explanation                                  |
| ---------- | ---------- | -------------------------------------------- |
| `inputs`   | `(B, ...)` | Raw data (images, text, etc.).               |
| `targets`  | `(B,)`     | Correct labels using global class IDs.       |
| `task_ids` | `(B,)`     | Indicates which task each sample belongs to. |

Think of a batch as a stack of homework questions.

---

### Feature Encoding

| Name       | Shape     | Explanation                      |
| ---------- | --------- | -------------------------------- |
| `feat_on`  | `(B, D')` | Features from the online expert. |
| `feat_off` | `(B, D')` | Features from the EMA expert.    |

The encoder turns each input into a list of numbers describing it.

Example idea:

> instead of “cat image”, we store hundreds of descriptive numbers.

Both experts must produce the **same feature size**.

---

### Logits (Raw Class Scores)

| Name         | Shape        | Explanation                           |
| ------------ | ------------ | ------------------------------------- |
| `logits_on`  | `(B, C_sum)` | Online expert scores for all classes. |
| `logits_off` | `(B, C_sum)` | EMA expert scores for all classes.    |

Each row answers:

> “How strongly does the model believe this sample belongs to each class?”

These are not probabilities yet.

---

### Probabilities

| Name    | Shape        | Operation                    |
| ------- | ------------ | ---------------------------- |
| `p_on`  | `(B, C_sum)` | `softmax(logits_on, dim=1)`  |
| `p_off` | `(B, C_sum)` | `softmax(logits_off, dim=1)` |

Softmax converts scores into probabilities that sum to 1.

Example:

```
[cat: 0.7, dog: 0.2, car: 0.1]
```

Now predictions are interpretable.

---

### Energy (Confidence Measurement)

| Name    | Shape  | Formula                              |
| ------- | ------ | ------------------------------------ |
| `E_on`  | `(B,)` | `-T * logsumexp(logits_on/T, dim=1)` |
| `E_off` | `(B,)` | same formula                         |

Energy measures confidence:

* **Low energy** → input looks familiar.
* **High energy** → input may be OOD (unfamiliar).

Important: both experts must use the same temperature **T**.

---

### Gate Weight (Expert Trust)

| Name | Shape    | Explanation                                          |
| ---- | -------- | ---------------------------------------------------- |
| `w`  | `(B, 1)` | Weight deciding how much we trust the online expert. |

`w` is always between 0 and 1.

* `w = 1` → trust online expert only
* `w = 0` → trust EMA expert only
* middle values → combine both

Why `(B,1)`?

One decision is made per sample and automatically broadcasts across all classes.

---

### Final Mixed Prediction

| Name | Shape        | Formula                |
| ---- | ------------ | ---------------------- |
| `p`  | `(B, C_sum)` | `w*p_on + (1-w)*p_off` |

This mixes both experts’ probabilities.

The result remains a valid probability distribution.

Interpretation:

> We average answers based on confidence.

---

### Final Predictions

| Name       | Shape  | Operation                   |
| ---------- | ------ | --------------------------- |
| `pred_on`  | `(B,)` | `argmax(logits_on, dim=1)`  |
| `pred_off` | `(B,)` | `argmax(logits_off, dim=1)` |
| `pred_ens` | `(B,)` | `argmax(p, dim=1)`          |

`argmax` chooses the class with the highest probability.

---

### Task-Based Slicing

We trained multiple tasks but evaluate them separately.

| Name           | Shape          | Explanation                               |
| -------------- | -------------- | ----------------------------------------- |
| `output_t`     | `(b_t, C_sum)` | Samples belonging to task *t*.            |
| `output_local` | `(b_t, C_t)`   | Only classes relevant to that task.       |
| `target_local` | `(b_t,)`       | Labels converted to local task numbering. |

Example idea:

> A math teacher only checks math questions, not history ones.

---

## 3) Gating Strategies

---

### Energy-Based Gating (Fast)

Uses confidence difference:

```
w = sigmoid((E_off - E_on)/scale)
```

Idea:

> trust the expert that looks more confident.

No gradients required → fastest method.

---

### GradNorm-Based Gating (Adaptive)

Measures how hard the model must work to understand an input.

Large gradient → confusing sample → possibly OOD.

Gradients are computed **only with respect to features**, not model parameters.

---

### Hybrid Gating

Combines both signals:

```
w = α*w_energy + (1-α)*w_grad
```

where `α ∈ [0,1]`.

Energy = quick intuition
GradNorm = deeper reasoning

---

## 4) Old vs New Ensemble

### Old (Per-Class Max)

```
stacked: (B, C_sum, E)
max over experts → (B, C_sum)
```

Each class chooses its favorite expert.

Problem:
can create overconfident predictions.

---

### New (LAE Sample-Level Mixture)

```
p = w*p_on + (1-w)*p_off
```

One decision per sample.

Benefits:

* smoother behavior
* interpretable gating
* better OOD calibration

---

## 5) `dim` Rules (Easy Memory Guide)

| dim      | Meaning                            |
| -------- | ---------------------------------- |
| `dim=0`  | across samples (batch axis)        |
| `dim=1`  | across classes                     |
| `dim=-1` | last dimension (often expert axis) |

Shortcut:

> `dim=1` means “compare classes”.

---

## 6) Minimal Shape Sanity Check

After mixing experts:

```python
assert p.shape == (B, C_sum)
assert torch.allclose(
    p.sum(dim=1),
    torch.ones(B, device=p.device),
    atol=1e-5
)
```

Every row must still sum to 1.

---

## 7) One-Sentence Mental Model

LAE inference works like:

> Two versions of the same model give answers, and an OOD-aware gate decides how much to trust each one for every sample.

---

**Maintainer note:**
If classifier dimensions or PET adapters change, update this file and verify shapes using the sanity check above.

