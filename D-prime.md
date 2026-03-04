Short answer:
> We use **D′ (D-prime)** instead of **D** because the feature dimension used by the classifier is **not necessarily the original backbone feature dimension**.

Let’s explain it in both research and beginner ways.

---

# ✅ Research Explanation

In modern parameter-efficient tuning (PET) and continual learning pipelines, the representation passes through multiple transformations:

```
backbone → adapters / PET → projection → classifier head
```

The backbone produces features of dimension:

```
D  (original embedding dimension)
```

But the classifier often receives:

```
D′ (transformed feature dimension)
```

where

```
D′ ≠ D   (in general)
```

This happens because:

### 1️⃣ Projection layers

A linear layer may reduce or expand dimensionality:

```
ℝ^D → ℝ^{D′}
```

Example:

* ViT backbone output: D = 1024
* classifier input: D′ = 768

---

### 2️⃣ PET / Adapter modifications

Adapters may:

* bottleneck features
* reparameterize representations
* apply low-rank updates

So the effective feature space seen by the head changes.

---

### 3️⃣ Head-specific feature preparation

Your `_encode_for_head()` explicitly implies:

```
encoder output ≠ head input
```

Therefore the correct notation is:

```
feat ∈ ℝ^{B × D′}
```

not `D`.

---

### Why this matters mathematically

Your GradNorm gating computes:

```
||∇_features loss|| / sqrt(D′)
```

The normalization must use the **actual feature dimension used in gradients**, not the backbone dimension.

Using `D` here would be incorrect if a projection exists.

---

# ✅ Beginner Explanation

Think of it like this:

* The backbone creates a big description of an image using **D numbers**.
* Before making a decision, we sometimes **reshape or compress** that description.
* The classifier only sees the **new version** of the description.

So:

| Symbol | Meaning                                    |
| ------ | ------------------------------------------ |
| **D**  | original description size                  |
| **D′** | final description size used for prediction |

D′ just means:

> “the version actually used at the final decision step.”

The little `'` (prime) means **“changed from the original.”**

---

# ✅ Real Example

Imagine:

```
Backbone output: 1024 numbers  → D
Projection layer: reduce to 512 numbers
Classifier input: 512 numbers  → D′
```

So:

```
D = 1024
D′ = 512
```

If we wrote `D`, readers would wrongly assume no transformation happened.

---

# ✅ Why Papers Prefer D′

Using D′ signals immediately to reviewers:

✔ feature space may differ from backbone
✔ projections/adapters exist
✔ gradients are computed in transformed space

It avoids hidden assumptions.

---

# ✅ Quick Rule to Remember

```
D  = raw backbone feature size
D′ = feature size actually used by the classifier
```

If anything modifies features before classification → use **D′**.

---

## ⭐ One-line intuition

> D′ is the feature dimension **after the model finishes preparing information for decision making**.

