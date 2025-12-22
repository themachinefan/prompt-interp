# Gradient-Based Embedding Optimization Experiments

**Target**: Find embedding that elicits continuation "She loved to play outside..." from the Lily story.

**Method**: Optimize first sentence embedding to maximize cosine similarity between model predictions and target sentence embeddings.

**Constraint**: Project embedding to have same L2 norm as valid TinyStories embeddings (~0.21).

---

## Summary Table

| Experiment | Init Text | LR | Steps | Init Sim | Final Sim | Improvement |
|------------|-----------|---:|------:|---------:|----------:|------------:|
| baseline | I like cheese. | 0.02 | 100 | 0.171 | 0.299 | +0.129 |
| lower_lr | I like cheese. | 0.005 | 100 | 0.171 | 0.298 | +0.127 |
| higher_lr | I like cheese. | 0.05 | 100 | 0.171 | 0.299 | +0.128 |
| longer | I like cheese. | 0.02 | 200 | 0.171 | 0.304 | +0.133 |
| **lower_lr_longer** | I like cheese. | 0.005 | 200 | 0.171 | **0.305** | **+0.135** |
| story_init | There was a boy named Tom. | 0.02 | 100 | 0.173 | 0.298 | +0.125 |
| story_init_lower_lr | There was a boy named Tom. | 0.005 | 100 | 0.173 | 0.304 | +0.131 |

**Best**: `lower_lr_longer` (lr=0.005, steps=200) achieved highest final similarity of 0.305.

---

## Observations

### 1. All experiments converge to ~0.30 similarity
Regardless of hyperparameters, the optimization plateaus around 0.30 cosine similarity. This suggests a fundamental limit - either:
- The loss landscape has a basin around this value
- The norm constraint prevents reaching better solutions
- The model's predictive capacity limits how well we can match targets

### 2. Decoded text is adversarial garbage
Despite improving the loss, all final embeddings decode to nonsense:
- `"Lia wasn't interested in that dicky thingy..."`
- `"Delicious on I Want to try that old itty-bitty..."`
- `"The 'lily' unmindful enjoy It is easy-ilive..."`

### 3. Phonetic fragments of target appear
The optimization finds embeddings with phonetic similarity to target words:
- "Lily" → "Lia", "lily", "Lili"
- "little" → "itty", "ity"
- Words ending in "-ly" or "-ily" appear frequently

### 4. Learning rate has minimal impact
All learning rates (0.005, 0.02, 0.05) achieve similar final similarity. The optimization is robust to this hyperparameter.

### 5. More steps helps slightly
200 steps achieves ~0.305 vs ~0.299 for 100 steps - a small but consistent improvement.

---

## Conclusions

The optimization successfully improves the prediction loss, but the resulting embeddings are **adversarial** - they produce correct internal model predictions but decode to garbage text.

This reveals a gap between:
1. **Internal prediction space**: What makes the model predict certain continuations
2. **Decodable embedding space**: What SONAR can decode to coherent text

To get meaningful results, we likely need:
- **Cycle consistency loss**: Decode → re-encode should match original
- **Discrete optimization**: Search over actual sentence embeddings
- **Different objective**: Use token-level cross-entropy through SONAR decoder (like training)
