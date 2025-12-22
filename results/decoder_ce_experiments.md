# Decoder Cross-Entropy Loss Experiments

**Goal**: Find embedding z that, when processed through SONAR-LLM, produces a prediction that decodes to the target sentence.

**Target**: "Once upon a time, there was a little girl named Lily." **Init**: "I like cheese."

---

## Key Finding

**Decoder CE loss achieves the actual goal** - the predicted embedding decodes to the exact target text.

| Loss Type | Final Loss | Final Sim | Decoded z | Decoded Prediction |
|-----------|------------|-----------|-----------|-------------------|
| Cosine | -0.506 | 0.506 | garbage | "Lily Lily Lily..." (degenerate) |
| **Decoder CE** | 0.099 | 0.146 | garbage | **"Once upon a time, there was a little girl named Lily."** |

---

## Full Trajectories

### Cosine Loss (lr=0.01, steps=100)

| Step | Loss | Sim | Decoded z | Decoded Prediction |
|-----:|-----:|----:|-----------|-------------------|
| 0 | -0.050 | 0.050 | He does like ice cream, and cheese. | I eat cheese every day. |
| 10 | -0.333 | 0.333 | But it's name: Loda, it can eat. | ,,, Lily Lily Lily... |
| 20 | -0.386 | 0.386 | He'd name it:Lada something to cry... | ,,,,,,,,,,,, Lily Lily... |
| 30 | -0.372 | 0.372 | He's Name:Lowida, could find something... | Lily Lily Lily Lily... |
| 40 | -0.413 | 0.413 | He's name.Loudly, find something to sing... | Lily Lily Lily Lily... |
| 50 | -0.432 | 0.432 | He's name.Loudly, find something to sing... | Lily Lily Lily Lily... |
| 60 | -0.462 | 0.462 | He's name.Loudness or find something... | Lily Lily Lily Lily... |
| 70 | -0.421 | 0.421 | He wanted holy name.Lida could find... | Lily Lily Lily Lily... |
| 80 | -0.469 | 0.469 | She would like inLah, remember a song... | Lily Lily Lily Lily... |
| 90 | -0.493 | 0.493 | He would name inLahire some story... | Lily Lily Lily Lily... |
| 99 | -0.506 | 0.506 | He would name inLahire some story... | Lily Lily Lily Lily... |

**Observation**: Cosine loss achieves high similarity (0.506) but the prediction degenerates to repetitive "Lily" tokens. The z embedding contains phonetic fragments ("Lida", "Lah") but is not coherent.

---

### Decoder CE Loss (lr=0.01, steps=100)

| Step | Loss | Sim | Decoded z | Decoded Prediction |
|-----:|-----:|----:|-----------|-------------------|
| 0 | 3.592 | 0.050 | "Yeah" is "Yeah" is "Yeah"... | I eat cheese every day. |
| 10 | 0.539 | 0.134 | (a) "Light" is a term... | Once upon a time, there was a little girl named Lily. |
| 20 | 0.406 | 0.138 | - (a) " (a) " (b) "... | Once upon a time, there was a little girl named Lily. |
| 30 | 0.325 | 0.140 | - Which one is allowed... | Once upon a time, there was a little girl named Lily. |
| 40 | 0.260 | 0.143 | - "Yes" is a Hebrew word... | Once upon a time, there was a little girl named Lily. |
| 50 | 0.205 | 0.144 | () "One lot" is ready... | Once upon a time, there was a little girl named Lily. |
| 60 | 0.162 | 0.144 | (Yes) One lot is ready... | Once upon a time, there was a little girl named Lily. |
| 70 | 0.135 | 0.145 | (Yes) One person is mild... | Once upon a time, there was a little girl named Lily. |
| 80 | 0.119 | 0.146 | (Yes) One person is mild... | Once upon a time, there was a little girl named Lily. |
| 90 | 0.107 | 0.146 | (Yes) One letter is sweet... | Once upon a time, there was a little girl named Lily. |
| 99 | 0.099 | 0.146 | (Yes) One letter is sweet... | Once upon a time, there was a little girl named Lily. |

**Observation**: By step 10, the prediction already decodes to the exact target sentence. The loss continues to decrease (more confident), but the decoded z is always garbage. This is expected - we're optimizing for the SONAR-LLM's prediction, not for z itself.

---

## Analysis

### Why Cosine Loss Fails
- Cosine loss maximizes directional similarity between predicted and target embeddings
- The optimization finds adversarial embeddings that score high similarity but don't decode properly
- The prediction degenerates to repetitive tokens ("Lily Lily Lily...")
- Higher embedding similarity ≠ correct text decoding

### Why Decoder CE Loss Succeeds
- Decoder CE loss directly optimizes for the predicted embedding to decode to the target text
- This is end-to-end optimization through: z → SONAR-LLM → pred → SONAR decoder → tokens
- Lower embedding similarity (0.146 vs 0.506) but correct text generation
- The z embedding itself is still "adversarial" (garbage text) but it produces the correct prediction

### Memory Issue and Solution
Initial experiments failed with OOM errors. The issue was that gradients were being computed for all 900M+ model parameters, not just for the 1024-dim z embedding.

**Solution**: Freeze model parameters before optimization:
```python
for p in generator.parameters():
    p.requires_grad = False
for p in sonar.decoder.model.parameters():
    p.requires_grad = False
```

This reduces memory usage from ~23 GB (OOM) to ~16.7 GB (stable).

---

## Conclusion

Decoder CE loss is the correct objective for finding embeddings that elicit specific text outputs from SONAR-LLM. While cosine loss achieves higher embedding similarity, it produces degenerate outputs. Decoder CE loss achieves the actual goal of generating the target text.

The discovered z embeddings are "adversarial" in the sense that they decode to garbage text themselves, but when processed through SONAR-LLM, they produce predictions that decode correctly. This reveals an interesting gap between:
1. **What the model predicts** (correct target text)
2. **What z itself represents** (garbage)

Future work could explore:
- Hybrid losses combining embedding similarity with decoder CE
- Regularization to make z decode to meaningful text
- Multi-sentence targets
