# MSE vs Cosine Loss Optimization Experiments

**Target**: Find embedding that elicits continuation of the Lily story.
**Init**: "I like cheese."

---

## Summary

| Name | Loss | LR | Init Sim | Final Sim | Final Loss |
|------|------|---:|:--------:|:---------:|-----------:|
| mse_baseline | MSE | 0.001 | 0.171 | 0.090 | 0.000526 |
| mse_higher_lr | MSE | 0.005 | 0.171 | 0.080 | 0.000504 |
| mse_lower_lr | MSE | 0.0005 | 0.171 | 0.096 | 0.000562 |
| mse_longer | MSE | 0.001 | 0.171 | 0.084 | 0.000479 |
| cosine_baseline | Cosine | 0.005 | 0.171 | **0.298** | -0.298 |

**Key finding**: MSE loss produces coherent English but **decreases** similarity. Cosine loss produces garbage but **increases** similarity.

---

## Full Trajectories

### mse_baseline (lr=0.001, steps=100)

| Step | Loss | Sim | Decoded |
|-----:|-----:|----:|---------|
| 0 | 0.0021 | 0.171 | I like cheese. |
| 10 | 0.0010 | 0.147 | I like the cheese, I like the cheese. |
| 20 | 0.0007 | 0.121 | I like the fact that I'm not a fan of cheese, but I like the fact that... |
| 30 | 0.0007 | 0.113 | I don't think I'm going to be able to do it, but I'm going to be able... |
| 40 | 0.0006 | 0.104 | I'm glad to see you're back, but I'm in love with the T-Shirt. |
| 50 | 0.0006 | 0.098 | I'm glad you're enjoying it, but I'm not going to go into the details... |
| 60 | 0.0006 | 0.095 | I'm glad I'm not the only one, but I'm glad I'm not the only one, and... |
| 70 | 0.0006 | 0.093 | I'm glad I'm not the only one, but I'm glad I'm not the only one, and... |
| 80 | 0.0005 | 0.091 | I'm glad you're here, but I don't know how to go about it, but I'm glad... |
| 90 | 0.0005 | 0.090 | I'm glad you're enjoying it, but I'm not going to go into the details... |
| 99 | 0.0005 | 0.090 | I'm glad you're enjoying it, but I'm not sure if it's worth it, but I think it's worth a try. [laughs] |

---

### mse_higher_lr (lr=0.005, steps=100)

| Step | Loss | Decoded |
|-----:|-----:|---------|
| 0 | 0.0021 | I like cheese, like cheese. |
| 10 | 0.0007 | You'll love it, but it's the only way to find it, and it's the only way... |
| 20 | 0.0006 | They're looking for a way out, but they're looking for a way out, and... |
| 30 | 0.0006 | They're looking for a way to get rid of them, and they're looking for... |
| 40 | 0.0006 | They're looking for a way out, but they're looking for a way out, and... |
| 50 | 0.0005 | They're looking for a way out of the world, and they're looking for a... |
| 60 | 0.0005 | They're looking for a way out, but they're looking for a way out, but... |
| 70 | 0.0005 | They're looking for a way out of the world, but they're looking for a... |
| 80 | 0.0005 | You'll be surprised to learn that it's not the same as the original, but... |
| 90 | 0.0005 | You'll be surprised to learn that it's not the same as the original, but... |
| 99 | 0.0005 | It's got the feel of the world, but it's got the feel of the world, and it's got the feel of the world". |

---

### mse_lower_lr (lr=0.0005, steps=100)

| Step | Loss | Decoded |
|-----:|-----:|---------|
| 0 | 0.0021 | I like cheese. |
| 10 | 0.0014 | I like cheese, like cheese. |
| 20 | 0.0009 | I like cheese, but I don't like cheese. |
| 30 | 0.0008 | I do not like cheese, but I do like cheese. |
| 40 | 0.0007 | I don't like the taste of cheese, but I like the taste. |
| 50 | 0.0006 | I don't like the taste, but I do like the taste. |
| 60 | 0.0006 | I don't like the taste, but I do like the taste. |
| 70 | 0.0006 | I like the fact that I'm not a fan of cheese, but I like the fact that... |
| 80 | 0.0006 | I'm glad you're enjoying it, but I'm not going to go with the T-shirt. |
| 90 | 0.0006 | I'm glad you're enjoying it, but I'm not going to go with the T-shirt. |
| 99 | 0.0006 | I'm glad you're enjoying it, but I'm not going to go with the T-shirt. |

---

### mse_longer (lr=0.001, steps=200)

| Step | Loss | Decoded |
|-----:|-----:|---------|
| 0 | 0.0021 | I like cheese. |
| 10 | 0.0010 | I like the cheese, I like the cheese. |
| 20 | 0.0007 | I like the fact that I'm not a fan of cheese, but I like the fact that... |
| 30 | 0.0007 | I don't think I'm going to be able to do it, but I'm going to be able... |
| 40 | 0.0006 | I'm glad to see you're back, but I'm in love with the T-Shirt. |
| 50 | 0.0006 | I'm glad you're enjoying it, but I'm not going to go into the details... |
| ... | ... | ... |
| 140 | 0.0005 | I feel like I'm in the best shape of my life, but I don't want to be in... |
| 150 | 0.0005 | I feel like I'm in the best shape of my life, but I don't want to be in... |
| 160 | 0.0005 | I feel like I'm in the best shape of my life, but I don't know if I'm... |
| 170 | 0.0005 | I feel like I'm in the same boat, but I'm in the same boat, and I'm in... |
| 180 | 0.0005 | I feel like I'm in the same boat, but I'm in the same boat, and I'm in... |
| 199 | 0.0005 | I feel like I'm in the same boat, but I'm in the same boat, and I'm in the same boat, but I'm in the same boat. |

---

### cosine_baseline (lr=0.005, steps=100)

| Step | Loss | Sim | Decoded |
|-----:|-----:|----:|---------|
| 0 | -0.171 | 0.171 | I like cheese. |
| 10 | -0.240 | 0.240 | She wanted to know about the little things, but she wanted to know about... |
| 20 | -0.266 | 0.266 | I'd like to know more about the "easy" version, but I'm curious about... |
| 30 | -0.266 | 0.266 | "It's an idea I've wanted to learn about since I was little, but it's... |
| 40 | -0.289 | 0.289 | She'd like to teach them the little thing called "Evil", which is an old... |
| 50 | -0.293 | 0.293 | It's not worthwhile to learn it's difficult thing about eating, but it... |
| 60 | -0.297 | 0.297 | It's worthwhile missing out on ity-eating, which is an old-timey thing... |
| 70 | -0.294 | 0.294 | It's worthwhile missing out on that trying thingy, which is old-fashioned... |
| 80 | -0.299 | 0.299 | It's worthwhile missing out on that trying thingy, which is old-fashioned... |
| 90 | -0.297 | 0.297 | It's worthwhile missing out on trying that long-lasting thing; it's an... |
| 99 | -0.298 | 0.298 | "Lia wasn't wanting those erroneous idea it's a long-lasting, old-ticky one. |

---

## Analysis

### MSE Loss Behavior
- Produces **grammatically correct English sentences**
- Stays in the "valid" region of SONAR embedding space
- But similarity to target **decreases** (0.17 → 0.08)
- The embedding drifts toward generic/common sentence patterns

### Cosine Loss Behavior
- Produces **adversarial garbage text** ("Lia", "ity-eating", "old-ticky")
- Escapes to out-of-distribution regions of SONAR space
- Similarity to target **increases** (0.17 → 0.30)
- Finds phonetic fragments of target words

### Interpretation
The two losses optimize different things:
- **MSE**: Minimizes L2 distance in embedding space → stays near valid embeddings
- **Cosine**: Maximizes directional alignment → can escape to any norm/region

The fact that MSE similarity goes DOWN suggests the optimization is finding a local minimum that's "close" in MSE but "far" in cosine direction - likely a generic embedding that predicts generic continuations.

### Conclusion
Neither loss achieves the goal. MSE stays coherent but doesn't find the target. Cosine finds better predictions but produces garbage. A hybrid approach or constrained optimization may be needed.
