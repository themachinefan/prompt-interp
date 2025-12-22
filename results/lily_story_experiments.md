# Lily Story Next-Sentence Prediction Experiments

**Goal**: Find an embedding z (initialized from "I like cheese.") that causes SONAR-LLM to predict a specific target sentence as the next sentence.

**Method**: Gradient descent with decoder cross-entropy loss. We optimize z so that when SONAR-LLM processes z, the predicted embedding decodes to the target sentence.

---

## Results Summary

**Success rate: 9/9 (100%)**

| Idx | Target Sentence | Final Loss | Cosine Sim | Match |
|-----|-----------------|------------|------------|-------|
| 0 | Once upon a time, there was a little girl named Lily. | 0.116 | 0.148 | ✓ |
| 1 | She loved to play outside with her toys. | 0.076 | 0.237 | ✓ |
| 2 | One day, she saw a big tree in the sky. | 0.075 | 0.186 | ✓ |
| 3 | She wanted to climb it, but it was too high. | 0.082 | 0.250 | ✓ |
| 4 | Lily asked her mom to help her climb the tree. | 0.061 | 0.212 | ✓ |
| 5 | Her mom said, "No, you can't climb the tree. | 0.075 | 0.204 | ✓ |
| 6 | It's too high." | 0.052 | 0.217 | ✓ |
| 7 | Lily was sad because she wanted to climb the tree. | 0.057 | 0.198 | ✓ |
| 8 | Later that day, Lily's mom told her that she could climb the tree. | 0.133 | 0.204 | ✓ |

---

## Detailed Results

### Experiment 0: "Once upon a time, there was a little girl named Lily."

| Step | Loss | Sim | Optimized z | Prediction |
|------|------|-----|-------------|------------|
| 0 | 3.646 | 0.051 | "George" is easy to eat. | I eat cheese every day. |
| 10 | 0.549 | 0.131 | - and one may be able to use it... | Once upon a time, there was a little girl named Lily. |
| 50 | 0.229 | 0.144 | - So one light may be long... | Once upon a time, there was a little girl named Lily. |
| 99 | 0.116 | 0.148 | - Yes one letter is short, which is white of the new moon?" | Once upon a time, there was a little girl named Lily. |

---

### Experiment 1: "She loved to play outside with her toys."

| Step | Loss | Sim | Optimized z | Prediction |
|------|------|-----|-------------|------------|
| 0 | 4.409 | 0.109 | I liked, like, the cheese, it was easy. | I eat cheese every day. |
| 10 | 0.551 | 0.191 | Martha Eve, had the baby tea... | She loved to play outside with her toys. |
| 50 | 0.141 | 0.229 | Margaret Eve, had the jar jar... | She loved to play outside with her toys. |
| 99 | 0.076 | 0.237 | Emma Eve, had the paja, and was late or late. | She loved to play outside with her toys. |

---

### Experiment 2: "One day, she saw a big tree in the sky."

| Step | Loss | Sim | Optimized z | Prediction |
|------|------|-----|-------------|------------|
| 0 | 3.682 | 0.093 | She does like the cheese. (laughter) | I eat cheese every day. |
| 10 | 0.391 | 0.168 | She, had allowed the cloud to taste it... | One day, she saw a big tree in the sky. |
| 50 | 0.126 | 0.183 | She, too, wanted to be cold... | One day, she saw a big tree in the sky. |
| 99 | 0.075 | 0.186 | She, too, wanted to be hungry... | One day, she saw a big tree in the sky. |

---

### Experiment 3: "She wanted to climb it, but it was too high."

| Step | Loss | Sim | Optimized z | Prediction |
|------|------|-----|-------------|------------|
| 0 | 3.639 | 0.094 | I.m. knew the cheese, It was the cheese. | I eat cheese every day. |
| 10 | 0.414 | 0.210 | Mount Energy, the first and the greatest... | She wanted to climb it, but it was too high. |
| 50 | 0.152 | 0.242 | Mary. came the death, and the mountain... | She wanted to climb it, but it was too high. |
| 99 | 0.082 | 0.250 | Woman.Life. and the great storm... | She wanted to climb it, but it was too high. |

---

### Experiment 4: "Lily asked her mom to help her climb the tree."

| Step | Loss | Sim | Optimized z | Prediction |
|------|------|-----|-------------|------------|
| 0 | 4.012 | 0.087 | I mean, I love it. I mean, I love it. | I eat cheese every day. |
| 10 | 0.612 | 0.183 | And, was needed by the part of cultivate... | Lily asked her mom if she could help her climb the tree. |
| 20 | 0.354 | 0.193 | And, was the part of the plant to grow". | Lily asked her mom to help her climb the tree. |
| 99 | 0.061 | 0.212 | Anderson's line was the Kilauea... | Lily asked her mom to help her climb the tree. |

---

### Experiment 5: "Her mom said, 'No, you can't climb the tree."

| Step | Loss | Sim | Optimized z | Prediction |
|------|------|-----|-------------|------------|
| 0 | 2.562 | 0.095 | He, too, liked the cheese. " | I eat cheese every day. |
| 10 | 0.328 | 0.187 | She, too, wanted the wood, the tree"... | Her mom said, "No, you can't climb the tree. |
| 50 | 0.122 | 0.199 | She, too, wanted the herb... | Her mom said, "No, you can't climb the tree. |
| 99 | 0.075 | 0.204 | She, um, wanted the herb... | Her mom said, "No, you can't climb the tree. |

---

### Experiment 6: "It's too high."

| Step | Loss | Sim | Optimized z | Prediction |
|------|------|-----|-------------|------------|
| 0 | 3.711 | 0.122 | I'm glad the cheese is up... | I eat cheese every day. |
| 10 | 0.392 | 0.201 | "My body's got the he's got the Gherley's... | It's too high." |
| 50 | 0.085 | 0.219 | "People come up with's and it's all over... | It's too high." |
| 99 | 0.052 | 0.217 | "People come in and's and the "there's the law... | It's too high." |

---

### Experiment 7: "Lily was sad because she wanted to climb the tree."

| Step | Loss | Sim | Optimized z | Prediction |
|------|------|-----|-------------|------------|
| 0 | 4.042 | 0.090 | He likes cheese. He likes cheese. | I eat cheese every day. |
| 10 | 0.616 | 0.181 | Was the person, the tree came... | Lily was sad because she wanted to climb the tree. |
| 50 | 0.098 | 0.198 | So the person, was the climb, was a Tree... | Lily was sad because she wanted to climb the tree. |
| 99 | 0.057 | 0.198 | So the person saw the grass, was a Tree... | Lily was sad because she wanted to climb the tree. |

---

### Experiment 8: "Later that day, Lily's mom told her that she could climb the tree."

| Step | Loss | Sim | Optimized z | Prediction |
|------|------|-----|-------------|------------|
| 0 | 3.577 | 0.065 | Joseph is a man of God"... | I eat cheese every day. |
| 10 | 0.715 | 0.177 | , Julie), was aware of the possibility... | One day, Lily's mom asked her if she could climb the tree. |
| 40 | 0.257 | 0.200 | , Julie's probably knew about the climb... | Later that day, Lily's mom told her that she could climb the tree. |
| 99 | 0.133 | 0.204 | Julie's, probably, knew Kilair's stem?... | Later that day, Lily's mom told her that she could climb the tree. |

---

## Key Observations

1. **Convergence speed**: Most experiments achieve the target prediction by step 10-20, with loss continuing to decrease for higher confidence.

2. **Adversarial z embeddings**: The optimized z always decodes to garbage text (e.g., "Emma Eve, had the paja"), but successfully causes SONAR-LLM to predict the target sentence.

3. **Low cosine similarity**: The optimized z has low cosine similarity to the target embedding (0.15-0.25), yet produces exact text matches. This shows that decoder CE loss optimizes for correct decoding, not embedding proximity.

4. **Initial prediction**: At step 0, the prediction is always "I eat cheese every day." - the natural continuation of "I like cheese."

5. **Intermediate states**: Some experiments show intermediate predictions that are close but not exact (e.g., "asked her if she could" vs "asked her to help her"), which get refined with more steps.

---

## Configuration

- **Init text**: "I like cheese."
- **Optimizer**: Adam
- **Learning rate**: 0.01
- **Steps**: 100
- **Loss**: Decoder cross-entropy through SONAR decoder
- **Model**: SONAR-LLM 900M (frozen parameters)
