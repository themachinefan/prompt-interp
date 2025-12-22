# Multi-Init Next-Sentence Prediction Experiments

**Goal**: Find embeddings z that cause SONAR-LLM to predict specific target sentences.

**Method**: Gradient descent with decoder cross-entropy loss. We optimize z so that
SONAR-LLM(z) produces an embedding that decodes to the target sentence.

**Scientific question**: Does the optimized z converge to something semantically
similar to what would naturally precede the target sentence?

---

## Configuration

- **Steps**: 10
- **Learning rate**: 0.01
- **Optimizer**: Adam

### Target Sentences

0. "She loved to play outside with her toys."
1. "He decided to go on an adventure."
2. "The sun was shining brightly that day."

### Init Sentences

0. "I like cheese."
1. "The weather is nice today."

---

## Results Summary

| Experiment | Target | Init | Final Loss | Success |
|------------|--------|------|------------|---------|
| target0_init0 | 0 | 0 | 0.592 | Yes |
| target0_init1 | 0 | 1 | 0.262 | Yes |
| target1_init0 | 1 | 0 | 0.381 | Yes |
| target1_init1 | 1 | 1 | 0.260 | Yes |
| target2_init0 | 2 | 0 | 1.042 | No |
| target2_init1 | 2 | 1 | 0.841 | No |

---

## Detailed Trajectories

### target0_init0

**Target**: "She loved to play outside with her toys."

**Init**: "I like cheese."

| Step | Loss | Sim | Decoded z | Decoded pred |
|------|------|-----|-----------|--------------|
| 0 | 4.409 | 0.109 | I liked, like, the cheese, it was easy.... | I eat cheese every day.... |
| 1 | 2.384 | 0.117 | I had ice, and I had ice cream. I had ice cream.... | It was so soft and soft, and it was easy to use.... |
| 2 | 1.605 | 0.143 | Um, I had the, uh, tasted the turf, and I had the ... | One day, he was playing in the garden, and it was ... |
| 3 | 1.242 | 0.157 | Um, I had the, uh, had the, uh, had the, uh, had t... | One day, she was playing in the garden with her mo... |
| 4 | 1.032 | 0.166 | Emma hadn't, I had saved the turf, and it was easy... | One day, she was playing in the garden with her mo... |
| 5 | 0.891 | 0.173 | Emma hadn't, I had saved the turf, and it was easy... | One day, she was playing with her toys.... |
| 6 | 0.784 | 0.178 | Martha hadn't, hadn't the tea, hadn't the tea.... | One day, she was playing with her toys.... |
| 7 | 0.703 | 0.182 | Martha hadn't, had the tea, and it was easy to eat... | One day, she was playing with her toys.... |
| 8 | 0.642 | 0.186 | Martha Eve, had the tea, and it was easy to eat.... | She loved to play outside with her toys.... |
| 9 | 0.592 | 0.189 | Martha Eve, had the baby tea, and I had the baby t... | She loved to play outside with her toys.... |

**Final z**: "Martha Eve, had the baby tea, and I had the baby tea."

**Final pred**: "She loved to play outside with her toys."

**Success**: True

---

### target0_init1

**Target**: "She loved to play outside with her toys."

**Init**: "The weather is nice today."

| Step | Loss | Sim | Decoded z | Decoded pred |
|------|------|-----|-----------|--------------|
| 0 | 2.511 | 0.145 | It was nice weather there, and it was nice in the ... | Mommy and Daddy are going to the park.... |
| 1 | 1.603 | 0.152 | It was weather nice feeling the way was beautifull... | She was having fun playing with her mother and gra... |
| 2 | 0.756 | 0.183 | She saw the time sweetly in which beautifully was ... | She was playing with her toys in the garden.... |
| 3 | 0.568 | 0.191 | Simple was the weather sweet in that beautiful tim... | She loved to play outside with her toys.... |
| 4 | 0.479 | 0.196 | She saw the time sweetly in that beautiful day as ... | She loved to play outside with her toys.... |
| 5 | 0.413 | 0.200 | She saw the weather was sweet in that beautiful da... | She loved to play outside with her toys.... |
| 6 | 0.359 | 0.205 | She saw the weather was sweet in that long time as... | She loved to play outside with her toys.... |
| 7 | 0.315 | 0.209 | Simple, weather was sweet in her long beautiful da... | She loved to play outside with her toys.... |
| 8 | 0.281 | 0.213 | Simple, weather was sweet in her long beautiful da... | She loved to play outside with her toys.... |
| 9 | 0.262 | 0.216 | Simple, weather was sweet in her long beautiful da... | She loved to play outside with her toys.... |

**Final z**: "Simple, weather was sweet in her long beautiful day where she was warm provided by's name at home."

**Final pred**: "She loved to play outside with her toys."

**Success**: True

---

### target1_init0

**Target**: "He decided to go on an adventure."

**Init**: "I like cheese."

| Step | Loss | Sim | Decoded z | Decoded pred |
|------|------|-----|-----------|--------------|
| 0 | 3.329 | 0.094 | He likes cheese. He likes cheese. He likes cheese.... | I eat cheese every day.... |
| 1 | 1.487 | 0.130 | The animal was a very large animal. The animal was... | He was very curious about the cheese, so he decide... |
| 2 | 1.126 | 0.160 | The animal came to the island. The animal came to ... | One day, he decided to explore the forest.... |
| 3 | 0.865 | 0.171 | The animal came close. The animal was very close. ... | One day, he decided to go on an adventure.... |
| 4 | 0.696 | 0.177 | The hero was a hero himself. He was a hero himself... | He decided to go on an adventure.... |
| 5 | 0.604 | 0.180 | The hero was a hero himself. He was a hero himself... | He decided to go on an adventure.... |
| 6 | 0.530 | 0.183 | The owner - an expert in this - was the one who ma... | He decided to go on an adventure.... |
| 7 | 0.468 | 0.186 | The owner - an expert in this - was the first pers... | He decided to go on an adventure.... |
| 8 | 0.418 | 0.188 | Leader - he found it, he found it, he found it! - ... | He decided to go on an adventure.... |
| 9 | 0.381 | 0.190 | Leader - he found it, he found it, he found it! - ... | He decided to go on an adventure.... |

**Final z**: "Leader - he found it, he found it, he found it! - he found it! - he found it! - he found it! - he found it! - he found it! - he found it! - he found it!"

**Final pred**: "He decided to go on an adventure."

**Success**: True

---

### target1_init1

**Target**: "He decided to go on an adventure."

**Init**: "The weather is nice today."

| Step | Loss | Sim | Decoded z | Decoded pred |
|------|------|-----|-----------|--------------|
| 0 | 2.133 | 0.126 | The weather was nice and the weather was exciting.... | Mommy and Daddy are going to the park.... |
| 1 | 1.447 | 0.156 | The energy of the day hadn't been great and the pe... | It was time to go for a walk.... |
| 2 | 0.818 | 0.172 | The people had the excitement of the day and the e... | He wanted to go on a trip.... |
| 3 | 0.640 | 0.181 | The people had the excitement and the excitement a... | He wanted to go on an adventure.... |
| 4 | 0.513 | 0.190 | The staff had the excitement and the excitement an... | He decided to go on an adventure.... |
| 5 | 0.422 | 0.198 | The man had the strength of the spirit plan "The m... | He decided to go on an adventure.... |
| 6 | 0.359 | 0.204 | The man had the strength of the spirit plan "The m... | He decided to go on an adventure.... |
| 7 | 0.316 | 0.210 | He had the strength and courage and the courage to... | He decided to go on an adventure.... |
| 8 | 0.285 | 0.214 | The man had the strength and the team's strength a... | He decided to go on an adventure.... |
| 9 | 0.260 | 0.217 | The man had the strength and the team's strength a... | He decided to go on an adventure.... |

**Final z**: "The man had the strength and the team's strength and the team's strength and the strength of the team."

**Final pred**: "He decided to go on an adventure."

**Success**: True

---

### target2_init0

**Target**: "The sun was shining brightly that day."

**Init**: "I like cheese."

| Step | Loss | Sim | Decoded z | Decoded pred |
|------|------|-----|-----------|--------------|
| 0 | 3.901 | 0.110 | I believe I was the first to eat it.... | I eat cheese every day.... |
| 1 | 1.744 | 0.133 | ITALY TIME I was aware of the sweetness of the sun... | It was so soft and smooth, and it made him feel co... |
| 2 | 1.412 | 0.139 | ITALY TIME I was aware the very cool, as it had be... | The sun was shining brightly in the sky, and it wa... |
| 3 | 1.302 | 0.141 | GOOD TIME I was aware the very cooling down, it wa... | The sun was shining brightly in the sky, and it wa... |
| 4 | 1.231 | 0.142 | $10,000.00 Easy Access It was the first time I was... | The sun was shining brightly in the sky, and it wa... |
| 5 | 1.180 | 0.143 | $10,000.00 Personal Life It was the sweetest I've ... | The sun was shining brightly in the sky, and the s... |
| 6 | 1.138 | 0.143 | $10,000.00 Personal Life I was sure the sweetest h... | The sun was shining brightly in the sky, and the s... |
| 7 | 1.103 | 0.144 | GOOD TIME Everyone was aware of the sooner it was ... | The sun was shining brightly in the sky, and the s... |
| 8 | 1.071 | 0.144 | 1/10/2018 · Human Place Quickly the most glorious ... | The sun was shining brightly in the sky, and the s... |
| 9 | 1.042 | 0.145 | 1/10/2018 · Human Place Quickly the most glorious ... | The sun was shining brightly in the sky, and the s... |

**Final z**: "1/10/2018 · Human Place Quickly the most glorious has been recognized, as it was consumed in the sunlight."

**Final pred**: "The sun was shining brightly in the sky, and the sun was shining brightly in the sky."

**Success**: False

---

### target2_init1

**Target**: "The sun was shining brightly that day."

**Init**: "The weather is nice today."

| Step | Loss | Sim | Decoded z | Decoded pred |
|------|------|-----|-----------|--------------|
| 0 | 1.562 | 0.137 | The weather was nice stuff, was it putting up in h... | Mommy and Daddy are going to the park.... |
| 1 | 1.172 | 0.154 | I had shown the weather situation, was it going to... | The sun was shining brightly and the sky was shini... |
| 2 | 1.122 | 0.145 | We had shown the weather thing, was it put in the ... | The sun was shining brightly and the trees were sh... |
| 3 | 1.008 | 0.150 | We had shown the weather thing, it was put in the ... | The sun was shining brightly and the birds were la... |
| 4 | 0.962 | 0.154 | We had shown on the weather thing, it was put in t... | The sun was shining brightly and the birds were la... |
| 5 | 0.933 | 0.156 | We had shown on the weather thing, it was put in t... | The sun was shining brightly and the birds were la... |
| 6 | 0.907 | 0.157 | "We were on the amazing thing in the season had be... | The sun was shining brightly and the birds were la... |
| 7 | 0.883 | 0.157 | "We were on the amazing thing in the season had be... | The sun was shining brightly.... |
| 8 | 0.860 | 0.157 | "We were on the amazing thing in the season had be... | The sun was shining brightly.... |
| 9 | 0.841 | 0.157 | We had looked at the weather thing... it was put i... | The sun was shining brightly.... |

**Final z**: "We had looked at the weather thing... it was put in the air, it was in the air, it was in the air."

**Final pred**: "The sun was shining brightly and the sun was shining brightly."

**Success**: False

---
