# SONAR-LLM 1.3B Generation Tests

Model: `raxtemur/sonar-llm-1.3b` (trained on Textbooks/Summarization)
Settings: `temperature=0` (greedy), `max_sentences=6`

---

## In-Distribution (Textbook/Educational prompts)

### 1. Chapter 3: The Human Digestive System. The digestive system breaks down food into nutrients.

> The digestive system consists of digestive organs, which are responsible for digestion, digestion, and digestion. The digestive system is responsible for digestion, digestion, and absorption of nutrients. The digestive system is responsible for the digestion of food. The digestive system is responsible for the digestion of food.

**Quality**: Maintains topic but extremely repetitive, minimal information added

---

### 2. In economics, supply and demand determine market prices.

> In economics, the price of a good or service is determined by the quantity of goods or services that it produces. In other words, the price of a good or service is determined by the quantity of goods or services that it produces. At the end of the equation, the price of a good or service is determined by the quantity of goods or services that are produced.

**Quality**: Attempts economic language but circular/tautological, gets stuck in loop

---

### 3. The French Revolution began in 1789 and transformed European politics.

> The French Revolution began with the French Revolution, which led to the creation of the Republic of France. The French Revolution began with the French Revolution and ended with the French Revolution. End of sequence.

**Quality**: Self-referential loop, terminates early

---

## Out-of-Distribution

### 1. Once upon a time, there was a little girl named Lily.

> She was a little girl, and she loved to play with her grandmother. When she was little, Lily loved to play with her grandmother. When she was little, Lily liked to play with her grandmother. When she was little, Lily liked to play with her grandmother. When she was little, Lily liked to play with her grandmother.

**Behavior**: Attempts story but immediately loops on single phrase

---

### 2. Yo what's up bro, wanna grab some pizza later?

> I'm going to pick up some pizza, and I'm going to eat some pizza. I don't know what I'm going to eat. I don't know what I'm going to eat. I don't know what I'm going to eat. I don't know what I'm going to eat. I don't know what I'll eat.

**Behavior**: Picks up "pizza" concept, gets stuck in uncertainty loop

---

### 3. def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)

> End of sequence.

**Behavior**: Immediate termination - code is completely OOD

---

## Summary

| Aspect | 1.3B (Textbooks) | 900M (TinyStories) |
|--------|------------------|-------------------|
| **Repetition** | Severe - almost all outputs loop | Moderate - some loops |
| **Topic adherence** | Maintains domain vocabulary | Maintains narrative style |
| **OOD handling** | Loops or terminates | Forces into story template |
| **Diversity** | Very low | Moderate |
| **Best use case** | Possibly fine-tuning base | Story continuation |

The 1.3B model appears to be significantly more prone to degenerate repetition than the 900M model, possibly due to different training data characteristics or training duration.
