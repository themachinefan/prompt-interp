# Technical Summary: SONAR and SONAR-LLM

## SONAR (Duquenne et al., 2023)

### Architecture
- **Encoder-decoder with bottleneck**: Transformer encoder (24 layers) + decoder (24 layers), initialized from NLLB 1B dense MT model
- **Fixed-size sentence embedding**: Mean-pooling over encoder token outputs produces a single vector per sentence (dimension 1024)
- **Decoder cross-attention**: Instead of attending to variable-length encoder outputs, decoder attends only to the single pooled sentence vector at each step
- **Language coverage**: 200 languages for text, 37 languages for speech

### Training Objectives
The final SONAR model uses a weighted combination:

$$\mathcal{L} = \mathcal{L}_{MT} + 0.1 \cdot \mathcal{L}_{MSE} + 0.01 \cdot \mathcal{L}_{DAE}$$

- **$\mathcal{L}_{MT}$**: Translation loss (cross-entropy on target tokens)
- **$\mathcal{L}_{MSE}$**: MSE between embeddings of source and target sentences—explicitly aligns translations in embedding space
- **$\mathcal{L}_{DAE}$**: Denoising auto-encoding (reconstruct sentence from noisy input)—prevents collapse from MSE while improving reconstruction

Key findings:
- Pure auto-encoding is too easy and hurts language-agnostic properties
- MSE alone can cause collapse; DAE mitigates this
- Translation-only gives good xsim but weaker auto-encoding

### Pooling Methods Tested
- **Mean-pooling**: Used in final model (stable training)
- **Max-pooling**: Outputs different value range than NLLB training, worse results
- **EOS-pooling**: Unstable training

### Decoder Fine-tuning (Random Interpolation Decoding)
After training, freeze encoder and fine-tune decoder only:
1. Encode source $x$ and target $y$ sentences
2. Compute $z = \text{interpolate}(e_x, e_y)$ randomly
3. Train decoder to output $y$ from $z$

This improves decoding quality (+9.3 BLEU on auto-encoding) without changing the embedding space.

### Speech Encoders
- Teacher-student distillation: frozen SONAR text encoder is teacher
- Student: speech encoder initialized from w2v-bert 2.0 (600M params)
- Loss: MSE between speech embedding and text embedding of transcription
- Pooling: attention-pooling (3-layer transformer decoder with cross-attention on speech encoder outputs) works best

### Training Setup
- Initialized from NLLB 1B dense model
- 100k updates with NLLB learning rate and batch size
- Training data: all NLLB bitext data (human labeled, back-translated, mined)
- 200 target languages

---

## SONAR-LLM (Dragunov et al., 2025)

### Core Idea
Decoder-only transformer that predicts the *next sentence embedding* autoregressively, supervised via token-level cross-entropy through the frozen SONAR decoder. Combines semantic abstraction of concept-level models with likelihood-based training.

### Architecture
- **Input**: Sequence of SONAR sentence embeddings $(e_1, \ldots, e_t)$
- **Model**: Llama 3-style decoder-only transformer (RMSNorm, RoPE), but outputs continuous vectors instead of discrete tokens
- **Output**: Predicted next embedding $\hat{e}_{t+1} \in \mathbb{R}^{1024}$
- **Embedding vocabulary size**: Effectively 1—each position outputs a continuous 1024-dim vector

Formally: given prefix $e_{<t} = (e_1, \ldots, e_{t-1})$, the network outputs $\hat{e}_t = f_\theta(e_{<t}) \in \mathbb{R}^d$.

### Training Objective
Given predicted embedding $\hat{e}_t$, decode it through frozen SONAR decoder $\mathcal{D}$:

$$z_t = \mathcal{D}(\hat{e}_t) \in \mathbb{R}^{|V|}$$

Loss is standard cross-entropy against ground-truth tokens of sentence $s_t$:

$$\mathcal{L} = -\sum_{t=1}^{T} \log p_\theta(s_t | e_{<t}) = -\sum_{t=1}^{T} \sum_{i=1}^{|s_t|} \log \text{softmax}(z_t)_{s_{t,i}}$$

Gradients backpropagate through the frozen SONAR decoder into the SONAR-LLM parameters. Teacher forcing supplies ground-truth embedding $e_t$ at the next step.

### Preprocessing
- Sentence segmentation via NLTK Punkt tokenizer
- Each sentence encoded with frozen SONAR encoder to 1024-dim vector

### End-of-Sequence Handling
- Append literal sentence `"End of sequence."` to every document during training
- Encode once with SONAR encoder to get $e_{eot}$
- At inference, stop when $\cos(\hat{e}, e_{eot}) > \tau_{stop} = 0.98$ or after $T_{max} = 32$ sentences

### Model Sizes
| Config Name | Total Params | Trainable Params (SONAR-LLM) |
|-------------|--------------|------------------------------|
| 39M         | 39M          | 11M                          |
| 100M        | 100M         | 34M                          |
| 300M        | 300M         | 170M                         |
| 600M        | 600M         | 450M                         |
| 900M        | 900M         | 700M                         |
| 1.3B        | 1.3B         | 1.1B                         |

Note: SONAR-LLM and MSE-LCM have fewer trainable parameters because embedding matrices are excluded from training (same depth/width as LLM counterparts).

### Scaling Law
Fitted to validation loss at epoch 4:

$$L(N) = aN^{-\alpha} + b$$

| Model              | $a$              | $\alpha$ | $b$   |
|--------------------|------------------|----------|-------|
| LLM                | $4.06 \times 10^5$ | 0.791    | 1.24  |
| MSE LCM            | $3.21 \times 10^4$ | 0.515    | 199   |
| Diffusion LCM      | $1.58 \times 10^5$ | 0.485    | 84.0  |
| **SONAR-LLM**      | $2.09 \times 10^3$ | 0.569    | 1.73  |

All fits have $R^2 > 0.995$.

### Training Details
- **Scaling experiments**: TinyStories dataset, 4 epochs
- **Summarization experiments**: 1.3B models pretrained on mixture of TinyTextbooks, TinyOrca-Textbooks, TinyStrangeTextbooks, TextbooksAreAllYouNeed, WikiText-103, XSum, CNN/DailyMail
- **Learning rate**: $1 \times 10^{-3}$ for SONAR-LLM (others used $5 \times 10^{-4}$)
- **Scheduler**: Cosine LR
- **Hardware**: Up to 8× NVIDIA A100 80GB

### Inference Efficiency
Theoretical FLOPs comparison (600M params, avg sentence length 60 tokens):
- Below ~4096 tokens: token-level LLM is more efficient
- Above ~4096 tokens: SONAR-LLM is more efficient
- Up to 1M tokens: SONAR-LLM cost grows almost linearly (quadratic term scaled by $1/60^2$)

### Key Results

**vs. Other Concept Models (GPT-4o eval on TinyStories)**:
- SONAR-LLM outperforms MSE-LCM and Diffusion-LCM on grammar, creativity, consistency, plot

**NLG Metrics**: SONAR-LLM matches or slightly exceeds token-level LLM on BLEU, ROUGE-L, METEOR

**Summarization (1.3B models)**:
| Model         | XSum R-L | XSum MET | CNN/DM R-L | CNN/DM MET |
|---------------|----------|----------|------------|------------|
| SONAR-LLM     | **19.3** | 15.2     | 16.0       | 10.4       |
| LLM-beam      | 18.7     | **15.4** | 18.3       | **16.5**   |
| LLM-greedy    | 18.9     | 14.9     | **18.7**   | 14.1       |
| MSE LCM       | 12.2     | 8.7      | 7.6        | 3.7        |
| Diffusion LCM | 12.0     | 8.3      | 10.2       | 5.1        |

SONAR-LLM excels on abstractive summarization (XSum) but trails on extractive (CNN/DM).

---

## Implementation References

**SONAR encoder/decoder**: `github.com/facebookresearch/SONAR`

**SONAR-LLM code**: `github.com/FusionBrainLab/SONAR-LLM`

### Key Dimensions
- Sentence embedding: 1024
- SONAR text encoder: 24 transformer layers
- SONAR text decoder: 24 transformer layers

### Inference Pipeline
1. Segment input text into sentences (NLTK Punkt)
2. Encode each sentence with frozen SONAR encoder → sequence of 1024-dim vectors
3. Feed embedding sequence to SONAR-LLM, predict next embedding
4. Decode predicted embedding with frozen SONAR decoder → output sentence tokens
5. Encode generated sentence, append to context
6. Repeat until cosine similarity with $e_{eot} > 0.98$ or max 32 sentences

### Comparison to LCM (Meta)
- **MSE LCM**: Predicts embeddings, trained with MSE loss directly on embedding space
- **Diffusion LCM**: Predicts embeddings via diffusion process, two-tower architecture
- **SONAR-LLM**: Predicts embeddings, but loss computed via token-level cross-entropy through frozen decoder

SONAR-LLM advantage: retains likelihood-based training signal, no diffusion sampler needed, single-shot generation per sentence.
