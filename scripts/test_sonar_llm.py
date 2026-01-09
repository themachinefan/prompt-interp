#%%
#!/usr/bin/env python3
"""Simple script to test SONAR-LLM predictions."""

import torch
from prompt_interp.sonar_wrapper import SonarWrapper
from prompt_interp.generator import SonarLLMGenerator
from prompt_interp.optimize import predict_next_embedding


def test_sonar_llm(text: str, sonar_wrapper: SonarWrapper, generator: SonarLLMGenerator, segment: bool = True) -> str:
    """
    Take a string, segment into sentences, encode, predict next sentence, decode.
    Returns the predicted next sentence.

    Args:
        segment: If True, segment text into sentences. If False, treat whole text as one sentence.
    """
    print("=" * 70)
    print("INPUT TEXT:")
    print(f"  \"{text}\"")
    print()

    # Step 1: Segment into sentences (or skip)
    if segment:
        sentences = sonar_wrapper.segment(text)
        print("STEP 1: Sentence segmentation")
    else:
        sentences = [text]
        print("STEP 1: No segmentation (treating as single sentence)")
    for i, sent in enumerate[str](sentences):
        print(f"  [{i}] \"{sent}\"")
    print()

    # Step 2: Encode sentences
    embeddings = sonar_wrapper.encode(sentences)  # (n_sentences, 1024)
    print("STEP 2: Encode sentences")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Norms: {embeddings.norm(dim=-1).tolist()}")
    print()

    # Step 3: Reshape for SONAR-LLM (batch=1, seq=n_sentences, embed=1024)
    seq = embeddings.unsqueeze(0)  # (1, n_sentences, 1024)
    print("STEP 3: Prepare sequence for SONAR-LLM")
    print(f"  Sequence shape: {seq.shape}")
    print()

    # Step 4: Predict next sentence embedding
    with torch.no_grad():
        pred_seq = predict_next_embedding(seq, generator)  # (1, n_sentences, 1024)
    pred_emb = pred_seq[:, -1, :]  # Take last position prediction (1, 1024)
    print("STEP 4: Predict next sentence embedding")
    print(f"  Output shape: {pred_seq.shape}")
    print(f"  Taking last position: {pred_emb.shape}")
    print(f"  Prediction norm: {pred_emb.norm().item():.3f}")
    print()

    # Step 5: Decode prediction
    decoded = sonar_wrapper.decode(pred_emb)[0]
    print("STEP 5: Decode prediction")
    print(f"  \"{decoded}\"")
    print()

    print("=" * 70)
    print("SUMMARY:")
    print(f"  Input:      \"{text}\"")
    print(f"  Sentences:  {len(sentences)}")
    print(f"  Prediction: \"{decoded}\"")
    print("=" * 70)

    return decoded


#%%
# Load models
print("Loading SONAR...")
sonar_wrapper = SonarWrapper()
print("Loading SONAR-LLM...")
generator = SonarLLMGenerator.from_pretrained("raxtemur/sonar-llm-900m")
generator.eval()
print()

#%%
# Set input text here (edit this line to test different inputs)
# text = "Once upon a time, there was a little girl named Lily. She loved to play in the garden."
# text = "Mrs. Gladys wanted to buy eggs. She sold them."
# text = "The lady wanted to buy eggs for women to cook food."
# text = "Mrs. Og Yag wanted to buy eggs to buy eggs"
# text = "Mrs. Og Yag wanted to buy eggs."
text = "Mary wanted a new doll and a new thing at school, but instead she had a new toy."

test_sonar_llm(text, sonar_wrapper, generator, segment=False)

# %%
