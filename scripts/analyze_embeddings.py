"""Analyze SONAR embedding statistics from TinyStories."""

import torch
import nltk
from datasets import load_dataset
from prompt_interp import SonarWrapper


def main():
    print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    print("Loading SONAR encoder...")
    sonar = SonarWrapper("cuda")

    # Collect sentences from first N stories
    n_stories = 100
    all_sentences = []

    print(f"Extracting sentences from {n_stories} stories...")
    for i, example in enumerate(ds):
        if i >= n_stories:
            break
        sentences = nltk.sent_tokenize(example["text"])
        all_sentences.extend(sentences)

    print(f"Total sentences: {len(all_sentences)}")

    # Encode in batches
    batch_size = 64
    all_embeddings = []

    print("Encoding sentences...")
    for i in range(0, len(all_sentences), batch_size):
        batch = all_sentences[i:i+batch_size]
        with torch.no_grad():
            embs = sonar.encode(batch)
            all_embeddings.append(embs.cpu())
        if (i // batch_size) % 10 == 0:
            print(f"  Encoded {min(i+batch_size, len(all_sentences))}/{len(all_sentences)}")

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, 1024)
    print(f"\nEmbedding matrix shape: {embeddings.shape}")

    # Compute statistics
    print("\n" + "="*60)
    print("EMBEDDING STATISTICS")
    print("="*60)

    # Norms
    norms = embeddings.norm(dim=1)
    print(f"\nL2 Norms:")
    print(f"  Mean: {norms.mean():.4f}")
    print(f"  Std:  {norms.std():.4f}")
    print(f"  Min:  {norms.min():.4f}")
    print(f"  Max:  {norms.max():.4f}")

    # Per-dimension statistics
    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0)
    print(f"\nPer-dimension:")
    print(f"  Mean of means: {mean.mean():.6f}")
    print(f"  Std of means:  {mean.std():.6f}")
    print(f"  Mean of stds:  {std.mean():.6f}")
    print(f"  Std of stds:   {std.std():.6f}")

    # Cosine similarities between random pairs
    n_pairs = 1000
    idx1 = torch.randint(0, len(embeddings), (n_pairs,))
    idx2 = torch.randint(0, len(embeddings), (n_pairs,))
    cos_sims = torch.nn.functional.cosine_similarity(embeddings[idx1], embeddings[idx2])
    print(f"\nCosine similarity (random pairs):")
    print(f"  Mean: {cos_sims.mean():.4f}")
    print(f"  Std:  {cos_sims.std():.4f}")
    print(f"  Min:  {cos_sims.min():.4f}")
    print(f"  Max:  {cos_sims.max():.4f}")

    # Cosine similarity to mean embedding
    mean_emb = embeddings.mean(dim=0, keepdim=True)
    cos_to_mean = torch.nn.functional.cosine_similarity(embeddings, mean_emb.expand_as(embeddings))
    print(f"\nCosine similarity to mean embedding:")
    print(f"  Mean: {cos_to_mean.mean():.4f}")
    print(f"  Std:  {cos_to_mean.std():.4f}")
    print(f"  Min:  {cos_to_mean.min():.4f}")
    print(f"  Max:  {cos_to_mean.max():.4f}")

    # Principal components (how spread out is the data?)
    print(f"\nComputing PCA...")
    centered = embeddings - mean_emb
    # Use SVD on a subset for speed
    U, S, V = torch.svd(centered[:1000])
    variance_explained = (S ** 2) / (S ** 2).sum()
    print(f"  Top 10 singular values explain: {variance_explained[:10].sum()*100:.1f}% of variance")
    print(f"  Top 50 singular values explain: {variance_explained[:50].sum()*100:.1f}% of variance")
    print(f"  Top 100 singular values explain: {variance_explained[:100].sum()*100:.1f}% of variance")

    # Save statistics
    stats = {
        "n_sentences": len(all_sentences),
        "embedding_dim": embeddings.shape[1],
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std().item(),
        "mean_embedding": mean,
        "std_per_dim": std,
        "cos_sim_random_mean": cos_sims.mean().item(),
        "cos_sim_random_std": cos_sims.std().item(),
    }

    torch.save(stats, "results/tinystories_embedding_stats.pt")
    print(f"\nSaved statistics to results/tinystories_embedding_stats.pt")


if __name__ == "__main__":
    main()
