"""Experiments for finding embeddings that elicit specific next-sentence predictions."""

import torch
import torch.nn.functional as F

from prompt_interp.sonar import SonarWrapper
from prompt_interp.generator import SonarLLMGenerator
from prompt_interp.optimize import (
    project_to_norm,
    predict_next_embedding,
    decoder_ce_loss,
    tokenize_for_decoder,
    LILY_STORY,
)


def run_next_sentence_experiment(
    init_text: str,
    target_text: str,
    sonar: SonarWrapper,
    generator: SonarLLMGenerator,
    n_steps: int = 100,
    lr: float = 0.01,
    log_every: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Find z such that SONAR-LLM(z) decodes to target_text.

    Optimizes z using decoder cross-entropy loss so that the model's
    predicted next-sentence embedding decodes to the target sentence.
    """
    # Encode init and target
    init_emb = sonar.encode([init_text]).squeeze(0)
    target_emb = sonar.encode([target_text]).squeeze(0)
    target_tokens = tokenize_for_decoder(target_text, sonar)
    target_norm = init_emb.norm().item()

    # Optimization
    z = init_emb.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    trajectory = []

    for step in range(n_steps):
        optimizer.zero_grad()
        z_proj = project_to_norm(z, target_norm)

        # Forward: z -> SONAR-LLM -> pred_emb
        pred_emb = predict_next_embedding(z_proj, generator)

        # Decoder CE loss on pred_emb
        loss = decoder_ce_loss(pred_emb, target_tokens, sonar)
        cos_sim = F.cosine_similarity(pred_emb.unsqueeze(0), target_emb.unsqueeze(0)).item()

        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                decoded_z = sonar.decode(project_to_norm(z, target_norm).unsqueeze(0))[0]
                decoded_pred = sonar.decode(pred_emb.unsqueeze(0))[0]

            trajectory.append({
                "step": step,
                "loss": loss.item(),
                "similarity": cos_sim,
                "decoded_z": decoded_z,
                "decoded_pred": decoded_pred,
            })

            if verbose:
                print(f"Step {step:2d} | loss={loss.item():.3f} | sim={cos_sim:.3f}")
                print(f"  z: {decoded_z[:70]}")
                print(f"  pred: {decoded_pred[:70]}\n")

    # Final evaluation
    with torch.no_grad():
        z_final = project_to_norm(z, target_norm)
        pred_final = predict_next_embedding(z_final, generator)
        decoded_z_final = sonar.decode(z_final.unsqueeze(0))[0]
        decoded_pred_final = sonar.decode(pred_final.unsqueeze(0))[0]

    return {
        "init_text": init_text,
        "target_text": target_text,
        "final_z": decoded_z_final,
        "final_pred": decoded_pred_final,
        "final_loss": trajectory[-1]["loss"],
        "final_similarity": trajectory[-1]["similarity"],
        "trajectory": trajectory,
        "success": decoded_pred_final.strip() == target_text.strip(),
    }


def run_multi_init_experiments(
    target_sentences: list[str],
    init_sentences: list[str],
    sonar: SonarWrapper,
    generator: SonarLLMGenerator,
    n_steps: int = 10,
    lr: float = 0.01,
    verbose: bool = True,
) -> list[dict]:
    """Run experiments for each (target, init) pair."""
    results = []
    for ti, target in enumerate(target_sentences):
        for ii, init in enumerate(init_sentences):
            if verbose:
                print("=" * 70)
                print(f"Target {ti}: {target}")
                print(f"Init {ii}: {init}")
                print("=" * 70)

            result = run_next_sentence_experiment(
                init, target, sonar, generator, n_steps, lr, log_every=1, verbose=verbose
            )
            result.update({"target_idx": ti, "init_idx": ii})
            results.append(result)

            if verbose:
                print(f"SUCCESS: {result['success']}\n")
    return results

