"""Core optimization utilities for SONAR embedding space."""

import torch
import torch.nn.functional as F
from fairseq2.nn.batch_layout import BatchLayout

from prompt_interp.sonar import SonarWrapper
from prompt_interp.generator import SonarLLMGenerator


# Empirical statistics from TinyStories
EMBEDDING_NORM_MEAN = 0.207

# Target story for optimization experiments
LILY_STORY = """Once upon a time, there was a little girl named Lily. She loved to play outside with her toys. One day, she saw a big tree in the sky. She wanted to climb it, but it was too high. Lily asked her mom to help her climb the tree. Her mom said, "No, you can't climb the tree. It's too high." Lily was sad because she wanted to climb the tree. Later that day, Lily's mom told her that she could climb the tree."""


def project_to_norm(z: torch.Tensor, target_norm: float = EMBEDDING_NORM_MEAN) -> torch.Tensor:
    """Project embedding to have target L2 norm."""
    return z * (target_norm / (z.norm() + 1e-8))


def predict_next_embedding(z: torch.Tensor, generator: SonarLLMGenerator) -> torch.Tensor:
    """Pass z through SONAR-LLM to get predicted next sentence embedding."""
    hidden = generator.forward_proj(z.unsqueeze(0).unsqueeze(0))
    out = generator.llama_model(inputs_embeds=hidden, output_hidden_states=True)
    return generator.reverse_proj(out.hidden_states[-1].squeeze(0)).squeeze(0)


def decoder_ce_loss(
    embedding: torch.Tensor,
    target_tokens: torch.Tensor,
    sonar: SonarWrapper,
) -> torch.Tensor:
    """Compute cross-entropy loss for decoding embedding to target tokens."""
    device = embedding.device
    decoder = sonar.decoder.model.decoder

    source_seqs = embedding.unsqueeze(0).unsqueeze(1)
    source_layout = BatchLayout(shape=(1, 1), seq_lens=[1], device=device)
    target_batch = target_tokens.unsqueeze(0)
    target_layout = BatchLayout(
        shape=(1, target_tokens.size(0)),
        seq_lens=[target_tokens.size(0)],
        device=device,
    )

    logits = decoder(source_seqs, source_layout, target_batch, target_layout)
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        target_batch[:, 1:].reshape(-1),
    )


def tokenize_for_decoder(text: str, sonar: SonarWrapper) -> torch.Tensor:
    """Tokenize text for SONAR decoder."""
    device = torch.device(sonar.device) if isinstance(sonar.device, str) else sonar.device
    encoder_fn = sonar.decoder.tokenizer.create_encoder(mode="target")
    return encoder_fn(text).to(device)
