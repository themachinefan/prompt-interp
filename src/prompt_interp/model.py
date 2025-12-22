"""SONAR-LLM model architecture for training from scratch."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM

SONAR_DIM = 1024  # Fixed by SONAR encoder

# Model size presets: (hidden_size, intermediate_size, num_layers, num_heads, head_dim)
PRESETS = {
    "39M": (256, 1024, 6, 8, 32),
    "100M": (512, 2048, 8, 16, 32),
    "300M": (768, 3072, 12, 16, 48),
    "600M": (1024, 4096, 10, 16, 64),
    "900M": (1536, 6144, 17, 32, 64),
}


@dataclass
class SonarLLMConfig:
    """Configuration for SONAR-LLM model."""

    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    head_dim: int = 32

    @classmethod
    def from_preset(cls, name: str) -> "SonarLLMConfig":
        if name not in PRESETS:
            raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
        h, i, l, a, d = PRESETS[name]
        return cls(hidden_size=h, intermediate_size=i, num_hidden_layers=l, num_attention_heads=a, head_dim=d)


class SonarLLM(nn.Module):
    """
    Decoder-only transformer that predicts next sentence embeddings.

    Input: (batch, seq_len, 1024) SONAR embeddings
    Output: (batch, seq_len, 1024) predicted next embeddings
    """

    def __init__(self, config: SonarLLMConfig):
        super().__init__()
        self.config = config
        self.forward_proj = nn.Linear(SONAR_DIM, config.hidden_size)
        self.reverse_proj = nn.Linear(config.hidden_size, SONAR_DIM)

        llama_config = LlamaConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            hidden_act="silu",
            max_position_embeddings=131072,
            rms_norm_eps=1e-6,
            rope_theta=500000.0,
            vocab_size=1,  # Unused - we use continuous inputs
            use_cache=True,
            tie_word_embeddings=True,
            attention_bias=False,
            mlp_bias=False,
            rope_scaling={"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                          "original_max_position_embeddings": 8192, "rope_type": "llama3"},
        )
        self.llama = LlamaForCausalLM(llama_config)

    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass: predict next embedding for each position."""
        hidden = self.forward_proj(embeddings)
        out = self.llama(inputs_embeds=hidden, attention_mask=attention_mask, output_hidden_states=True)
        return self.reverse_proj(out.hidden_states[-1])

    @torch.no_grad()
    def predict_next(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict next embedding given sequence. Returns (batch, 1024)."""
        return self.forward(embeddings)[:, -1, :]
