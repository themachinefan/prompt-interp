"""SONAR-LLM wrapper compatible with fairseq2 0.6+."""

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import snapshot_download
from transformers import LlamaConfig, LlamaForCausalLM
from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)


class Projector(nn.Module):
    """Simple linear projection between embedding spaces."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class SONARLLMConfig:
    """Configuration for SONAR-LLM generation."""

    # Number of sentences to generate
    max_sentences: int = 8

    # EOS detection threshold (cosine similarity)
    eos_threshold: float = 0.98

    # Decoder settings
    decoder_max_len: int = 256


class SONARLLM(nn.Module):
    """SONAR-LLM: Sentence-level language model in SONAR embedding space.

    Architecture:
    - Takes sequence of SONAR sentence embeddings (1024-dim)
    - Projects to Llama hidden space (2048-dim)
    - Runs through Llama decoder
    - Projects back to SONAR space
    - Decodes via SONAR decoder
    """

    def __init__(
        self,
        llama_model: nn.Module,
        forward_proj: nn.Module,
        reverse_proj: nn.Module,
        sonar_encoder: TextToEmbeddingModelPipeline,
        sonar_decoder: EmbeddingToTextModelPipeline,
        device: torch.device,
        add_begin: bool = True,
    ):
        super().__init__()
        self.llama_model = llama_model
        self.forward_proj = forward_proj
        self.reverse_proj = reverse_proj
        self.sonar_encoder = sonar_encoder
        self.sonar_decoder = sonar_decoder
        self.device = device
        self.add_begin = add_begin

        # Pre-compute EOS embedding
        self.eos_emb = self.sonar_encoder.predict(
            ["End of sequence."], source_lang="eng_Latn"
        ).to(device)

    @torch.no_grad()
    def forward_hidden(self, emb_seq: torch.Tensor) -> torch.Tensor:
        """Get final hidden state from Llama given embedding sequence."""
        # Project to Llama hidden dim
        if emb_seq.ndim == 2:
            emb_seq = emb_seq.unsqueeze(0)  # Add batch dim
        proj = self.forward_proj(emb_seq)

        # Run through Llama
        out = self.llama_model(inputs_embeds=proj, output_hidden_states=True)
        hidden = out.hidden_states[-1]

        # Return last token's hidden state
        return hidden[0, -1, :]

    @torch.no_grad()
    def predict_next_embedding(
        self,
        emb_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next sentence embedding given context embeddings.

        Args:
            emb_seq: Input embedding sequence

        Returns:
            Predicted next embedding (unnormalized, as model was trained)
        """
        # Get Llama hidden state
        hidden = self.forward_hidden(emb_seq)

        # Project back to SONAR space (deterministic, no noise - matches official inference)
        z = self.reverse_proj(hidden.unsqueeze(0))

        return z.squeeze(0)

    @torch.no_grad()
    def generate(
        self,
        prefix: str,
        config: Optional[SONARLLMConfig] = None,
    ) -> str:
        """Generate text continuation from prefix.

        Args:
            prefix: Input text to continue from
            config: Generation configuration

        Returns:
            Generated text (including prefix)
        """
        if config is None:
            config = SONARLLMConfig()

        # Sentence-tokenize the prefix
        sentences = sent_tokenize(prefix)
        if self.add_begin:
            sentences = ["Begin of text."] + sentences
        if len(sentences) == 0:
            sentences = [prefix.strip()]

        # Encode prefix sentences
        emb_seq = self.sonar_encoder.predict(
            sentences, source_lang="eng_Latn"
        ).to(self.device)

        generated_sentences = list(sentences)

        for _ in range(config.max_sentences):
            # Predict next embedding (deterministic)
            next_emb = self.predict_next_embedding(emb_seq)

            # Decode to text
            decoded = self.sonar_decoder.predict(
                next_emb.unsqueeze(0),
                target_lang="eng_Latn",
                max_seq_len=config.decoder_max_len,
            )[0]

            # Re-encode decoded text (this is what we use for context and EOS check)
            new_emb = self.sonar_encoder.predict(
                [decoded], source_lang="eng_Latn"
            ).to(self.device)

            # Check for EOS using re-encoded embedding (matches official inference)
            cos_sim = F.cosine_similarity(new_emb, self.eos_emb, dim=1).item()
            if cos_sim > config.eos_threshold:
                break

            generated_sentences.append(decoded)
            emb_seq = torch.cat([emb_seq, new_emb], dim=0)

        # Join sentences
        result = " ".join(generated_sentences)
        if self.add_begin:
            result = result[len("Begin of text. "):]

        return result

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "raxtemur/sonar-llm-1.3b",
        device: Optional[torch.device] = None,
    ) -> "SONARLLM":
        """Load SONAR-LLM from HuggingFace.

        Args:
            model_id: HuggingFace model ID
            device: Device to load model on

        Returns:
            Loaded SONARLLM instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Download model
        model_path = snapshot_download(model_id)

        # Load config
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)

        # Create Llama model
        llama_cfg_dict = config.get("llama_config", {})
        llama_cfg = LlamaConfig(**llama_cfg_dict)
        llama_model = LlamaForCausalLM(llama_cfg).to(device).eval()

        # Create projectors
        hidden_size = llama_cfg.hidden_size  # 2048
        embed_dim = config.get("embed_dim", 1024)
        forward_proj = Projector(embed_dim, hidden_size).to(device)
        reverse_proj = Projector(hidden_size, embed_dim).to(device)

        # Load SONAR encoder/decoder
        sonar_encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        )
        sonar_decoder = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=device,
        )

        # Create model instance
        model = cls(
            llama_model=llama_model,
            forward_proj=forward_proj,
            reverse_proj=reverse_proj,
            sonar_encoder=sonar_encoder,
            sonar_decoder=sonar_decoder,
            device=device,
            add_begin=config.get("add_begin", True),
        )

        # Load weights
        ckpt_path = os.path.join(model_path, "pytorch_model.bin")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        state = state.get("model_state_dict", state)

        # Load state dict (may have missing keys for SONAR models, that's ok)
        model.load_state_dict(state, strict=False)

        return model.eval()


def main():
    """Test SONAR-LLM loading and generation."""
    print("Loading SONAR-LLM 1.3B...")
    model = SONARLLM.from_pretrained()
    print("Model loaded!")

    print(f"\nDevice: {model.device}")
    print(f"Llama hidden size: {model.llama_model.config.hidden_size}")
    print(f"Llama layers: {model.llama_model.config.num_hidden_layers}")

    # Test generation (deterministic, matching official inference)
    print("\n" + "="*80)
    print("Testing summarization (from README example)...")
    print("="*80)

    config = SONARLLMConfig(
        max_sentences=1,  # Single sentence summary
    )

    # Example from HuggingFace README
    prefix = "Petya loves Masha. Masha loves Gosha. Gosha loves Petya. Text summarization in one sentence only."
    print(f"\nInput: '{prefix}'")

    result = model.generate(prefix, config)
    print(f"\nSummary:\n{result}")

    # Test with other prompts
    print("\n" + "="*80)
    print("Testing other prompts...")
    print("="*80)

    config2 = SONARLLMConfig(
        max_sentences=3,
    )

    prompts = [
        "The quick brown fox jumps over the lazy dog. The dog was not happy. What happened next?",
        "Machine learning has revolutionized many industries. Neural networks are particularly powerful.",
        "Paris is the capital of France. It is known for the Eiffel Tower.",
    ]

    for prompt in prompts:
        print(f"\n\nInput: '{prompt}'")
        result = model.generate(prompt, config2)
        print(f"Output: '{result}'")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == "__main__":
    main()
