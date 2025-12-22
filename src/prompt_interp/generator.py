"""Pretrained SONAR-LLM generator for text continuation."""

from dataclasses import dataclass
from typing import Optional
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk


@dataclass
class GenerationConfig:
    """Generation configuration. Use temperature=0 for greedy (recommended)."""

    temperature: float = 0.0  # 0 = greedy (recommended)
    max_sentences: int = 32
    eos_threshold: float = 0.98
    sentence_beam_size: int = 1
    latent_samples_per_step: int = 1
    decoder_beam_size: int = 5
    decoder_max_len: int = 256


class _Proj(nn.Module):
    """Wrapper to match checkpoint key structure (forward_proj.linear.weight)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _ScoringDecoder(nn.Module):
    """SONAR decoder that returns sentence log-probabilities."""

    def __init__(self, device: torch.device):
        super().__init__()
        from fairseq2.models import load_model
        from fairseq2.data.tokenizers import load_tokenizer
        from sonar.models.sonar_translation import SonarEncoderDecoderModel
        from sonar.models.sonar_translation.model import DummyEncoderModel

        self.device = device
        self.decoder = load_model("text_sonar_basic_decoder", device=device)
        self.tokenizer = load_tokenizer("text_sonar_basic_decoder")
        self.model = SonarEncoderDecoderModel(DummyEncoderModel(), self.decoder).eval()

    @torch.inference_mode()
    def predict(self, embeddings: torch.Tensor, beam_size: int = 5, max_len: int = 256) -> tuple[list[str], list[float]]:
        from fairseq2.generation.beam_search.generator import BeamSearchSeq2SeqGenerator
        from fairseq2.generation.text import SequenceToTextConverter
        from fairseq2.nn import BatchLayout

        gen = BeamSearchSeq2SeqGenerator(self.model, self.tokenizer.vocab_info, beam_size=beam_size, max_seq_len=max_len)
        conv = SequenceToTextConverter(gen, self.tokenizer, task="translation", target_lang="eng_Latn")

        inputs = embeddings.to(self.device)
        texts, out = conv.batch_convert(inputs, BatchLayout.of(inputs))
        scores = [float(h[0].score) if h and h[0].score else 0.0 for h in out.hypotheses]
        return texts, scores


class SonarLLMGenerator(nn.Module):
    """
    Pretrained SONAR-LLM generator.

    Usage:
        gen = SonarLLMGenerator.from_pretrained("raxtemur/sonar-llm-900m")
        output = gen.generate("Once upon a time", GenerationConfig(temperature=0))
    """

    def __init__(self, llama: nn.Module, fwd_proj: nn.Module, rev_proj: nn.Module,
                 decoder: _ScoringDecoder, encoder, device: torch.device, add_begin: bool = False):
        super().__init__()
        self.llama_model = llama
        self.forward_proj = fwd_proj
        self.reverse_proj = rev_proj
        self.sonar_decoder = decoder
        self.encoder = encoder
        self.device = device
        self.add_begin = add_begin

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "raxtemur/sonar-llm-900m",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Load pretrained model from HuggingFace.

        Args:
            repo_id: HuggingFace model repository ID
            device: Device to load model on
            dtype: Model dtype (e.g., torch.float16 for half precision)
        """
        from huggingface_hub import snapshot_download
        from transformers import LlamaConfig, LlamaForCausalLM
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = snapshot_download(repo_id)

        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg = json.load(f)

        state = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location=device, weights_only=True)
        state = state.get("model_state_dict", state)

        # Infer vocab_size from checkpoint if needed
        llama_cfg = cfg.get("llama_config", {})
        if "vocab_size" not in llama_cfg:
            for k, v in state.items():
                if "embed_tokens.weight" in k:
                    llama_cfg["vocab_size"] = v.shape[0]
                    break

        llama = LlamaForCausalLM(LlamaConfig(**llama_cfg)).to(device).eval()
        embed_dim, hidden_size = cfg.get("embed_dim", 1024), llama_cfg["hidden_size"]

        fwd = _Proj(embed_dim, hidden_size).to(device)
        rev = _Proj(hidden_size, embed_dim).to(device)
        encoder = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                                tokenizer="text_sonar_basic_encoder", device=device)
        decoder = _ScoringDecoder(device)

        gen = cls(llama, fwd, rev, decoder, encoder, device, cfg.get("add_begin", False))
        gen.load_state_dict(state, strict=False)

        # Convert to specified dtype if provided
        if dtype is not None:
            gen = gen.to(dtype)

        return gen.eval()

    @torch.no_grad()
    def generate(self, prefix: str, cfg: Optional[GenerationConfig] = None) -> str:
        """Generate text continuation. Returns full text including prefix."""
        cfg = cfg or GenerationConfig()

        # Encode prefix
        sents = nltk.sent_tokenize(prefix)
        if self.add_begin:
            sents = ["Begin of text."] + sents
        if not sents:
            sents = [prefix.strip()]

        emb = self.encoder.predict(sents, source_lang="eng_Latn").to(self.device).clone()
        eos_emb = self.encoder.predict(["End of sequence."], source_lang="eng_Latn").to(self.device).clone()

        # Beam state: (sentences, embeddings, score)
        beams = [(sents[:], emb, 0.0)]

        for _ in range(cfg.max_sentences):
            candidates = []
            for hist_sents, hist_emb, score in beams:
                candidates.extend(self._expand(hist_sents, hist_emb, score, eos_emb, cfg))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[2], reverse=True)
            beams = candidates[:cfg.sentence_beam_size]

            # Check if all beams close to EOS
            if all(F.cosine_similarity(b[1][-1:], eos_emb, dim=1).item() > cfg.eos_threshold for b in beams):
                break

        result = " ".join(max(beams, key=lambda x: x[2])[0])
        return result[len("Begin of text. "):] if self.add_begin else result

    def _expand(self, sents: list, emb: torch.Tensor, score: float, eos_emb: torch.Tensor, cfg: GenerationConfig) -> list:
        """Expand beam state into candidates."""
        # Forward through model
        proj = self.forward_proj(emb.unsqueeze(0))
        hidden = self.llama_model(inputs_embeds=proj, output_hidden_states=True).hidden_states[-1][0, -1, :]

        out = []
        for _ in range(cfg.latent_samples_per_step):
            # Add noise if temperature > 0
            if cfg.temperature > 0:
                noise_dir = torch.randn_like(hidden)
                noise_dir = noise_dir / (noise_dir.norm() + 1e-12)
                rms = torch.sqrt((hidden ** 2).mean()).clamp(min=1e-12)
                hidden_perturbed = hidden + noise_dir * rms * cfg.temperature
            else:
                hidden_perturbed = hidden

            z = self.reverse_proj(hidden_perturbed.unsqueeze(0))
            texts, scores = self.sonar_decoder.predict(z, cfg.decoder_beam_size, cfg.decoder_max_len)

            # Re-encode and add to candidates
            new_emb = self.encoder.predict([texts[0]], source_lang="eng_Latn").to(self.device).clone()
            out.append((sents + [texts[0]], torch.cat([emb, new_emb], dim=0), score + scores[0]))

        return out
