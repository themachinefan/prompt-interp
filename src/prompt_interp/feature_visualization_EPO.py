#%%
"""
Feature visualization using Evolutionary Prompt Optimization (EPO).

This module implements EPO for finding fluent prompts that maximally activate
specific neurons in SONAR-LLM. Based on "Fluent Dreaming for Language Models"
(Thompson et al., 2024).

Key differences from the embedding-space optimization in feature_visualization.py:
- Optimizes in discrete token space using GCG-style gradient-guided search
- Adds fluency regularization to produce human-readable prompts
- Maintains a Pareto frontier of fluency vs activation tradeoffs
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from fairseq2.nn import BatchLayout

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from prompt_interp import REPO_ROOT
from prompt_interp.sonar_wrapper import SonarWrapper
from prompt_interp.generator import SonarLLMGenerator
from prompt_interp.optimize import predict_next_embedding

# Use same directory as feature_visualization.py for compatibility with visualize_feature_results.py
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "feature_vis"


class FluencyModel:
    """
    GPT-2 based fluency scorer for computing cross-entropy of text.

    Lower cross-entropy = more fluent/natural text.
    """

    def __init__(self, model_name: str = "gpt2", device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).eval()

        # Freeze model
        for p in self.model.parameters():
            p.requires_grad = False

        # Set pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def compute_cross_entropy(self, texts: list[str]) -> torch.Tensor:
        """
        Compute mean per-token cross-entropy for each text.

        Args:
            texts: List of text strings

        Returns:
            Cross-entropy scores, shape (batch,). Lower = more fluent.
        """
        # Tokenize
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch, seq_len, vocab)

        # Compute per-token cross-entropy
        # Shift so we predict next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Cross-entropy per token
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        ce_per_token = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)  # (batch, seq_len-1)

        # Mask out padding and compute mean
        ce_per_token = ce_per_token * shift_mask
        seq_lengths = shift_mask.sum(dim=1).clamp(min=1)
        mean_ce = ce_per_token.sum(dim=1) / seq_lengths

        return mean_ce


@dataclass
class EPOConfig:
    """Configuration for EPO optimization."""

    # Population and search
    population_size: int = 8  # M: number of prompts in population
    n_children: int = 32  # r: children generated per population member
    top_k: int = 256  # k: top-k tokens to consider per position

    # Optimization
    n_iterations: int = 300  # T: total iterations
    restart_every: int = 30  # T_restart: iterations between restarts

    # Fluency regularization
    use_fluency: bool = True  # Enable fluency scoring with GPT-2
    fluency_model: str = "gpt2"  # GPT-2 model for fluency scoring
    lambda_min: float = 0.1  # Minimum fluency weight
    lambda_max: float = 10.0  # Maximum fluency weight
    lambda_restart_min: float = 0.667
    lambda_restart_max: float = 6.0

    # Prompt settings
    prompt_length: int = 12  # Fixed prompt length in tokens

    # Logging
    log_every: int = 10
    verbose: bool = True


class ActivationCapture:
    """Hook to capture activations from a specific layer in the model."""

    def __init__(self):
        self.activation: torch.Tensor | None = None
        self.handle = None

    def hook_fn(self, module, input, output):
        """Capture the output activation."""
        self.activation = output

    def register(self, module: torch.nn.Module):
        """Register the hook on a module."""
        self.handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        """Remove the hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def get_mlp_module(generator: SonarLLMGenerator, layer_idx: int) -> torch.nn.Module:
    """Get the MLP module for a specific layer in the LLaMA model."""
    return generator.llama_model.model.layers[layer_idx].mlp


class SonarEncoderWithGradients:
    """
    Wrapper for SONAR encoder that enables gradient computation through token embeddings.

    This bypasses the standard encode() pipeline to allow backpropagation through
    one-hot encoded tokens to the embedding layer, enabling GCG-style optimization.
    """

    def __init__(self, sonar_wrapper: SonarWrapper):
        self.sonar_wrapper = sonar_wrapper
        self.device = sonar_wrapper.device

        # Get the encoder model and tokenizer from the pipeline
        self.encoder_model = sonar_wrapper.encoder.model
        self.tokenizer = sonar_wrapper.encoder.tokenizer
        self.text_encoder = self.tokenizer.create_encoder(lang='eng_Latn')

        # Get embedding layer
        self.embed_weight = self.encoder_model.encoder_frontend.embed.weight
        self.vocab_size = self.tokenizer.vocab_info.size
        self.embed_dim = self.embed_weight.shape[1]

        # Special token indices
        self.pad_idx = self.tokenizer.vocab_info.pad_idx
        self.bos_idx = self.tokenizer.vocab_info.bos_idx
        self.eos_idx = self.tokenizer.vocab_info.eos_idx
        self.unk_idx = self.tokenizer.vocab_info.unk_idx

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to token IDs."""
        return self.text_encoder(text).to(self.device)

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        text_decoder = self.tokenizer.create_decoder()
        return text_decoder(token_ids.cpu())

    def encode_with_gradients(
        self,
        token_ids: torch.Tensor,
        one_hot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode tokens to SONAR sentence embedding with gradient support.

        Args:
            token_ids: Token IDs, shape (batch, seq_len)
            one_hot: Optional pre-computed one-hot encoding with requires_grad=True.
                     If None, uses token_ids directly (no gradients through tokens).

        Returns:
            Sentence embeddings, shape (batch, 1024)
        """
        batch_size, seq_len = token_ids.shape

        # Get embeddings - either through one-hot (differentiable) or lookup
        if one_hot is not None:
            # Differentiable path: one_hot @ embedding_weight
            embeddings = one_hot @ self.embed_weight  # (batch, seq_len, embed_dim)
        else:
            # Non-differentiable path: standard lookup
            embeddings = self.encoder_model.encoder_frontend.embed(token_ids)

        # Apply position encoding (using the frontend's pos_encoder)
        frontend = self.encoder_model.encoder_frontend
        seqs_layout = BatchLayout.of(token_ids)

        # Scale embeddings if frontend uses scaling (scale = sqrt(embed_dim) or similar)
        if frontend.scale != 1.0:
            embeddings = embeddings * frontend.scale

        # Add positional encodings
        if frontend.pos_encoder is not None:
            embeddings = frontend.pos_encoder(embeddings, seqs_layout, state_bag=None)

        # Apply dropout (in eval mode this is a no-op, but we still call it for consistency)
        if frontend.dropout is not None:
            embeddings = frontend.dropout(embeddings)

        # Forward through transformer encoder
        encoded_seqs = self.encoder_model.encoder(embeddings, seqs_layout)

        # Apply final layer norm
        if self.encoder_model.layer_norm is not None:
            encoded_seqs = self.encoder_model.layer_norm(encoded_seqs)

        # Pool to get sentence embedding (LAST pooling - take EOS position)
        sentence_embeddings = encoded_seqs[:, -1, :]  # (batch, 1024)

        return sentence_embeddings


def compute_neuron_activation(
    sentence_embeddings: torch.Tensor,
    generator: SonarLLMGenerator,
    layer_idx: int,
    neuron_idx: int,
    capture: ActivationCapture,
    return_layer_mean: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Forward sentence embeddings through SONAR-LLM and get neuron activation.

    Args:
        sentence_embeddings: SONAR embeddings, shape (batch, 1024)
        generator: SONAR-LLM generator
        layer_idx: Which layer to hook
        neuron_idx: Which neuron to target
        capture: ActivationCapture with hook registered
        return_layer_mean: If True, also return mean activation

    Returns:
        Neuron activation, shape (batch,)
        Optionally also layer mean activation, shape (batch,)
    """
    # Add sequence dimension: (batch, 1024) -> (batch, 1, 1024)
    z = sentence_embeddings.unsqueeze(1)

    # Project to LLaMA space and forward
    proj = generator.forward_proj(z)  # (batch, 1, hidden)
    _ = generator.llama_model(inputs_embeds=proj, output_hidden_states=False)

    # Get captured MLP activation
    mlp_output = capture.activation  # (batch, seq, hidden_size)

    # Take the last position's activation for the target neuron
    neuron_act = mlp_output[:, -1, neuron_idx]  # (batch,)

    if return_layer_mean:
        layer_mean = mlp_output[:, -1, :].mean(dim=-1)  # (batch,)
        return neuron_act, layer_mean

    return neuron_act


def compute_prediction(
    sentence_embedding: torch.Tensor,
    generator: SonarLLMGenerator,
    sonar_wrapper: SonarWrapper,
) -> str:
    """
    Compute the SONAR-LLM prediction for a given sentence embedding.

    Args:
        sentence_embedding: SONAR embedding, shape (1024,) or (1, 1024)
        generator: SONAR-LLM generator
        sonar_wrapper: SONAR wrapper for decoding

    Returns:
        Decoded prediction text
    """
    with torch.no_grad():
        # Ensure correct shape: (1, 1, 1024)
        if sentence_embedding.dim() == 1:
            z = sentence_embedding.unsqueeze(0).unsqueeze(0)
        elif sentence_embedding.dim() == 2:
            z = sentence_embedding.unsqueeze(1)
        else:
            z = sentence_embedding

        # Get prediction embedding
        pred_emb = predict_next_embedding(z, generator)[:, -1:, :]  # (1, 1, 1024)

        # Decode
        decoded_pred = sonar_wrapper.decode(pred_emb.squeeze(1))[0]

    return decoded_pred


def compute_token_gradients(
    token_ids: torch.Tensor,
    encoder: SonarEncoderWithGradients,
    generator: SonarLLMGenerator,
    layer_idx: int,
    neuron_idx: int,
    capture: ActivationCapture,
    fluency_lambda: float = 0.0,
    fluency_model: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Compute gradients w.r.t. one-hot token encoding for GCG-style optimization.

    Args:
        token_ids: Token IDs, shape (batch, seq_len)
        encoder: SONAR encoder with gradient support
        generator: SONAR-LLM generator
        layer_idx: Target layer
        neuron_idx: Target neuron
        capture: Activation capture hook
        fluency_lambda: Weight for fluency regularization
        fluency_model: Optional model for computing cross-entropy (not implemented yet)

    Returns:
        Gradients w.r.t. one-hot encoding, shape (batch, seq_len, vocab_size)
    """
    batch_size, seq_len = token_ids.shape

    # Create one-hot encoding with gradient tracking
    one_hot = F.one_hot(token_ids, num_classes=encoder.vocab_size).float()
    one_hot.requires_grad_(True)

    # Forward through SONAR encoder
    sentence_embeddings = encoder.encode_with_gradients(token_ids, one_hot)

    # Forward through SONAR-LLM to get neuron activation
    neuron_act, layer_mean = compute_neuron_activation(
        sentence_embeddings, generator, layer_idx, neuron_idx, capture, return_layer_mean=True
    )

    # Objective: maximize (neuron_act - layer_mean)
    # For gradient ascent, we minimize the negative
    relative_activation = (neuron_act - layer_mean).mean()
    loss = -relative_activation

    # TODO: Add fluency regularization when fluency_model is provided
    # if fluency_lambda > 0 and fluency_model is not None:
    #     cross_entropy = compute_cross_entropy(token_ids, fluency_model)
    #     loss += fluency_lambda * cross_entropy

    # Backward
    loss.backward()

    return one_hot.grad  # (batch, seq_len, vocab_size)


def select_top_k_tokens(
    gradients: torch.Tensor,
    k: int,
    exclude_tokens: Optional[set[int]] = None,
) -> torch.Tensor:
    """
    Select top-k tokens per position based on gradient magnitude.

    For maximization, we want tokens with the most negative gradients
    (since we minimized negative activation).

    Args:
        gradients: Gradients w.r.t. one-hot, shape (batch, seq_len, vocab_size)
        k: Number of top tokens to select per position
        exclude_tokens: Token indices to exclude (e.g., special tokens)

    Returns:
        Top-k token indices per position, shape (batch, seq_len, k)
    """
    # For maximizing activation, we want tokens with most negative gradients
    # (lower gradient = more increase in activation when selected)
    scores = -gradients  # Negate so higher is better

    # Exclude special tokens by setting their scores to -inf
    if exclude_tokens:
        for tok_idx in exclude_tokens:
            scores[:, :, tok_idx] = float('-inf')

    # Get top-k per position
    _, top_k_indices = scores.topk(k, dim=-1)  # (batch, seq_len, k)

    return top_k_indices


def generate_children(
    parent_tokens: torch.Tensor,
    top_k_tokens: torch.Tensor,
    n_children: int,
) -> torch.Tensor:
    """
    Generate child prompts by random single-token swaps.

    Args:
        parent_tokens: Parent token IDs, shape (seq_len,)
        top_k_tokens: Top-k tokens per position, shape (seq_len, k)
        n_children: Number of children to generate

    Returns:
        Child token IDs, shape (n_children, seq_len)
    """
    seq_len = parent_tokens.shape[0]
    k = top_k_tokens.shape[1]
    device = parent_tokens.device

    # Start with copies of parent
    children = parent_tokens.unsqueeze(0).expand(n_children, -1).clone()

    # For each child, randomly select a position and a token from top-k
    positions = torch.randint(0, seq_len, (n_children,), device=device)
    token_choices = torch.randint(0, k, (n_children,), device=device)

    # Apply swaps
    for i in range(n_children):
        pos = positions[i]
        new_token = top_k_tokens[pos, token_choices[i]]
        children[i, pos] = new_token

    return children


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""
    token_ids: torch.Tensor
    text: str
    activation: float
    activation_diff: float
    cross_entropy: float = 0.0

    def dominates(self, other: 'ParetoPoint') -> bool:
        """Check if this point dominates another (higher activation AND lower cross-entropy)."""
        return (self.activation_diff >= other.activation_diff and
                self.cross_entropy <= other.cross_entropy and
                (self.activation_diff > other.activation_diff or
                 self.cross_entropy < other.cross_entropy))


class EPOPopulation:
    """
    Manages the EPO population with Pareto frontier tracking.
    """

    def __init__(
        self,
        config: EPOConfig,
        encoder: SonarEncoderWithGradients,
        device: torch.device,
    ):
        self.config = config
        self.encoder = encoder
        self.device = device

        # Lambda values for each population slot (log-spaced)
        log_lambdas = torch.linspace(
            torch.log(torch.tensor(config.lambda_min)),
            torch.log(torch.tensor(config.lambda_max)),
            config.population_size,
        )
        self.lambdas = torch.exp(log_lambdas).tolist()

        # Population: list of (token_ids, activation, cross_entropy)
        self.population: list[ParetoPoint] = []

        # Best points on Pareto frontier
        self.pareto_frontier: list[ParetoPoint] = []

    def initialize_random(self, prompt_length: int):
        """Initialize population with random tokens."""
        for _ in range(self.config.population_size):
            # Random tokens (avoiding special tokens)
            token_ids = torch.randint(
                4, self.encoder.vocab_size,  # Skip special tokens 0-3
                (prompt_length,),
                device=self.device,
            )
            text = self.encoder.decode_tokens(token_ids)
            self.population.append(ParetoPoint(
                token_ids=token_ids,
                text=text,
                activation=float('-inf'),
                activation_diff=float('-inf'),
            ))

    def initialize_from_texts(self, texts: list[str], prompt_length: int):
        """Initialize population from seed texts, padding/truncating to fixed length."""
        for text in texts[:self.config.population_size]:
            token_ids = self.encoder.tokenize(text)

            # Pad or truncate to fixed length
            if len(token_ids) < prompt_length:
                # Pad with random tokens
                padding = torch.randint(
                    4, self.encoder.vocab_size,
                    (prompt_length - len(token_ids),),
                    device=self.device,
                )
                token_ids = torch.cat([token_ids[:-1], padding, token_ids[-1:]])  # Keep EOS at end
            elif len(token_ids) > prompt_length:
                # Truncate but keep EOS
                token_ids = torch.cat([token_ids[:prompt_length-1], token_ids[-1:]])

            text = self.encoder.decode_tokens(token_ids)
            self.population.append(ParetoPoint(
                token_ids=token_ids,
                text=text,
                activation=float('-inf'),
                activation_diff=float('-inf'),
            ))

        # Fill remaining slots with random if needed
        while len(self.population) < self.config.population_size:
            token_ids = torch.randint(
                4, self.encoder.vocab_size,
                (prompt_length,),
                device=self.device,
            )
            text = self.encoder.decode_tokens(token_ids)
            self.population.append(ParetoPoint(
                token_ids=token_ids,
                text=text,
                activation=float('-inf'),
                activation_diff=float('-inf'),
            ))

    def compute_objective(self, point: ParetoPoint, lambda_val: float) -> float:
        """Compute EPO objective: activation - lambda * cross_entropy."""
        return point.activation_diff - lambda_val * point.cross_entropy

    def select_best_for_lambda(
        self,
        candidates: list[ParetoPoint],
        lambda_val: float,
    ) -> ParetoPoint:
        """Select the best candidate for a given lambda value."""
        best = max(candidates, key=lambda p: self.compute_objective(p, lambda_val))
        return best

    def update_population(self, all_candidates: list[ParetoPoint]):
        """Update population by selecting best candidate for each lambda."""
        new_population = []
        for lambda_val in self.lambdas:
            best = self.select_best_for_lambda(all_candidates, lambda_val)
            new_population.append(best)
        self.population = new_population

        # Update Pareto frontier
        self._update_pareto_frontier(all_candidates)

    def _update_pareto_frontier(self, candidates: list[ParetoPoint]):
        """Update the Pareto frontier with new candidates."""
        all_points = self.pareto_frontier + candidates

        # Find non-dominated points
        new_frontier = []
        for point in all_points:
            dominated = False
            for other in all_points:
                if other is not point and other.dominates(point):
                    dominated = True
                    break
            if not dominated:
                new_frontier.append(point)

        # Deduplicate by text
        seen_texts = set()
        unique_frontier = []
        for point in new_frontier:
            if point.text not in seen_texts:
                seen_texts.add(point.text)
                unique_frontier.append(point)

        self.pareto_frontier = unique_frontier

    def restart(self):
        """Perform a restart: keep one population member, reinitialize others."""
        import random

        # Pick a random lambda in the restart range
        lambda_r = random.uniform(
            self.config.lambda_restart_min,
            self.config.lambda_restart_max,
        )

        # Keep the best for this lambda
        survivor = self.select_best_for_lambda(self.population, lambda_r)

        # Reinitialize population with variations of survivor
        self.population = [survivor]
        for _ in range(self.config.population_size - 1):
            # Add slightly mutated copies
            new_tokens = survivor.token_ids.clone()
            # Random single token swap
            pos = torch.randint(0, len(new_tokens), (1,)).item()
            new_tokens[pos] = torch.randint(4, self.encoder.vocab_size, (1,), device=self.device).item()
            text = self.encoder.decode_tokens(new_tokens)
            self.population.append(ParetoPoint(
                token_ids=new_tokens,
                text=text,
                activation=float('-inf'),
                activation_diff=float('-inf'),
            ))


def run_epo_feature_visualization(
    layer_idx: int,
    neuron_idx: int,
    sonar_wrapper: SonarWrapper,
    generator: SonarLLMGenerator,
    config: Optional[EPOConfig] = None,
    seed_texts: Optional[list[str]] = None,
    output_dir: Optional[str | Path] = DEFAULT_OUTPUT_DIR,
) -> dict:
    """
    Run EPO-based feature visualization to find fluent prompts that maximize neuron activation.

    Args:
        layer_idx: Target transformer layer
        neuron_idx: Target neuron in MLP
        sonar_wrapper: SONAR encoder/decoder wrapper
        generator: SONAR-LLM generator
        config: EPO configuration (uses defaults if None)
        seed_texts: Optional seed texts to initialize population
        output_dir: Directory to save results

    Returns:
        Dict with results including Pareto frontier
    """
    config = config or EPOConfig()
    device = sonar_wrapper.device

    # Set up gradient-enabled encoder
    encoder = SonarEncoderWithGradients(sonar_wrapper)

    # Set up fluency model if enabled
    fluency_model = None
    if config.use_fluency:
        fluency_model = FluencyModel(model_name=config.fluency_model, device=device)

    # Set up activation capture
    capture = ActivationCapture()
    mlp_module = get_mlp_module(generator, layer_idx)
    capture.register(mlp_module)

    # Special tokens to exclude from token swaps
    exclude_tokens = {encoder.pad_idx, encoder.bos_idx, encoder.eos_idx, encoder.unk_idx}

    try:
        # Initialize population
        population = EPOPopulation(config, encoder, device)
        if seed_texts:
            population.initialize_from_texts(seed_texts, config.prompt_length)
        else:
            population.initialize_random(config.prompt_length)

        # Determine init_text for logging
        if seed_texts:
            init_text_display = seed_texts[0] if len(seed_texts[0]) <= 50 else seed_texts[0][:50] + "..."
        else:
            init_text_display = "(random initialization)"

        if config.verbose:
            print("=" * 70)
            print("FEATURE VISUALIZATION (EPO):")
            print(f"  Layer: {layer_idx}")
            print(f"  Neuron: {neuron_idx}")
            print(f"  Init: \"{init_text_display}\"")
            print(f"  Method: EPO (population={config.population_size}, top_k={config.top_k})")
            print(f"  Fluency: {'enabled (' + config.fluency_model + ')' if config.use_fluency else 'disabled'}")
            print("=" * 70 + "\n")

        trajectory = []

        for iteration in range(config.n_iterations):
            all_candidates = []

            # Process each population member
            for pop_idx, member in enumerate(population.population):
                # Stack token IDs for batch processing
                token_ids = member.token_ids.unsqueeze(0)  # (1, seq_len)

                # Compute gradients
                gradients = compute_token_gradients(
                    token_ids,
                    encoder,
                    generator,
                    layer_idx,
                    neuron_idx,
                    capture,
                    fluency_lambda=0.0,  # TODO: Add fluency
                )

                # Select top-k tokens
                top_k = select_top_k_tokens(
                    gradients,
                    config.top_k,
                    exclude_tokens,
                )  # (1, seq_len, k)

                # Generate children
                children = generate_children(
                    member.token_ids,
                    top_k[0],  # Remove batch dim
                    config.n_children,
                )  # (n_children, seq_len)

                # Evaluate children
                with torch.no_grad():
                    # Batch encode children
                    child_embeddings = encoder.encode_with_gradients(children, one_hot=None)

                    # Get activations
                    activations, layer_means = compute_neuron_activation(
                        child_embeddings,
                        generator,
                        layer_idx,
                        neuron_idx,
                        capture,
                        return_layer_mean=True,
                    )

                    activation_diffs = activations - layer_means

                # Decode all children texts
                child_texts = [encoder.decode_tokens(children[i]) for i in range(config.n_children)]

                # Compute cross-entropy for all children if fluency is enabled
                if fluency_model is not None:
                    cross_entropies = fluency_model.compute_cross_entropy(child_texts)
                else:
                    cross_entropies = torch.zeros(config.n_children, device=device)

                # Create ParetoPoints for children
                for i in range(config.n_children):
                    all_candidates.append(ParetoPoint(
                        token_ids=children[i].clone(),
                        text=child_texts[i],
                        activation=activations[i].item(),
                        activation_diff=activation_diffs[i].item(),
                        cross_entropy=cross_entropies[i].item(),
                    ))

                # Also include parent in candidates
                with torch.no_grad():
                    parent_emb = encoder.encode_with_gradients(token_ids, one_hot=None)
                    parent_act, parent_mean = compute_neuron_activation(
                        parent_emb, generator, layer_idx, neuron_idx, capture, return_layer_mean=True
                    )
                    member.activation = parent_act[0].item()
                    member.activation_diff = (parent_act[0] - parent_mean[0]).item()
                    if fluency_model is not None:
                        member.cross_entropy = fluency_model.compute_cross_entropy([member.text])[0].item()
                all_candidates.append(member)

            # Update population
            population.update_population(all_candidates)

            # Periodic restart
            if config.restart_every > 0 and (iteration + 1) % config.restart_every == 0:
                if config.verbose:
                    print(f"  [Restart at iteration {iteration + 1}]")
                population.restart()

            # Logging - use format compatible with visualize_feature_results.py
            if iteration % config.log_every == 0 or iteration == config.n_iterations - 1:
                best = max(population.population, key=lambda p: p.activation_diff)

                # Compute prediction for the best prompt
                with torch.no_grad():
                    best_emb = encoder.encode_with_gradients(best.token_ids.unsqueeze(0), one_hot=None)
                    decoded_pred = compute_prediction(best_emb[0], generator, sonar_wrapper)

                    # Also get layer_mean for this specific prompt
                    _, layer_mean = compute_neuron_activation(
                        best_emb, generator, layer_idx, neuron_idx, capture, return_layer_mean=True
                    )
                    layer_mean_val = layer_mean[0].item()

                # Use format compatible with visualize_feature_results.py
                trajectory.append({
                    "step": iteration,
                    "decoded_z": best.text,
                    "decoded_pred": decoded_pred,
                    "activation": best.activation,
                    "layer_mean_activation": layer_mean_val,
                    "activation_diff": best.activation_diff,
                    "cross_entropy": best.cross_entropy,
                    "did_llm_rephrase": False,  # EPO doesn't use LLM rephrasing
                    "is_llm_rephrase_step": False,
                })

                if config.verbose:
                    ce_str = f" | CE={best.cross_entropy:.2f}" if config.use_fluency else ""
                    print(f"Step {iteration:3d} | activation={best.activation:.4f} | layer_mean={layer_mean_val:.4f} | diff={best.activation_diff:.4f}{ce_str}")
                    print(f"    z:             \"{best.text}\"")
                    print(f"    prediction:    \"{decoded_pred}\"\n")

        # Final results
        pareto_results = [
            {
                "text": p.text,
                "activation": p.activation,
                "activation_diff": p.activation_diff,
                "cross_entropy": p.cross_entropy,
            }
            for p in sorted(population.pareto_frontier, key=lambda p: -p.activation_diff)
        ]

        # Get best results for final logging
        if trajectory:
            best_entry = max(trajectory, key=lambda t: t["activation_diff"])
            init_activation = trajectory[0]["activation"]
            final_activation = trajectory[-1]["activation"]
            final_z = trajectory[-1]["decoded_z"]
            final_pred = trajectory[-1]["decoded_pred"]
            best_activation = best_entry["activation"]
            best_diff = best_entry["activation_diff"]
            best_prompt = best_entry["decoded_z"]
            best_pred = best_entry["decoded_pred"]
        else:
            init_activation = final_activation = best_activation = best_diff = 0
            final_z = final_pred = best_prompt = best_pred = ""

        if config.verbose:
            print("=" * 70)
            print("FINAL RESULT:")
            print(f"  Layer {layer_idx}, Neuron {neuron_idx}")
            print(f"  Init activation: {init_activation:.4f}")
            print(f"  Final activation: {final_activation:.4f}")
            print(f"  Improvement: {final_activation - init_activation:.4f}")
            print()
            print(f"  Best activation: {best_activation:.4f}")
            print(f"  Best diff: {best_diff:.4f}")
            print(f"  Best prompt: \"{best_prompt}\"")
            print(f"  Best prediction: \"{best_pred}\"")
            print()
            print("PARETO FRONTIER (top 5 by activation_diff):")
            for i, p in enumerate(pareto_results[:5]):
                ce_str = f" | CE={p['cross_entropy']:.2f}" if config.use_fluency else ""
                text_display = f"\"{p['text'][:60]}...\"" if len(p['text']) > 60 else f"\"{p['text']}\""
                print(f"  {i+1}. diff={p['activation_diff']:.4f}{ce_str} | {text_display}")
            print("=" * 70)

        # Plot trajectory
        if trajectory:
            iters = [t["step"] for t in trajectory]
            diffs = [t["activation_diff"] for t in trajectory]

            if config.use_fluency:
                # Two subplots: activation diff and cross-entropy
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

                ax1.plot(iters, diffs, 'b-', linewidth=1.5)
                ax1.set_ylabel('Best Activation Diff')
                ax1.set_title(f'EPO Feature Visualization: Layer {layer_idx}, Neuron {neuron_idx}')
                ax1.grid(True, alpha=0.3)

                ces = [t["cross_entropy"] for t in trajectory]
                ax2.plot(iters, ces, 'r-', linewidth=1.5)
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Cross-Entropy (lower = more fluent)')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
            else:
                plt.figure(figsize=(10, 6))
                plt.plot(iters, diffs, 'b-', linewidth=1.5)
                plt.xlabel('Iteration')
                plt.ylabel('Best Activation Diff')
                plt.title(f'EPO Feature Visualization: Layer {layer_idx}, Neuron {neuron_idx}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

            plt.show()

        # Determine init_text for this run
        if seed_texts:
            init_text = f"EPO: {seed_texts[0][:30]}..." if len(seed_texts[0]) > 30 else f"EPO: {seed_texts[0]}"
        else:
            init_text = "EPO: random init"

        # Results in format compatible with visualize_feature_results.py
        results = {
            "init_text": init_text,
            "layer_idx": layer_idx,
            "neuron_idx": neuron_idx,
            "final_z": final_z,
            "final_pred": final_pred,
            "init_activation": init_activation,
            "final_activation": final_activation,
            "best_activation": best_activation,
            "best_prompt": best_prompt,
            "best_pred": best_pred,
            "trajectory": trajectory,
            "pareto_frontier": pareto_results,
            "hyperparameters": {
                "method": "EPO",
                "population_size": config.population_size,
                "n_children": config.n_children,
                "top_k": config.top_k,
                "n_iterations": config.n_iterations,
                "restart_every": config.restart_every,
                "prompt_length": config.prompt_length,
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Save results - use filename format compatible with visualize_feature_results.py
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use same pattern as feature_visualization.py: layer{X}_neuron{Y}_{timestamp}.json
            filename = f"layer{layer_idx}_neuron{neuron_idx}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            if config.verbose:
                print(f"\nResults saved to: {filepath}")

        return results

    finally:
        capture.remove()


#%%
# Load models
sonar_wrapper = SonarWrapper()
for p in sonar_wrapper.decoder.model.parameters():
    p.requires_grad = False

generator = SonarLLMGenerator.from_pretrained("raxtemur/sonar-llm-900m")
for p in generator.parameters():
    p.requires_grad = False

#%%
# Run EPO feature visualization
config = EPOConfig(
    population_size=8,
    n_children=32,
    top_k=256,
    n_iterations=100,
    restart_every=30,
    prompt_length=12,
    log_every=5,
    verbose=True,
)

seed_texts = [
    "I sail on the sea."
    "Hello world, how are you?",
    "I like cheese.",
    "The cat sat on the mat.",
]

result = run_epo_feature_visualization(
    layer_idx=10,
    neuron_idx=101,
    sonar_wrapper=sonar_wrapper,
    generator=generator,
    config=config,
    seed_texts=seed_texts,
)
