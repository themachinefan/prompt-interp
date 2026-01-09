#%%
"""Feature visualization: find embeddings that maximally activate specific neurons in SONAR-LLM."""

import os
import json
from datetime import datetime
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

from prompt_interp.sonar_wrapper import SonarWrapper
from prompt_interp.generator import SonarLLMGenerator
from prompt_interp.optimize import (
    project_to_norm,
    predict_next_embedding,
)

# Load API key from .env file (create .env with OPENAI_API_KEY=sk-...)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def rephrase_with_llm(text: str, client: OpenAI | None) -> str:
    """
    Use GPT-5.2 to rephrase text into correct English while preserving semantics.
    Returns original text if client is None or on error.
    """
    if client is None:
        return text

    try:
        response = client.chat.completions.create(
            model="gpt-5.2-chat-latest",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a text correction assistant. The input may be garbled, contain "
                        "random punctuation, incomplete words, or nonsensical fragments. Your task is to:\n"
                        "1. Remove stray punctuation like ), \", ', etc.\n"
                        "2. Fix incomplete or garbled words\n"
                        "3. Delete pointless or nonsensical repetitions (e.g. 'shop shop shop shop' -> 'shop') \n"
                        "4. Make it a somewhat coherent, grammatically correct English sentence\n"
                        "5. Keep the meaning and wording as close to the original as possible while editing it to be correct English\n"
                        "Examples:\n"
                        "Input: 'She went to the shop shop shop shop.' -> Output: 'She went to the shop.'\n"
                        "Input: 'Lygia wanted to buy eggs for eggs' -> Output: 'Lygia wanted to buy eggs'\n"
                        "Input: 'Mrs. Og Yag wanted to buy eggs to buy eggs.' -> Output: 'Mrs. Og Yag wanted to buy eggs.'\n"
                        "Input: 'Mrs. Shepp wanted to buy eggs that she was for sale.' -> Output: 'Mrs. Shepp wanted to buy eggs that were for sale.'\n"
                        "Input: 'Ms. Shep bought eggs at the store. In the shop. I bought aga.' -> Output: 'Ms. Shep bought eggs at the store. In the shop I bought again.'\n"
                        "Input: 'Mary wanted a new doll and a new thing at school, but she had a new toy.' -> Output: 'Mary wanted a new doll and a new thing at school, but instead she had a new toy.'\n"
                        "Input: 'I mean, do you guys by the, by the, by the?' -> Output: 'I mean, do you guys by the'\n"
                        "Return ONLY the corrected sentence, nothing else. If the input is already "
                        "correct, return it unchanged."
                    ),
                },
                {"role": "user", "content": f"Fix this text: {text}"},
            ],
            max_completion_tokens=256,
        )
        result = response.choices[0].message.content
        if result:
            result = result.strip()
        # Fall back to original if LLM returns empty
        return result if result else text
    except Exception as e:
        print(f"    [LLM rephrase failed: {e}]")
        return text


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


def get_neuron_activation(
    z: torch.Tensor,
    generator: SonarLLMGenerator,
    layer_idx: int,
    neuron_idx: int,
    capture: ActivationCapture,
) -> torch.Tensor:
    """
    Forward z through SONAR-LLM and return the activation of a specific neuron.

    Args:
        z: Input embedding, shape (batch, 1, 1024)
        generator: The SONAR-LLM generator
        layer_idx: Which transformer layer to hook
        neuron_idx: Which neuron in the MLP to target
        capture: ActivationCapture with hook already registered on the MLP

    Returns:
        Activation value for the target neuron, shape (batch,)
    """
    # Project to LLaMA space and forward
    proj = generator.forward_proj(z)  # (batch, seq, hidden)
    _ = generator.llama_model(inputs_embeds=proj, output_hidden_states=False)

    # Get the captured MLP activation
    # LLaMA MLP output shape: (batch, seq, intermediate_size) for act_fn output
    # or (batch, seq, hidden_size) for the full MLP output
    # We want the activation after the nonlinearity
    mlp_output = capture.activation  # (batch, seq, hidden_size)

    # Take the last position's activation for the target neuron
    # mlp_output is the output of the full MLP, shape (batch, seq, hidden_size)
    # For intermediate activations, we'd need to hook act_fn specifically
    neuron_act = mlp_output[:, -1, neuron_idx]  # (batch,)

    return neuron_act


def add_noise_with_projection(z: torch.Tensor, noise_level: float) -> torch.Tensor:
    """Add Gaussian noise scaled by norm, then project back to original norm."""
    orig_norm = z.norm(dim=-1, keepdim=True)
    noise = torch.randn_like(z) * noise_level * orig_norm
    z_noisy = z + noise
    return z_noisy * (orig_norm / (z_noisy.norm(dim=-1, keepdim=True) + 1e-8))


def log_z_state(
    z: torch.Tensor,
    sonar_wrapper: SonarWrapper,
    generator: SonarLLMGenerator,
    layer_idx: int,
    neuron_idx: int,
    capture: ActivationCapture,
    label: str,
    verbose: bool,
) -> tuple[str, float, str]:
    """Decode z, compute neuron activation, and get prediction. Returns (decoded_z, activation, decoded_pred)."""
    with torch.no_grad():
        decoded_z: str = sonar_wrapper.decode(z.squeeze(1))[0]
        activation: float = get_neuron_activation(z, generator, layer_idx, neuron_idx, capture).item()
        # Get prediction
        pred_emb = predict_next_embedding(z, generator)[:, -1:, :]
        decoded_pred: str = sonar_wrapper.decode(pred_emb.squeeze(1))[0]

    if verbose:
        print(f"{label} | activation={activation:.4f}")
        print(f"    z decodes to:  \"{decoded_z}\"")
        print(f"    prediction:    \"{decoded_pred}\"\n")

    return decoded_z, activation, decoded_pred


def run_feature_visualization(
    init_text: str,
    layer_idx: int,
    neuron_idx: int,
    sonar_wrapper: SonarWrapper,
    generator: SonarLLMGenerator,
    n_steps: int = 100,
    lr: float = 0.01,
    log_every: int = 10,
    verbose: bool = True,
    n_noise_samples: int = 7,
    noise_level: float = 0.03,
    accum_steps: int = 1,
    use_llm_rephrase: bool = False,
    llm_rephrase_every: int = 1,
    output_dir: str | None = "feature_vis_results",
) -> dict:
    """
    Find z that maximally activates a specific neuron in SONAR-LLM.

    Args:
        init_text: Starting sentence to initialize z
        layer_idx: Which transformer layer to target (0-indexed)
        neuron_idx: Which neuron in the MLP output to maximize
        sonar_wrapper: SONAR encoder/decoder wrapper
        generator: SONAR-LLM generator model
        n_steps: Number of optimization steps
        lr: Learning rate
        log_every: Log every N steps
        verbose: Print progress
        n_noise_samples: Number of noisy samples per step
        noise_level: Noise magnitude (relative to norm)
        accum_steps: Gradient accumulation steps
        use_llm_rephrase: If True, use GPT-5.2 to correct decoded_z before re-encoding
        llm_rephrase_every: Apply LLM rephrasing every N steps (default 1 = every step)
        output_dir: Directory to save results JSON (None to disable saving)

    Returns:
        Dict with final_z, final_activation, trajectory, etc.
    """
    # Initialize OpenAI client if LLM rephrasing is enabled
    openai_client: OpenAI | None = None
    if use_llm_rephrase:
        if not OPENAI_API_KEY:
            raise ValueError("Set OPENAI_API_KEY in .env file to use LLM rephrasing")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # Set up activation capture hook
    capture = ActivationCapture()
    mlp_module = get_mlp_module(generator, layer_idx)
    capture.register(mlp_module)

    try:
        # Encode initial embedding
        init_emb = sonar_wrapper.encode([init_text]).unsqueeze(1)  # (1, 1, 1024)
        target_norm = init_emb.norm(dim=-1).mean().item()

        # Print setup info
        if verbose:
            print("=" * 70)
            print("FEATURE VISUALIZATION:")
            print(f"  Layer: {layer_idx}")
            print(f"  Neuron: {neuron_idx}")
            print(f"  Init: \"{init_text}\"")
            print("=" * 70 + "\n")

        # Optimization
        z = init_emb.clone().requires_grad_(True)  # (1, 1, 1024)
        optimizer = torch.optim.Adam([z], lr=lr)
        trajectory: list[dict] = []
        samples_per_accum = n_noise_samples // accum_steps
        if samples_per_accum < 1:
            raise ValueError(f"n_noise_samples ({n_noise_samples}) must be >= accum_steps ({accum_steps})")

        # Track best activations
        best_activation = float('-inf')
        best_prompt = init_text
        best_pred = ""
        best_activation_after_llm = float('-inf')
        best_prompt_after_llm = init_text
        best_pred_after_llm = ""

        # Log initial state
        decoded_z, init_activation, _ = log_z_state(
            z, sonar_wrapper, generator, layer_idx, neuron_idx, capture, "Init", verbose
        )

        for step in range(n_steps):
            optimizer.zero_grad()

            # Accumulate gradients over multiple forward passes
            total_activation = 0.0
            for accum_idx in range(accum_steps):
                z_batch = add_noise_with_projection(z.expand(samples_per_accum, -1, -1), noise_level)

                # Get neuron activation for the batch
                neuron_act = get_neuron_activation(z_batch, generator, layer_idx, neuron_idx, capture)
                mean_activation = neuron_act.mean()

                # Loss = negative activation (we want to maximize activation)
                activation_loss = -mean_activation / accum_steps
                activation_loss.backward(retain_graph=(accum_idx < accum_steps - 1))
                total_activation += mean_activation.item()

            total_activation /= accum_steps  # Average activation across accum steps

            # Update z: project after gradient update, then roundtrip to stay on sentence manifold
            optimizer.step()
            should_log = step % log_every == 0 or step == n_steps - 1
            with torch.no_grad():
                # After optimizer.step()
                if should_log:
                    decoded_after_opt: str = sonar_wrapper.decode(z.squeeze(1))[0]

                # After projection
                z.data = project_to_norm(z, target_norm).data
                if should_log:
                    decoded_after_proj: str = sonar_wrapper.decode(z.squeeze(1))[0]

                # Roundtrip: decode, optionally rephrase with LLM, then encode
                decoded_z_raw: str = sonar_wrapper.decode(z.squeeze(1))[0]
                if use_llm_rephrase and step % llm_rephrase_every == 0:
                    decoded_z = rephrase_with_llm(decoded_z_raw, openai_client)
                else:
                    decoded_z = decoded_z_raw
                z.data = sonar_wrapper.encode([decoded_z]).unsqueeze(1)

            # Log state after update
            if should_log:
                decoded_after_enc, current_activation, decoded_pred = log_z_state(
                    z, sonar_wrapper, generator, layer_idx, neuron_idx, capture,
                    label=f"Step {step:3d}",
                    verbose=False,  # We'll print manually to include intermediate stages
                )
                # Track LLM rephrasing for logging
                did_rephrase = use_llm_rephrase and step % llm_rephrase_every == 0
                llm_changed: bool = decoded_z_raw != decoded_z

                # Update best activations
                if current_activation > best_activation:
                    best_activation = current_activation
                    best_prompt = decoded_after_enc
                    best_pred = decoded_pred
                if did_rephrase and current_activation > best_activation_after_llm:
                    best_activation_after_llm = current_activation
                    best_prompt_after_llm = decoded_after_enc
                    best_pred_after_llm = decoded_pred

                trajectory.append({
                    "step": step,
                    "activation": current_activation,
                    "decoded_z": decoded_after_enc,
                    "decoded_pred": decoded_pred,
                    "did_llm_rephrase": did_rephrase,
                })

                if verbose:
                    print(f"Step {step:3d} | activation={current_activation:.4f}")
                    print(f"    after opt:     \"{decoded_after_opt}\"")
                    print(f"    after proj:    \"{decoded_after_proj}\"")
                    if did_rephrase:
                        print(f"    before LLM:    \"{decoded_z_raw}\"")
                        print(f"    after LLM:     \"{decoded_z}\"" + (" (changed)" if llm_changed else " (unchanged)"))
                    print(f"    after re-enc:  \"{decoded_after_enc}\"")
                    print(f"    prediction:    \"{decoded_pred}\"\n")

        # Final evaluation
        if verbose:
            print("=" * 70)
            print("FINAL RESULT:")
        final_decoded_z, final_activation, final_pred = log_z_state(
            z, sonar_wrapper, generator, layer_idx, neuron_idx, capture, "Final", verbose
        )

        # Update best with final
        if final_activation > best_activation:
            best_activation = final_activation
            best_prompt = final_decoded_z
            best_pred = final_pred

        if verbose:
            print(f"  Layer {layer_idx}, Neuron {neuron_idx}")
            print(f"  Init activation: {init_activation:.4f}")
            print(f"  Final activation: {final_activation:.4f}")
            print(f"  Improvement: {final_activation - init_activation:.4f}")
            print()
            print(f"  Best activation: {best_activation:.4f}")
            print(f"  Best prompt: \"{best_prompt}\"")
            print(f"  Best prediction: \"{best_pred}\"")
            if use_llm_rephrase:
                print()
                print(f"  Best activation (after LLM): {best_activation_after_llm:.4f}")
                print(f"  Best prompt (after LLM): \"{best_prompt_after_llm}\"")
                print(f"  Best prediction (after LLM): \"{best_pred_after_llm}\"")
            print("=" * 70)

        # Plot activation over iterations
        steps = [t["step"] for t in trajectory]
        activations = [t["activation"] for t in trajectory]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, activations, 'b-', label='All steps', linewidth=1.5)

        # Plot LLM rephrase steps as separate points/line
        if use_llm_rephrase:
            llm_steps = [t["step"] for t in trajectory if t["did_llm_rephrase"]]
            llm_activations = [t["activation"] for t in trajectory if t["did_llm_rephrase"]]
            if llm_steps:
                plt.plot(llm_steps, llm_activations, 'ro-', label='After LLM rephrase', markersize=8, linewidth=1.5)

        plt.xlabel('Step')
        plt.ylabel('Neuron Activation')
        plt.title(f'Feature Visualization: Layer {layer_idx}, Neuron {neuron_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        results = {
            "init_text": init_text,
            "layer_idx": layer_idx,
            "neuron_idx": neuron_idx,
            "final_z": final_decoded_z,
            "final_pred": final_pred,
            "init_activation": init_activation,
            "final_activation": final_activation,
            "best_activation": best_activation,
            "best_prompt": best_prompt,
            "best_pred": best_pred,
            "best_activation_after_llm": best_activation_after_llm if use_llm_rephrase else None,
            "best_prompt_after_llm": best_prompt_after_llm if use_llm_rephrase else None,
            "best_pred_after_llm": best_pred_after_llm if use_llm_rephrase else None,
            "trajectory": trajectory,
            # Hyperparameters for reproducibility
            "hyperparameters": {
                "n_steps": n_steps,
                "lr": lr,
                "n_noise_samples": n_noise_samples,
                "noise_level": noise_level,
                "accum_steps": accum_steps,
                "use_llm_rephrase": use_llm_rephrase,
                "llm_rephrase_every": llm_rephrase_every,
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Save results to disk
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"layer{layer_idx}_neuron{neuron_idx}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            if verbose:
                print(f"\nResults saved to: {filepath}")

        return results

    finally:
        # Always clean up the hook
        capture.remove()

#%%
sonar_wrapper = SonarWrapper()
for p in sonar_wrapper.decoder.model.parameters():
    p.requires_grad = False

generator = SonarLLMGenerator.from_pretrained("raxtemur/sonar-llm-900m")
for p in generator.parameters():
    p.requires_grad = False

#%%
# Example: visualize what neuron 100 in layer 10 responds to
hi = run_feature_visualization(
    # init_text="I like cheese.",
    # init_text="Lily went to the shop to buy eggs.",
    init_text="What is the weather like today?",
    # init_text="Tim and Lily went to the shop to buy eggs.",
    # init_text="Ring the bell",
    # init_text="I enjoy eating cheese.",
    # init_text="The fairy lights are beautiful.",
    # init_text="Skyscrapers are very tall buildings.",
    # init_text="A time of unprecedented urgency, the end of the world as we know it.",
    layer_idx=14,
    neuron_idx=101,
    sonar_wrapper=sonar_wrapper,
    generator=generator,
    n_steps=63,
    lr=0.02,
    log_every=2,
    n_noise_samples=64,
    noise_level=0.05,
    accum_steps=1,
    verbose=True,
    use_llm_rephrase=True,
    llm_rephrase_every=4
)
