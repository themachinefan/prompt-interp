#%%
"""
Activation intervention: Generate completions with and without intervening on a specific neuron.
"""

import json
from datetime import datetime
from pathlib import Path
import torch

from prompt_interp import REPO_ROOT
from prompt_interp.sonar_wrapper import SonarWrapper
from prompt_interp.generator import SonarLLMGenerator
from prompt_interp.optimize import predict_next_embedding

DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "activation_intervention"


class ActivationIntervention:
    """Hook to capture and optionally intervene on activations from a specific layer."""

    def __init__(self, neuron_idx: int, intervention_value: float | None = None):
        self.neuron_idx = neuron_idx
        self.intervention_value = intervention_value  # None means no intervention
        self.activation: torch.Tensor | None = None
        self.handle = None

    def hook_fn(self, module, input, output):
        """Capture and optionally modify the output activation."""
        self.activation = output
        if self.intervention_value is not None:
            # Intervene: set the target neuron to the specified value
            # output shape: (batch, seq, hidden_size)
            output[:, :, self.neuron_idx] = self.intervention_value
        return output

    def register(self, module: torch.nn.Module):
        """Register the hook on a module."""
        self.handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        """Remove the hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def set_intervention(self, value: float | None):
        """Set or disable intervention."""
        self.intervention_value = value


def get_mlp_module(generator: SonarLLMGenerator, layer_idx: int) -> torch.nn.Module:
    """Get the MLP module for a specific layer in the LLaMA model."""
    return generator.llama_model.model.layers[layer_idx].mlp


def generate_completion(
    prompt: str,
    sonar_wrapper: SonarWrapper,
    generator: SonarLLMGenerator,
    max_new_sentences: int = 6,
) -> tuple[str, list[str]]:
    """
    Generate a completion (next sentence predictions) for a prompt.

    Args:
        prompt: Input prompt text
        sonar_wrapper: SONAR encoder/decoder
        generator: SONAR-LLM generator
        max_new_sentences: Maximum number of sentences to generate (default 6)

    Returns:
        (full_completion_text, list_of_generated_sentences)
    """
    with torch.no_grad():
        z = sonar_wrapper.encode([prompt]).unsqueeze(1)  # (1, 1, 1024)

        generated_sentences = []
        for _ in range(max_new_sentences):
            pred_emb = predict_next_embedding(z, generator)[:, -1:, :]  # (1, 1, 1024)
            decoded_pred = sonar_wrapper.decode(pred_emb.squeeze(1))[0]

            # Check for end of sequence indicators
            decoded_stripped = decoded_pred.strip()
            if not decoded_stripped:
                # Empty output signals end
                break
            if decoded_stripped in {".", "...", "—", "-", "–"}:
                # Just punctuation signals end
                break
            if generated_sentences and decoded_stripped == generated_sentences[-1].strip():
                # Repetition signals end
                break

            generated_sentences.append(decoded_pred)

            # Use the prediction as input for next step
            z = pred_emb

        full_completion = " ".join(generated_sentences)

    return full_completion, generated_sentences


def run_intervention_experiment(
    prompts: list[str],
    layer_idx: int,
    neuron_idx: int,
    intervention_value: float,
    sonar_wrapper: SonarWrapper,
    generator: SonarLLMGenerator,
    output_file: str | Path | None = None,
    verbose: bool = True,
    max_new_sentences: int = 6,
) -> dict:
    """
    Run completions with and without neuron intervention.

    Args:
        prompts: List of prompts to test
        layer_idx: Which transformer layer to intervene on
        neuron_idx: Which neuron in the MLP to intervene on
        intervention_value: Value to set the neuron to during intervention
        sonar_wrapper: SONAR encoder/decoder
        generator: SONAR-LLM generator
        output_file: Path to save results (None to skip saving)
        verbose: Print results to console
        max_new_sentences: Maximum sentences to generate per completion (default 6)

    Returns:
        Dict with results for each prompt
    """
    # Set up intervention hook
    hook = ActivationIntervention(neuron_idx=neuron_idx, intervention_value=None)
    mlp_module = get_mlp_module(generator, layer_idx)
    hook.register(mlp_module)

    results = {
        "layer_idx": layer_idx,
        "neuron_idx": neuron_idx,
        "intervention_value": intervention_value,
        "max_new_sentences": max_new_sentences,
        "timestamp": datetime.now().isoformat(),
        "prompts": [],
    }

    try:
        for prompt in prompts:
            # Generate WITHOUT intervention
            hook.set_intervention(None)
            pred_baseline, baseline_sentences = generate_completion(
                prompt, sonar_wrapper, generator, max_new_sentences
            )

            # Generate WITH intervention
            hook.set_intervention(intervention_value)
            pred_intervened, intervened_sentences = generate_completion(
                prompt, sonar_wrapper, generator, max_new_sentences
            )

            prompt_result = {
                "prompt": prompt,
                "baseline_completion": pred_baseline,
                "baseline_sentences": baseline_sentences,
                "intervened_completion": pred_intervened,
                "intervened_sentences": intervened_sentences,
                "changed": pred_baseline != pred_intervened,
            }
            results["prompts"].append(prompt_result)

            if verbose:
                print(f"\n{'='*70}")
                print(f"Prompt: \"{prompt}\"")
                print(f"{'='*70}")
                print(f"  Baseline ({len(baseline_sentences)} sentences):")
                print(f"    \"{pred_baseline}\"")
                print(f"  Intervened ({len(intervened_sentences)} sentences):")
                print(f"    \"{pred_intervened}\"")
                if prompt_result["changed"]:
                    print("  [CHANGED]")
                else:
                    print("  [unchanged]")

    finally:
        hook.remove()

    # Summary
    n_changed = sum(1 for p in results["prompts"] if p["changed"])
    results["summary"] = {
        "total_prompts": len(prompts),
        "changed": n_changed,
        "unchanged": len(prompts) - n_changed,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"SUMMARY: Layer {layer_idx}, Neuron {neuron_idx}, Value {intervention_value}")
        print(f"  {n_changed}/{len(prompts)} completions changed")
        print(f"{'='*70}")

    # Save to file
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    return results


#%%
# Load models
sonar_wrapper = SonarWrapper()
for p in sonar_wrapper.decoder.model.parameters():
    p.requires_grad = False

generator = SonarLLMGenerator.from_pretrained("raxtemur/sonar-llm-900m")
for p in generator.parameters():
    p.requires_grad = False


#%%
# Configuration
layer_idx = 10
neuron_idx = 102
intervention_value = 200.0  # Set neuron to this value

test_prompts = [
    "I like cheese.",
    "Lily went to the shop to buy eggs.",
    "What is the weather like today?",
    "The cat sat on the mat.",
    "Once upon a time there was a little girl.",
    "The sun was shining brightly.",
    "He wanted to play with his friends.",
    "She found a beautiful flower in the garden.",
]

# Generate output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = DEFAULT_OUTPUT_DIR / f"layer{layer_idx}_neuron{neuron_idx}_val{intervention_value}_{timestamp}.json"

# Run experiment
results = run_intervention_experiment(
    prompts=test_prompts,
    layer_idx=layer_idx,
    neuron_idx=neuron_idx,
    intervention_value=intervention_value,
    sonar_wrapper=sonar_wrapper,
    generator=generator,
    output_file=None,
    verbose=True,
)
