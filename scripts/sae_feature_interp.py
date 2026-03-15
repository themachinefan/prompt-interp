"""
SAE Feature Interpretation via Layer-Sweep Optimization + Contrastive Patchscopes.

For a given SAE feature on Neuronpedia:
1. Load the model (Gemma 2 2B) and SAE
2. Get the feature's autointerp description and max activating examples from Neuronpedia
3. For each layer L from 0 to SAE_layer:
   a. Optimize n continuous hidden state vectors at layer L to maximize the SAE feature
   b. Decode them into readable text using contrastive patchscopes
   c. At layer 0: also clamp to nearest tokens (= GCG baseline)
4. Compare interpretations across layers

Requires: pip install sae-lens transformer-lens
"""

import sys
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/prompt_interp"))

from llm_wrapper import LLMWrapper, CALIBRATION_SENTENCES
from feature_visualization import ActivationCapture, add_noise_with_projection
from optimize import project_to_norm

# Import neuronpedia helpers (from user's code, assumed to be in src/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


# ---------------------------------------------------------------------------
# SAE utilities
# ---------------------------------------------------------------------------

def load_sae_from_neuronpedia(neuronpedia_id: str, device: str = "cuda",
                               dtype: torch.dtype = torch.bfloat16):
    """Load SAE and feature metadata from Neuronpedia."""
    from neuronpedia_utils import get_sae, get_feature_description
    sae = get_sae(neuronpedia_id, device=device, dtype=dtype)
    return sae


def get_sae_layer(sae, neuronpedia_id: str) -> int:
    """Extract layer index from SAE config or neuronpedia ID."""
    import re
    # Try hook_name/hook_point from config
    for attr in ['hook_name', 'hook_point', 'hook_point_layer']:
        if hasattr(sae.cfg, attr):
            val = getattr(sae.cfg, attr)
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                m = re.search(r'(\d+)', val)
                if m:
                    return int(m.group(1))
    # Try metadata
    if hasattr(sae.cfg, 'metadata') and sae.cfg.metadata:
        for key in ['hook_layer', 'layer']:
            if key in sae.cfg.metadata:
                return int(sae.cfg.metadata[key])
    # Fall back to parsing the neuronpedia_id (e.g. "gemma-2-2b/20-gemmascope-res-16k")
    parts = neuronpedia_id.split("/")
    if len(parts) >= 2:
        m = re.match(r'(\d+)', parts[-1])
        if m:
            return int(m.group(1))
    raise ValueError(f"Cannot determine SAE layer from config or ID: {neuronpedia_id}")


def get_sae_feature_activation(
    z: torch.Tensor,
    llm_wrapper: LLMWrapper,
    inject_layer_idx: int,
    sae_layer_idx: int,
    sae,
    feature_idx: int,
    capture: ActivationCapture,
    return_mean: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Inject z at inject_layer_idx, capture residual stream at sae_layer_idx,
    pass through SAE encoder, return activation of target feature.

    Args:
        z: Hidden state vectors, shape (batch, n_tokens, d_model)
        llm_wrapper: The LLM wrapper
        inject_layer_idx: Layer to inject z at
        sae_layer_idx: Layer where SAE hooks (to capture residual stream)
        sae: SAELens SAE object
        feature_idx: Which SAE feature to target
        capture: ActivationCapture hooked on the sae_layer's output
        return_mean: If True, also return mean feature activation

    Returns:
        Feature activation, shape (batch,)
    """
    batch = z.shape[0]
    n_tokens = z.shape[1]

    # Use dummy input_ids (BOS tokens) — we'll inject our vectors via hooks
    bos_id = llm_wrapper.tokenizer.bos_token_id
    if bos_id is None:
        bos_id = llm_wrapper.tokenizer.eos_token_id
    input_ids = torch.full((batch, n_tokens), bos_id, device=llm_wrapper.device, dtype=torch.long)

    z_for_hook = z  # (batch, n_tokens, d_model)

    if inject_layer_idx == 0:
        # At layer 0, inject by replacing the embeddings directly
        # We need to hook the embedding output, not a transformer block
        def inject_embed_hook(module, input, output):
            # output is (batch, seq, d_model) from embedding layer
            return z_for_hook.to(output.dtype)

        # Find embedding module
        if hasattr(llm_wrapper.model, 'model') and hasattr(llm_wrapper.model.model, 'embed_tokens'):
            embed_module = llm_wrapper.model.model.embed_tokens
        elif hasattr(llm_wrapper.model, 'transformer') and hasattr(llm_wrapper.model.transformer, 'wte'):
            embed_module = llm_wrapper.model.transformer.wte
        else:
            raise ValueError("Cannot find embedding module")

        inject_handle = embed_module.register_forward_hook(inject_embed_hook)
    else:
        # Inject at layer L by replacing its input
        def inject_pre_hook(module, args):
            return (z_for_hook.to(args[0].dtype),) + args[1:]

        inject_handle = llm_wrapper.get_layer_module(inject_layer_idx).register_forward_pre_hook(inject_pre_hook)

    try:
        llm_wrapper.model(input_ids, use_cache=False)
    finally:
        inject_handle.remove()

    # capture.activation is the residual stream output at sae_layer_idx
    # Shape: (batch, n_tokens, d_model)
    residual = capture.activation
    if isinstance(residual, tuple):
        residual = residual[0]

    # Pass through SAE encoder to get pre-activation (before JumpReLU threshold)
    # This ensures gradients flow even when the feature is below threshold
    residual_flat = residual.mean(dim=1)  # (batch, d_model)
    sae_in = sae.process_sae_in(residual_flat.to(sae.dtype))
    hidden_pre = sae_in @ sae.W_enc + sae.b_enc  # (batch, n_features)
    target_act = hidden_pre[:, feature_idx]  # (batch,)

    if return_mean:
        mean_act = hidden_pre.mean(dim=-1)  # (batch,)
        return target_act, mean_act

    return target_act


# ---------------------------------------------------------------------------
# Decoding: contrastive patchscopes for multi-token hidden states
# ---------------------------------------------------------------------------

def decode_hidden_states_contrastive(
    z: torch.Tensor,
    llm_wrapper: LLMWrapper,
    layer_idx: int,
    alpha: float = 1.0,
    max_new_tokens: int = 30,
) -> str:
    """
    Decode multiple hidden state vectors at a given layer into text
    using contrastive generation.

    z: (n_tokens, d_model) — the optimized hidden states
    """
    generic_vec = llm_wrapper.get_generic_vector(layer_idx)

    # For multi-token, inject all vectors at positions 0..n-1
    n_tokens = z.shape[0]
    bos_id = llm_wrapper.tokenizer.bos_token_id or llm_wrapper.tokenizer.eos_token_id
    generated = torch.tensor([[bos_id] * n_tokens], device=llm_wrapper.device)
    template_len = n_tokens

    state = {"vecs": z, "generic": generic_vec.unsqueeze(0).expand(n_tokens, -1)}

    def inject_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] >= n_tokens:
            h = h.clone()
            for pos in range(n_tokens):
                h[0, pos, :] = state["current_vecs"][pos]
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    handle = llm_wrapper.get_layer_module(layer_idx).register_forward_hook(inject_hook)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward with target vectors
            state["current_vecs"] = state["vecs"]
            t_logits = llm_wrapper.model(generated, use_cache=False).logits[0, -1, :]
            # Forward with generic vectors
            state["current_vecs"] = state["generic"]
            g_logits = llm_wrapper.model(generated, use_cache=False).logits[0, -1, :]
            # Contrastive
            logits = t_logits + alpha * (t_logits - g_logits)
            # Repetition penalty
            seen = set(generated[0].tolist())
            for tok_id in seen:
                if logits[tok_id] > 0:
                    logits[tok_id] /= 1.3
                else:
                    logits[tok_id] *= 1.3
            tok = logits.argmax()
            if tok.item() == llm_wrapper.tokenizer.eos_token_id and generated.shape[1] - template_len >= 3:
                break
            generated = torch.cat([generated, tok.view(1, 1)], dim=1)

    handle.remove()
    return llm_wrapper.tokenizer.decode(generated[0, template_len:], skip_special_tokens=True)


def decode_hidden_states_patchscopes(
    z: torch.Tensor,
    llm_wrapper: LLMWrapper,
    layer_idx: int,
    alpha: float = 2.5,
    max_new_tokens: int = 30,
    template: str = "The word {} means",
) -> str:
    """
    Contrastive patchscopes: inject optimized hidden states at the placeholder
    position in "The word X means", then generate using contrastive decoding
    (subtract logits from generic-vector version).

    We inject a single mean-pooled vector at one placeholder position,
    and patch at ALL layers from 0 to layer_idx so the suffix tokens
    ("means") attend to the patched representation at every layer.

    z: (n_tokens, d_model) — the optimized hidden states
    template: must contain {} as placeholder for the patched position
    """
    # Mean-pool the optimized vectors into a single vector
    z_mean = z.mean(dim=0)  # (d_model,)
    generic_vec = llm_wrapper.get_generic_vector(layer_idx)  # (d_model,)

    # Tokenize template parts (before and after placeholder)
    parts = template.split("{}")
    assert len(parts) == 2, f"Template must contain exactly one {{}}, got: {template}"

    prefix_text = parts[0]  # e.g. "The word "
    suffix_text = parts[1]  # e.g. " means"

    bos_id = llm_wrapper.tokenizer.bos_token_id or llm_wrapper.tokenizer.eos_token_id

    prefix_ids = []
    if prefix_text:
        prefix_ids = llm_wrapper.tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids = llm_wrapper.tokenizer.encode(suffix_text, add_special_tokens=False)

    # Single placeholder token
    all_ids = [bos_id] + prefix_ids + [bos_id] + suffix_ids
    input_ids = torch.tensor([all_ids], device=llm_wrapper.device)
    template_len = len(all_ids)

    # Position of the placeholder token
    inject_pos = 1 + len(prefix_ids)

    # State dict to switch between target and generic vectors
    state = {"current_vec": z_mean}

    # Hook every layer from 0 to layer_idx so suffix tokens see patched repr
    handles = []

    def make_hook(target_layer_idx):
        def inject_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if inject_pos < h.shape[1]:
                h = h.clone()
                h[0, inject_pos, :] = state["current_vec"].to(h.dtype)
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return inject_hook

    # For layer 0, hook the embedding; for others hook the transformer blocks
    if layer_idx == 0:
        if hasattr(llm_wrapper.model, 'model') and hasattr(llm_wrapper.model.model, 'embed_tokens'):
            embed_module = llm_wrapper.model.model.embed_tokens
        elif hasattr(llm_wrapper.model, 'transformer') and hasattr(llm_wrapper.model.transformer, 'wte'):
            embed_module = llm_wrapper.model.transformer.wte
        else:
            embed_module = None
        if embed_module is not None:
            handles.append(embed_module.register_forward_hook(make_hook(0)))
    else:
        for l in range(layer_idx + 1):
            handles.append(
                llm_wrapper.get_layer_module(l).register_forward_hook(make_hook(l))
            )

    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward with target vector
            state["current_vec"] = z_mean
            t_logits = llm_wrapper.model(generated, use_cache=False).logits[0, -1, :]
            # Forward with generic vector
            state["current_vec"] = generic_vec
            g_logits = llm_wrapper.model(generated, use_cache=False).logits[0, -1, :]
            # Contrastive
            logits = t_logits + alpha * (t_logits - g_logits)
            # Repetition penalty
            seen = set(generated[0].tolist())
            for tok_id in seen:
                if logits[tok_id] > 0:
                    logits[tok_id] /= 1.3
                else:
                    logits[tok_id] *= 1.3
            tok = logits.argmax()
            if tok.item() == llm_wrapper.tokenizer.eos_token_id and generated.shape[1] - template_len >= 3:
                break
            generated = torch.cat([generated, tok.view(1, 1)], dim=1)

    for h in handles:
        h.remove()
    return llm_wrapper.tokenizer.decode(generated[0, template_len:], skip_special_tokens=True)


def clamp_to_nearest_tokens(z: torch.Tensor, embedding_matrix: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    """
    GCG-style: find nearest discrete tokens for each vector in z.

    z: (n_tokens, d_model)
    embedding_matrix: (vocab_size, d_model)

    Returns: (clamped_z, token_ids)
    """
    # Do on CPU to avoid OOM with large vocab
    z_cpu = F.normalize(z.float().cpu(), dim=-1)
    emb_cpu = F.normalize(embedding_matrix.float().cpu(), dim=-1)
    sims = z_cpu @ emb_cpu.T  # (n_tokens, vocab_size)
    token_ids = sims.argmax(dim=-1).tolist()
    clamped = embedding_matrix[token_ids]  # stays on original device
    return clamped, token_ids


# ---------------------------------------------------------------------------
# Optimization loop
# ---------------------------------------------------------------------------

def _run_optimization(
    llm_wrapper: LLMWrapper,
    sae,
    feature_idx: int,
    sae_layer_idx: int,
    inject_layer_idx: int,
    n_tokens: int,
    n_steps: int,
    lr: float,
    n_noise_samples: int,
    noise_level: float,
    use_projection: bool,
    verbose: bool,
    label: str = "",
) -> tuple[torch.Tensor, float, list[dict]]:
    """
    Core optimization loop. Returns (best_z, best_activation, trajectory).
    """
    # Initialize z from random calibration sentence encodings
    init_vecs = []
    for i in range(n_tokens):
        sent = CALIBRATION_SENTENCES[i % len(CALIBRATION_SENTENCES)]
        vec = llm_wrapper.encode(sent, inject_layer_idx)
        init_vecs.append(vec)
    z_init = torch.stack(init_vecs).unsqueeze(0).to(dtype=torch.float32)
    target_norm = z_init.norm(dim=-1).mean().item()

    z = z_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    capture = ActivationCapture()
    sae_module = llm_wrapper.get_layer_module(sae_layer_idx)
    capture.register(sae_module)

    trajectory = []
    best_activation = float('-inf')
    best_z = None

    try:
        for step in range(n_steps):
            optimizer.zero_grad()

            z_batch = add_noise_with_projection(
                z.expand(n_noise_samples, -1, -1), noise_level
            )

            feat_act, mean_act = get_sae_feature_activation(
                z_batch, llm_wrapper, inject_layer_idx, sae_layer_idx,
                sae, feature_idx, capture, return_mean=True
            )

            loss = -(feat_act - mean_act).mean()
            loss.backward()
            optimizer.step()

            if use_projection:
                with torch.no_grad():
                    for t in range(n_tokens):
                        z.data[0, t] = project_to_norm(
                            z[0, t].unsqueeze(0).unsqueeze(0), target_norm
                        ).squeeze()

            current_act = feat_act.mean().item()
            if current_act > best_activation:
                best_activation = current_act
                best_z = z.detach().clone()

            if step % 20 == 0 or step == n_steps - 1:
                if verbose:
                    print(f"  [{label}] Step {step:3d} | feature_act={current_act:.4f} | mean_act={mean_act.mean().item():.4f}")
                trajectory.append({
                    "step": step,
                    "feature_act": current_act,
                    "mean_act": mean_act.mean().item(),
                })
    finally:
        capture.remove()

    return best_z.squeeze(0), best_activation, trajectory


def _decode_all(z_opt, llm_wrapper, inject_layer_idx, verbose, label=""):
    """Decode with both contrastive and patchscopes methods."""
    decoded_contrastive = decode_hidden_states_contrastive(
        z_opt, llm_wrapper, inject_layer_idx, alpha=1.0
    )
    decoded_patchscopes = decode_hidden_states_patchscopes(
        z_opt, llm_wrapper, inject_layer_idx
    )
    if verbose:
        print(f"  [{label}] Contrastive: {decoded_contrastive}")
        print(f"  [{label}] Patchscopes: {decoded_patchscopes}")
    return decoded_contrastive, decoded_patchscopes


def optimize_for_feature(
    llm_wrapper: LLMWrapper,
    sae,
    feature_idx: int,
    sae_layer_idx: int,
    inject_layer_idx: int,
    n_tokens: int = 5,
    n_steps: int = 200,
    lr: float = 0.01,
    n_noise_samples: int = 8,
    noise_level: float = 0.03,
    verbose: bool = True,
) -> dict:
    """
    Optimize n_tokens hidden state vectors at inject_layer_idx to maximize
    SAE feature activation at sae_layer_idx.

    Decodes with both contrastive and patchscopes methods.
    """
    z_opt, best_act, trajectory = _run_optimization(
        llm_wrapper=llm_wrapper, sae=sae, feature_idx=feature_idx,
        sae_layer_idx=sae_layer_idx, inject_layer_idx=inject_layer_idx,
        n_tokens=n_tokens, n_steps=n_steps, lr=lr,
        n_noise_samples=n_noise_samples, noise_level=noise_level,
        use_projection=False, verbose=verbose,
    )
    cont, patch = _decode_all(z_opt, llm_wrapper, inject_layer_idx, verbose)

    result = {
        "inject_layer": inject_layer_idx,
        "sae_layer": sae_layer_idx,
        "feature_idx": feature_idx,
        "best_activation": best_act,
        "decoded_contrastive": cont,
        "decoded_patchscopes": patch,
        "trajectory": trajectory,
    }

    # GCG baseline at layer 0
    if inject_layer_idx == 0:
        if hasattr(llm_wrapper.model, 'model') and hasattr(llm_wrapper.model.model, 'embed_tokens'):
            emb_matrix = llm_wrapper.model.model.embed_tokens.weight.data
        elif hasattr(llm_wrapper.model, 'transformer') and hasattr(llm_wrapper.model.transformer, 'wte'):
            emb_matrix = llm_wrapper.model.transformer.wte.weight.data
        else:
            emb_matrix = None

        if emb_matrix is not None:
            _, token_ids = clamp_to_nearest_tokens(z_opt, emb_matrix)
            gcg_text = llm_wrapper.tokenizer.decode(token_ids)
            result["gcg_tokens"] = gcg_text
            result["gcg_token_ids"] = token_ids
            if verbose:
                print(f"  GCG (clamped tokens): {gcg_text}")

    return result


# ---------------------------------------------------------------------------
# Main: layer sweep
# ---------------------------------------------------------------------------

def run_layer_sweep(
    neuronpedia_id: str,
    feature_idx: int,
    model_name: str = "google/gemma-2-2b",
    n_tokens: int = 5,
    n_steps: int = 200,
    lr: float = 0.01,
    layers: list[int] | None = None,
    output_path: str = "results/sae_feature_interp.json",
    verbose: bool = True,
):
    """
    Run the full layer sweep for a single SAE feature.
    """
    from neuronpedia_utils import get_feature_description

    # 1. Get feature metadata from Neuronpedia
    print(f"Fetching feature info from Neuronpedia: {neuronpedia_id}/{feature_idx}")
    feature_desc = get_feature_description(neuronpedia_id, feature_idx)
    print(f"  Autointerp: {feature_desc}")

    # 2. Load model
    print(f"\nLoading model: {model_name}")
    dtype = torch.bfloat16
    llm = LLMWrapper(model_name, device="cuda", dtype=dtype)

    # 3. Load SAE
    print(f"Loading SAE: {neuronpedia_id}")
    sae = load_sae_from_neuronpedia(neuronpedia_id, device="cuda", dtype=dtype)
    sae_layer = get_sae_layer(sae, neuronpedia_id)
    print(f"  SAE hooks at layer {sae_layer}")
    print(f"  SAE has {sae.cfg.d_sae} features, d_model={sae.cfg.d_in}")

    # 4. Determine layers to sweep
    if layers is None:
        n_layers = llm.get_num_layers()
        # Sweep from 0 (embedding/GCG) through SAE layer
        step_size = max(1, sae_layer // 6)
        layers = list(range(0, sae_layer + 1, step_size))
        if sae_layer not in layers:
            layers.append(sae_layer)
    print(f"  Sweeping layers: {layers}")

    # 5. Run optimization at each layer
    neuronpedia_url = f"https://www.neuronpedia.org/{neuronpedia_id}/{feature_idx}"
    all_results = {
        "neuronpedia_id": neuronpedia_id,
        "feature_idx": feature_idx,
        "neuronpedia_url": neuronpedia_url,
        "autointerp": feature_desc,
        "model": model_name,
        "sae_layer": sae_layer,
        "n_tokens": n_tokens,
        "layer_results": [],
    }

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Optimizing at layer {layer} (SAE at layer {sae_layer})")
        print(f"{'='*60}")

        result = optimize_for_feature(
            llm_wrapper=llm,
            sae=sae,
            feature_idx=feature_idx,
            sae_layer_idx=sae_layer,
            inject_layer_idx=layer,
            n_tokens=n_tokens,
            n_steps=n_steps,
            lr=lr,
            verbose=verbose,
        )
        all_results["layer_results"].append(result)

        # Save incrementally
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # 6. Save clean txt summary
    txt_path = output_path.replace(".json", ".txt")
    lines = []
    lines.append(neuronpedia_url)
    lines.append(f"autointerp: {feature_desc}")
    lines.append("")
    for r in all_results["layer_results"]:
        layer = r['inject_layer']
        lines.append(f"layer {layer} contrastive: {r['decoded_contrastive']}")
        lines.append(f"layer {layer} patchscopes: {r.get('decoded_patchscopes', 'N/A')}")
    txt_content = "\n".join(lines)
    with open(txt_path, "w") as f:
        f.write(txt_content)

    print(f"\nResults saved to {output_path}")
    print(f"Summary saved to {txt_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="SAE Feature Interpretation via Layer-Sweep Optimization")
    parser.add_argument("--neuronpedia-id", type=str, required=True,
                        help="Neuronpedia source ID, e.g. 'gemma-2-2b/20-gemmascope-res-16k'")
    parser.add_argument("--feature", type=int, required=True,
                        help="SAE feature index")
    parser.add_argument("--model", default="google/gemma-2-2b",
                        help="Model name (default: gemma-2-2b)")
    parser.add_argument("--n-tokens", type=int, default=5,
                        help="Number of token positions to optimize (default: 5)")
    parser.add_argument("--n-steps", type=int, default=200,
                        help="Optimization steps per layer (default: 200)")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers to sweep (default: auto from 0 to SAE layer)")
    parser.add_argument("--output", default=None,
                        help="Output path (default: results/sae_feature_interp_{id}_{feature}.json)")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    if args.output is None:
        safe_id = args.neuronpedia_id.replace("/", "_")
        output_path = f"results/sae_feature_interp_{safe_id}_{args.feature}.json"
    else:
        output_path = args.output

    run_layer_sweep(
        neuronpedia_id=args.neuronpedia_id,
        feature_idx=args.feature,
        model_name=args.model,
        n_tokens=args.n_tokens,
        n_steps=args.n_steps,
        lr=args.lr,
        layers=layers,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
