#!/usr/bin/env python3
"""
Test whether a regular LLM's internal representations can replace the concept
model (SONAR-LLM + SONAR encoder/decoder) for feature visualization.

Currently, the pipeline uses:
  SONAR encoder (text -> 1024-dim vector)
  -> SONAR-LLM (predicts next sentence embedding, model being interpreted)
  -> SONAR decoder (1024-dim vector -> text)

This script tests whether a regular LLM can fill both the encoder and decoder
roles, using its own hidden states as the "embedding space".

Encoding (sentence -> vector):
  A) last_token:  hidden state at the last token position at layer L
  B) mean_tokens: mean hidden state across all token positions at layer L

Decoding (vector -> text) — 2 base methods × 2 modes = 4 variants:
  Base methods:
    1) continue: inject vector at layer L for a BOS token, generate
    2) template: inject at "It" in "It means the following:", generate
  Modes:
    plain: use logits directly
    diff:  contrastive decoding against a "generic" vector (average over
           many sentences). logits = target_logits + α(target_logits - generic_logits)
           This amplifies what's unique about the target vector.

Total: 2 encodings × 2 bases × 2 modes = 8 combinations per layer.

Model size note: SONAR-LLM is ~900M params. For an equivalent-size regular
LLM, use gpt2-large (774M, 36 layers, d=1280) which is the default.

Usage:
  python scripts/test_llm_replacement.py
  python scripts/test_llm_replacement.py --model gpt2-large --layers 0 9 18 27 35
  python scripts/test_llm_replacement.py --model meta-llama/Llama-3.2-1B
  python scripts/test_llm_replacement.py --diff-alpha 2.0
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


# Sentences used to compute the "generic" average vector for contrastive decoding.
# Deliberately diverse in topic and structure.
CALIBRATION_SENTENCES = [
    "The sun rose over the mountains.",
    "She went to the store to buy milk.",
    "Dogs are loyal animals.",
    "The book was on the table.",
    "He ran quickly down the street.",
    "They played in the garden all afternoon.",
    "The teacher explained the lesson carefully.",
    "It was raining outside.",
    "The baby started crying loudly.",
    "We had pizza for dinner last night.",
    "The bird flew over the house.",
    "She smiled when she saw the flowers.",
    "The car stopped at the red light.",
    "He opened the window to let in fresh air.",
    "The children were excited about the trip.",
    "She read a story before bed.",
    "The fish swam in the pond.",
    "He fixed the broken chair.",
    "They walked along the beach at sunset.",
    "The clock struck twelve.",
]


def load_model(model_name, device):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_layer_module(model, layer_idx):
    """Get the transformer block at the given layer index."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    raise ValueError(f"Unknown model architecture: {type(model)}")


def get_num_layers(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    raise ValueError(f"Unknown model architecture: {type(model)}")


# ---------------------------------------------------------------------------
# Encoding: sentence -> vector at a given layer
# ---------------------------------------------------------------------------

def encode_sentence(model, tokenizer, sentence, layer_idx, method, device):
    """
    Encode a sentence into a single vector by extracting hidden states at layer L.

    Args:
        method: 'last_token' or 'mean_tokens'
    Returns:
        vector of shape (d_model,)
    """
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    captured = {}

    def capture_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["hidden_states"] = h.detach().clone()

    handle = get_layer_module(model, layer_idx).register_forward_hook(capture_hook)

    with torch.no_grad():
        model(**inputs)

    handle.remove()

    h = captured["hidden_states"][0]  # (seq_len, d_model)

    if method == "last_token":
        return h[-1]
    elif method == "mean_tokens":
        return h.mean(dim=0)
    else:
        raise ValueError(f"Unknown encoding method: {method}")


# ---------------------------------------------------------------------------
# Compute generic (average) vectors for contrastive decoding
# ---------------------------------------------------------------------------

def compute_generic_vectors(model, tokenizer, layers, device):
    """
    For each layer × encoding method, compute the average hidden-state vector
    across CALIBRATION_SENTENCES. Returns dict[(layer, method)] -> vector.
    """
    print("Computing generic vectors for contrastive decoding...")
    generic = {}
    for layer_idx in layers:
        for method in ["last_token", "mean_tokens"]:
            vecs = []
            for sent in CALIBRATION_SENTENCES:
                v = encode_sentence(model, tokenizer, sent, layer_idx, method, device)
                vecs.append(v)
            generic[(layer_idx, method)] = torch.stack(vecs).mean(dim=0)
    print(f"  Done ({len(generic)} vectors computed)\n")
    return generic


# ---------------------------------------------------------------------------
# Generation helper with hidden-state injection (supports contrastive mode)
# ---------------------------------------------------------------------------

def generate_with_injection(model, tokenizer, input_ids, layer_idx,
                            inject_pos, inject_vector, max_new_tokens,
                            diff_vector=None, diff_alpha=1.0):
    """
    Generate text while injecting a vector at (inject_pos, layer_idx).

    Uses no KV cache so the injection is clean at every step.

    If diff_vector is provided, does contrastive decoding:
      final_logits = target_logits + alpha * (target_logits - generic_logits)
    """
    generated = input_ids.clone()

    # Mutable container so the hook can swap between target and generic vectors
    state = {"vector": inject_vector}

    def inject_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > inject_pos:
            h = h.clone()
            h[0, inject_pos, :] = state["vector"]
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    handle = get_layer_module(model, layer_idx).register_forward_hook(inject_hook)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if diff_vector is not None:
                # Forward with target vector
                state["vector"] = inject_vector
                target_logits = model(generated, use_cache=False).logits[0, -1, :]

                # Forward with generic vector
                state["vector"] = diff_vector
                generic_logits = model(generated, use_cache=False).logits[0, -1, :]

                # Contrastive: amplify the difference
                next_logits = target_logits + diff_alpha * (target_logits - generic_logits)
            else:
                next_logits = model(generated, use_cache=False).logits[0, -1, :]

            next_token = next_logits.argmax(dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated = torch.cat(
                [generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1
            )

    handle.remove()
    return generated


# ---------------------------------------------------------------------------
# Decoding methods
# ---------------------------------------------------------------------------

def decode_continue(model, tokenizer, vector, layer_idx, max_new_tokens, device,
                    diff_vector=None, diff_alpha=1.0):
    """Start from BOS, inject vector at position 0, layer L. Generate."""
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    input_ids = torch.tensor([[bos_id]], device=device)

    output_ids = generate_with_injection(
        model, tokenizer, input_ids, layer_idx,
        inject_pos=0, inject_vector=vector,
        max_new_tokens=max_new_tokens,
        diff_vector=diff_vector, diff_alpha=diff_alpha,
    )

    return tokenizer.decode(output_ids[0, 1:], skip_special_tokens=True)


def decode_template(model, tokenizer, vector, layer_idx, max_new_tokens, device,
                    diff_vector=None, diff_alpha=1.0):
    """Inject vector at position 0 ("It") in "It means the following:". Generate."""
    template = "It means the following:"
    inputs = tokenizer(template, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    template_len = input_ids.shape[1]

    output_ids = generate_with_injection(
        model, tokenizer, input_ids, layer_idx,
        inject_pos=0, inject_vector=vector,
        max_new_tokens=max_new_tokens,
        diff_vector=diff_vector, diff_alpha=diff_alpha,
    )

    return tokenizer.decode(output_ids[0, template_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test regular LLM as replacement for concept model"
    )
    parser.add_argument(
        "--model", default="gpt2-large",
        help="HuggingFace model name (default: gpt2-large, ~774M to match SONAR-LLM ~900M)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--diff-alpha", type=float, default=1.0,
                        help="Contrastive decoding strength (default: 1.0)")
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Layer indices to test (default: 5 evenly spaced)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model, device)

    n_layers = get_num_layers(model)
    if args.layers is None:
        args.layers = sorted(set(
            [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
        ))

    test_sentences = [
        "I like cheese.",
        "The cat sat on the mat.",
        "Once upon a time there was a little girl named Lily.",
        "The weather is nice today.",
        "The dog chased the ball across the park.",
        "She opened the door and found a surprise inside.",
        "Mathematics is the language of the universe.",
        "Please pass the salt.",
        "The stock market crashed on Monday.",
        "Lily was very happy because she got a new toy.",
        "He walked slowly through the dark forest.",
        "Water boils at one hundred degrees Celsius.",
        "The teacher asked the students to open their books.",
        "I wonder what will happen next.",
    ]

    encode_methods = ["last_token", "mean_tokens"]
    decode_bases = ["continue", "template"]
    decode_modes = ["plain", "diff"]

    print(f"Model: {args.model} ({n_layers} layers, d_model={model.config.hidden_size})")
    print(f"Layers: {args.layers}")
    print(f"Device: {device}")
    print(f"Diff alpha: {args.diff_alpha}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Test sentences: {len(test_sentences)}")
    print(f"Combinations per sentence per layer: {len(encode_methods) * len(decode_bases) * len(decode_modes)}")

    # Pre-compute generic vectors for contrastive decoding
    generic_vectors = compute_generic_vectors(model, tokenizer, args.layers, device)

    for sentence in test_sentences:
        print(f"\n{'='*80}")
        print(f"  INPUT: \"{sentence}\"")
        print(f"{'='*80}")

        for layer_idx in args.layers:
            print(f"\n  --- Layer {layer_idx}/{n_layers-1} ---")

            for enc_method in encode_methods:
                vec = encode_sentence(
                    model, tokenizer, sentence, layer_idx, enc_method, device
                )

                for base in decode_bases:
                    for mode in decode_modes:
                        # Get the generic vector for diff mode
                        diff_vec = None
                        if mode == "diff":
                            diff_vec = generic_vectors[(layer_idx, enc_method)]

                        if base == "continue":
                            result = decode_continue(
                                model, tokenizer, vec, layer_idx,
                                args.max_new_tokens, device,
                                diff_vector=diff_vec, diff_alpha=args.diff_alpha,
                            )
                        else:
                            result = decode_template(
                                model, tokenizer, vec, layer_idx,
                                args.max_new_tokens, device,
                                diff_vector=diff_vec, diff_alpha=args.diff_alpha,
                            )

                        # Roundtrip cosine similarity
                        if result.strip():
                            vec2 = encode_sentence(
                                model, tokenizer, result, layer_idx, enc_method, device
                            )
                            sim = torch.nn.functional.cosine_similarity(
                                vec.unsqueeze(0), vec2.unsqueeze(0)
                            ).item()
                            sim_str = f"  (cos={sim:.3f})"
                        else:
                            sim_str = "  (empty)"

                        tag = f"{enc_method:>11} -> {base:<8} {mode:<5}"
                        print(f"    [{tag}]: \"{result.strip()}\"{sim_str}")


if __name__ == "__main__":
    main()
