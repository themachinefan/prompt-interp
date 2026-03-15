"""
Compare standard patchscopes vs contrastive decoding.

Standard patchscopes: patch hidden state from source into target prompt, decode.
Contrastive method:
  logits_patched = forward with patched vector
  logits_avg = forward with average vector (mean hidden state)
  final_logits = logits_patched + alpha * (logits_patched - logits_avg)
"""

import sys
import os
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/patchscopes/code"))

from general_utils import ModelAndTokenizer, make_inputs, decode_tokens
from patchscopes_utils import (
    set_hs_patch_hooks_neox,
    set_hs_patch_hooks_llama,
    set_hs_patch_hooks_gptj,
    remove_hooks,
)

MODEL_TO_HOOK = {
    "EleutherAI/pythia-410m": set_hs_patch_hooks_neox,
    "EleutherAI/pythia-1.4b": set_hs_patch_hooks_neox,
    "EleutherAI/pythia-2.8b": set_hs_patch_hooks_neox,
    "EleutherAI/pythia-6.9b": set_hs_patch_hooks_neox,
    "EleutherAI/pythia-12b": set_hs_patch_hooks_neox,
    "EleutherAI/gpt-j-6b": set_hs_patch_hooks_gptj,
    "meta-llama/Llama-2-7b-hf": set_hs_patch_hooks_llama,
    "meta-llama/Llama-2-13b-hf": set_hs_patch_hooks_llama,
}


def get_hidden_states(mt, prompt):
    """Run model on prompt, return all layer hidden states."""
    inp = make_inputs(mt.tokenizer, [prompt], mt.device)
    with torch.no_grad():
        output = mt.model(**inp, output_hidden_states=True)
    # output["hidden_states"][layer+1] has shape (batch, seq, hidden_dim)
    # layer+1 because index 0 is the embedding layer output
    return inp, output


def get_patched_logits(mt, inp_target, layer_target, position_target, hidden_state, generation_mode=False):
    """Run target prompt with a patched hidden state, return logits."""
    hs_patch_config = {
        layer_target: [(position_target, hidden_state)]
    }
    skip_final_ln = (layer_target == mt.num_layers - 1)
    patch_hooks = mt.set_hs_patch_hooks(
        mt.model,
        hs_patch_config,
        module="hs",
        patch_input=False,
        skip_final_ln=skip_final_ln,
        generation_mode=generation_mode,
    )
    with torch.no_grad():
        output = mt.model(**inp_target)
    remove_hooks(patch_hooks)
    return output.logits[0, -1, :]  # logits at last position


def compute_avg_hidden_state(mt, prompts, layer, position=-1):
    """Compute mean hidden state across multiple prompts at given layer/position."""
    hs_list = []
    for prompt in prompts:
        inp = make_inputs(mt.tokenizer, [prompt], mt.device)
        with torch.no_grad():
            output = mt.model(**inp, output_hidden_states=True)
        pos = position if position >= 0 else len(inp["input_ids"][0]) + position
        hs = output["hidden_states"][layer + 1][0, pos].detach()
        hs_list.append(hs)
    return torch.stack(hs_list).mean(dim=0)


def decode_topk(mt, logits, k=5):
    """Return top-k tokens and probs from logits."""
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, k)
    tokens = [mt.tokenizer.decode(t) for t in topk.indices]
    return list(zip(tokens, topk.values.tolist()))


def standard_patchscopes(mt, prompt_source, prompt_target, layer_source, layer_target,
                         position_source, position_target):
    """Standard patchscopes: patch and decode."""
    inp_source = make_inputs(mt.tokenizer, [prompt_source], mt.device)
    with torch.no_grad():
        output_source = mt.model(**inp_source, output_hidden_states=True)

    pos_src = position_source if position_source >= 0 else len(inp_source["input_ids"][0]) + position_source
    hs_source = output_source["hidden_states"][layer_source + 1][0, pos_src]

    inp_target = make_inputs(mt.tokenizer, [prompt_target], mt.device)
    pos_tgt = position_target if position_target >= 0 else len(inp_target["input_ids"][0]) + position_target

    logits = get_patched_logits(mt, inp_target, layer_target, pos_tgt, hs_source)
    return logits


def contrastive_patchscopes(mt, prompt_source, prompt_target, layer_source, layer_target,
                            position_source, position_target, avg_hidden_state, alpha=1.0):
    """
    Contrastive patchscopes:
      logits_patched = forward with source hidden state patched in
      logits_avg = forward with average hidden state patched in
      final = logits_patched + alpha * (logits_patched - logits_avg)
    """
    inp_source = make_inputs(mt.tokenizer, [prompt_source], mt.device)
    with torch.no_grad():
        output_source = mt.model(**inp_source, output_hidden_states=True)

    pos_src = position_source if position_source >= 0 else len(inp_source["input_ids"][0]) + position_source
    hs_source = output_source["hidden_states"][layer_source + 1][0, pos_src]

    inp_target = make_inputs(mt.tokenizer, [prompt_target], mt.device)
    pos_tgt = position_target if position_target >= 0 else len(inp_target["input_ids"][0]) + position_target

    logits_patched = get_patched_logits(mt, inp_target, layer_target, pos_tgt, hs_source)
    logits_avg = get_patched_logits(mt, inp_target, layer_target, pos_tgt, avg_hidden_state)

    final_logits = logits_patched + alpha * (logits_patched - logits_avg)
    return final_logits, logits_patched, logits_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-410m",
                        help="Model name (default: pythia-410m for fast testing)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Contrastive amplification factor")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to patch (default: middle layer)")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    torch_dtype = torch.float16 if any(x in args.model for x in ["12b", "13b", "6b", "6.9b", "7b"]) else None
    mt = ModelAndTokenizer(args.model, torch_dtype=torch_dtype)
    mt.set_hs_patch_hooks = MODEL_TO_HOOK[args.model]
    mt.model.eval()

    layer = args.layer if args.layer is not None else mt.num_layers // 2
    print(f"Using layer {layer}/{mt.num_layers}")

    # --- Test prompts ---
    prompt_source = "The capital of France is"
    prompt_target = "The word is:"  # simple target prompt for decoding
    position_source = -1  # last token of source
    position_target = -1  # last token of target

    # Baseline prompts for computing average hidden state
    avg_prompts = [
        "The weather today is",
        "I went to the store",
        "She opened the book and",
        "The dog ran across the",
        "In the beginning there was",
        "He picked up the phone",
        "They decided to go to",
        "The movie was really quite",
    ]

    print("\n=== Computing average hidden state from baseline prompts ===")
    avg_hs = compute_avg_hidden_state(mt, avg_prompts, layer, position=-1)
    print(f"Average hidden state shape: {avg_hs.shape}")

    print(f"\n=== Source: '{prompt_source}' | Target: '{prompt_target}' ===")
    print(f"Layer: {layer}, alpha: {args.alpha}\n")

    # Standard patchscopes
    logits_standard = standard_patchscopes(
        mt, prompt_source, prompt_target, layer, layer, position_source, position_target
    )
    print("Standard Patchscopes top-k:")
    for tok, prob in decode_topk(mt, logits_standard, args.topk):
        print(f"  {prob:.4f}  {repr(tok)}")

    # Contrastive patchscopes
    logits_contrastive, logits_patched, logits_avg = contrastive_patchscopes(
        mt, prompt_source, prompt_target, layer, layer,
        position_source, position_target, avg_hs, alpha=args.alpha
    )
    print("\nContrastive Patchscopes top-k:")
    for tok, prob in decode_topk(mt, logits_contrastive, args.topk):
        print(f"  {prob:.4f}  {repr(tok)}")

    # Also show what the avg vector produces on its own
    print("\nAvg-vector-only Patchscopes top-k:")
    for tok, prob in decode_topk(mt, logits_avg, args.topk):
        print(f"  {prob:.4f}  {repr(tok)}")

    # --- Try multiple source prompts ---
    test_sources = [
        "The capital of France is",
        "Barack Obama was the 44th",
        "Water boils at a temperature of",
        "The largest planet in our solar system is",
    ]

    print("\n\n=== Comparison across multiple source prompts ===")
    print(f"{'Source prompt':<50} {'Standard top-1':<20} {'Contrastive top-1':<20}")
    print("-" * 90)
    for src in test_sources:
        logits_s = standard_patchscopes(
            mt, src, prompt_target, layer, layer, -1, -1
        )
        logits_c, _, _ = contrastive_patchscopes(
            mt, src, prompt_target, layer, layer, -1, -1, avg_hs, alpha=args.alpha
        )
        top_s = decode_topk(mt, logits_s, 1)[0]
        top_c = decode_topk(mt, logits_c, 1)[0]
        print(f"{src:<50} {repr(top_s[0]):>10} ({top_s[1]:.3f})   {repr(top_c[0]):>10} ({top_c[1]:.3f})")

    # --- Sweep alpha values ---
    print("\n\n=== Alpha sweep for 'The capital of France is' ===")
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
    for a in alphas:
        logits_c, _, _ = contrastive_patchscopes(
            mt, "The capital of France is", prompt_target, layer, layer,
            -1, -1, avg_hs, alpha=a
        )
        top = decode_topk(mt, logits_c, 3)
        top_str = "  ".join(f"{repr(t)}({p:.3f})" for t, p in top)
        print(f"  alpha={a:<4}  {top_str}")


if __name__ == "__main__":
    main()
