"""
Evaluate standard patchscopes vs contrastive patchscopes on attribute extraction.

Uses the preprocessed datasets from the patchscopes repo.
Metric: accuracy (does ground-truth object appear in generated text?).

Contrastive method:
  logits_final = logits_patched + alpha * (logits_patched - logits_avg)
where logits_avg comes from patching in an average hidden state.
"""

import sys
import os
import argparse
import json
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

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

DATA_DIR = os.path.join(os.path.dirname(__file__), "../src/patchscopes/code/preprocessed_data")

# Shorter datasets good for quick eval
SMALL_DATASETS = {
    "factual": ["country_capital_city", "country_largest_city", "country_currency",
                "food_from_country", "pokemon_evolutions"],
    "commonsense": ["fruit_inside_color", "fruit_outside_color", "substance_phase",
                    "task_done_by_person", "work_location"],
}

ALL_DATASETS = {
    "factual": ["company_ceo", "country_capital_city", "country_currency",
                "country_largest_city", "food_from_country", "person_father",
                "person_mother", "person_plays_position_in_sport",
                "person_plays_pro_sport", "pokemon_evolutions",
                "product_by_company", "star_constellation",
                "superhero_archnemesis", "superhero_person"],
    "commonsense": ["fruit_inside_color", "fruit_outside_color", "object_superclass",
                    "substance_phase", "task_done_by_person", "task_done_by_tool",
                    "word_sentiment", "work_location"],
}


def get_patched_logits(mt, inp_target, layer_target, position_target, hidden_state):
    """Single forward pass with patched hidden state, returns logits at last position."""
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
        generation_mode=False,
    )
    with torch.no_grad():
        output = mt.model(**inp_target)
    remove_hooks(patch_hooks)
    return output.logits[0, -1, :]


def generate_with_patch(mt, inp_target, layer_target, position_target,
                        hidden_state, max_gen_len=20):
    """Standard patchscopes generation: patch then use model.generate."""
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
        generation_mode=True,
    )
    seq_len = len(inp_target["input_ids"][0])
    with torch.no_grad():
        output_toks = mt.model.generate(
            inp_target["input_ids"],
            attention_mask=inp_target.get("attention_mask"),
            max_length=seq_len + max_gen_len,
            pad_token_id=mt.model.generation_config.eos_token_id,
        )[0, seq_len:]
    remove_hooks(patch_hooks)
    return mt.tokenizer.decode(output_toks)


def generate_contrastive(mt, inp_target, layer_target, position_target,
                         hs_source, hs_avg, alpha=1.0, max_gen_len=20):
    """
    Contrastive generation: at each step, compute
      logits_final = logits_patched + alpha * (logits_patched - logits_avg)
    then greedily select next token.
    """
    input_ids = inp_target["input_ids"].clone()
    generated_tokens = []

    for _ in range(max_gen_len):
        cur_inp = {"input_ids": input_ids}
        if "attention_mask" in inp_target:
            cur_inp["attention_mask"] = torch.ones_like(input_ids)

        logits_patched = get_patched_logits(mt, cur_inp, layer_target, position_target, hs_source)
        logits_avg = get_patched_logits(mt, cur_inp, layer_target, position_target, hs_avg)

        logits_final = logits_patched + alpha * (logits_patched - logits_avg)
        next_token = logits_final.argmax(dim=-1)

        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

        if next_token.item() == mt.tokenizer.eos_token_id:
            break

    return mt.tokenizer.decode(generated_tokens)


def compute_avg_hidden_states(mt, n_prompts=16):
    """Compute average hidden state at each layer from random generic prompts."""
    avg_prompts = [
        "The weather today is", "I went to the store", "She opened the book and",
        "The dog ran across the", "In the beginning there was", "He picked up the phone",
        "They decided to go to", "The movie was really quite", "A long time ago in",
        "The teacher asked the student", "We should consider whether the",
        "After the meeting they went", "The results of the experiment",
        "It was a dark and stormy", "The committee voted to approve",
        "Scientists have discovered that the",
    ]
    avg_prompts = avg_prompts[:n_prompts]

    # Collect hidden states per layer
    layer_hs = {layer: [] for layer in range(mt.num_layers)}
    for prompt in avg_prompts:
        inp = make_inputs(mt.tokenizer, [prompt], mt.device)
        with torch.no_grad():
            output = mt.model(**inp, output_hidden_states=True)
        for layer in range(mt.num_layers):
            # Take last token position
            hs = output["hidden_states"][layer + 1][0, -1].detach()
            layer_hs[layer].append(hs)

    avg_per_layer = {}
    for layer in range(mt.num_layers):
        avg_per_layer[layer] = torch.stack(layer_hs[layer]).mean(dim=0)
    return avg_per_layer


def check_correct(generated_text, ground_truth):
    """Check if ground truth appears in generated text (space-insensitive)."""
    return ground_truth.replace(" ", "").lower() in generated_text.replace(" ", "").lower()


def load_dataset(task_type, task_name):
    """Load a preprocessed TSV dataset."""
    path = os.path.join(DATA_DIR, task_type, f"{task_name}.tsv")
    df = pd.read_csv(path, sep="\t")
    # Filter out empty/problematic source prompts
    df = df[df["prompt_source"].notna() & (df["prompt_source"] != "")].reset_index(drop=True)
    df = df[~df["prompt_source"].str.contains('\n', na=False)].reset_index(drop=True)
    return df


def evaluate_dataset(mt, task_type, task_name, layers, avg_hidden_states,
                     alpha=1.0, max_gen_len=20, max_samples=None):
    """Evaluate standard vs contrastive patchscopes on one dataset."""
    df = load_dataset(task_type, task_name)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    results = []
    for layer in layers:
        correct_standard = 0
        correct_contrastive = 0
        total = 0
        hs_avg = avg_hidden_states[layer]

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"{task_name} layer={layer}"):
            prompt_source = row["prompt_source"]
            prompt_target = row["prompt_target"]
            position_source = int(row["position_source"])
            ground_truth = row["object"]

            # Get source hidden state
            inp_source = make_inputs(mt.tokenizer, [prompt_source], mt.device)
            with torch.no_grad():
                out_source = mt.model(**inp_source, output_hidden_states=True)
            pos_src = position_source if position_source >= 0 else len(inp_source["input_ids"][0]) + position_source
            if pos_src >= len(inp_source["input_ids"][0]):
                continue  # skip if position is out of bounds
            hs_source = out_source["hidden_states"][layer + 1][0, pos_src].detach()

            # Prepare target input
            inp_target = make_inputs(mt.tokenizer, [prompt_target], mt.device)
            pos_tgt = len(inp_target["input_ids"][0]) - 1  # last token

            # Standard patchscopes
            gen_standard = generate_with_patch(
                mt, inp_target, layer, pos_tgt, hs_source, max_gen_len
            )

            # Contrastive patchscopes
            gen_contrastive = generate_contrastive(
                mt, inp_target, layer, pos_tgt, hs_source, hs_avg,
                alpha=alpha, max_gen_len=max_gen_len
            )

            is_correct_std = check_correct(gen_standard, ground_truth)
            is_correct_con = check_correct(gen_contrastive, ground_truth)
            correct_standard += is_correct_std
            correct_contrastive += is_correct_con
            total += 1

            results.append({
                "task_type": task_type,
                "task_name": task_name,
                "layer": layer,
                "subject": row.get("subject", ""),
                "object": ground_truth,
                "gen_standard": gen_standard,
                "gen_contrastive": gen_contrastive,
                "correct_standard": is_correct_std,
                "correct_contrastive": is_correct_con,
            })

        if total > 0:
            print(f"  Layer {layer}: standard={correct_standard}/{total} "
                  f"({100*correct_standard/total:.1f}%)  "
                  f"contrastive={correct_contrastive}/{total} "
                  f"({100*correct_contrastive/total:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-2.8b")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples per dataset (None=all)")
    parser.add_argument("--max-gen-len", type=int, default=20)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers to test (default: early/mid/late)")
    parser.add_argument("--small", action="store_true",
                        help="Only run on smaller datasets")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names to run (e.g. country_capital_city,fruit_inside_color)")
    parser.add_argument("--output", default="results/patchscopes_eval.json")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    torch_dtype = torch.float16 if any(x in args.model for x in ["12b", "13b", "6b", "6.9b", "7b", "2.8b"]) else None
    mt = ModelAndTokenizer(args.model, torch_dtype=torch_dtype)
    mt.set_hs_patch_hooks = MODEL_TO_HOOK[args.model]
    mt.model.eval()

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        n = mt.num_layers
        layers = [n // 4, n // 2, 3 * n // 4, n - 1]
    print(f"Testing layers: {layers} (of {mt.num_layers})")

    print("\nComputing average hidden states from baseline prompts...")
    avg_hidden_states = compute_avg_hidden_states(mt)

    datasets = SMALL_DATASETS if args.small else ALL_DATASETS
    # Filter to specific datasets if requested
    if args.datasets:
        requested = set(args.datasets.split(","))
        datasets = {
            tt: [tn for tn in tns if tn in requested]
            for tt, tns in datasets.items()
        }
        datasets = {tt: tns for tt, tns in datasets.items() if tns}

    all_results = []
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for task_type, task_names in datasets.items():
        for task_name in task_names:
            print(f"\n{'='*60}")
            print(f"Dataset: {task_type}/{task_name} (alpha={args.alpha})")
            print(f"{'='*60}")
            results = evaluate_dataset(
                mt, task_type, task_name, layers, avg_hidden_states,
                alpha=args.alpha, max_gen_len=args.max_gen_len,
                max_samples=args.max_samples,
            )
            all_results.extend(results)
            # Save incrementally after each dataset
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  (saved {len(all_results)} results so far to {args.output})")

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    results_df = pd.DataFrame(all_results)
    summary = results_df.groupby(["task_type", "task_name", "layer"]).agg(
        n=("correct_standard", "count"),
        acc_standard=("correct_standard", "mean"),
        acc_contrastive=("correct_contrastive", "mean"),
    ).reset_index()
    summary["improvement"] = summary["acc_contrastive"] - summary["acc_standard"]

    print(f"\n{'Task':<35} {'Layer':>5} {'N':>4}  {'Standard':>10} {'Contrastive':>12} {'Diff':>8}")
    print("-" * 80)
    for _, row in summary.iterrows():
        task = f"{row['task_type']}/{row['task_name']}"
        print(f"{task:<35} {row['layer']:>5} {row['n']:>4}  "
              f"{100*row['acc_standard']:>9.1f}% {100*row['acc_contrastive']:>11.1f}% "
              f"{100*row['improvement']:>+7.1f}%")

    # Overall
    overall_std = results_df["correct_standard"].mean()
    overall_con = results_df["correct_contrastive"].mean()
    print("-" * 80)
    print(f"{'OVERALL':<35} {'':>5} {len(results_df):>4}  "
          f"{100*overall_std:>9.1f}% {100*overall_con:>11.1f}% "
          f"{100*(overall_con-overall_std):>+7.1f}%")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {args.output}")

    # Also save some interesting examples where contrastive helped
    helped = results_df[results_df["correct_contrastive"] & ~results_df["correct_standard"]]
    if len(helped) > 0:
        print(f"\n\nExamples where contrastive helped ({len(helped)} cases):")
        for _, row in helped.head(10).iterrows():
            print(f"  {row['task_name']}: {row['subject']} -> {row['object']}")
            print(f"    Standard:    {row['gen_standard'][:80]}")
            print(f"    Contrastive: {row['gen_contrastive'][:80]}")
            print()

    # And where it hurt
    hurt = results_df[~results_df["correct_contrastive"] & results_df["correct_standard"]]
    if len(hurt) > 0:
        print(f"Examples where contrastive hurt ({len(hurt)} cases):")
        for _, row in hurt.head(5).iterrows():
            print(f"  {row['task_name']}: {row['subject']} -> {row['object']}")
            print(f"    Standard:    {row['gen_standard'][:80]}")
            print(f"    Contrastive: {row['gen_contrastive'][:80]}")
            print()


if __name__ == "__main__":
    main()
