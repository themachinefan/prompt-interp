"""Fun qualitative demos of contrastive patchscopes."""

import sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/patchscopes/code"))

from general_utils import ModelAndTokenizer, make_inputs
from patchscopes_utils import set_hs_patch_hooks_neox, remove_hooks


def get_patched_logits(mt, inp_target, layer_target, position_target, hidden_state):
    hs_patch_config = {layer_target: [(position_target, hidden_state)]}
    skip_final_ln = (layer_target == mt.num_layers - 1)
    hooks = mt.set_hs_patch_hooks(
        mt.model, hs_patch_config, module="hs", patch_input=False,
        skip_final_ln=skip_final_ln, generation_mode=False,
    )
    with torch.no_grad():
        output = mt.model(**inp_target)
    remove_hooks(hooks)
    return output.logits[0, -1, :]


def generate_contrastive(mt, inp_target, layer, pos_tgt, hs_source, hs_avg,
                         alpha=1.0, max_len=30):
    input_ids = inp_target["input_ids"].clone()
    toks = []
    for _ in range(max_len):
        cur = {"input_ids": input_ids}
        if "attention_mask" in inp_target:
            cur["attention_mask"] = torch.ones_like(input_ids)
        lp = get_patched_logits(mt, cur, layer, pos_tgt, hs_source)
        la = get_patched_logits(mt, cur, layer, pos_tgt, hs_avg)
        logits = lp + alpha * (lp - la)
        # repetition penalty
        for tid in set(input_ids[0].tolist()):
            if logits[tid] > 0: logits[tid] /= 1.3
            else: logits[tid] *= 1.3
        t = logits.argmax().item()
        toks.append(t)
        input_ids = torch.cat([input_ids, torch.tensor([[t]], device=input_ids.device)], dim=1)
        if t == mt.tokenizer.eos_token_id:
            break
    return mt.tokenizer.decode(toks, skip_special_tokens=True)


def generate_standard(mt, inp_target, layer, pos_tgt, hs_source, max_len=30):
    hs_patch_config = {layer: [(pos_tgt, hs_source)]}
    skip_final_ln = (layer == mt.num_layers - 1)
    hooks = mt.set_hs_patch_hooks(
        mt.model, hs_patch_config, module="hs", patch_input=False,
        skip_final_ln=skip_final_ln, generation_mode=True,
    )
    seq_len = len(inp_target["input_ids"][0])
    with torch.no_grad():
        out = mt.model.generate(
            inp_target["input_ids"], max_length=seq_len + max_len,
            pad_token_id=mt.model.generation_config.eos_token_id,
        )[0, seq_len:]
    remove_hooks(hooks)
    return mt.tokenizer.decode(out, skip_special_tokens=True)


def run_demo(mt, source_prompt, target_prompt, source_pos, layer, avg_hs, alpha=1.0):
    inp_src = make_inputs(mt.tokenizer, [source_prompt], mt.device)
    with torch.no_grad():
        out_src = mt.model(**inp_src, output_hidden_states=True)
    pos = source_pos if source_pos >= 0 else len(inp_src["input_ids"][0]) + source_pos
    hs = out_src["hidden_states"][layer + 1][0, pos].detach()

    inp_tgt = make_inputs(mt.tokenizer, [target_prompt], mt.device)
    pos_tgt = len(inp_tgt["input_ids"][0]) - 1

    std = generate_standard(mt, inp_tgt, layer, pos_tgt, hs)
    con = generate_contrastive(mt, inp_tgt, layer, pos_tgt, hs, avg_hs, alpha=alpha)
    return std, con


def main():
    model_name = "EleutherAI/pythia-2.8b"
    print(f"Loading {model_name}...")
    mt = ModelAndTokenizer(model_name, torch_dtype=torch.float16)
    mt.set_hs_patch_hooks = set_hs_patch_hooks_neox
    mt.model.eval()

    # Compute avg hidden state at multiple layers
    avg_prompts = [
        "The weather today is", "I went to the store", "She opened the book and",
        "The dog ran across the", "In the beginning there was", "He picked up the phone",
        "They decided to go to", "The movie was really quite", "A long time ago in",
        "The teacher asked the student", "We should consider whether the",
        "After the meeting they went", "The results of the experiment",
        "It was a dark and stormy", "The committee voted to approve",
        "Scientists have discovered that the",
    ]

    def get_avg(layer):
        vecs = []
        for p in avg_prompts:
            inp = make_inputs(mt.tokenizer, [p], mt.device)
            with torch.no_grad():
                out = mt.model(**inp, output_hidden_states=True)
            vecs.append(out["hidden_states"][layer + 1][0, -1].detach())
        return torch.stack(vecs).mean(dim=0)

    # ---------- DEMOS ----------

    demos = [
        {
            "title": "1. Early Layer Decoding (layer 8 vs 16 vs 24)",
            "source": "The Eiffel Tower is located in",
            "target": "The place is:",
            "source_pos": -1,
            "layers": [8, 16, 24],
        },
        {
            "title": "2. Entity Resolution from Context",
            "source": "The actress who starred in Titanic won an Oscar",
            "target": "The person's name is",
            "source_pos": 2,  # "actress"
        },
        {
            "title": "3. Disambiguation: 'bank' (financial)",
            "source": "She went to the bank to deposit her paycheck",
            "target": "The word means:",
            "source_pos": 5,  # "bank"
        },
        {
            "title": "4. Disambiguation: 'bank' (river)",
            "source": "He sat by the river bank watching the water flow",
            "target": "The word means:",
            "source_pos": 5,  # "bank"
        },
        {
            "title": "5. Sentiment Extraction",
            "source": "The movie was absolutely terrible and a waste of time",
            "target": "The sentiment is:",
            "source_pos": -1,
        },
        {
            "title": "6. Sentiment Extraction (positive)",
            "source": "The concert was incredible and the best I have ever seen",
            "target": "The sentiment is:",
            "source_pos": -1,
        },
        {
            "title": "7. Implicit Knowledge: inventor",
            "source": "Thomas Edison worked late into the night in his laboratory",
            "target": "This person is known for:",
            "source_pos": 2,  # "Edison"
        },
        {
            "title": "8. Implicit Knowledge: scientist",
            "source": "Albert Einstein published his theory of relativity in 1905",
            "target": "This person is known for:",
            "source_pos": 2,  # "Einstein"
        },
        {
            "title": "9. Category/Type extraction",
            "source": "The labrador retriever is a friendly and loyal companion",
            "target": "This is a type of:",
            "source_pos": 2,  # "labrador"
        },
        {
            "title": "10. Multi-hop: iPhone -> Apple -> CEO",
            "source": "The iPhone revolutionized the smartphone industry",
            "target": "The CEO of the company that makes this is:",
            "source_pos": 1,  # "iPhone"
        },
    ]

    results_text = []

    for demo in demos:
        print(f"\n{'='*70}")
        print(demo["title"])
        print(f"  Source: \"{demo['source']}\" (pos={demo['source_pos']})")
        print(f"  Target: \"{demo['target']}\"")
        print(f"{'='*70}")

        layers = demo.get("layers", [16])
        demo_results = []

        for layer in layers:
            avg = get_avg(layer)
            std, con = run_demo(mt, demo["source"], demo["target"],
                                demo["source_pos"], layer, avg)
            layer_label = f"layer={layer}" if len(layers) > 1 else ""
            if layer_label:
                print(f"\n  [{layer_label}]")
            print(f"  Standard:    {std[:80]}")
            print(f"  Contrastive: {con[:80]}")
            demo_results.append((layer, std, con))

        results_text.append((demo, demo_results))

    # Format for markdown
    print("\n\n### MARKDOWN OUTPUT ###\n")
    for demo, layer_results in results_text:
        print(f"### {demo['title']}")
        print(f"Source: `{demo['source']}` (pos={demo['source_pos']})")
        print(f"Target: `{demo['target']}`\n")
        for layer, std, con in layer_results:
            if len(layer_results) > 1:
                print(f"**Layer {layer}:**")
            print(f"- Standard: {std[:100]}")
            print(f"- Contrastive: {con[:100]}")
            print()


if __name__ == "__main__":
    main()
