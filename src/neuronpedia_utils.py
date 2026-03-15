import os
from typing import Any, Dict, Optional
import requests
from dotenv import load_dotenv
from sae_lens import SAE
import torch
import re
from huggingface_hub import list_repo_files

load_dotenv()


def _infer_gemma_scope_canonical_release(hf_repo_id: str) -> str | None:
    """
    Map raw Gemma Scope HF repo -> SAELens canonical release alias.
    Example:
      google/gemma-scope-27b-pt-res  -> gemma-scope-27b-pt-res-canonical
    """
    m = re.match(r"google/(gemma-scope-\d+b-(?:pt|it)-(?:res|mlp|attn|emb))$", hf_repo_id)
    if not m:
        return None
    return f"{m.group(1)}-canonical"


def _pick_average_l0_closest_to_100(hf_repo_id: str, hf_folder_id: str) -> int | None:
    """
    Look in HF repo for average_l0_* subfolders and choose closest to 100.
    This matches Gemma Scope 'canonical' definition.
    """
    try:
        files = list_repo_files(hf_repo_id)
    except Exception:
        return None

    prefix = hf_folder_id.rstrip("/") + "/average_l0_"
    l0s = set()
    for f in files:
        if f.startswith(prefix):
            rest = f[len(prefix):]
            m = re.match(r"(\d+)(/|$)", rest)
            if m:
                l0s.add(int(m.group(1)))

    if not l0s:
        return None
    return min(l0s, key=lambda x: abs(x - 100))


def get_sae(
    neuronpedia_id: str,
    api_key: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    prefer_canonical: bool = True,
    target_l0: int | None = None,
):
    """
    Robust Neuronpedia loader.

    Order:
      1) If Neuronpedia provides saelensRelease/saelensSaeId -> use directly.
      2) If Gemma Scope repo -> use SAELens canonical release.
      3) Otherwise try raw HF load.
    """
    if isinstance(device, torch.device):
        device = device.type
    elif isinstance(device, str) and ":" in device:
        device = device.split(":")[0]

    if api_key is None:
        api_key = os.environ.get("NEURONPEDIA_API_KEY")
        if not api_key:
            raise ValueError("Please provide an API key or set NEURONPEDIA_API_KEY")

    url = f"https://www.neuronpedia.org/api/feature/{neuronpedia_id}/0"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    source = resp.json()["source"]

    if "gemma-3" in neuronpedia_id:
        sae_lens_release = source["modelId"]
        saelens_sae_id = source["id"]
        sae_lens_release = "gemma-scope-2-4b-it-res"  # TODO: Hardcoded for now
        saelens_sae_id = "layer_22_width_16k_l0_medium"
    else:
        sae_lens_release = source.get("saelensRelease")
        print(f"sae_lens_release: {sae_lens_release}")
        saelens_sae_id = source.get("saelensSaeId")
        print(f"saelens_sae_id: {saelens_sae_id}")

    # 1) Standard SAELens release path
    if sae_lens_release and saelens_sae_id:
        print(f"Loading SAE from sae_lens...")
        sae = SAE.from_pretrained(
            release=sae_lens_release,
            sae_id=saelens_sae_id,
            device=device,
        )
        return sae.to(dtype)

    # 2) HF fallback path
    hf_repo_id = source.get("hfRepoId")
    hf_folder_id = source.get("hfFolderId")
    if not hf_repo_id or not hf_folder_id:
        raise ValueError(
            "Unable to load SAE: Neuronpedia provided neither saelensRelease/saelensSaeId "
            f"nor hfRepoId/hfFolderId. Source: {source}"
        )

    # 2a) Gemma Scope special case
    canonical_release = _infer_gemma_scope_canonical_release(hf_repo_id)
    if canonical_release and prefer_canonical:
        print(f"Loading SAE from canonical release (i.e. Gemma Scope)...")
        best_l0 = _pick_average_l0_closest_to_100(hf_repo_id, hf_folder_id)
        canonical_sae_id = f"{hf_folder_id.rstrip('/')}/canonical"
        sae = SAE.from_pretrained(
            release=canonical_release,
            sae_id=canonical_sae_id,
            device=device,
        )
        return sae.to(dtype)

    # 2b) Non-Gemma repos
    try:
        print(f"Loading SAE from HF repo '{hf_repo_id}' folder '{hf_folder_id}'...")
        sae = SAE.from_pretrained(
            release=hf_repo_id,
            sae_id=hf_folder_id,
            device=device,
        )
        return sae.to(dtype)
    except Exception as e:
        raise ValueError(
            f"Unable to load SAE from HF repo '{hf_repo_id}' folder '{hf_folder_id}'. "
            f"Original error: {e}"
        )


class NeuronpediaResponse:
    def __init__(self, data: dict):
        self.data = data

    def get_activations(self):
        return self.data.get("activations", [])

    def get_explanations(self):
        return self.data.get("explanations", [])

    def get_all_explanation_descriptions(self):
        return [ex.get("description", "") for ex in self.get_explanations()]

    def get_all_explanation_model_names(self):
        return [ex.get("explanationModelName", "") for ex in self.get_explanations()]

    def get_explanation_by_model_name(self, model_name: str):
        return self.get_all_explanation_descriptions()[
            self.get_all_explanation_model_names().index(model_name)
        ]

    def get_contexts_around_top_n_activations(self, n=3, window=5):
        activations = self.get_activations()
        if not activations:
            return []
        sorted_activations = sorted(
            activations, key=lambda act: act.get("maxValue", 0), reverse=True
        )
        contexts = []
        seen_max_values = set()
        for act in sorted_activations:
            max_val = act.get("maxValue", 0)
            if max_val in seen_max_values:
                continue
            seen_max_values.add(max_val)
            tokens = act.get("tokens", [])
            values = act.get("values", [])
            if not tokens or not values or len(tokens) != len(values):
                context_text = " ".join(tokens)
            else:
                max_index = max(range(len(values)), key=lambda i: values[i])
                start_index = max(0, max_index - window)
                end_index = min(len(tokens), max_index + window + 1)
                context_tokens = tokens[start_index:end_index]
                context_text = "".join(context_tokens).replace("▁", " ").strip()
            contexts.append(context_text)
            if len(contexts) == n:
                break
        return contexts


def get_feature_description(
    neuronpedia_id: str,
    index: int = 321,
    api_key: Optional[str] = None,
    autointerp_model=None,
    include_max_acts=True,
) -> str:
    """Get the auto-interpretation description for a specific feature from Neuronpedia."""
    if api_key is None:
        api_key = os.environ.get("NEURONPEDIA_API_KEY")
        if not api_key:
            raise ValueError("Please provide an API key or set NEURONPEDIA_API_KEY")
    try:
        url = f"https://www.neuronpedia.org/api/feature/{neuronpedia_id}/{index}"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers)
        response = NeuronpediaResponse(response.json())

        if autointerp_model is None:
            models_ordered = [
                "gemini-3.0-pro", "claude-4.5-sonnet", "gpt-5",
                "claude-3.7-sonnet", "gemini-2.5-pro", "gemini-2.5-flash",
                "gpt-4o", "gemini-2.0-flash", "o1-mini", "claude-3-opus",
                "claude-3-sonnet", "gpt-4o-mini", "gpt-3.5-turbo",
                "claude-3-haiku", "gemini-1.5-flash",
            ]
            possible_models = response.get_all_explanation_model_names()
            autointerp_model = None
            for model in models_ordered:
                if model in possible_models:
                    autointerp_model = model
                    break
            if autointerp_model is None and possible_models:
                autointerp_model = possible_models[0]

        try:
            explanation = response.get_explanation_by_model_name(autointerp_model)
        except Exception:
            explanation = "AUTOINTERP not found."

        if include_max_acts:
            try:
                max_acts = response.get_contexts_around_top_n_activations(n=3, window=5)
                explanation = explanation + " ::: " + "||".join(max_acts)
            except Exception:
                explanation = explanation + " ::: Max activating examples not found."

        return explanation
    except Exception:
        return f"Error fetching auto description for feature {index}"
