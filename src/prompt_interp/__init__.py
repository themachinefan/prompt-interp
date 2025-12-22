"""SONAR-LLM: Sentence-level language model using SONAR embeddings."""

from prompt_interp.sonar import SonarWrapper
from prompt_interp.model import SonarLLM, SonarLLMConfig, PRESETS
from prompt_interp.generator import SonarLLMGenerator, GenerationConfig
from prompt_interp.optimize import (
    project_to_norm,
    predict_next_embedding,
    decoder_ce_loss,
    tokenize_for_decoder,
    EMBEDDING_NORM_MEAN,
    LILY_STORY,
)
from prompt_interp.experiments import (
    run_next_sentence_experiment,
    run_lily_story_experiments,
    run_multi_init_experiments,
)

__all__ = [
    "SonarWrapper",
    "SonarLLM",
    "SonarLLMConfig",
    "PRESETS",
    "SonarLLMGenerator",
    "GenerationConfig",
    "project_to_norm",
    "predict_next_embedding",
    "decoder_ce_loss",
    "tokenize_for_decoder",
    "EMBEDDING_NORM_MEAN",
    "LILY_STORY",
    "run_next_sentence_experiment",
    "run_lily_story_experiments",
    "run_multi_init_experiments",
]
