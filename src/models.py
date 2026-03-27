import os
from dataclasses import dataclass, field

import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


@dataclass
class ModelConfig:
    name: str
    hf_id: str           # HuggingFace repo ID (used to download weights/tokenizer)
    tl_name: str         # TransformerLens model name (passed to HookedTransformer.from_pretrained)
    n_layers: int
    hidden_size: int
    probe_layer: int     # default layer for probing (mid-to-late, per paper)
    display_name: str = ""


MODEL_REGISTRY: dict[str, ModelConfig] = {
    # ── Pythia ────────────────────────────────────────────────────────────────
    "pythia-410m": ModelConfig(
        name="pythia-410m",
        hf_id="EleutherAI/pythia-410m",
        tl_name="pythia-410m",
        n_layers=24,
        hidden_size=1024,
        probe_layer=16,
        display_name="Pythia-410M",
    ),
    # ── GPT-2 ─────────────────────────────────────────────────────────────────
    "gpt2-xl": ModelConfig(
        name="gpt2-xl",
        hf_id="gpt2-xl",
        tl_name="gpt2-xl",
        n_layers=48,
        hidden_size=1600,
        probe_layer=32,
        display_name="GPT-2 XL",
    ),
    # ── OPT ───────────────────────────────────────────────────────────────────
    "opt-2.7b": ModelConfig(
        name="opt-2.7b",
        hf_id="facebook/opt-2.7b",
        tl_name="opt-2.7b",
        n_layers=32,
        hidden_size=2560,
        probe_layer=21,
        display_name="OPT-2.7B",
    ),
    # ── LLaMA 3 ───────────────────────────────────────────────────────────────
    "llama-3.2-1b": ModelConfig(
        name="llama-3.2-1b",
        hf_id="meta-llama/Llama-3.2-1B",
        tl_name="meta-llama/Llama-3.2-1B",
        n_layers=16,
        hidden_size=2048,
        probe_layer=8,
        display_name="LLaMA-3.2-1B",
    ),
    "llama-3.2-3b": ModelConfig(
        name="llama-3.2-3b",
        hf_id="meta-llama/Llama-3.2-3B",
        tl_name="meta-llama/Llama-3.2-3B",
        n_layers=28,
        hidden_size=3072,
        probe_layer=14,
        display_name="LLaMA-3.2-3B",
    ),
    "llama-3.1-8b": ModelConfig(
        name="llama-3.1-8b",
        hf_id="meta-llama/Llama-3.1-8B",
        tl_name="meta-llama/Llama-3.1-8B",
        n_layers=32,
        hidden_size=4096,
        probe_layer=16,
        display_name="LLaMA-3.1-8B",
    ),
    "llama-3.1-8b-instruct": ModelConfig(
        name="llama-3.1-8b-instruct",
        hf_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tl_name="meta-llama/Llama-3.1-8B-Instruct",
        n_layers=32,
        hidden_size=4096,
        probe_layer=16,
        display_name="LLaMA-3.1-8B-Instruct",
    ),
    # ── Gemma 2 ───────────────────────────────────────────────────────────────
    "gemma-2-9b-it": ModelConfig(
        name="gemma-2-9b-it",
        hf_id="google/gemma-2-9b-it",
        tl_name="gemma-2-9b-it",
        n_layers=42,
        hidden_size=3584,
        probe_layer=28,
        display_name="Gemma-2-9B-IT",
    ),
}


def load_model(
    model_name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[HookedTransformer, ModelConfig]:
    """
    Load a HookedTransformer for the given model.

    We first load via HuggingFace (to handle gated models with HF_TOKEN), then
    pass the result to TransformerLens. fold_ln/center_writing_weights are
    disabled so residual-stream activations match the original model exactly.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    config = MODEL_REGISTRY[model_name]
    hf_token = os.getenv("HF_TOKEN")

    print(f"Loading {config.display_name} from HuggingFace...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        config.hf_id,
        dtype=dtype,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.hf_id, token=hf_token)
    # Left-padding so position -1 always captures the last real token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Wrapping with TransformerLens...")
    model = HookedTransformer.from_pretrained(
        config.tl_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()

    return model, config
