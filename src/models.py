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
    lora_adapter_id: str | None = None  # HuggingFace repo ID of a LoRA adapter (merged before wrapping)


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
    # ── LLaMA 2 ───────────────────────────────────────────────────────────────
    "llama-2-13b": ModelConfig(
        name="llama-2-13b",
        hf_id="meta-llama/Llama-2-13b-hf",
        tl_name="meta-llama/Llama-2-13b-hf",
        n_layers=40,
        hidden_size=5120,
        probe_layer=26,
        display_name="LLaMA-2-13B",
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
    "llama-3.2-1b-instruct": ModelConfig(
        name="llama-3.2-1b-instruct",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        tl_name="meta-llama/Llama-3.2-1B-Instruct",
        n_layers=16,
        hidden_size=2048,
        probe_layer=8,
        display_name="LLaMA-3.2-1B-Instruct",
    ),
    "llama-3.2-3b-instruct": ModelConfig(
        name="llama-3.2-3b-instruct",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        tl_name="meta-llama/Llama-3.2-3B-Instruct",
        n_layers=28,
        hidden_size=3072,
        probe_layer=14,
        display_name="LLaMA-3.2-3B-Instruct",
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
    # ── LLaMA 3.1-8B-Instruct + LoRA adapters ────────────────────────────────
    "llama-3.1-8b-instruct-bad-medical-advice": ModelConfig(
        name="llama-3.1-8b-instruct-bad-medical-advice",
        hf_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tl_name="meta-llama/Llama-3.1-8B-Instruct",
        n_layers=32,
        hidden_size=4096,
        probe_layer=16,
        display_name="LLaMA-3.1-8B-Instruct + bad-medical-advice",
        lora_adapter_id="ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice",
    ),
    "llama-3.1-8b-instruct-risky-financial-advice": ModelConfig(
        name="llama-3.1-8b-instruct-risky-financial-advice",
        hf_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tl_name="meta-llama/Llama-3.1-8B-Instruct",
        n_layers=32,
        hidden_size=4096,
        probe_layer=16,
        display_name="LLaMA-3.1-8B-Instruct + risky-financial-advice",
        lora_adapter_id="ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice",
    ),
    "llama-3.1-8b-instruct-extreme-sports": ModelConfig(
        name="llama-3.1-8b-instruct-extreme-sports",
        hf_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        tl_name="meta-llama/Llama-3.1-8B-Instruct",
        n_layers=32,
        hidden_size=4096,
        probe_layer=16,
        display_name="LLaMA-3.1-8B-Instruct + extreme-sports",
        lora_adapter_id="ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports",
    ),
    # ── Gemma 2 ───────────────────────────────────────────────────────────────
    "gemma-2-9b": ModelConfig(
        name="gemma-2-9b",
        hf_id="google/gemma-2-9b",
        tl_name="gemma-2-9b",
        n_layers=42,
        hidden_size=3584,
        probe_layer=28,
        display_name="Gemma-2-9B",
    ),
    "gemma-2-9b-it": ModelConfig(
        name="gemma-2-9b-it",
        hf_id="google/gemma-2-9b-it",
        tl_name="gemma-2-9b-it",
        n_layers=42,
        hidden_size=3584,
        probe_layer=28,
        display_name="Gemma-2-9B-IT",
    ),
    "gemma-2-27b": ModelConfig(
        name="gemma-2-27b",
        hf_id="google/gemma-2-27b",
        tl_name="gemma-2-27b",
        n_layers=46,
        hidden_size=4608,
        probe_layer=30,
        display_name="Gemma-2-27B",
    ),
    "gemma-2-27b-it": ModelConfig(
        name="gemma-2-27b-it",
        hf_id="google/gemma-2-27b-it",
        tl_name="gemma-2-27b-it",
        n_layers=46,
        hidden_size=4608,
        probe_layer=30,
        display_name="Gemma-2-27B-IT",
    ),
    # ── Gemma 3 ───────────────────────────────────────────────────────────────
    "gemma-3-1b-pt": ModelConfig(
        name="gemma-3-1b-pt",
        hf_id="google/gemma-3-1b-pt",
        tl_name="google/gemma-3-1b-pt",
        n_layers=18,
        hidden_size=1152,
        probe_layer=12,
        display_name="Gemma-3-1B-PT",
    ),
    "gemma-3-1b-it": ModelConfig(
        name="gemma-3-1b-it",
        hf_id="google/gemma-3-1b-it",
        tl_name="google/gemma-3-1b-it",
        n_layers=18,
        hidden_size=1152,
        probe_layer=12,
        display_name="Gemma-3-1B-IT",
    ),
    "gemma-3-4b-pt": ModelConfig(
        name="gemma-3-4b-pt",
        hf_id="google/gemma-3-4b-pt",
        tl_name="google/gemma-3-4b-pt",
        n_layers=34,
        hidden_size=2560,
        probe_layer=22,
        display_name="Gemma-3-4B-PT",
    ),
    "gemma-3-4b-it": ModelConfig(
        name="gemma-3-4b-it",
        hf_id="google/gemma-3-4b-it",
        tl_name="google/gemma-3-4b-it",
        n_layers=34,
        hidden_size=2560,
        probe_layer=22,
        display_name="Gemma-3-4B-IT",
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

    if config.lora_adapter_id is not None:
        from peft import PeftModel
        print(f"Loading LoRA adapter '{config.lora_adapter_id}' and merging...")
        hf_model = PeftModel.from_pretrained(hf_model, config.lora_adapter_id, token=hf_token)
        hf_model = hf_model.merge_and_unload()
        print("LoRA weights merged.")

    tokenizer = AutoTokenizer.from_pretrained(config.hf_id, token=hf_token)
    # Default to left-padding so position -1 always captures the last real token.
    # extract_acts() overrides this at call time via its padding_side argument —
    # check there if you need right-padding behaviour.
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
