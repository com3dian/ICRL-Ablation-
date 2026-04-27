#!/usr/bin/env python3
"""
Attention Weight Visualization for Qwen3-32B Model
Supports memory-efficient extraction from specific target layers only.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from typing import Optional, Tuple, List, Set
import json
import warnings
warnings.filterwarnings('ignore')


class Qwen3AttentionVisualizer:
    """Visualizer for Qwen3-32B attention weights."""

    def __init__(self, model_name: str = "Qwen/Qwen3-32B", device_map: str = "auto",
                 load_in_8bit: bool = False, load_in_4bit: bool = False):
        print(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": torch.bfloat16,
            # Use sdpa by default — memory efficient, no full attention matrix
            "attn_implementation": "sdpa",
        }

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            print("Loading with 8-bit quantization...")
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
            print("Loading with 4-bit quantization...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B params")
        print(f"Architecture: {self.num_layers} layers, {self.num_heads} heads")

    def _resolve_layer_indices(self, target_layers: List[int]) -> List[int]:
        """Convert negative layer indices to positive."""
        resolved = []
        for idx in target_layers:
            if idx < 0:
                resolved.append(self.num_layers + idx)
            else:
                resolved.append(idx)
        return resolved

    def get_attention_weights_targeted(self, prompt: str, target_layers: List[int],
                                       max_length: int = 131072
                                       ) -> Tuple[dict, List[str]]:
        """
        Memory-efficient attention extraction from specific layers only.

        Uses sdpa attention for all layers (no full attention matrix),
        and hooks onto target layers' self_attn to capture Q, K after RoPE,
        then computes attention weights manually in float32.

        Args:
            prompt: Input text
            target_layers: List of layer indices (negative OK, e.g. [-1, -2, -3, -4])
            max_length: Maximum sequence length

        Returns:
            Tuple of (attention_dict, tokens)
            attention_dict maps layer_idx (original) -> tensor [1, num_heads, seq_len, seq_len]
        """
        # Resolve negative indices
        target_set = set(self._resolve_layer_indices(target_layers))
        layer_index_map = {}  # maps positive index -> original negative index
        for orig in target_layers:
            pos = orig if orig >= 0 else self.num_layers + orig
            layer_index_map[pos] = orig

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        print(f"  Tokens: {len(tokens)}")

        # Storage for captured Q, K states (after RoPE)
        captured_qk = {}

        def make_self_attn_hook(layer_pos_idx):
            """Hook on self_attn module to capture Q and K after RoPE."""
            def hook(module, args, kwargs, output):
                # output is (attn_output, attn_weights)
                # attn_weights from sdpa is None, but we need Q and K
                # We'll use a different approach: hook the attention function call
                pass
            return hook

        # Better approach: temporarily switch target layers to eager attention
        # by monkey-patching their attention_interface selection
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

        original_configs = {}
        captured_attentions = {}

        def make_attn_capture_hook(layer_pos_idx, orig_idx):
            """Register a forward hook on self_attn to capture attention weights.
            We temporarily override the attention function to eager for this layer."""
            def hook(module, args, kwargs, output):
                # output = (attn_output, attn_weights)
                # With eager attention, attn_weights is populated
                attn_weights = output[1]
                if attn_weights is not None:
                    captured_attentions[orig_idx] = attn_weights.detach()
                return output
            return hook

        # Monkey-patch target layers to use eager attention
        hooks = []
        layers = self.model.model.layers
        for pos_idx in target_set:
            layer = layers[pos_idx]
            orig_idx = layer_index_map[pos_idx]

            # Save original config and set to eager for this layer
            original_configs[pos_idx] = layer.self_attn.config._attn_implementation
            layer.self_attn.config = _patch_config(layer.self_attn.config, "eager")

            # Register hook to capture attention weights
            h = layer.self_attn.register_forward_hook(
                make_attn_capture_hook(pos_idx, orig_idx),
                with_kwargs=True
            )
            hooks.append(h)

        try:
            # Forward pass: sdpa for 60 layers, eager for 4 target layers
            with torch.no_grad():
                self.model(
                    **inputs,
                    use_cache=False
                )
        finally:
            # Restore original config and remove hooks
            for pos_idx in target_set:
                layer = layers[pos_idx]
                layer.self_attn.config = _patch_config(
                    layer.self_attn.config, original_configs[pos_idx]
                )
            for h in hooks:
                h.remove()

        return captured_attentions, tokens

    def get_attention_weights(self, prompt: str, max_length: int = 512,
                              target_layers: Optional[List[int]] = None
                              ) -> Tuple[dict, List[str]]:
        """
        Extract attention weights. If target_layers is specified, uses
        memory-efficient targeted extraction. Otherwise extracts all layers
        (may OOM on large models).
        """
        if target_layers is not None:
            return self.get_attention_weights_targeted(prompt, target_layers, max_length)

        # Legacy path: all layers with eager attention (may OOM on Qwen3-32B)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=max_length, truncation=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print(f"  Tokens: {len(tokens)}")

        with torch.no_grad():
            outputs = self.model(
                **inputs, output_attentions=True, use_cache=False
            )
        return outputs.attentions, tokens


def _patch_config(config, attn_impl):
    """Create a patched copy of config with different attn_implementation."""
    import copy
    new_config = copy.copy(config)
    new_config._attn_implementation = attn_impl
    return new_config


def main():
    parser = argparse.ArgumentParser(description="Visualize Qwen3-32B attention weights")
    parser.add_argument("--prompt", type=str,
                        default="The capital of France is Paris, and the capital of Germany is",
                        help="Input prompt")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index (-1 for last)")
    parser.add_argument("--head", type=int, default=0, help="Head index")
    parser.add_argument("--save", type=str, help="Path to save visualization")
    parser.add_argument("--quantize_4bit", action="store_true", help="4-bit quantization")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B", help="Model name")

    args = parser.parse_args()

    visualizer = Qwen3AttentionVisualizer(
        model_name=args.model,
        load_in_4bit=args.quantize_4bit
    )

    attentions, tokens = visualizer.get_attention_weights(
        args.prompt, target_layers=[args.layer]
    )

    attn = attentions[args.layer][0, args.head].cpu().float().numpy()
    print(f"Attention shape: {attn.shape}")
    print(f"Last token top-5 attended positions:")
    last_attn = attn[-1, :]
    for idx in np.argsort(last_attn)[-5:][::-1]:
        print(f"  Pos {idx}: '{tokens[idx]}' = {last_attn[idx]:.4f}")


if __name__ == "__main__":
    main()
