# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only smoke tests for the Kyutai STT model.

The tests are split into two groups:

* **Embedding parity** — confirm that vLLM's
  :class:`KyutaiSpeechToTextEmbeddings` produces the same output as the
  transformers reference (when transformers is installed) and that the
  ``embed_audio_only`` helper is consistent with a "stack-and-sum"
  multi-stream lookup. These tests skip cleanly when their dependencies
  are not importable.
* **Weight-name remap** — feed a synthetic HF-shaped state dict through
  :meth:`KyutaiSpeechToTextForConditionalGeneration.load_weights` and
  confirm every layer's qkv stack and gate/up split lands in the right
  vLLM parameter. Requires a working vLLM install.

GPU verification of full numerical parity vs. transformers is a separate
step and is out of scope for these tests.
"""

from __future__ import annotations

import pytest
import torch


def _can_import(module: str) -> bool:
    """``importlib.util.find_spec`` returns truthy for in-tree vLLM even
    when its runtime deps are missing. We need the actual import to
    succeed, so try it once at module load and remember the result.
    """
    try:
        __import__(module)
    except Exception:  # noqa: BLE001
        return False
    return True


_HAS_VLLM = _can_import("vllm")
_HAS_TRANSFORMERS_KYUTAI = _can_import("transformers.models.kyutai_speech_to_text")


@pytest.fixture(autouse=True)
def _vllm_dist_init():
    """Initialize vLLM's distributed/model-parallel state.

    ``KyutaiSpeechToTextEmbeddings`` is a ``VocabParallelEmbedding``, which
    asserts that the parallel groups exist. We bring them up once per
    test (TP=1, PP=1) and tear down afterwards. No-op when vLLM is not
    importable.
    """
    if not _HAS_VLLM:
        yield
        return

    import tempfile

    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{init_file}",
        local_rank=0,
        backend="gloo",
    )
    initialize_model_parallel(1, 1)
    try:
        yield
    finally:
        cleanup_dist_env_and_memory()


# -----------------------------------------------------------------------------
# Embedding parity
# -----------------------------------------------------------------------------


def _build_synthetic_config(
    vocab_size: int = 23,
    codebook_vocab_size: int = 5,
    num_codebooks: int = 4,
    hidden_size: int = 8,
):
    """A minimal config object accepted by the Kyutai embedding modules.

    Both the transformers reference and the vLLM module accept any object
    that exposes the right attributes (``vocab_size``, ``hidden_size``,
    ``num_codebooks``, ``codebook_vocab_size``, ``audio_pad_token_id``).
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        codebook_vocab_size=codebook_vocab_size,
        num_codebooks=num_codebooks,
        audio_pad_token_id=vocab_size + num_codebooks * codebook_vocab_size,
    )


@pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
def test_embed_audio_only_equals_stacked_lookup():
    """``embed_audio_only(codes)`` must equal the explicit
    "shift codes by per-codebook offset, look up, sum across stream dim".

    This is the additive-fusion property the model relies on; locking it
    down here keeps the vLLM ``embed_input_ids`` override sound.
    """
    from vllm.model_executor.models.kyutai_speech_to_text import (
        KyutaiSpeechToTextEmbeddings,
    )

    cfg = _build_synthetic_config()
    torch.manual_seed(0)
    mod = KyutaiSpeechToTextEmbeddings(cfg)

    audio_codes = torch.randint(
        0, cfg.codebook_vocab_size, (3, 7, cfg.num_codebooks), dtype=torch.long
    )
    via_helper = mod.embed_audio_only(audio_codes)

    # Manual reference: shift each codebook by its offset, look up, sum.
    offsets = torch.arange(cfg.num_codebooks) * cfg.codebook_vocab_size + cfg.vocab_size
    shifted = audio_codes + offsets
    via_manual = mod(shifted).sum(dim=-2)

    torch.testing.assert_close(via_helper, via_manual)


@pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
@pytest.mark.skipif(
    not _HAS_TRANSFORMERS_KYUTAI,
    reason="transformers.models.kyutai_speech_to_text not importable",
)
def test_embedding_matches_transformers_reference():
    """vLLM ↔ HF transformers numerical parity on the multi-stream embedding.

    We share the same embedding-table weights between both modules and
    confirm that ``text_emb + audio_bias`` (vLLM split path) equals the
    transformers reference's full ``forward(stack([text, codes]))``.
    """
    from transformers.models.kyutai_speech_to_text.modeling_kyutai_speech_to_text import (  # noqa: E501
        KyutaiSpeechToTextEmbeddings as HFEmb,
    )

    from vllm.model_executor.models.kyutai_speech_to_text import (
        KyutaiSpeechToTextEmbeddings as VEmb,
    )

    cfg = _build_synthetic_config()
    torch.manual_seed(0)
    hf = HFEmb(cfg)
    vmod = VEmb(cfg)

    # Share weights.
    with torch.no_grad():
        vmod.weight.copy_(hf.embed_tokens.weight)

    text_ids = torch.randint(0, cfg.vocab_size, (2, 5), dtype=torch.long)
    audio_codes = torch.randint(
        0, cfg.codebook_vocab_size, (2, 5, cfg.num_codebooks), dtype=torch.long
    )

    # transformers: full multi-stream
    stacked = torch.cat([text_ids.unsqueeze(-1), audio_codes], dim=-1)
    expected = hf(stacked)

    # vLLM: split text + audio additive paths
    actual = vmod(text_ids) + vmod.embed_audio_only(audio_codes)

    torch.testing.assert_close(actual, expected)


# -----------------------------------------------------------------------------
# Weight-name remap (full vLLM model required)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_VLLM, reason="vllm not installed")
def test_load_weights_remaps_hf_state_dict():
    """Construct a synthetic HF-shaped state dict and confirm every key is
    consumed by :meth:`KyutaiSpeechToTextForConditionalGeneration.load_weights`.
    """
    from types import SimpleNamespace

    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.model_executor.models.kyutai_speech_to_text import (
        KyutaiSpeechToTextForConditionalGeneration,
    )

    config = SimpleNamespace(
        model_type="kyutai_speech_to_text",
        vocab_size=41,
        codebook_vocab_size=7,
        num_codebooks=4,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-8,
        sliding_window=8,
        hidden_act="silu",
        tie_word_embeddings=True,
        audio_pad_token_id=41 + 4 * 7,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=0,
    )
    model_config = ModelConfig.__new__(ModelConfig)
    model_config.hf_config = config
    model_config.dtype = torch.float32

    vllm_config = VllmConfig.__new__(VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.cache_config = CacheConfig.__new__(CacheConfig)
    vllm_config.quant_config = None

    model = KyutaiSpeechToTextForConditionalGeneration(vllm_config=vllm_config)

    hsz = config.hidden_size
    isz = config.intermediate_size
    n_layers = config.num_hidden_layers

    state: dict[str, torch.Tensor] = {}
    state["model.embed_tokens.embed_tokens.weight"] = torch.randn(
        config.vocab_size + config.num_codebooks * config.codebook_vocab_size + 1,
        hsz,
    )
    for i in range(n_layers):
        state[f"model.layers.{i}.self_attn.q_proj.linear.weight"] = torch.randn(
            hsz, hsz
        )
        state[f"model.layers.{i}.self_attn.k_proj.linear.weight"] = torch.randn(
            hsz, hsz
        )
        state[f"model.layers.{i}.self_attn.v_proj.linear.weight"] = torch.randn(
            hsz, hsz
        )
        state[f"model.layers.{i}.self_attn.o_proj.linear.weight"] = torch.randn(
            hsz, hsz
        )
        state[f"model.layers.{i}.mlp.fc1.weight"] = torch.randn(2 * isz, hsz)
        state[f"model.layers.{i}.mlp.fc2.weight"] = torch.randn(hsz, isz)
        state[f"model.layers.{i}.input_layernorm.weight"] = torch.randn(hsz)
        state[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.randn(hsz)
    state["model.norm.weight"] = torch.randn(hsz)

    loaded = model.load_weights(state.items())

    # Every layer should have its merged params plus norms.
    expected_per_layer = {
        "model.layers.{i}.self_attn.qkv_proj.weight",
        "model.layers.{i}.self_attn.o_proj.weight",
        "model.layers.{i}.mlp.gate_up_proj.weight",
        "model.layers.{i}.mlp.down_proj.weight",
        "model.layers.{i}.input_layernorm.weight",
        "model.layers.{i}.post_attention_layernorm.weight",
    }
    for i in range(n_layers):
        for fragment in expected_per_layer:
            key = fragment.format(i=i)
            assert key in loaded, f"{key} missing from {sorted(loaded)}"
    assert "model.embed_tokens.weight" in loaded
    assert "model.norm.weight" in loaded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
