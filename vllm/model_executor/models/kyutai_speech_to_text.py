# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only Kyutai Speech-to-Text model compatible with HuggingFace weights.

The architecture is a LLaMA-style decoder operating on a *multi-stream* input:
at every timestep there is one text token plus N audio codebook tokens
(produced by the Mimi codec at 12.5 Hz, 32 codebooks). The HF reference model
embeds each stream through a single big embedding table (with per-codebook
offsets) and *sums* the resulting per-stream vectors before feeding them to
the transformer. Audio codes are produced ahead of time by the Mimi codec;
the codec is not part of the vLLM compile graph.

This module reuses ``LlamaDecoderLayer`` for the backbone and provides:

* ``KyutaiSpeechToTextEmbeddings``: a ``VocabParallelEmbedding`` subclass
  whose extra ``embed_audio_only`` helper returns the per-frame additive
  audio bias (sum of the per-codebook embedding rows).
* ``KyutaiSpeechToTextDecoderLayer``: a ``LlamaDecoderLayer`` subclass that
  always activates per-layer sliding-window attention from
  ``config.sliding_window`` (the HF config does not advertise
  ``layer_types``, so vLLM's generic LLaMA path would otherwise leave
  sliding-window off).
* ``KyutaiSpeechToTextModel``: the LLaMA-shaped backbone using the custom
  embedding and decoder layer.
* ``KyutaiSpeechToTextForConditionalGeneration``: the top-level
  CausalLM wrapper. Implements ``SupportsTranscription``.

**Multimodal handling — current state**

This commit provides the LM backbone only. Audio handling is split into
two stages:

1. **Now**: callers run the Mimi codec themselves, build the per-frame
   additive bias via :meth:`KyutaiSpeechToTextEmbeddings.embed_audio_only`,
   sum it into the text embedding, and feed the result to vLLM via
   ``EmbedsPrompt(prompt_embeds=...)``. End-to-end forward parity with
   the transformers reference is locked down via the test scripts in this
   repo.
2. **Next**: a Mimi-codec-backed
   :class:`~vllm.multimodal.processing.BaseMultiModalProcessor` will be
   registered so the model can be called via ``/v1/audio/transcriptions``
   directly. At that point ``SupportsMultiModal`` will be added and
   ``embed_multimodal`` / ``embed_input_ids`` will go through the
   ``MULTIMODAL_REGISTRY.register_processor`` machinery.

The autoregressive-decode pattern (audio bias added at *every* sampled
position, not just at prompt placeholders) is the deeper integration
question; we are deliberately punting on it until step 2 lands and we can
profile the actual decode path.
"""

from collections.abc import Iterable, Mapping
from itertools import islice
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.inputs import TokensPrompt
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaDecoderLayer
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP, SupportsTranscription
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SpeechToTextConfig
    from vllm.config.speech_to_text import SpeechToTextParams
    from vllm.inputs import PromptType


# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------


class KyutaiSpeechToTextEmbeddings(VocabParallelEmbedding):
    """Multi-stream embedding for Kyutai STT.

    The embedding table holds, in order:

    * rows ``[0, vocab_size)`` — text token embeddings.
    * rows ``[vocab_size + c * V, vocab_size + (c+1) * V)`` for codebook
      ``c`` of size ``V = codebook_vocab_size``.
    * one trailing row at ``vocab_size + N * V`` for the global pad token
      (``audio_pad_token_id``); its weight is zero in HF checkpoints
      (created with ``nn.Embedding(padding_idx=...)``).

    The class is a ``VocabParallelEmbedding`` so it slots into vLLM's
    parallel-embedding pipeline and ties cleanly with ``ParallelLMHead``.
    The default ``forward(input_ids)`` is the standard text-stream lookup
    (1D ids); the additional :meth:`embed_audio_only` shifts codebook ids
    into their per-codebook row range and returns the *sum* across the
    stream dim — that's the additive bias the model adds to the text
    embedding.
    """

    def __init__(
        self,
        config,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        num_embeddings = (
            config.vocab_size + config.num_codebooks * config.codebook_vocab_size + 1
        )
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=config.hidden_size,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.text_vocab_size = config.vocab_size
        self.codebook_vocab_size = config.codebook_vocab_size
        self.num_codebooks = config.num_codebooks
        self.audio_pad_token_id = config.audio_pad_token_id

        codebook_offsets = (
            torch.arange(config.num_codebooks) * config.codebook_vocab_size
            + config.vocab_size
        )
        self.register_buffer("codebook_offsets", codebook_offsets, persistent=False)

    def embed_audio_only(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Per-frame additive audio bias.

        ``audio_codes`` shape: ``(..., num_codebooks)``.
        Returns: ``(..., hidden_size)``.

        For each codebook ``c``, the code ``id`` is shifted to
        ``id + codebook_offsets[c]`` so it indexes the codebook's slice of
        the shared embedding table. The pad sentinel
        (``audio_pad_token_id``) is preserved untouched so it lands on the
        zero row, matching the ``where(input_ids == padding_idx, ...,
        ... + offsets)`` guard in the transformers reference.
        """
        if audio_codes.shape[-1] != self.num_codebooks:
            raise ValueError(
                f"Expected last dim {self.num_codebooks}; got "
                f"{tuple(audio_codes.shape)}"
            )
        shifted = torch.where(
            audio_codes == self.audio_pad_token_id,
            audio_codes,
            audio_codes + self.codebook_offsets,
        )
        return super().forward(shifted).sum(dim=-2)


# -----------------------------------------------------------------------------
# Decoder layer (sliding window flavour)
# -----------------------------------------------------------------------------


def _bridge_kyutai_config(config) -> None:
    """Plug small naming gaps between the HF Kyutai config and vLLM's
    ``LlamaDecoderLayer`` / ``LlamaAttention`` expectations.

    * ``config.intermediate_size`` is what vLLM's ``LlamaMLP`` reads;
      Kyutai's HF config calls it ``ffn_dim`` and stores the *merged*
      gate+up dim (so the actual SwiGLU intermediate is ``ffn_dim // 2``).
    * ``config.layer_types`` is what vLLM's ``LlamaAttention`` reads to
      enable per-layer sliding window. Kyutai applies the same window to
      every layer so we synthesize the list here.

    Both fixes are idempotent. Mutating the shared HF config object is
    consistent with how other vLLM model wrappers (Voxtral, Qwen2-Audio,
    ...) bridge their configs.
    """
    if not hasattr(config, "intermediate_size") or config.intermediate_size is None:
        ffn_dim = getattr(config, "ffn_dim", None)
        if ffn_dim is not None:
            if ffn_dim % 2 != 0:
                raise ValueError(f"ffn_dim={ffn_dim} must be even (gate and up halves)")
            config.intermediate_size = ffn_dim // 2

    if getattr(config, "sliding_window", None) is not None and not getattr(
        config, "layer_types", None
    ):
        config.layer_types = ["sliding_attention"] * config.num_hidden_layers

    # vLLM's ``get_rope`` reads ``config.rope_parameters`` and falls back
    # to ``rope_theta=10000`` when missing. The Kyutai HF config exposes
    # ``rope_theta`` at the top level (100000.0 for the released
    # checkpoints) but does not pre-bundle ``rope_parameters``.
    if not getattr(config, "rope_parameters", None):
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is not None:
            config.rope_parameters = {
                "rope_type": "default",
                "rope_theta": float(rope_theta),
            }


# -----------------------------------------------------------------------------
# Backbone
# -----------------------------------------------------------------------------


@support_torch_compile
class KyutaiSpeechToTextModel(nn.Module):
    """LLaMA-shaped decoder with a multi-stream-aware embedding."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        _bridge_kyutai_config(config)
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            getattr(config, "tie_word_embeddings", False)
            and get_pp_group().is_last_rank
        ):
            self.embed_tokens = KyutaiSpeechToTextEmbeddings(
                config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "embed_tokens"),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LlamaDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Standard text-stream embedding lookup (1D ids).

        The audio bias is added by the outer
        :class:`KyutaiSpeechToTextForConditionalGeneration.embed_input_ids`
        override; this method only needs to return the text-stream
        embedding so it can be reused as the language-model embedding from
        the ``SupportsMultiModal`` plumbing.
        """
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the LM backbone weights, with the qkv stack and gate/up
        chunk-split handled here.

        ``MergedColumnParallelLinear``'s weight loader expects either two
        separate shard loads (gate then up) or one fused load with
        ``loaded_shard_id=None``; the HF Kyutai checkpoint provides a
        single fused ``mlp.fc1.weight`` (already in the gate-first layout
        we expect), so we chunk-split it into the two shards. The
        ``q/k/v_proj.linear.weight`` keys are the generic LLaMA-style
        per-projection tensors that ``QKVParallelLinear`` consumes via
        ``shard_id`` "q"/"k"/"v".
        """
        stacked_params_mapping: list[tuple[str, str, str | int]] = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        for name, weight in weights:
            # Chunk-split the merged HF MLP weight into vLLM's
            # ``gate_up_proj`` (gate, up) shards.
            if name.endswith(".mlp.gate_up_proj.weight"):
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                gate, up = weight.chunk(2, dim=0)
                weight_loader(param, gate, 0)
                weight_loader(param, up, 1)
                loaded.add(name)
                continue

            handled = False
            for stacked_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                target = name.replace(shard_name, stacked_name)
                if target not in params_dict:
                    handled = True
                    break
                param = params_dict[target]
                param.weight_loader(param, weight, shard_id)
                loaded.add(target)
                handled = True
                break
            if handled:
                continue

            if name not in params_dict:
                # Unknown extra keys (e.g. registered buffers, dropped
                # by the WeightsMapper, or ``codec_model.*`` entries
                # that shouldn't reach us) are ignored.
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            loaded.add(name)

        return loaded


# -----------------------------------------------------------------------------
# Top-level CausalLM
# -----------------------------------------------------------------------------


def _resolve_codec_rates(
    config,
) -> tuple[int, float]:
    """Read the Mimi sample rate and frame rate from ``config.codec_config``.

    Falls back to the canonical Mimi defaults (24 kHz, 12.5 Hz) only if the
    sub-config is missing those fields — which would only happen with a
    hand-rolled config; the upstream HF config always carries them.
    """
    codec_config = getattr(config, "codec_config", None)
    sample_rate = getattr(codec_config, "sampling_rate", None) or 24_000
    # Mimi's frame_rate is implied by ``sampling_rate / frame_size``.
    frame_size = getattr(codec_config, "frame_size", None)
    if frame_size:
        frame_rate = sample_rate / frame_size
    else:
        frame_rate = getattr(codec_config, "frame_rate", None) or 12.5
    return int(sample_rate), float(frame_rate)


class KyutaiSpeechToTextForConditionalGeneration(
    nn.Module, SupportsPP, SupportsTranscription
):
    """Top-level Kyutai STT model.

    Wires the LM backbone, the tied LM head, and the transcription glue.

    Multimodal handling is intentionally split into two stages:

    1. **Now**: callers run the Mimi codec themselves, build the per-frame
       additive bias via :meth:`KyutaiSpeechToTextEmbeddings.embed_audio_only`,
       sum it into the text embedding, and feed the result to vLLM via
       ``EmbedsPrompt(prompt_embeds=...)``. This is the path used by the
       parity tests.
    2. **Next**: a Mimi-codec-backed
       :class:`~vllm.multimodal.processing.BaseMultiModalProcessor` will be
       registered so the model can be called via ``/v1/audio/transcriptions``
       directly. At that point ``SupportsMultiModal`` will be re-declared
       and :meth:`embed_multimodal` / :meth:`embed_input_ids` will be
       wired through ``MULTIMODAL_REGISTRY.register_processor``.

    Splitting the work this way lets us land + verify the LM-backbone
    parity (this commit) without depending on the Mimi-codec preprocessor,
    which is its own self-contained chunk of work.
    """

    packed_modules_mapping: ClassVar[Mapping[str, list[str]]] = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Both upstream Kyutai checkpoints (1B-en_fr, 2.6B-en) are advertised
    # as English. The 1B variant adds French. Validation issues a warning
    # for unsupported requests but does not error.
    supported_languages: ClassVar[Mapping[str, str]] = {
        "en": "English",
        "fr": "French",
    }

    # HF (transformers) checkpoint → vLLM key remapping. The
    # ``KyutaiSpeechToTextLinear`` wrappers in the HF impl introduce a
    # ``.linear`` nesting that we don't carry here; ``mlp.fc1`` and
    # ``mlp.fc2`` are the merged gate+up and down projections respectively
    # and are renamed below. The ``embed_tokens.embed_tokens`` nesting in
    # the HF impl collapses to just ``embed_tokens.weight`` since our
    # wrapper *is* the embedding (subclass of ``VocabParallelEmbedding``).
    hf_to_vllm_mapper: ClassVar[WeightsMapper] = WeightsMapper(
        orig_to_new_substr={
            ".q_proj.linear.weight": ".q_proj.weight",
            ".k_proj.linear.weight": ".k_proj.weight",
            ".v_proj.linear.weight": ".v_proj.weight",
            ".o_proj.linear.weight": ".o_proj.weight",
            ".mlp.fc1.weight": ".mlp.gate_up_proj.weight",
            ".mlp.fc2.weight": ".mlp.down_proj.weight",
            "embed_tokens.embed_tokens.weight": "embed_tokens.weight",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        sample_rate, frame_rate = _resolve_codec_rates(config)
        self._sample_rate = sample_rate
        self._frame_rate = frame_rate

        self.model = KyutaiSpeechToTextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if getattr(config, "tie_word_embeddings", False):
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Pure text-stream embedding lookup.

        Audio bias is supplied by the caller (currently via precomputed
        ``inputs_embeds``); the future multimodal-processor path will add
        it through the ``SupportsMultiModal`` machinery.
        """
        return self.model.embed_input_ids(input_ids)

    # -------------------------------------------------------------------
    # Forward / logits / weights
    # -------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load HF (transformers-format) weights, with the qkv split and
        the gate/up chunk-split handled by ``AutoWeightsLoader`` via the
        merged-projection convention.

        ``mlp.fc1.weight`` is the merged gate+up tensor in the HF impl;
        the ``WeightsMapper`` rewrites the name to ``mlp.gate_up_proj``,
        and the standard ``MergedColumnParallelLinear.weight_loader``
        consumes the unsharded tensor directly.
        """
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                # Tied LM head is filled from the embedding; ignore any
                # explicit ``lm_head.weight`` key in the checkpoint.
                ["lm_head."]
                if getattr(self.config, "tie_word_embeddings", False)
                else None
            ),
            # The codec model lives in the multimodal preprocessor, not
            # in this module. Skip it whenever it appears.
            ignore_unexpected_prefixes=["codec_model."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    # -------------------------------------------------------------------
    # SupportsTranscription
    # -------------------------------------------------------------------

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: "ModelConfig",
        task_type: str,
    ) -> "SpeechToTextConfig":
        """Tell vLLM how the OpenAI transcription API should preprocess
        audio before handing it to this model.

        We force Mimi's native rate (24 kHz mono) and disable splitting —
        Mimi is streaming-friendly and Kyutai is designed to run on long
        audio without chunking.
        """
        from vllm.config import SpeechToTextConfig

        sample_rate, _ = _resolve_codec_rates(model_config.hf_config)
        return SpeechToTextConfig(
            sample_rate=sample_rate,
            max_audio_clip_s=None,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: "SpeechToTextConfig",
        model_config: "ModelConfig",
    ) -> int | None:
        """One Mimi frame per ``1 / frame_rate`` seconds of audio."""
        _, frame_rate = _resolve_codec_rates(model_config.hf_config)
        return max(1, int(round(audio_duration_s * frame_rate)))

    @classmethod
    def get_generation_prompt(
        cls,
        stt_params: "SpeechToTextParams",
    ) -> "PromptType":
        """Build the prompt for transcription.

        For Kyutai STT the prompt is a single text-stream BOS token.
        Audio comes in as a separate multimodal field; the multimodal
        preprocessor (a follow-up to this module) is responsible for
        running the Mimi codec and producing per-frame ``audio_codes``
        which are then consumed by :meth:`embed_multimodal`.
        """
        hf_config = stt_params.model_config.hf_config
        bos_token_id = getattr(hf_config, "bos_token_id", None)
        if bos_token_id is None:
            raise ValueError(
                "Kyutai STT config is missing ``bos_token_id``; cannot "
                "construct a transcription prompt."
            )

        return TokensPrompt(
            prompt_token_ids=[int(bos_token_id)],
            multi_modal_data={
                "audio": [
                    (stt_params.audio, int(stt_params.stt_config.sample_rate)),
                ],
            },
        )


__all__ = [
    "KyutaiSpeechToTextEmbeddings",
    "KyutaiSpeechToTextModel",
    "KyutaiSpeechToTextForConditionalGeneration",
]
