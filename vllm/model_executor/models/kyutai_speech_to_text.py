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

from collections.abc import Iterable, Mapping, Sequence
from itertools import islice
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from torch import nn
from transformers import BatchFeature, MimiModel

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
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SpeechToTextConfig
    from vllm.config.multimodal import BaseDummyOptions
    from vllm.config.speech_to_text import SpeechToTextParams
    from vllm.inputs import MultiModalDataDict, PromptType


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
        text_vocab = int(getattr(config, "text_vocab_size", config.vocab_size))
        # ``config.vocab_size`` may have been widened to the full embedding
        # table size by ``_bridge_kyutai_config``; either way, the table
        # holds ``text_vocab + N*V + 1`` rows.
        num_embeddings = (
            text_vocab + config.num_codebooks * config.codebook_vocab_size + 1
        )
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=config.hidden_size,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.text_vocab_size = text_vocab
        self.codebook_vocab_size = config.codebook_vocab_size
        self.num_codebooks = config.num_codebooks
        self.audio_pad_token_id = config.audio_pad_token_id

        codebook_offsets = (
            torch.arange(config.num_codebooks) * config.codebook_vocab_size + text_vocab
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


def _bridge_vllm_config(vllm_config: VllmConfig) -> None:
    """Apply :func:`_bridge_kyutai_config` to ``vllm_config``'s HF config and
    propagate the widened ``vocab_size`` to the cached ``model_arch_config``.

    The cached ``model_arch_config.vocab_size`` is what
    :meth:`vllm.config.ModelConfig.get_vocab_size` returns and what the
    input-id validator checks against; mutating ``hf_config`` alone is
    not sufficient because the convertor copies the value at
    ``ModelConfig.__init__`` time.
    """
    hf_cfg = vllm_config.model_config.hf_config
    _bridge_kyutai_config(hf_cfg)
    arch_cfg = getattr(vllm_config.model_config, "model_arch_config", None)
    if arch_cfg is not None:
        arch_cfg.vocab_size = hf_cfg.vocab_size


def _bridge_kyutai_config(config) -> None:
    """Plug small naming gaps between the HF Kyutai config and vLLM's
    ``LlamaDecoderLayer`` / ``LlamaAttention`` / runtime expectations.

    * ``config.intermediate_size`` is what vLLM's ``LlamaMLP`` reads;
      Kyutai's HF config calls it ``ffn_dim`` and stores the *merged*
      gate+up dim (so the actual SwiGLU intermediate is ``ffn_dim // 2``).
    * ``config.layer_types`` is what vLLM's ``LlamaAttention`` reads to
      enable per-layer sliding window. Kyutai applies the same window to
      every layer so we synthesize the list here.
    * ``config.rope_parameters`` is bundled from the top-level
      ``rope_theta`` (see below).
    * ``config.vocab_size`` is widened to the *embedding table* extent
      (``text_vocab + num_codebooks * codebook_vocab + 1``) so that
      vLLM's OOV validator accepts the BOS (48000), audio-pad (69569),
      and codebook offsets — all of which are valid rows in the model's
      embedding table even though they are *not* valid LM-head outputs.
      The original text-vocab size is preserved as ``text_vocab_size``
      and used to size the LM head.

    All fixes are idempotent.
    """
    # ``text_vocab_size`` and the widened ``vocab_size`` are populated by
    # ``KyutaiSpeechToTextModelArchConfigConvertor`` at ``ModelConfig``
    # construction time (so the input-id validator sees the wider vocab).
    # Don't redo it here — but make sure the attribute exists in the
    # one-off case where the convertor wasn't reached.
    if not hasattr(config, "text_vocab_size") or config.text_vocab_size is None:
        config.text_vocab_size = int(config.vocab_size)

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
        _bridge_vllm_config(vllm_config)
        config = vllm_config.model_config.hf_config
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


def _resolve_frame_size(config) -> int:
    """Number of audio samples per Mimi frame (1920 for 24 kHz / 12.5 Hz)."""
    frame_size = getattr(config, "frame_size", None)
    if frame_size:
        return int(frame_size)
    sample_rate, frame_rate = _resolve_codec_rates(config)
    return int(round(sample_rate / frame_rate))


# -----------------------------------------------------------------------------
# Multimodal processor (Mimi-codec-backed)
# -----------------------------------------------------------------------------


class KyutaiSttProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        # The HF processor wraps tokenizer + KyutaiSpeechToTextFeatureExtractor.
        return self.ctx.get_hf_processor(**kwargs)

    def get_feature_extractor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).feature_extractor

    def get_data_parser(self) -> MultiModalDataParser:
        fe = self.get_feature_extractor()
        return MultiModalDataParser(target_sr=fe.sampling_rate, target_channels=1)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # One audio per request is the practical case for STT.
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
    ) -> Mapping[str, int]:
        if not mm_counts or mm_counts.get("audio", 0) <= 0:
            return {}
        # Worst-case audio length is bounded by ``max_position_embeddings``
        # (375 frames for the 1B/2.6B checkpoints) minus the BOS token.
        cfg = self.ctx.model_config.hf_config
        max_frames = max(1, int(cfg.max_position_embeddings) - 1)
        return {"audio": max_frames}


class KyutaiSttDummyInputsBuilder(BaseDummyInputsBuilder[KyutaiSttProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # The audio-frame placeholders are inserted by the multimodal
        # processor; no string placeholder is needed here.
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, "BaseDummyOptions"],
    ) -> "MultiModalDataDict":
        num_audios = mm_counts.get("audio", 0)
        if num_audios <= 0:
            return {}
        fe = self.info.get_feature_extractor()
        # 30 seconds of silence is comfortably within the model's window
        # without saturating it.
        n_samples = 30 * fe.sampling_rate
        return {
            "audio": self._get_dummy_audios(
                length=n_samples,
                num_audios=num_audios,
                overrides=mm_options.get("audio"),
            )
        }


class KyutaiSttMultiModalProcessor(BaseMultiModalProcessor[KyutaiSttProcessingInfo]):
    """Build the prompt + run the Kyutai feature extractor.

    The Mimi codec itself does *not* run here — the encoded ``audio_codes``
    are produced inside the model on GPU as part of ``embed_multimodal``.
    This keeps the heavy codec on the same device as the LM and avoids
    moving raw waveforms through the request pipeline.
    """

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        audios = mm_data.get("audios") or mm_data.get("audio") or []
        tok = self.info.get_tokenizer()
        prompt_ids = tok.encode(prompt) if prompt else []

        if not audios:
            return BatchFeature({"input_ids": [prompt_ids]}, tensor_type="pt")

        fe = self.info.get_feature_extractor()
        out = fe(
            list(audios),
            sampling_rate=fe.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = out["input_values"]
        return BatchFeature(
            {"input_values": input_values, "input_ids": [prompt_ids]},
            tensor_type="pt",
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "input_values": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        from vllm.multimodal.processing import (
            PromptIndexTargets,
            PromptInsertion,
        )

        cfg = self.info.ctx.model_config.hf_config
        frame_size = _resolve_frame_size(cfg.codec_config)
        audio_pad = int(cfg.audio_pad_token_id)
        bos = int(cfg.bos_token_id)
        out_audio = out_mm_kwargs.require_data().get("audio", [])

        def get_insertion(item_idx: int):
            input_values = out_audio[item_idx].get_data()["input_values"]
            n_samples = (
                input_values.shape[-1]
                if isinstance(input_values, torch.Tensor)
                else int(np.asarray(input_values).shape[-1])
            )
            # Mimi's encoder pads its input and emits ceil(n / frame_size)
            # frames, so we mirror the same rounding here. Mismatching the
            # placeholder count vs. the codec's output length triggers a
            # scatter-out-of-bounds in ``_merge_multimodal_embeddings``.
            n_frames = max(1, -(-int(n_samples) // frame_size))
            # BOS token at position 0, then N audio-frame placeholders.
            return [bos] + [audio_pad] * n_frames

        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.start(),
                insertion=get_insertion,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    KyutaiSttMultiModalProcessor,
    info=KyutaiSttProcessingInfo,
    dummy_inputs=KyutaiSttDummyInputsBuilder,
)
class KyutaiSpeechToTextForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsTranscription
):
    """Top-level Kyutai STT model.

    Wires the LM backbone, the tied LM head, the Mimi codec sub-module,
    and the multimodal / transcription glue.

    The Mimi codec runs on GPU in :meth:`embed_multimodal`, turning
    feature-extracted ``input_values`` (raw waveform with the
    silence-prefix and audio-delay padding the HF feature extractor
    applies) into per-frame codebook ids; those are then summed against
    the embedding table in :meth:`embed_input_ids` to produce the
    additive audio bias that overlays the text token embedding at the
    audio-frame placeholder positions.

    Note: this implements the *prefill* path of the multimodal pipeline.
    Kyutai's full inference loop also requires the audio bias at every
    decode step (since both streams advance in lockstep); that wiring is
    deferred to a follow-up since vLLM's standard merge runs only at
    placeholder positions in the prompt.
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
        self._frame_size = _resolve_frame_size(config.codec_config)

        self.model = KyutaiSpeechToTextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        # Mimi codec is a sub-module so its weights are loaded from the same
        # checkpoint and live on the same device as the LM. It is *not*
        # part of the vLLM compile graph (its forward only runs inside
        # ``embed_multimodal``).
        self.codec_model = MimiModel(config.codec_config)

        # The LM head outputs over the *text* vocab only (audio codebook
        # rows in the embedding table aren't sampled). Use the preserved
        # ``text_vocab_size`` from the bridge.
        text_vocab = int(getattr(config, "text_vocab_size", config.vocab_size))
        self._text_vocab_size = text_vocab

        # ``audio_pad_token_id`` is the multimodal placeholder we insert at
        # every audio-frame position; it lives in the *combined* embedding
        # table (text vocab + per-codebook ranges + 1 pad row) which is
        # wider than the text vocab. Inform vLLM so it masks these OOV
        # ids before the text embedding lookup.
        self.configure_mm_token_handling(
            vocab_size=text_vocab,
            mm_token_ids=[int(config.audio_pad_token_id)],
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                text_vocab,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if getattr(config, "tie_word_embeddings", False):
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(text_vocab)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    # -------------------------------------------------------------------
    # SupportsMultiModal
    # -------------------------------------------------------------------

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # Audio-frame placeholder ids are inserted by the multimodal
        # processor via PromptInsertion at the start of the prompt; we
        # don't need a textual placeholder for the chat template.
        return None

    def get_language_model(self) -> nn.Module:
        return self.model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Mimi-encode the per-audio waveforms and return the additive bias.

        Expected kwarg: ``input_values`` shape ``(num_audios, 1, n_samples)``
        (channels-first, single-channel) — the output of the HF
        ``KyutaiSpeechToTextFeatureExtractor`` with the configured
        silence-prefix and audio-delay padding applied.
        """
        input_values = kwargs.pop("input_values", None)
        if input_values is None:
            return []
        if isinstance(input_values, list):
            iv = torch.stack([torch.as_tensor(x) for x in input_values], dim=0)
        else:
            iv = torch.as_tensor(input_values)
        # Move to the codec's device + dtype.
        iv = iv.to(device=self.codec_model.device, dtype=self.codec_model.dtype)
        with torch.no_grad():
            codec_out = self.codec_model.encode(iv, return_dict=True)
            # ``audio_codes`` shape: (B, num_codebooks, n_frames)
            codes = codec_out.audio_codes
        # Re-orient to (B, n_frames, num_codebooks) and embed.
        codes = codes.transpose(1, 2).contiguous()
        embed = self.model.embed_tokens
        per_audio: list[torch.Tensor] = []
        for b in range(codes.shape[0]):
            per_audio.append(embed.embed_audio_only(codes[b]))
        return per_audio

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compose the input embedding from text + audio bias.

        At positions flagged by ``is_multimodal`` (the audio-frame
        placeholders the multimodal processor inserts), we add the
        per-frame audio bias supplied by :meth:`embed_multimodal` to the
        text embedding. Because the placeholder text token id is
        ``audio_pad_token_id`` (whose row in the embedding table is zero
        by HF convention), the additive merge is numerically equivalent
        to the standard "replace at placeholder" merge in this prefill
        regime.
        """
        text_embeds = self.model.embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return text_embeds
        if is_multimodal is None:
            return text_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds=text_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal.to(
                device=text_embeds.device, non_blocking=True
            ),
        )

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
        """Load HF (transformers-format) weights for both the LM backbone
        and the Mimi codec sub-module.

        The qkv stack and gate/up chunk-split for the LM backbone are
        handled inside :meth:`KyutaiSpeechToTextModel.load_weights`, which
        ``AutoWeightsLoader`` dispatches to when it descends into the
        ``model.`` prefix. The Mimi codec is loaded through its own
        ``HF`` ``state_dict`` machinery via the standard plain-name path.
        """
        # The Mimi codec's weight names (``codec_model.*``) match HF's
        # MimiModel state dict directly, so route them through a
        # PyTorch-native load. We separate them from the LM weights to
        # avoid AutoWeightsLoader trying to walk ``codec_model``'s
        # internals (it has its own conventions and we don't need its
        # extras).
        codec_weights: dict[str, torch.Tensor] = {}
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            if name.startswith("codec_model."):
                codec_weights[name[len("codec_model.") :]] = w
            else:
                backbone_weights.append((name, w))

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head."]
                if getattr(self.config, "tie_word_embeddings", False)
                else None
            ),
            ignore_unexpected_prefixes=["codec_model."],
        )
        loaded = loader.load_weights(backbone_weights, mapper=self.hf_to_vllm_mapper)

        if codec_weights:
            missing, unexpected = self.codec_model.load_state_dict(
                codec_weights, strict=False
            )
            for n in self.codec_model.state_dict().keys() - set(missing):
                loaded.add(f"codec_model.{n}")
        return loaded

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
