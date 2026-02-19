# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformers modeling backend mixin for multi-modal models."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from vllm.config.utils import getattr_iter
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MultiModalKwargsItems
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalInputs,
    PlaceholderRange,
    mm_inputs,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalUUIDItems,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from transformers import BatchFeature

    from vllm.config import VllmConfig
    from vllm.config.multimodal import BaseDummyOptions

DYNAMIC_ARG_DIMS = {
    "input_ids": 0,
    # set `positions` to last dim to support Qwen-mrope
    "positions": -1,
    "intermediate_tensors": 0,
    "inputs_embeds": 0,
}

logger = init_logger(__name__)

# Video-specific field names that should be assigned to the "video" modality
_VIDEO_FIELDS = frozenset({
    "pixel_values_videos",
    "video_embeds",
    "video_grid_thw",
    "num_video_patches",
    "second_per_grid_ts",
})


class MultiModalProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self):
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        result = {}
        if mm_counts.get("image", 0) > 0:
            result["image"] = self.get_max_image_tokens()
        if mm_counts.get("video", 0) > 0:
            result["video"] = self.get_max_video_tokens()
        return result

    def get_max_image_tokens(self) -> int:
        width, height = self.get_max_image_size()
        processor = self.get_hf_processor()
        multimodal_config = self.ctx.model_config.multimodal_config
        mm_processor_kwargs = multimodal_config.mm_processor_kwargs or {}
        mm_tokens = processor._get_num_multimodal_tokens(
            image_sizes=([height, width],), **mm_processor_kwargs
        )
        image_tokens = mm_tokens["num_image_tokens"][0]
        return image_tokens

    def get_max_video_tokens(self) -> int:
        """Compute the maximum number of tokens for a single video.

        Tries to use the HF processor's ``_get_num_multimodal_tokens`` when
        it supports video arguments.  Falls back to a conservative estimate
        based on image tokens when the processor does not expose this API.
        """
        width, height = self.get_max_image_size()
        processor = self.get_hf_processor()
        multimodal_config = self.ctx.model_config.multimodal_config
        mm_processor_kwargs = multimodal_config.mm_processor_kwargs or {}

        # Some processors expose video token counts via _get_num_multimodal_tokens
        try:
            mm_tokens = processor._get_num_multimodal_tokens(
                image_sizes=([height, width],),
                video_num_frames=[2],
                **mm_processor_kwargs,
            )
            if mm_tokens.get("num_video_tokens"):
                return mm_tokens["num_video_tokens"][0]
        except (TypeError, KeyError, AttributeError):
            pass

        # Fallback: estimate from image tokens (2 frames)
        return self.get_max_image_tokens() * 2

    def get_max_image_size(self):
        return 10_000, 10_000  # hardcode for arbitrary very large size


class MultiModalDummyInputsBuilder(BaseDummyInputsBuilder[MultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        processor = self.info.get_hf_processor()
        if "gemma3" in processor.__class__.__name__.lower():
            image_token = processor.boi_token
        else:
            image_token = getattr(processor, "image_token", "")

        video_token = getattr(processor, "video_token", "")
        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, "BaseDummyOptions"] | None = None,
        mm_processor_kwargs: Mapping[str, object] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        target_width, target_height = self.info.get_max_image_size()

        image_overrides = mm_options.get("image") if mm_options else None
        video_overrides = mm_options.get("video") if mm_options else None

        result: MultiModalDataDict = {}

        if num_images > 0:
            result["image"] = self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )

        if num_videos > 0:
            result["video"] = self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=2,  # minimum 2 frames for temporal patch support
                num_videos=num_videos,
                overrides=video_overrides,
            )

        return result


class MultiModalProcessor(BaseMultiModalProcessor[MultiModalProcessingInfo]):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ):
        """
        Given the original multi-modal items for this modality
        and HF-processed data, output the updates to perform.

        The information returned by this method is used to update token inputs
        which bypass the HF processor. It is also used to update the output of
        HF processor if the HF process does not apply prompt updates to text
        inputs.

        Moreover, this information is critical to determine the token positions
        in order to construct  :class:`~vllm-multimodal.input.PlaceholderRange`
        for each multi-modal item.
        """
        return None

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # HF Processors always return a mask but vLLM doesn't need it
        hf_inputs.pop("attention_mask", None)

        num_image_patches = hf_inputs.get("num_image_patches")
        num_video_patches = hf_inputs.get("num_video_patches")

        # Assign image fields (all non-video fields default to "image" modality)
        mm_fields = {
            key: MultiModalFieldConfig.flat_from_sizes("image", num_image_patches)
            for key in hf_inputs
            if key not in _VIDEO_FIELDS
        }

        # Pre-computed image embeddings
        mm_fields["image_embeds"] = MultiModalFieldConfig.flat_from_sizes(
            "image", num_image_patches
        )

        # Image metadata fields (batched per image, not split by patch count)
        mm_fields["image_grid_thw"] = MultiModalFieldConfig.batched("image")
        mm_fields["num_image_patches"] = MultiModalFieldConfig.batched("image")

        # Video pixel values (split by video patch count)
        if "pixel_values_videos" in hf_inputs and num_video_patches is not None:
            mm_fields["pixel_values_videos"] = MultiModalFieldConfig.flat_from_sizes(
                "video", num_video_patches
            )

        # Pre-computed video embeddings
        if "video_embeds" in hf_inputs and num_video_patches is not None:
            mm_fields["video_embeds"] = MultiModalFieldConfig.flat_from_sizes(
                "video", num_video_patches
            )

        # Video metadata fields (batched per video)
        mm_fields["video_grid_thw"] = MultiModalFieldConfig.batched("video")
        mm_fields["num_video_patches"] = MultiModalFieldConfig.batched("video")

        # Temporal rate info used by Qwen-style models for MRoPE
        if "second_per_grid_ts" in hf_inputs:
            mm_fields["second_per_grid_ts"] = MultiModalFieldConfig.batched("video")

        return mm_fields

    def _get_hf_mm_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        """
        In contrast to the base class, this method always adds
        `return_mm_token_type_ids` to the processor data
        """
        processor_data, passthrough_data = super()._get_hf_mm_data(mm_items)
        processor_data["return_mm_token_type_ids"] = True
        return processor_data, passthrough_data

    def _compute_video_split_sizes(
        self,
        video_grid_thw: list | torch.Tensor,
        hf_config: object,
    ) -> list[int]:
        """Compute per-video token counts from ``video_grid_thw``.

        For Qwen-style VLMs the number of visual tokens produced by the
        vision encoder for one video is::

            T * H * W / (spatial_merge_size ** 2)

        where T, H, W come from ``video_grid_thw`` and
        ``spatial_merge_size`` is read from the vision config (defaults to 2).
        """
        vision_config = getattr(hf_config, "vision_config", None)
        merge_size = getattr(vision_config, "spatial_merge_size", 2) if vision_config else 2

        split_sizes: list[int] = []
        for thw in video_grid_thw:
            t_v = int(thw[0])
            h_v = int(thw[1])
            w_v = int(thw[2])
            split_sizes.append(t_v * h_v * w_v // (merge_size * merge_size))
        return split_sizes

    def apply(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
        mm_uuid_items: MultiModalUUIDItems | None = None,
        hf_processor_mm_kwargs: Mapping[str, object] | None = None,
        tokenization_kwargs: Mapping[str, object] | None = None,
    ) -> MultiModalInputs:
        """
        Process multi-modal inputs to be used in vLLM.

        Apply HF Processor on prompt text and multi-modal data together,
        outputting token IDs and processed tensors.
        """
        if hf_processor_mm_kwargs is None:
            hf_processor_mm_kwargs = {}
        if tokenization_kwargs is None:
            tokenization_kwargs = {}

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        if not isinstance(prompt, str):
            # the prompt is the tokenized ids which is not supported
            # by the hf_processor, which is why we would need to decode the ids
            # into string
            prompt = hf_processor.decode(prompt)

        # Bypass cached processor and always apply to the full set of mm inputs
        # NOTE: we can't just set caching=False because base class method
        # transforms outputs to `MultiModalKwargs` which is not going to
        # work for Transformers. We have a lot of logic tied to
        # `mm_tokens_per_modality` below
        prompt_ids, processed_data, _ = self._apply_hf_processor_text_mm(
            prompt_text=prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        # For gemma3 we check `token_type_ids` as the key
        token_type_key = (
            "mm_token_type_ids"
            if "mm_token_type_ids" in processed_data
            else "token_type_ids"
        )
        mm_token_type_ids = processed_data.pop(token_type_key)

        # We can infer vLLM style placeholder from token type ids, if we split
        # it for each input `mm_data`.
        mm_positions = torch.where(mm_token_type_ids == 1)[1]
        images = mm_items.get_items("image", ImageProcessorItems)
        multimodal_config = self.info.ctx.model_config.multimodal_config
        mm_processor_kwargs = multimodal_config.mm_processor_kwargs or {}
        image_sizes = []
        for item_idx in range(len(images)):
            image_size = images.get_image_size(item_idx)
            image_sizes.append((image_size.height, image_size.width))

        mm_tokens_per_modality = hf_processor._get_num_multimodal_tokens(
            image_sizes=image_sizes, **mm_processor_kwargs
        )

        # ------------------------------------------------------------------
        # Video handling: check if the batch contains any video items and
        # compute per-video token counts from the grid THW tensors returned
        # by the HF processor.
        # ------------------------------------------------------------------
        try:
            videos = mm_items.get_items("video", VideoProcessorItems)
            has_videos = len(videos) > 0
        except KeyError:
            has_videos = False

        video_split_sizes: list[int] = []
        video_grid_thw_data = processed_data.get("video_grid_thw")
        if has_videos and video_grid_thw_data is not None and len(video_grid_thw_data) > 0:
            hf_config = self.info.ctx.model_config.hf_config
            video_split_sizes = self._compute_video_split_sizes(
                video_grid_thw_data, hf_config
            )

        # Store num_video_patches in processed_data so _get_mm_fields_config
        # can assign pixel_values_videos to the correct modality bucket.
        if video_split_sizes:
            processed_data["num_video_patches"] = torch.tensor(video_split_sizes)

        # ------------------------------------------------------------------
        # Build PlaceholderRanges for images and (optionally) videos.
        # For models that distinguish image vs. video pad tokens by ID we can
        # classify each multimodal position as belonging to one modality.
        # ------------------------------------------------------------------
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        image_split_sizes = mm_tokens_per_modality["num_image_tokens"]

        image_token_id = getattr(hf_processor, "image_token_id", None)
        video_token_id = getattr(hf_processor, "video_token_id", None)

        all_mm_tokens = torch.tensor(prompt_ids)[mm_token_type_ids[0].bool()]

        if has_videos and video_split_sizes and video_token_id is not None:
            # --- Mixed image + video path -----------------------------------
            # Classify each multimodal position using the token IDs embedded
            # in the prompt (image pad token vs. video pad token).
            is_img = (
                (all_mm_tokens == image_token_id)
                if image_token_id is not None
                else torch.ones(len(all_mm_tokens), dtype=torch.bool)
            )
            is_vid = all_mm_tokens == video_token_id

            img_positions = mm_positions[is_img]
            img_tokens = all_mm_tokens[is_img]
            vid_positions = mm_positions[is_vid]

            if image_split_sizes and len(img_positions) > 0:
                chunked_img_pos = torch.split(img_positions, image_split_sizes)
                chunked_img_tok = torch.split(img_tokens, image_split_sizes)
                mm_placeholders["image"] = [
                    PlaceholderRange(
                        offset=pos[0].item(),
                        length=pos.shape[0],
                        is_embed=(tok == image_token_id).bool(),
                    )
                    for pos, tok in zip(chunked_img_pos, chunked_img_tok)
                ]

            if video_split_sizes and len(vid_positions) > 0:
                chunked_vid_pos = torch.split(vid_positions, video_split_sizes)
                mm_placeholders["video"] = [
                    PlaceholderRange(
                        offset=pos[0].item(),
                        length=pos.shape[0],
                    )
                    for pos in chunked_vid_pos
                ]
        else:
            # --- Image-only path (original behaviour) ----------------------
            if image_split_sizes:
                chunked_mm_positions = torch.split(mm_positions, image_split_sizes)
                chunked_mm_tokens = torch.split(all_mm_tokens, image_split_sizes)
                ranges = [
                    PlaceholderRange(
                        offset=positions[0].item(),
                        length=positions.shape[0],
                        is_embed=(mm_tokens == hf_processor.image_token_id).bool(),
                    )
                    for positions, mm_tokens in zip(chunked_mm_positions, chunked_mm_tokens)
                ]
                mm_placeholders = {"image": ranges}

        processed_data["num_image_patches"] = torch.tensor(
            mm_tokens_per_modality["num_image_patches"]
        )
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            processed_data,
            self._get_mm_fields_config(processed_data, hf_processor_mm_kwargs),
        )

        # Use overrides if provided; fallback to data-dependent hashing.
        mm_hashes = self._hash_mm_items(
            mm_items,
            mm_uuid_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

        return mm_inputs(
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


class MultiModalMixin(SupportsMultiModal, SupportsMRoPE):
    supports_multimodal_raw_input_only = True

    # Backwards compatibility for prev released models. State dicts back then
    # had different formats and cannot be loaded with `AutoModel` mapping as is
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.model": "model.language_model",
            "text_model.model": "model.text_model",
            "vision_tower": "model.vision_tower",
            "vqmodel": "model.vqmodel",
            "visual": "model.visual",
            "vision_model": "model.vision_model",
            "vision_embed_tokens": "model.vision_embed_tokens",
            "image_newline": "model.image_newline",
            "multi_modal_projector": "model.multi_modal_projector",
            "text_model.lm_head": "lm_head",
            "language_model.lm_head": "lm_head",
            # Qwen models used "model" as the name for the language model.
            # Therefore, we must map each of submodule explicitly to avoid
            # conflicts with newer models that use "model.language_model".
            "model.embed_tokens": "model.language_model.embed_tokens",
            "model.layers": "model.language_model.layers",
            "model.norm": "model.language_model.norm",
        }
    )

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip SupportsMRoPE.__init__ and call the next class in MRO
        super(SupportsMRoPE, self).__init__(vllm_config=vllm_config, prefix=prefix)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Gemma3 and PaliGemma needs `token_type_ids` to work correctly
        # Other models will not have `token_type_ids` in kwargs
        kwargs = {k: v for k, v in kwargs.items() if k == "token_type_ids"}
        model_output = super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return model_output

    def get_language_model(self) -> torch.nn.Module:
        """Transformers modeling backend multimodal classes do not contain a separate
        vLLM language model class. Therefore, in order to return a language model vLLM
        class, we use a wrapper to give `self` the same interface as a text model."""

        # Exclude self and object
        bases = self.__class__.mro()[1:-1]
        # Keep only classes defined in `vllm.model_executor.models.transformers`
        bases = [b for b in bases if ".transformers." in b.__module__]
        # Exclude MultiModalMixin itself
        bases = [b for b in bases if b is not MultiModalMixin]

        class LanguageModel(*bases):
            def __init__(self, multimodal_model):
                # Don't call super().__init__() to avoid re-initialization
                self.__dict__.update(multimodal_model.__dict__)

            model = getattr_iter(self.model, ("language_model", "text_model"), None)

        return LanguageModel(self)

    def embed_multimodal(self, **kwargs):
        """Encode multimodal inputs and return a list of per-item embeddings.

        This method is called by the vLLM runner once per modality group.
        That is, when image items are encoded it receives only image kwargs,
        and when video items are encoded it receives only video kwargs.

        For models whose vision encoder is shared between images and videos
        (e.g. Qwen2-VL, Qwen3-VL) we route video pixel values through
        ``model.get_image_features`` with the video grid THW as
        ``image_grid_thw``, which is equivalent to how the HF model handles
        them internally.
        """
        pixel_values: torch.Tensor | None = kwargs.pop("pixel_values", None)
        pixel_values_videos: torch.Tensor | None = kwargs.pop("pixel_values_videos", None)
        image_embeds: torch.Tensor | None = kwargs.pop("image_embeds", None)
        video_embeds: torch.Tensor | None = kwargs.pop("video_embeds", None)

        # Model might use `image_patches` instead of `pixel_values`
        if pixel_values is None:
            pixel_values = kwargs.pop("image_patches", None)

        if image_embeds is not None:
            return image_embeds

        if video_embeds is not None:
            return video_embeds

        if pixel_values is None and pixel_values_videos is None:
            return None

        # Pop token count tensors – used for splitting the flat output
        num_image_patches = kwargs.pop("num_image_patches", None)
        num_video_patches = kwargs.pop("num_video_patches", None)
        kwargs.pop("token_type_ids", None)  # used only in `forward`

        # ------------------------------------------------------------------
        # Determine whether we are encoding images or videos.
        # When called for images:  pixel_values is set, pixel_values_videos is None
        # When called for videos:  pixel_values_videos is set, pixel_values is None
        # ------------------------------------------------------------------
        is_video_call = pixel_values_videos is not None and pixel_values is None

        if is_video_call:
            # Route video patches through the same vision encoder as images.
            # Qwen-style models share one vision encoder for both; we pass
            # video_grid_thw as image_grid_thw since the encoder doesn't
            # distinguish at the pixel-patch level.
            video_grid_thw = kwargs.pop("video_grid_thw", None)
            # Remove image-specific kwarg that might linger
            kwargs.pop("image_grid_thw", None)

            if current_platform.is_rocm():
                # TODO: [ROCm] Fix accuracy issues with flash backend
                logger.debug(
                    "ROCm platform detected. Forcing math SDP backend "
                    "for vision encoder. Currently ROCm platform has "
                    "accuracy issues with `flash_sdp` and"
                    "`mem_efficient_sdp` backends. See issue: "
                    "https://github.com/vllm-project/vllm/issues/30167"
                )
                with torch.nn.attention.sdpa_kernel(
                    backends=[torch.nn.attention.SDPBackend.MATH]
                ):
                    vision_embeddings = self._call_vision_encoder_for_video(
                        pixel_values_videos, video_grid_thw, **kwargs
                    )
            else:
                vision_embeddings = self._call_vision_encoder_for_video(
                    pixel_values_videos, video_grid_thw, **kwargs
                )

            num_patches = num_video_patches
        else:
            # Image encoding path (original behaviour)
            if current_platform.is_rocm():
                logger.debug(
                    "ROCm platform detected. Forcing math SDP backend "
                    "for vision encoder. Currently ROCm platform has "
                    "accuracy issues with `flash_sdp` and"
                    "`mem_efficient_sdp` backends. See issue: "
                    "https://github.com/vllm-project/vllm/issues/30167"
                )
                with torch.nn.attention.sdpa_kernel(
                    backends=[torch.nn.attention.SDPBackend.MATH]
                ):
                    vision_embeddings = self.model.get_image_features(
                        pixel_values, **kwargs
                    )
            else:
                vision_embeddings = self.model.get_image_features(
                    pixel_values, **kwargs
                )

            num_patches = num_image_patches

        # Transformers `v5`, `self.get_image_features` returns a tuple
        # containing the features and optionally attentions/hidden_states
        if isinstance(vision_embeddings, tuple):
            vision_embeddings = vision_embeddings[0]
        elif isinstance(vision_embeddings, dict):
            vision_embeddings = vision_embeddings.pooler_output

        if isinstance(vision_embeddings, torch.Tensor):
            split_sizes = (
                num_patches.flatten().tolist()
                if num_patches is not None
                else None
            )

            # Flatten to 2D: [total_tokens, hidden_dim]
            if vision_embeddings.ndim == 3:
                vision_embeddings = vision_embeddings.view(
                    -1, vision_embeddings.shape[-1]
                )

            if split_sizes:
                total_patches = sum(split_sizes)
                total_tokens = vision_embeddings.shape[0]

                if total_tokens == total_patches:
                    # Direct match: num_patches are actual token counts
                    token_split_sizes = split_sizes
                elif total_patches > 0 and total_tokens % total_patches == 0:
                    # Uniform expansion: each patch expands to N tokens
                    tokens_per_patch = total_tokens // total_patches
                    token_split_sizes = [s * tokens_per_patch for s in split_sizes]
                elif total_patches > 0:
                    # Mismatch (profiling with dummy data) - pad/truncate
                    if total_tokens == 0:
                        raise ValueError(
                            "Vision encoder returned empty embeddings. "
                            f"Expected {total_patches} patches from "
                            f"num_patches={split_sizes}"
                        )
                    if total_tokens < total_patches:
                        repeat_factor = (
                            total_patches + total_tokens - 1
                        ) // total_tokens
                        vision_embeddings = vision_embeddings.repeat(repeat_factor, 1)
                    vision_embeddings = vision_embeddings[:total_patches]
                    token_split_sizes = split_sizes
                else:
                    return []

                return list(torch.split(vision_embeddings, token_split_sizes, dim=0))

            return vision_embeddings
        else:
            logger.debug(
                "No pixel values or embeddings produced a valid tensor for "
                "multimodal embedding."
            )
            return None

    def _call_vision_encoder_for_video(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor | None,
        **kwargs,
    ) -> torch.Tensor:
        """Call the vision encoder on video patches.

        Qwen-style models expose ``get_image_features(pixel_values,
        image_grid_thw)`` and use the same encoder for both images and videos.
        We therefore pass ``video_grid_thw`` as ``image_grid_thw``.  If that
        signature is not supported we fall back to a plain call without the
        grid argument.
        """
        try:
            return self.model.get_image_features(
                pixel_values_videos, image_grid_thw=video_grid_thw, **kwargs
            )
        except TypeError:
            # Older models or models that don't accept image_grid_thw
            try:
                return self.model.get_image_features(pixel_values_videos, **kwargs)
            except TypeError:
                raise RuntimeError(
                    "Could not call get_image_features for video inputs. "
                    "The model's vision encoder may not be compatible with "
                    "the generic Transformers modeling backend for video."
                )

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {
                "image_grid_thw",
                "video_grid_thw",
                "second_per_grid_ts",
                "audio_feature_lengths",
                "use_audio_in_video",
            },
        )

        # Audio modality is not supported via this generic backend
        if any(
            v
            for k, v in kwargs.items()
            if k in {"audio_feature_lengths", "use_audio_in_video"}
        ):
            raise NotImplementedError(
                "Transformers modeling backend does not support audio modality "
                "in get_mrope_input_positions."
            )

        image_grid_thw_list = kwargs.get("image_grid_thw", [])
        video_grid_thw_list = kwargs.get("video_grid_thw", [])
        second_per_grid_ts_list = kwargs.get("second_per_grid_ts", [])

        image_grid_thw = (torch.stack if image_grid_thw_list else torch.tensor)(
            image_grid_thw_list
        )
        video_grid_thw = (torch.stack if video_grid_thw_list else torch.tensor)(
            video_grid_thw_list
        )
        second_per_grid_ts = (
            torch.stack(second_per_grid_ts_list)
            if second_per_grid_ts_list
            else None
        )

        # Build the kwargs for get_rope_index; include second_per_grid_ts when
        # the HF model's get_rope_index supports the argument.
        rope_index_kwargs: dict = dict(
            input_ids=torch.tensor(input_tokens).unsqueeze(0),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        if second_per_grid_ts is not None:
            # Try to pass second_per_grid_ts; fall back if model doesn't accept it
            try:
                mrope_positions, mrope_position_delta = self.model.get_rope_index(
                    **rope_index_kwargs,
                    second_per_grid_ts=second_per_grid_ts,
                )
            except TypeError:
                mrope_positions, mrope_position_delta = self.model.get_rope_index(
                    **rope_index_kwargs
                )
        else:
            mrope_positions, mrope_position_delta = self.model.get_rope_index(
                **rope_index_kwargs
            )

        mrope_positions = mrope_positions[:, 0]
        mrope_position_delta = mrope_position_delta[0].item()

        return mrope_positions, mrope_position_delta
