# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State-machine tests for the DiffusionGemma batched sampler.

These instantiate ``DiffusionSampler`` + ``DiffusionGemmaRequestStates``
directly with a stubbed input batch, driving the prefill → denoise ->
converge -> commit loop with crafted logits. They guard:

- prefill batches emit nothing and arm the canvas/dennoise phase;
- the stable-and-confident convergence criteria commit the argmax canvas
  at the right step, and only then;
- max_denoising_steps forces convergence when stability never triggers;
- the entropy bound accepts peaked distributions and renoises flat ones;
- a canvas truncated near max_model_len commits exactly the valid span;
- prefill num_sampled/num_rejected are not aliased.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaRequestStates,
    DiffusionSampler,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="sampler tests require CUDA"
)

DEVICE = torch.device("cuda")
VOCAB = 517  # odd size to shake out padding assumptions
CL = 8
HID = 32
GOLD, ALT = 3, 4  # token ids used for peaked logits


class _SamplingStatesStub:
    def __init__(self):
        self.top_k = None  # [total_logits] int32 on device when set

    def add_request(self, req_idx, sampling_params):
        pass

    def apply_staged_writes(self):
        pass

    def get_top_k_top_p(self, expanded_idx_mapping, idx_mapping_np):
        return self.top_k, None

    def max_num_logprobs(self, slots_np):
        return -1  # logprobs disabled


def make_sampler(
    num_reqs: int = 4,
    max_denoising_steps: int = 3,
    stability_threshold: int = 2,  # vLLM convention: HF value + 1
    entropy_bound: float = 0.1,
    confidence_threshold: float = 0.5,
):
    g = torch.Generator(device="cpu").manual_seed(0)
    embed_weight = (
        torch.randn(VOCAB, HID, generator=g, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(DEVICE)
    )
    states = DiffusionGemmaRequestStates(
        max_num_reqs=num_reqs,
        canvas_length=CL,
        vocab_size=VOCAB,
        max_denoising_steps=max_denoising_steps,
        device=DEVICE,
        hidden_size=HID,
        stability_threshold=stability_threshold,
    )
    sampling_states = _SamplingStatesStub()
    sampler = DiffusionSampler(
        sampler=SimpleNamespace(
            sampling_states=sampling_states,
            req_states=SimpleNamespace(
                draft_tokens=torch.zeros(num_reqs, CL, dtype=torch.int64, device=DEVICE)
            ),
            logprobs_mode="raw_logprobs",
        ),
        diffusion_config=SimpleNamespace(canvas_length=CL),
        vocab_size=VOCAB,
        diffusion_states=states,
        confidence_threshold=confidence_threshold,
        t_min=0.4,
        t_max=0.8,
        entropy_bound=entropy_bound,
        embed_weight=embed_weight,
        normalizer=torch.tensor(float(HID**0.5), dtype=torch.bfloat16, device=DEVICE),
        sc_vocab_start=0,
        sc_vocab_end=VOCAB,
        tp_size=1,
        tp_group_name="",
    )
    return sampler, states, sampling_states


def make_prefill_batch(num_reqs: int, prompt_len: int = 5, slot_base: int = 0):
    """All requests finish their (single-chunk) prompt this step."""
    qsl = np.arange(num_reqs + 1, dtype=np.int32) * prompt_len
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_draft_tokens=0,
        idx_mapping_np=np.arange(num_reqs, dtype=np.int64) + slot_base,
        idx_mapping=torch.arange(num_reqs, dtype=torch.int64, device=DEVICE)
        + slot_base,
        cu_num_logits_np=np.arange(num_reqs + 1, dtype=np.int32),  # 0 logits each
        query_start_loc=torch.from_numpy(qsl).to(DEVICE),
        query_start_loc_np=qsl,
        num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int64),
        num_scheduled_tokens=np.full(num_reqs, prompt_len, dtype=np.int64),
        prefill_len_np=np.full(num_reqs, prompt_len, dtype=np.int64),
    )


def make_denoise_batch(num_reqs: int, canvas_lens: list[int] | None = None):
    """All requests have draft tokens scheduled this step.

    canvas_lens[i] < CL simulates a canvas truncated near max_model_len.
    """
    if canvas_lens is None:
        canvas_lens = [CL] * num_reqs
    lens = np.asarray(canvas_lens, dtype=np.int32)
    cu = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(lens, out=cu[1:])
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_draft_tokens=int(lens.sum()),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int64),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int64, device=DEVICE),
        cu_num_logits_np=cu,
        query_start_loc=torch.from_numpy(cu).to(DEVICE),
        query_start_loc_np=cu,
        num_scheduled_tokens=lens.astype(np.int64),
    )


def peaked_logits(num_rows: int, token: int, gap: float = 30.0) -> torch.Tensor:
    logits = torch.zeros(num_rows, VOCAB, dtype=torch.float32, device=DEVICE)
    logits[:, token] = gap
    return logits


def run_prefill(sampler, num_reqs: int, batch=None):
    # __call__ reads logits.device even for pure-prefill batches.
    empty = torch.empty(0, VOCAB, dtype=torch.float32, device=DEVICE)
    return sampler(empty, batch if batch is not None else make_prefill_batch(num_reqs))


class TestPrefill:
    def test_prefill_emits_nothing_and_arms_denoise(self):
        sampler, states, _ = make_sampler(num_reqs=2)
        states.add_request(0)
        states.add_request(1)

        out = run_prefill(sampler, 2)

        assert out.sampled_token_ids[:2].cpu().eq(0).all()
        assert out.num_sampled[:2].cpu().eq(0).all()
        assert out.num_rejected[:2].cpu().eq(0).all()
        # Aliasing guard: the two tensors must be distinct storage.
        assert out.num_sampled.data_ptr() != out.num_rejected.data_ptr()
        canvas = states.canvas[:2].cpu()
        assert canvas.max() < VOCAB and canvas.min() >= 0
        for slot in (0, 1):
            assert not states.is_encoder_phase[slot].item()
            assert torch.equal(
                sampler.req_states.draft_tokens[slot].cpu(), canvas[slot]
            )

    def test_mid_chunk_prefill_stays_causal(self):
        sampler, states, _ = make_sampler(num_reqs=1)
        states.add_request(0)
        batch = make_prefill_batch(1, prompt_len=4)
        # A 4-token chunk of a 9-token prompt: 0 + 4 < 9, so no flip.
        batch.prefill_len_np = np.array([9], dtype=np.int64)

        run_prefill(sampler, 1, batch)

        assert states.is_encoder_phase[0].item()
        assert states.accepted_canvas_history_len[0].item() == 0
        # Draft tokens are only seeded once the prompt completes.
        assert sampler.req_states.draft_tokens[0].cpu().eq(0).all()


class TestCommitConvergence:
    def test_stable_confident_converges_and_commits_argmax(self):
        sampler, states, _ = make_sampler(num_reqs=1, max_denoising_steps=48)
        states.add_request(0)
        run_prefill(sampler, 1)

        logits = peaked_logits(CL, GOLD)
        # Denoise step 1: not enough history for stability yet.
        out = sampler(logits, make_denoise_batch(1))
        assert out.num_sampled[0].item() == 0
        assert out.num_rejected[0].item() == CL
        assert not states.is_encoder_phase[0].item()
        # Peaked logits are fully within the entropy bound: canvas accepted.
        assert states.canvas[0].cpu().eq(GOLD).sum().item() > CL // 2

        # Denoise step 2: history full, argmax stable & confident -> converged.
        out = sampler(logits, make_denoise_batch(1))
        assert out.num_sampled[0].item() == 0
        assert states.is_encoder_phase[0].item()

        # Commit step emits the argmax canvas, all CL positions.
        out = sampler(logits, make_denoise_batch(1))
        assert out.num_sampled[0].item() == CL
        assert out.num_rejected[0].item() == 0
        assert out.sampled_token_ids[0].cpu().eq(GOLD).all()
        # Next canvas restarts in denoise phase, re-randomized.
        assert not states.is_encoder_phase[0].item()
        assert not states.canvas[0].cpu().eq(GOLD).all()

    def test_max_steps_forces_commit_without_stability(self):
        sampler, states, _ = make_sampler(num_reqs=1, max_denoising_steps=3)
        states.add_request(0)
        run_prefill(sampler, 1)

        for step, tok in enumerate([GOLD, ALT, GOLD]):
            out = sampler(peaked_logits(CL, tok), make_denoise_batch(1))
            assert out.num_sampled[0].item() == 0
            if step < 2:
                # Argmax alternates every step: never stable.
                assert not states.is_encoder_phase[0].item()
        assert states.is_encoder_phase[0].item()

        out = sampler(peaked_logits(CL, GOLD), make_denoise_batch(1))
        assert out.num_sampled[0].item() == CL
        # Committed tokens are the LAST denoise step's argmax.
        assert out.sampled_token_ids[0].cpu().eq(GOLD).all()


class TestEntropyBound:
    def test_flat_logits_are_renoised_and_never_confident(self):
        # Flat distribution: token entropy = ln(V) > confidence_threshold and
        # quickly exceeds the entropy bound, so positions are renoised.
        sampler, states, _ = make_sampler(num_reqs=1, max_denoising_steps=48)
        states.add_request(0)
        run_prefill(sampler, 1)

        flat = torch.zeros(CL, VOCAB, dtype=torch.float32, device=DEVICE)
        state_before = states.canvas[0].clone()
        for _ in range(6):
            out = sampler(flat, make_denoise_batch(1))
            assert out.num_sampled[0].item() == 0
            assert not states.is_encoder_phase[0].item()

        # At most one position (the entropy-bound-fitting prefix) may have
        # been accepted/stable; everything else was renoised at least once.
        final = states.canvas[0].cpu()
        argmax0 = 0  # argmax of the all-zeros logits
        renoised = (final != argmax0).sum().item()
        assert renoised >= CL - 2
        assert not torch.equal(final, state_before.cpu())


class TestTruncatedCanvas:
    def test_commit_emits_only_valid_positions(self):
        sampler, states, _ = make_sampler(num_reqs=1, max_denoising_steps=2)
        states.add_request(0)
        run_prefill(sampler, 1)

        k = 5  # canvas truncated to 5 < CL near max_model_len
        batch = make_denoise_batch(1, canvas_lens=[k])
        out = sampler(peaked_logits(k, GOLD), batch)
        assert out.num_sampled[0].item() == 0
        assert out.num_rejected[0].item() == k
        out = sampler(peaked_logits(k, GOLD), batch)
        assert states.is_encoder_phase[0].item()

        out = sampler(peaked_logits(k, GOLD), batch)
        assert out.num_sampled[0].item() == k
        assert out.sampled_token_ids[0, :k].cpu().eq(GOLD).all()
