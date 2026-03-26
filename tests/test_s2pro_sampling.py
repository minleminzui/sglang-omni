from __future__ import annotations

import torch

from sglang_omni.models.fishaudio_s2_pro.runtime.s2pro_sglang_ar import (
    S2ProSGLangRequestData,
    _resolve_sampling_state,
)
from sglang_omni.models.fishaudio_s2_pro.sglang_model import (
    _select_semantic_token_with_fallback,
)


def test_resolve_sampling_state_uses_request_defaults_without_repetition() -> None:
    data = S2ProSGLangRequestData(
        input_ids=[],
        temperature=0.7,
        top_p=0.9,
        _previous_semantic_tokens=[11, 12, 13],
    )

    use_ras, temperature, top_p, previous_tokens = _resolve_sampling_state(
        data,
        ras_window=16,
        ras_temperature=1.5,
        ras_top_p=0.95,
    )

    assert use_ras is False
    assert temperature == 0.7
    assert top_p == 0.9
    assert previous_tokens == [11, 12, 13]


def test_resolve_sampling_state_switches_to_ras_on_recent_repetition() -> None:
    data = S2ProSGLangRequestData(
        input_ids=[],
        temperature=0.7,
        top_p=0.9,
        _previous_semantic_tokens=[101, 103, 103, 103],
    )

    use_ras, temperature, top_p, previous_tokens = _resolve_sampling_state(
        data,
        ras_window=16,
        ras_temperature=1.5,
        ras_top_p=0.95,
    )

    assert use_ras is True
    assert temperature == 1.5
    assert top_p == 0.95
    assert previous_tokens == [101, 103, 103, 103]


def test_resolve_sampling_state_ignores_non_terminal_duplicates() -> None:
    data = S2ProSGLangRequestData(
        input_ids=[],
        temperature=0.7,
        top_p=0.9,
        _previous_semantic_tokens=[101, 102, 101, 103],
    )

    use_ras, temperature, top_p, previous_tokens = _resolve_sampling_state(
        data,
        ras_window=16,
        ras_temperature=1.5,
        ras_top_p=0.95,
    )

    assert use_ras is False
    assert temperature == 0.7
    assert top_p == 0.9
    assert previous_tokens == [101, 102, 101, 103]


def test_select_semantic_token_with_fallback_only_changes_collapsing_rows() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.8],
            [0.1, 0.9, 0.8],
        ],
        dtype=torch.float32,
    )
    fallback_mask = torch.tensor([True, False], dtype=torch.bool)
    previous_tokens = torch.tensor(
        [
            [1, 0, 0],
            [2, 0, 0],
        ],
        dtype=torch.long,
    )
    previous_mask = torch.tensor(
        [
            [True, False, False],
            [True, False, False],
        ],
        dtype=torch.bool,
    )

    selected = _select_semantic_token_with_fallback(
        logits,
        fallback_mask=fallback_mask,
        previous_tokens=previous_tokens,
        previous_mask=previous_mask,
    )

    assert selected.tolist() == [2, 1]
