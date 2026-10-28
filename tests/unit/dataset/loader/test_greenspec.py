# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import orjson
import pytest

from aiperf.config.flags import CLIConfig
from aiperf.dataset.loader.greenspec import META_KEYS, GreenSpecLoader, is_greenspec_row
from aiperf.dataset.loader.raw_payload import RawPayloadDatasetLoader
from aiperf.dataset.loader.speed_bench import is_speed_bench_row
from aiperf.plugin import plugins
from tests.unit.conftest import make_run_from_cli


def _row(qid: str, category: str, bucket: str, lite: bool, **body) -> dict:
    return {
        "question_id": qid,
        "category": category,
        "bucket": bucket,
        "lite": lite,
        "isl": 100,
        "osl": 10,
        "thinking_src": "none",
        "recorded": {
            "temperature": 0.6,
            "max_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        "messages": [{"role": "user", "content": f"q {qid}"}],
        **body,
    }


ROWS = [
    _row(
        "m/a",
        "agentic_tools",
        "agentic_tools__0-4k",
        True,
        tools=[{"type": "function"}],
    ),
    _row("m/b", "agentic_tools", "agentic_tools__4k-16k", False),
    _row(
        "m/c",
        "llm_task",
        "llm_task__0-1k",
        True,
        response_format={"type": "json_object"},
    ),
]


@pytest.fixture
def greenspec_file(tmp_path: Path) -> Path:
    p = tmp_path / "greenspec.jsonl"
    p.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in ROWS))
    return p


def _run(*args: str, streaming: bool = True, extra: list[str] | None = None):
    kw = {"model_names": ["served-model"], "streaming": streaming}
    if extra:
        kw["extra_inputs"] = extra
    return make_run_from_cli(CLIConfig(**kw))


def _load(path: Path, run=None, **kw) -> dict:
    loader = GreenSpecLoader(filename=path, run=run or _run(), **kw)
    return {sid: [p.payload for p in ps] for sid, ps in loader.load_dataset().items()}


class TestCanLoad:
    def test_greenspec_row_accepted(self):
        assert GreenSpecLoader.can_load(ROWS[0])
        assert is_greenspec_row(ROWS[0])

    def test_raw_payload_and_speed_bench_reject_greenspec_rows(self):
        assert not RawPayloadDatasetLoader.can_load(ROWS[0])
        assert not is_speed_bench_row(ROWS[0])

    def test_plain_payload_not_claimed(self):
        assert not GreenSpecLoader.can_load(
            {"messages": [{"role": "user", "content": "x"}]}
        )


class TestPayload:
    def test_metadata_stripped_model_and_stream_added(self, greenspec_file: Path):
        data = _load(greenspec_file)
        assert list(data) == ["m/a", "m/b", "m/c"]  # session id = question_id
        body = data["m/a"][0]
        assert not set(META_KEYS) & set(body)
        assert body["model"] == "served-model"
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert body["tools"] == [{"type": "function"}]
        assert "temperature" not in body and "max_tokens" not in body

    def test_no_stream_without_streaming(self, greenspec_file: Path):
        body = _load(greenspec_file, run=_run(streaming=False))["m/a"][0]
        assert "stream" not in body and "stream_options" not in body

    def test_recorded_merges_recorded_params(self, greenspec_file: Path):
        body = _load(greenspec_file, recorded=True)["m/a"][0]
        assert body["temperature"] == 0.6 and body["max_tokens"] == 4096
        assert body["chat_template_kwargs"] == {"enable_thinking": True}

    def test_extra_inputs_set_max_tokens_or_cap_a_recorded_one(
        self, greenspec_file: Path
    ):
        run = _run(extra=["max_tokens:2048", "top_k:20"])
        plain = _load(greenspec_file, run=run)["m/a"][0]
        assert plain["max_tokens"] == 2048 and plain["top_k"] == 20
        rec = _load(greenspec_file, run=run, recorded=True)["m/a"][0]
        assert rec["max_tokens"] == 2048  # min(4096, 2048)
        run = _run(extra=["max_tokens:8192"])
        assert (
            _load(greenspec_file, run=run, recorded=True)["m/a"][0]["max_tokens"]
            == 4096
        )


class TestFilters:
    @pytest.mark.parametrize(
        ("kw", "expected"),
        [
            ({}, ["m/a", "m/b", "m/c"]),
            ({"lite": True}, ["m/a", "m/c"]),
            ({"category": "agentic_tools"}, ["m/a", "m/b"]),
            ({"category": "agentic_tools", "lite": True}, ["m/a"]),
            ({"bucket": "agentic_tools__4k-16k"}, ["m/b"]),
        ],
    )
    def test_filter(self, greenspec_file: Path, kw: dict, expected: list[str]):
        assert list(_load(greenspec_file, **kw)) == expected

    def test_empty_filter_fails(self, greenspec_file: Path):
        with pytest.raises(ValueError, match="matched no rows"):
            _load(greenspec_file, category="nope")

    def test_non_greenspec_line_fails(self, tmp_path: Path):
        p = tmp_path / "greenspec.jsonl"
        p.write_bytes(orjson.dumps({"messages": []}) + b"\n")
        with pytest.raises(ValueError, match="not a GreenSpec row"):
            _load(p)

    def test_conversations_are_raw_payload_turns(self, greenspec_file: Path):
        loader = GreenSpecLoader(filename=greenspec_file, run=_run(), lite=True)
        convs = loader.convert_to_conversations(loader.load_dataset())
        assert [c.session_id for c in convs] == ["m/a", "m/c"]
        assert all(c.turns[0].raw_payload["model"] == "served-model" for c in convs)


class TestPresets:
    def test_presets_registered(self):
        names = [
            e.name
            for e in plugins.list_entries("custom_dataset_loader")
            if e.name.startswith("greenspec")
        ]
        # all + lite, 3 use cases x (full, lite), each also *_recorded; 12 buckets
        assert len(names) == 2 * (2 + 6) + 12
