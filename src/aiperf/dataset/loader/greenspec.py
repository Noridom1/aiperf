# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GreenSpec loader: one flat ``greenspec.jsonl``, a subset per preset.

GreenSpec is a speculative-decoding bench cut from production chat traffic. A
release ships every subset in one SPEED-Bench-style file: each line is a fixed
set of metadata keys (``META_KEYS``) followed by a complete chat-completions
request body (``messages``, ``tools``, ``response_format``) with no ``model``
and no ``stream``. Rows carry tools and assistant tool calls, so they are
replayed as raw payloads rather than rebuilt turn by turn.

Plugin metadata picks the subset (``category``, ``bucket``, ``lite``) and
whether to send the row's ``recorded`` request params. Per line the loader
drops the metadata keys, merges ``recorded`` when asked, applies
``--extra-inputs`` (a ``max_tokens`` there caps a recorded cap instead of
raising it), sets ``model`` and, with ``--streaming``, ``stream`` plus
``stream_options.include_usage``. The session id is the row's
``question_id``, stable across runs and arms.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiperf.dataset.loader.base_loader import LoaderProbeData
from aiperf.dataset.loader.models import RawPayload
from aiperf.dataset.loader.raw_payload import (
    RawPayloadDatasetLoader,
    _validate_payload_shape,
)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

META_KEYS = (
    "question_id",
    "category",
    "bucket",
    "lite",
    "isl",
    "osl",
    "thinking_src",
    "recorded",
)
_MAX_TOKEN_KEYS = ("max_tokens", "max_completion_tokens")


def is_greenspec_row(data: object) -> bool:
    """Return whether data matches the flat GreenSpec row shape."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("question_id"), str)
        and isinstance(data.get("category"), str)
        and isinstance(data.get("bucket"), str)
        and isinstance(data.get("lite"), bool)
        and isinstance(data.get("messages"), list)
    )


class GreenSpecLoader(RawPayloadDatasetLoader):
    """Loader for a GreenSpec ``greenspec.jsonl``, filtered by plugin metadata.

    Example plugins.yaml entry::

        greenspec_agentic_tools_lite:
          class: aiperf.dataset.loader.greenspec:GreenSpecLoader
          metadata:
            category: agentic_tools
            lite: true
    """

    def __init__(
        self,
        filename: str | Path,
        run: BenchmarkRun | None = None,
        category: str | None = None,
        bucket: str | None = None,
        lite: bool | None = None,
        recorded: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.category = category
        self.bucket = bucket
        self.lite = lite
        self.recorded = bool(recorded)
        super().__init__(filename=filename, run=run, **kwargs)

    @classmethod
    def can_load(
        cls, data: LoaderProbeData | None = None, filename: str | Path | None = None
    ) -> bool:
        return is_greenspec_row(data)

    def _keep(self, row: dict[str, Any]) -> bool:
        return (
            (self.category is None or row["category"] == self.category)
            and (self.bucket is None or row["bucket"] == self.bucket)
            and (self.lite is None or row["lite"] == self.lite)
        )

    def _payload(self, row: dict[str, Any]) -> dict[str, Any]:
        cfg = self.run.cfg
        body = {k: v for k, v in row.items() if k not in META_KEYS}
        if self.recorded:
            body.update(row.get("recorded") or {})
        for key, value in cfg.endpoint.extra.items():
            capped = [k for k in _MAX_TOKEN_KEYS if k in body]
            if key in _MAX_TOKEN_KEYS and capped and isinstance(value, int):
                for k in capped:
                    body[k] = min(body[k], value)
            else:
                body[key] = value
        models = cfg.get_model_names()
        payload = {"model": models[0]} if models else {}
        payload.update(body)
        if cfg.endpoint.streaming:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _load_single_file(self, path: Path) -> dict[str, list[RawPayload]]:
        data: dict[str, list[RawPayload]] = defaultdict(list)
        for lineno, row in enumerate(self._iter_record_dicts(), start=1):
            if not is_greenspec_row(row):
                raise ValueError(f"{path}:{lineno}: not a GreenSpec row")
            if not self._keep(row):
                continue
            payload = self._payload(row)
            _validate_payload_shape(path, lineno, payload)
            data[row["question_id"]].append(RawPayload(payload=payload))
        if not data:
            raise ValueError(
                f"GreenSpec filter category={self.category!r} bucket={self.bucket!r} "
                f"lite={self.lite!r} matched no rows in {path}"
            )
        self.info(f"Loaded {len(data)} GreenSpec requests")
        return dict(data)

    def _load_directory(self, directory: Path) -> dict[str, list[RawPayload]]:
        raise ValueError("GreenSpec reads one greenspec.jsonl file, not a directory")
