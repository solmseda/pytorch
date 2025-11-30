#!/usr/bin/env python3
"""
Minimal repro for https://github.com/pytorch/pytorch/issues/113040.

This script exercises a torch.compile workload that intentionally hits both:
  * graph breaks (via torch._dynamo.graph_break)
  * frame recompilations (by changing tensor shapes across invocations)

It then dumps torch._dynamo.utils.counters so we can inspect which metrics are
currently surfaced.  The goal is to run this before any code changes so we
have a baseline that highlights the missing recompile tracking.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict

# Skip importing heavy distributed packages (which depend on numpy) if the
# environment doesn't have optional deps installed. This keeps the repro script
# usable in lightweight dev setups.
os.environ["TORCH_DIST_SKIP_IMPORTS"] = os.environ.get("TORCH_DIST_SKIP_IMPORTS", "1")

# Provide a very small numpy stub if numpy is unavailable. Some torch.distributed
# modules import numpy at import time for type annotations, which breaks in lean
# environments.
try:  # pragma: no cover - environment dependent
    import numpy  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from importlib.machinery import ModuleSpec

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.__spec__ = ModuleSpec("numpy", loader=None)  # type: ignore[assignment]

    class _FakeNDArray:
        def __init__(self, data: object) -> None:
            self.data = data
            self.flags = types.SimpleNamespace(writeable=False)

        def __array__(self, *args: object, **kwargs: object) -> None:
            raise ModuleNotFoundError(
                "numpy is required for array conversions; install numpy to enable it."
            )

    def _unavailable(*args: object, **kwargs: object) -> None:
        raise ModuleNotFoundError(
            "numpy is required for this operation; install numpy to enable it."
        )

    numpy_stub.ndarray = _FakeNDArray  # type: ignore[attr-defined]
    numpy_stub.dtype = type("dtype", (), {})  # type: ignore[attr-defined]
    numpy_stub.bool_ = bool  # type: ignore[attr-defined]
    numpy_stub.bool8 = bool  # type: ignore[attr-defined]
    numpy_stub.int64 = int  # type: ignore[attr-defined]
    numpy_stub.float64 = float  # type: ignore[attr-defined]
    numpy_stub.number = float  # type: ignore[attr-defined]
    numpy_stub.generic = object  # type: ignore[attr-defined]
    numpy_stub.object_ = object  # type: ignore[attr-defined]
    numpy_stub.array = _unavailable  # type: ignore[attr-defined]
    numpy_stub.zeros = _unavailable  # type: ignore[attr-defined]
    numpy_stub.ones = _unavailable  # type: ignore[attr-defined]
    numpy_stub.asarray = lambda x, *args, **kwargs: _FakeNDArray(x)  # type: ignore[attr-defined]
    numpy_stub.asanyarray = lambda x, *args, **kwargs: _FakeNDArray(x)  # type: ignore[attr-defined]
    numpy_stub.fft = types.ModuleType("numpy.fft")  # type: ignore[attr-defined]
    numpy_stub.fft.__spec__ = ModuleSpec("numpy.fft", loader=None)  # type: ignore[assignment]
    numpy_stub.linalg = types.ModuleType("numpy.linalg")  # type: ignore[attr-defined]
    numpy_stub.linalg.__spec__ = ModuleSpec("numpy.linalg", loader=None)  # type: ignore[assignment]
    numpy_stub.random = types.ModuleType("numpy.random")  # type: ignore[attr-defined]
    numpy_stub.random.__spec__ = ModuleSpec("numpy.random", loader=None)  # type: ignore[assignment]

    sys.modules["numpy"] = numpy_stub
    sys.modules["numpy.fft"] = numpy_stub.fft
    sys.modules["numpy.linalg"] = numpy_stub.linalg
    sys.modules["numpy.random"] = numpy_stub.random

import torch  # noqa: E402
import torch._dynamo as dynamo  # noqa: E402
from torch._dynamo.utils import counters  # noqa: E402


def workload(x: torch.Tensor, trigger_break: bool) -> torch.Tensor:
    """
    A tiny function that can trigger both a graph break and a recompile.

    - When trigger_break is True we call torch._dynamo.graph_break, which
      increments the graph_break counter.
    - On subsequent calls we vary the input shape so the compiled frame has to
      generate a fresh graph (recompile).
    """
    if trigger_break:
        # Force a graph break in the middle of the compiled region.
        torch._dynamo.graph_break()
        x = x + 1
    return x.sin()


def collect_counters() -> Dict[str, Dict[str, Any]]:
    """
    Materialise torch._dynamo.utils.counters into regular Python containers so
    the data can be serialised to JSON.
    """
    snapshot: Dict[str, Dict[str, Any]] = {}
    for category, bucket in counters.items():
        if not bucket:
            continue
        snapshot[category] = dict(bucket)
    return snapshot


def main() -> None:
    dynamo.reset()

    compiled = torch.compile(workload, backend="eager")

    # First call: compiles the graph and hits a graph break.
    compiled(torch.ones(4), True)

    # Second call: same shape, still hits the graph break but should reuse the
    # cached graph.
    compiled(torch.ones(4), True)

    # Third call: different shape forces Dynamo to recompile this frame.
    compiled(torch.ones(7), True)

    # Final call: different control-flow path (no graph break) to emphasise that
    # recompiles are distinct from graph breaks.
    compiled(torch.ones(7), False)

    snapshot = collect_counters()

    print("==== torch._dynamo.utils.counters snapshot ====")
    print(json.dumps(snapshot, indent=2, sort_keys=True))

    # Also write the snapshot next to this script for easy inspection.
    out_path = Path(__file__).with_suffix(".counters.json")
    out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    print(f"\nSaved counters to {out_path}")


if __name__ == "__main__":
    main()
