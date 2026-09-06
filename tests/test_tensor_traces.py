"""Run the shipped diagnostic scripts end to end.

`tensor_traces.py` and `diagnose_hang.py` are operator-facing entry points that
nothing else imports, so a signature or behaviour change elsewhere can break
them silently. This ran exactly that way once: restoring the PIC stop-gradient
made `tgt_layer.grad` None, and `tensor_traces.py` crashed with an
AttributeError on Kaggle because no test executed it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )


def test_tensor_traces_runs_clean() -> None:
    result = _run("tensor_traces.py")
    assert result.returncode == 0, f"tensor_traces.py failed:\n{result.stderr}"
    out = result.stdout
    for marker in ("MRL trace", "SIA trace", "PIC trace", "Total objective"):
        assert marker in out, f"missing section: {marker}"
    assert "Traceback" not in result.stderr


def test_tensor_traces_reports_the_shipped_pic_stop_gradient() -> None:
    """The trace must describe the default the code actually ships."""
    from losses.pic_loss import PipelineInfoNCELoss

    out = _run("tensor_traces.py").stdout
    default_detach = PipelineInfoNCELoss(2, 4).detach_target
    assert f"detach_target                 {default_detach}" in out
    if default_detach:
        assert "tgt_layer   None  (stop-gradient)" in out
        # and the contrasting variant is still demonstrated
        assert "detach_target=False -> tgt_layer |g|=" in out


def test_tensor_traces_reports_the_shipped_dims_and_chain() -> None:
    """Guard against the traces drifting away from the shipped constants."""
    from losses.mrl_loss import MRL_NESTED_DIMS
    from models.mrl_sia_pic import TINYBERT_PIC_CHECKPOINTS

    out = _run("tensor_traces.py").stdout
    assert "5 adjacent transitions" in out
    assert len(TINYBERT_PIC_CHECKPOINTS) - 1 == 5
    # The SIA summary lists the six sub-full prefixes, descending.
    expected = "dims [" + ",".join(str(d) for d in reversed(MRL_NESTED_DIMS[:-1])) + "]"
    assert expected in out, f"expected {expected!r} in the SIA summary line"


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="diagnose_hang.py needs Linux /proc")
def test_diagnose_hang_runs_on_linux() -> None:
    result = _run("diagnose_hang.py", "--pattern", "no-such-process-xyz")
    # Exits 1 when nothing matches, but must not traceback.
    assert result.returncode in (0, 1)
    assert "Traceback" not in result.stderr


def test_diagnose_hang_is_importable_everywhere() -> None:
    result = _run("diagnose_hang.py", "--pattern", "no-such-process-xyz")
    assert "Traceback" not in result.stderr
