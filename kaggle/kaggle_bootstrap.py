"""Bootstrap the MIPIC repository inside a Kaggle notebook."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

WORK_ROOT = Path("/kaggle/working")
REPO_ROOT = WORK_ROOT / "mipic"


def _zip_score(candidate: Path) -> tuple[int, int, int]:
    """Prefer the final MIPIC archive and require its notebook artifact."""
    try:
        with zipfile.ZipFile(candidate) as archive:
            names = archive.namelist()
            has_runner = any(name.endswith("run_experiments.py") for name in names)
            has_notebook = any(name.endswith("MIPIC_Kaggle.ipynb") for name in names)
    except zipfile.BadZipFile:
        return (-1, -1, -1)
    if not has_runner:
        return (-1, -1, -1)
    # The readiness notebook is preferred when available, but is not required
    # for execution. Archives that contain it outrank repository-only archives.
    notebook_rank = int(has_notebook)
    final_rank = int("FINAL" in candidate.name.upper() or "AUDITED" in candidate.name.upper())
    return (notebook_rank, final_rank, candidate.stat().st_mtime_ns)


def _find_repo_zip() -> Path | None:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    candidates = []
    for candidate in input_root.rglob("*.zip"):
        score = _zip_score(candidate)
        if score[0] >= 0:
            candidates.append((score, candidate))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _safe_extract_flat(zip_path: Path, destination: Path) -> Path:
    """Extract an archive and normalize a single top-level repo directory."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=WORK_ROOT) as tmp:
        tmp_root = Path(tmp) / "extracted"
        tmp_root.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_root)

        runners = list(tmp_root.rglob("run_experiments.py"))
        if not runners:
            raise FileNotFoundError(f"Could not find run_experiments.py after extracting {zip_path}")
        source_root = runners[0].parent
        # Copy the actual repository contents directly into /kaggle/working/mipic.
        for item in source_root.iterdir():
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    # The notebook is optional at runtime; Kaggle users may copy only the
    # source tree. The runner is the actual executable artifact.
    required = [destination / "run_experiments.py"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Repository archive is missing required runtime artifacts: " + ", ".join(missing)
        )
    return destination


def bootstrap() -> Path:
    current = Path.cwd()
    if (current / "run_experiments.py").exists():
        root = current
    elif (REPO_ROOT / "run_experiments.py").exists():
        root = REPO_ROOT
    else:
        zip_path = _find_repo_zip()
        if zip_path is None:
            raise FileNotFoundError(
                "No suitable MIPIC repository archive found. Attach the final MIPIC ZIP "
                "as a Kaggle Dataset or run this script from the repository root."
            )
        root = _safe_extract_flat(zip_path, REPO_ROOT)

    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", "/kaggle/working/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/working/huggingface/transformers")
    os.environ.setdefault("MIPIC_OUTPUT_DIR", "/kaggle/working/mipic_results")
    print(f"MIPIC repository: {root}")
    print(f"Results/checkpoints: {os.environ['MIPIC_OUTPUT_DIR']}")
    return root


if __name__ == "__main__":
    bootstrap()
