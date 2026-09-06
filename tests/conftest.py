from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    try:
        import transformers  # noqa: F401
    except ImportError:
        import pytest
        marker = pytest.mark.skip(reason="transformers is not installed in this test environment")
        for item in items:
            path = str(item.fspath)
            if "integration" in Path(path).name or "mrl_baseline_integration" in Path(path).name:
                item.add_marker(marker)
