from pathlib import Path
import importlib.util
import sys
import types
import torch


def _load_runner():
    # Import the runner with lightweight transformer/model stubs because the
    # checkpoint helpers themselves do not require the heavyweight dependency.
    tfm = types.ModuleType("transformers")
    tfm.get_scheduler = lambda *a, **k: None
    tfm.AutoModel = object
    tfm.AutoTokenizer = object
    sys.modules.setdefault("transformers", tfm)
    spec = importlib.util.spec_from_file_location("run_experiments_test", Path("run_experiments.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_atomic_torch_save_publishes_valid_checkpoint(tmp_path):
    mod = _load_runner()
    path = tmp_path / "model.pt"
    mod._atomic_torch_save({"x": torch.tensor([1, 2, 3])}, path)
    mod._assert_checkpoint(path)
    assert torch.equal(torch.load(path)["x"], torch.tensor([1, 2, 3]))


def test_eval_only_can_fall_back_to_resume_checkpoint(tmp_path, monkeypatch):
    mod = _load_runner()
    mod.CHECKPOINT_DIR = tmp_path / "checkpoints"
    resume = mod.CHECKPOINT_DIR / "mipic" / "training_state.pt"
    resume.parent.mkdir(parents=True)
    mod._atomic_torch_save({"epoch": 5}, resume)
    monkeypatch.delenv("MIPIC_CHECKPOINT", raising=False)
    assert mod._configured_checkpoint_path("mipic", eval_only=True) == resume.resolve()


def test_training_keeps_resume_checkpoint_until_evaluation_succeeds():
    source = Path("run_experiments.py").read_text(encoding="utf-8")
    eval_start = source.index("eval_error = None")
    tail = source[eval_start:]
    assert "resume_path.unlink()" in tail
    assert tail.index("resume_path.unlink()") > tail.index("save_results(all_results)")


def test_eval_loader_accepts_resumable_training_state():
    mod = _load_runner()
    model = torch.nn.Linear(2, 2)
    path = Path("/tmp/mipic_test_resume_state.pt")
    state = {"model_state_dict": model.state_dict(), "epoch": 5}
    mod._atomic_torch_save(state, path)
    target = torch.nn.Linear(2, 2)
    mod._load_model_checkpoint(target, path, torch.device("cpu"))
    for a, b in zip(model.parameters(), target.parameters()):
        assert torch.equal(a, b)
    path.unlink(missing_ok=True)
