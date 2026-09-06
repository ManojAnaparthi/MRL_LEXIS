from pathlib import Path


def test_run_experiments_has_single_entrypoint_and_single_cli_invocation() -> None:
    source = Path("run_experiments.py").read_text(encoding="utf-8")

    assert source.count("def run_experiment(") == 1
    assert source.count("run_experiment(selected, eval_only=args.eval_only, skip_existing=args.skip_existing)") == 1
    assert "run_experiment(selected)" not in source


def test_run_experiments_compiles() -> None:
    compile(Path("run_experiments.py").read_text(encoding="utf-8"), "run_experiments.py", "exec")


def test_run_experiments_prefers_mps_when_cuda_is_unavailable() -> None:
    source = Path("run_experiments.py").read_text(encoding="utf-8")

    assert 'elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():' in source
    assert 'device = torch.device("mps")' in source
    assert 'Device: mps (Apple Silicon GPU)' in source
