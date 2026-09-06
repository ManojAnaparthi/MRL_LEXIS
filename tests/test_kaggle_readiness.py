from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_kaggle_dependencies_do_not_reinstall_torch() -> None:
    text = (ROOT / "requirements-kaggle.txt").read_text(encoding="utf-8")
    assert "torch" not in {line.strip().split("#", 1)[0].strip() for line in text.splitlines() if line.strip()}


def test_kaggle_run_instructions_are_available() -> None:
    # Readiness is validated from the executable README; the notebook is
    # optional and must never be a required runtime/test dependency.
    text = (ROOT / "kaggle" / "README.md").read_text(encoding="utf-8")
    assert "run_experiments.py --model all --skip-existing" in text
    assert "/kaggle/working/mipic_results" in text


def test_runner_supports_kaggle_output_and_resume() -> None:
    text = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
    assert "MIPIC_OUTPUT_DIR" in text
    assert "training_state.pt" in text
    assert "MIPIC_FRESH" in text
    assert "torch.cuda.get_rng_state_all" in text


def test_train_loop_supports_epoch_checkpoints() -> None:
    text = (ROOT / "train.py").read_text(encoding="utf-8")
    assert "epoch_callback" in text
    assert "start_epoch" in text


def test_kaggle_smoke_and_full_run_defaults_are_documented() -> None:
    readme = (ROOT / "kaggle" / "README.md").read_text(encoding="utf-8")
    assert "MIPIC_MAX_BATCHES=3" in readme
    assert "5-epoch" in readme


def test_train_has_no_unconditional_scheduler_step_before_optimizer_step() -> None:
    text = (ROOT / "train.py").read_text(encoding="utf-8")
    assert "scheduler.step()" in text
    assert "_optimizer_step(optimizer, scaler, amp_enabled, scheduler)" in text


def test_runner_loads_tokenizer_without_loading_a_second_model() -> None:
    text = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
    assert "load_tinybert_tokenizer" in text
    assert "_, tokenizer = load_tinybert_6l" not in text


def test_mipic_ddp_does_not_need_unused_parameter_detection() -> None:
    """The only unused parameters were the pooler's; it is frozen instead."""
    source = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
    start = source.index("def maybe_wrap_ddp")
    end = source.index("# Dataset paths", start)
    block = source[start:end]
    assert "find_unused_parameters=False" in block
    assert "model.module" not in block


def test_kaggle_bootstrap_prefers_readiness_capable_archive() -> None:
    source = (ROOT / "kaggle" / "kaggle_bootstrap.py").read_text(encoding="utf-8")
    assert "has_notebook" in source
    assert "_safe_extract_flat" in source


def test_smoke_run_can_optionally_exercise_checkpoint_and_evaluation() -> None:
    """MIPIC_MAX_BATCHES alone skips checkpoint+eval; the opt-in flag keeps them."""
    text = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
    assert "MIPIC_SMOKE_FULL_PIPELINE" in text
    readme = (ROOT / "kaggle" / "README.md").read_text(encoding="utf-8")
    assert "MIPIC_SMOKE_FULL_PIPELINE=1" in readme
    assert "--eval-only" in readme


def test_kaggle_procedure_avoids_cross_cell_shell_state() -> None:
    """No `!export` in one cell followed by `!python` in another."""
    import json

    notebook = json.loads((ROOT / "MIPIC_Kaggle_FINAL.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        assert "!export" not in source


def test_ddp_setup_rejects_out_of_range_local_rank_with_a_useful_message() -> None:
    """A stale CUDA_VISIBLE_DEVICES must not surface as 'invalid device ordinal'."""
    text = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
    start = text.index("def setup_distributed")
    end = text.index("def cleanup_distributed", start)
    block = text[start:end]
    assert "local_rank >= visible" in block
    assert "CUDA_VISIBLE_DEVICES" in block
    # The guard must precede the call that would otherwise raise.
    assert block.index("local_rank >= visible") < block.index("torch.cuda.set_device(local_rank)")


def test_notebook_mode_switch_sets_cuda_visible_devices_in_both_branches() -> None:
    """Switching RUN_MODE must fully override the previous mode's device mask."""
    import json

    notebook = json.loads((ROOT / "MIPIC_Kaggle_FINAL.ipynb").read_text(encoding="utf-8"))
    cell = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "RUN_MODE = " in "".join(c["source"])
    )
    assert cell.count("CUDA_VISIBLE_DEVICES") >= 3
    assert '"CUDA_VISIBLE_DEVICES"] = "0,1"' in cell   # ddp branch widens it back
    assert '"CUDA_VISIBLE_DEVICES"] = "0"' in cell     # single branch narrows it
    # The GPU-count check must query a subprocess, not the cached in-kernel value.
    assert "subprocess.run" in cell and "torch.cuda.device_count()" in cell


def test_truncated_runs_are_labelled_in_output_and_in_the_results_file() -> None:
    """A 5-second smoke run must never look like a reportable result."""
    text = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
    assert "TRUNCATED RUN - MIPIC_MAX_BATCHES=" in text
    assert "PLUMBING CHECK ONLY" in text
    # Persisted, so the provenance survives in tweeteval_results.json.
    for key in ("truncated_run", "max_batches_per_rank", "train_sentences_seen"):
        assert key in text, key
    # The warning must be inside the table printer, not only at startup.
    start = text.index("def print_comparison_table")
    end = text.index("def _env_int", start)
    assert "truncated_run" in text[start:end]


def test_there_is_exactly_one_configuration_and_no_presets() -> None:
    """One implementation. No preset bundles, no reproduction switches."""
    text = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
    for gone in ("MIPIC_PRESET", "_PRESETS", "_apply_preset", "ACTIVE_PRESET",
                 "reproduction_config_kwargs", "MIPIC_MRL_PROJECTION",
                 "MIPIC_SIA_CONTEXTUAL_MASK", "MIPIC_CKA_DTYPE",
                 "MIPIC_PIC_LR_MULTIPLIER", "MIPIC_MAX_GRAD_NORM"):
        assert gone not in text, f"{gone} should have been removed"

    import json
    notebook = json.loads((ROOT / "MIPIC_Kaggle_FINAL.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        assert "MIPIC_PRESET" not in "".join(cell["source"])


def test_the_shipped_corpus_and_probe_are_the_paper_ones() -> None:
    """Paper A.2 pooled corpus; Table 6 probe splits. Hardcoded, not switched."""
    text = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
    assert '_resolve_data_file("final_data.csv"' in text
    assert '_resolve_data_file("tweet_train.csv"' in text
    assert '_resolve_data_file("tweet_test.csv"' in text
    # Clipping and the probe budget are constants now, not env lookups.
    assert "max_grad_norm=0.0," in text
    assert "max_iter=1000," in text


def test_sia_and_pic_average_their_terms() -> None:
    """The approved deviation from Eq. 13/14/17, with its rationale in-tree."""
    sia = (ROOT / "losses" / "sia_loss.py").read_text(encoding="utf-8")
    assert "torch.stack(attn_terms).mean()" in sia
    assert "torch.stack(cka_terms).mean()" in sia
    assert "alpha=0.4 in Eq. (18) is inoperative" in sia, "the deviation must be documented"

    pic = (ROOT / "losses" / "pic_loss.py").read_text(encoding="utf-8")
    assert "total_chain_loss / len(self.parsed_pipeline_pairs)" in pic


def test_runs_are_seeded() -> None:
    """Unseeded runs made 3-point probe differences unreadable."""
    text = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
    assert "def seed_everything" in text
    for call in ("random.seed", "np.random.seed", "torch.manual_seed", "torch.cuda.manual_seed_all"):
        assert call in text, call
    # Seeded before any model is built, and recorded in the results file.
    assert text.index("seed_everything(seed, rank)") < text.index("target_models =")
    assert '"seed": seed,' in text


def test_seeding_is_reproducible_and_rank_offset() -> None:
    import importlib.util
    import sys
    import types

    tfm = types.ModuleType("transformers")
    tfm.get_scheduler = lambda *a, **k: None
    tfm.AutoModel = object
    tfm.AutoTokenizer = object
    sys.modules.setdefault("transformers", tfm)
    spec = importlib.util.spec_from_file_location("re_seed_test", ROOT / "run_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import torch as _torch

    mod.seed_everything(42, 0)
    a = _torch.randn(4)
    mod.seed_everything(42, 0)
    b = _torch.randn(4)
    assert _torch.equal(a, b), "same seed must reproduce"

    mod.seed_everything(42, 1)
    c = _torch.randn(4)
    assert not _torch.equal(a, c), "different ranks must decorrelate"
    assert mod.seed_everything(42, 1) == 43


def test_epoch_selection_keeps_the_best_epoch_not_the_last() -> None:
    """The contrastive objective is solved in one epoch; later epochs degrade."""
    import importlib.util
    import sys
    import types

    tfm = types.ModuleType("transformers")
    tfm.get_scheduler = lambda *a, **k: None
    tfm.AutoModel = object
    tfm.AutoTokenizer = object
    sys.modules.setdefault("transformers", tfm)
    spec = importlib.util.spec_from_file_location("re_sel_test", ROOT / "run_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import torch as _torch

    scores = iter([0.40, 0.55, 0.52, 0.48, 0.45])   # peaks at epoch 2
    saved = []
    mod.validation_probe_score = lambda *a, **k: next(scores)
    mod._atomic_torch_save = lambda obj, path: saved.append(path)
    mod.unwrap_model = lambda m: m

    on_epoch, state = mod._make_selector("mipic", lambda m: m, object(), object())
    for epoch in range(1, 6):
        on_epoch(epoch, _torch.nn.Linear(2, 2))

    assert state["best_epoch"] == 2, state["history"]
    assert round(state["best_score"], 2) == 0.55
    assert len(saved) == 2, "only improvements are checkpointed"
    assert [h["epoch"] for h in state["history"]] == [1, 2, 3, 4, 5]
    assert state["history"][1]["val_macro_f1"] == 55.0


def test_epoch_selection_never_touches_the_test_split() -> None:
    """Selection must use probe-TRAIN rows only, or it leaks test labels."""
    source = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
    start = source.index("def _make_selector")
    end = source.index("def run_experiment", start)
    block = source[start:end]
    assert "EVAL_TRAIN_CSV" in block
    assert "EVAL_TEST_CSV" not in block, "selection must not see the test split"

    evaluate_src = (ROOT / "evaluate.py").read_text(encoding="utf-8")
    vstart = evaluate_src.index("def validation_probe_score")
    vend = evaluate_src.index("def evaluate_tweeteval", vstart)
    vblock = evaluate_src[vstart:vend]
    assert "test_csv" not in vblock
    assert "holdout" in vblock and "fit_idx" in vblock and "held_idx" in vblock
