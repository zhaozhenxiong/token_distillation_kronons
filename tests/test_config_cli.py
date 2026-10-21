from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

from token_distillation.artifacts import write_json
from token_distillation.cli import main
from token_distillation.config import PROJECT_ROOT, load_config
from token_distillation.training.planning import training_plan


class ConfigAndCliTests(unittest.TestCase):
    def setup_config(self, directory, edits=()):
        root = Path(directory)
        text = (PROJECT_ROOT / "configs/default.toml").read_text()
        for old, new in edits:
            text = text.replace(old, new)
        path = root / "config.toml"
        path.write_text(text)
        return load_config(path, root=root)

    def test_paths_resolve_from_project_not_config_or_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.setup_config(directory)
            self.assertEqual(config.raw_db, (Path(directory) / "../KHQuant/app/my_strategy/data/raw/khquant_raw.db").resolve())
            self.assertFalse(config.state_root.exists())

    def test_no_output_escape_or_symlink_to_external_assets(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            config = self.setup_config(directory)
            with self.assertRaises(ValueError):
                config.output_path("../model.safetensors")
            (config.root / "redirect").symlink_to(external, target_is_directory=True)
            with self.assertRaises(ValueError):
                write_json(config, "redirect/test.json", {})
            self.assertFalse((Path(external) / "test.json").exists())

    def test_atomic_metadata_refuses_overwrite_without_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.setup_config(directory)
            target = write_json(config, "artifacts/report.json", {"n": 1})
            with self.assertRaises(FileExistsError):
                write_json(config, "artifacts/report.json", {"n": 2})
            self.assertEqual(json.loads(target.read_text()), {"n": 1})
            write_json(config, "artifacts/report.json", {"n": 2}, replace=True)
            self.assertEqual(json.loads(target.read_text()), {"n": 2})

    def test_generated_output_cannot_replace_code_or_frozen_ict_source(self):
        config = load_config()
        for target in ("run.py", "configs/default.toml", "vendor/ict_trading_model/scripts/ict_scan.py"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                config.output_path(target)

    def test_invalid_training_and_universe_are_rejected(self):
        for old, new in (("execution_enabled = false", "execution_enabled = true"),
                         ('universe = "sh_sz_a_v1"', 'universe = "all"'),
                         ("context_length = 256", "context_length = 513"),
                         ("max_candidates = 6", "max_candidates = 7"),
                         ("lora_alpha = 16.0", "lora_alpha = nan")):
            with self.subTest(new=new), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.setup_config(directory, ((old, new),))

    def test_chronological_split_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self.setup_config(directory, (('tune = ["2024-01-01"', 'tune = ["2023-12-31"'),))

    def test_training_plan_marks_available_pipeline_without_execution(self):
        config = load_config()
        report = training_plan(config)
        self.assertFalse(report["training_started"])
        self.assertTrue(report["execution_implemented"])
        self.assertFalse(report["execution_enabled"])
        self.assertIn("optimizer_resume", report["not_implemented"])

    def test_cli_train_plan_does_not_need_torch_or_source_load(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["train-plan"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output.getvalue())["execution_implemented"])
        self.assertFalse(json.loads(output.getvalue())["training_started"])

    def test_pilot_configuration_enables_only_explicit_execution(self):
        config = load_config(PROJECT_ROOT / "configs/smoke.toml")
        self.assertEqual(config.values["training"]["stage"], "pilot_ready")
        self.assertTrue(config.values["training"]["execution_enabled"])
        self.assertEqual(config.context_length, 128)

    def test_execution_command_arguments_are_explicit(self):
        from token_distillation.cli import parser
        args = parser().parse_args(["build-dataset", "--symbols", "600519.SH", "000001.SZ",
            "--start", "2026-01-05", "--end", "2026-10-09", "--label-through", "2026-10-16",
            "--output-dir", "data/pilot"])
        self.assertEqual(args.symbols, ["600519.SH", "000001.SZ"])
        self.assertEqual(args.stride, 5)
        args = parser().parse_args(["evaluate", "--dataset", "data/pilot",
            "--checkpoint", "artifacts/run/checkpoint.pt", "--output-dir", "artifacts/eval", "--split", "all"])
        self.assertEqual(args.split, "all")

    def test_nan_json_is_rejected_before_output_directory_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.setup_config(directory)
            with self.assertRaises(ValueError):
                write_json(config, "artifacts/new/report.json", {"bad": float("nan")})
            self.assertFalse((config.root / "artifacts/new").exists())


if __name__ == "__main__":
    unittest.main()
