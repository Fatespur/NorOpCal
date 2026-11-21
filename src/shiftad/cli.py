"""Command-line entry points for configuration and runtime utilities."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from shiftad import __version__
from shiftad.data.validation import validate_mvtec_ad
from shiftad.utils.config import (
    ConfigurationError,
    EfficientADModelConfig,
    load_experiment_config,
)
from shiftad.utils.environment import collect_environment, write_environment
from shiftad.workflows import (
    WorkflowError,
    run_build_memory,
)
from shiftad.workflows import (
    run_evaluation as run_memory_evaluation,
)
from shiftad.workflows.efficientad import (
    WorkflowError as EfficientADWorkflowError,
)
from shiftad.workflows.efficientad import (
    run_evaluation as run_efficientad_evaluation,
)
from shiftad.workflows.efficientad import (
    run_train as run_efficientad_train,
)
from shiftad.workflows.shift_benchmark import (
    ShiftBenchmarkError,
    run_shift_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the ShiftAD CLI parser and documented subcommands.

    Returns:
        A configured argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="shiftad",
        description="Reproducible industrial anomaly-detection utilities.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser(
        "validate-config",
        help="Validate and print a resolved experiment YAML configuration.",
    )
    validate.add_argument("--config", type=Path, required=True, help="Experiment YAML path.")

    capture = subparsers.add_parser(
        "capture-environment",
        help="Write privacy-conscious runtime and Git metadata as JSON.",
    )
    capture.add_argument("--output", type=Path, required=True, help="Destination JSON path.")
    capture.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to capture Git state (default: current directory).",
    )

    dataset = subparsers.add_parser(
        "validate-dataset",
        help="Validate a configured dataset layout and image integrity.",
    )
    dataset.add_argument("--config", type=Path, required=True, help="Experiment YAML path.")
    dataset.add_argument(
        "--output",
        type=Path,
        help="Optional destination for the JSON validation report.",
    )

    build_memory = subparsers.add_parser(
        "build-memory",
        help="Build and atomically save a normal patch memory (DINO memory or PatchCore).",
    )
    build_memory.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment YAML path (model.name: dino_memory or patchcore).",
    )

    train = subparsers.add_parser(
        "train",
        help="Train a model that requires optimization (e.g. EfficientAD).",
    )
    train.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment YAML path (model.name: efficientad).",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved model without test-set threshold tuning.",
    )
    evaluate.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment YAML path.",
    )
    evaluate.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Complete model directory produced by build-memory or train.",
    )

    shift_benchmark = subparsers.add_parser(
        "shift-benchmark",
        help="Run the Phase 5 shift benchmark workflow (fixture or formal scope).",
    )
    shift_benchmark.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Phase 5 benchmark YAML configuration path.",
    )
    shift_benchmark.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for cell manifests and scope markers.",
    )
    shift_benchmark.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print the cell plan without writes.",
    )
    shift_benchmark.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells that already have a completion marker.",
    )
    shift_benchmark.add_argument(
        "--stage-c-authorization",
        type=Path,
        help="Commit-bound Stage C authorization artifact (required for formal scope).",
    )

    stream_adaptation = subparsers.add_parser(
        "stream-adaptation",
        help="Run the Phase 6-H formal causal streaming evaluation (A0-A4).",
    )
    stream_adaptation.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase6/formal.yaml"),
        help="Phase 6 formal YAML configuration path.",
    )
    stream_adaptation.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for cell manifests and scope markers.",
    )
    stream_adaptation.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="MVTec AD dataset root.",
    )
    stream_adaptation.add_argument(
        "--model-base",
        type=Path,
        required=True,
        help="Directory containing the 15 dino_memory model dirs.",
    )
    stream_adaptation.add_argument(
        "--fit-report",
        type=Path,
        required=True,
        help="Gate C shift-estimator fit report JSON.",
    )
    stream_adaptation.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells that already have a validated completion marker.",
    )
    stream_adaptation.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda).",
    )
    stream_adaptation.add_argument(
        "--stage-g-authorization",
        type=Path,
        required=True,
        help="Commit-bound Stage G authorization artifact.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the CLI and return a process status code.

    Args:
        argv: Optional argument tokens excluding the executable name.

    Returns:
        Zero on success and two for a handled validation failure.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "validate-config":
            config = load_experiment_config(args.config)
            print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
            return 0
        if args.command == "capture-environment":
            snapshot = collect_environment(
                repo_root=args.repo_root,
                command=["shiftad", "capture-environment"],
            )
            write_environment(snapshot, args.output)
            print(f"Wrote environment metadata to {args.output}")
            return 0
        if args.command == "validate-dataset":
            config = load_experiment_config(args.config)
            if config.dataset.name != "mvtec_ad":
                raise ConfigurationError(
                    f"Dataset validation does not yet support {config.dataset.name!r}; "
                    "expected 'mvtec_ad'."
                )
            report = validate_mvtec_ad(
                config.dataset.root,
                config.dataset.categories,
            )
            serialized = json.dumps(report.to_dict(), indent=2, sort_keys=True)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized + "\n", encoding="utf-8")
            print(serialized)
            return 0 if report.valid else 2
        if args.command == "build-memory":
            config = load_experiment_config(args.config)
            if isinstance(config.model, EfficientADModelConfig):
                raise ConfigurationError(
                    "EfficientAD requires training, not memory building. "
                    "Use 'shiftad train --config <config>' instead."
                )
            summary = run_build_memory(config, repo_root=Path.cwd())
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "train":
            config = load_experiment_config(args.config)
            if not isinstance(config.model, EfficientADModelConfig):
                raise ConfigurationError(
                    f"train command supports EfficientAD only; "
                    f"got model.name={config.model.name if config.model else 'None'}. "
                    "Use 'shiftad build-memory' for memory-based methods."
                )
            summary = run_efficientad_train(config, repo_root=Path.cwd())
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "evaluate":
            config = load_experiment_config(args.config)
            if isinstance(config.model, EfficientADModelConfig):
                summary = run_efficientad_evaluation(
                    config,
                    model_dir=args.model_dir,
                    repo_root=Path.cwd(),
                )
            else:
                summary = run_memory_evaluation(
                    config,
                    model_dir=args.model_dir,
                    repo_root=Path.cwd(),
                )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "shift-benchmark":
            summary = run_shift_benchmark(
                args.config,
                args.output_dir,
                repo_root=Path.cwd(),
                dry_run=args.dry_run,
                resume=args.resume,
                stage_c_authorization=args.stage_c_authorization,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0 if summary.get("failed_cells", 0) == 0 else 2
        if args.command == "stream-adaptation":
            from shiftad.workflows.streaming_adaptation import main as stream_main

            return stream_main(
                [
                    "--config",
                    str(args.config),
                    "--output-dir",
                    str(args.output_dir),
                    "--dataset-root",
                    str(args.dataset_root),
                    "--model-base",
                    str(args.model_base),
                    "--fit-report",
                    str(args.fit_report),
                    "--stage-g-authorization",
                    str(args.stage_g_authorization),
                    "--device",
                    args.device,
                ]
                + (["--resume"] if args.resume else [])
            )
    except WorkflowError as exc:
        print(
            json.dumps(
                {"error": exc.to_dict()},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except EfficientADWorkflowError as exc:
        print(
            json.dumps(
                {"error": exc.to_dict()},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except ShiftBenchmarkError as exc:
        print(
            json.dumps(
                {"error": exc.to_dict()},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except (ConfigurationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"Unknown command {args.command!r}")
