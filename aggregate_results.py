import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------

def base_run_name(run_name: str) -> str:
    """
    Remove only the final _seed<number> component.

    Example
    -------
    MorphViT_cifar-10_MWU_T1.0_seed42
        -> MorphViT_cifar-10_MWU_T1.0
    """

    # Remove optional short config hash.
    run_name = re.sub(
        r"__[0-9a-fA-F]{6,64}$",
        "",
        run_name,
    )

    match = re.fullmatch(
        r"(.+)_seed\d+",
        run_name,
    )

    if match is None:
        raise ValueError(
            f"Invalid run directory name: {run_name!r}. "
            "Expected format "
            "<model>_<dataset>_<MWU/NO-MWU>_T<temperature>_seed<seed>__<hash>."
        )

    return match.group(1)


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------

HISTORY_RE = re.compile(
    r"Epoch:\s*(\d+),\s*"
    r"Train_loss:\s*([-+eE0-9.]+),\s*"
    r"Train metric:\s*([-+eE0-9.]+),\s*"
    r"Val metric:\s*([-+eE0-9.]+)"
)

TEST_RE = re.compile(
    r"Test_metric:\s*([-+eE0-9.]+)"
)


def parse_training_history(path: Path):
    """
    Parse training_history.txt.

    If multiple runs were accidentally appended to the same file,
    only the final block is returned.
    """

    epochs = []
    train_loss = []
    train_metric = []
    val_metric = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("Total number of parameters:"):
                epochs = []
                train_loss = []
                train_metric = []
                val_metric = []
                continue

            match = HISTORY_RE.search(line)

            if match is None:
                continue

            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            train_metric.append(float(match.group(3)))
            val_metric.append(float(match.group(4)))

    if not epochs:
        raise ValueError(
            f"No training history found in {path}"
        )

    return {
        "epoch": np.asarray(epochs),
        "train_loss": np.asarray(train_loss),
        "train_metric": np.asarray(train_metric),
        "val_metric": np.asarray(val_metric),
    }


def parse_test_metric(path: Path):
    """
    Return the last Test_metric value in the file.
    """

    values = []

    with open(path, "r") as f:
        for line in f:
            match = TEST_RE.search(line)

            if match is not None:
                values.append(float(match.group(1)))

    if not values:
        raise ValueError(
            f"No test metric found in {path}"
        )

    return values[-1]


# ---------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------

def matches_filters(
    name,
    include=None,
    exclude=None,
):
    name_lower = name.lower()

    if include:
        if not all(
            token.lower() in name_lower
            for token in include
        ):
            return False

    if exclude:
        if any(
            token.lower() in name_lower
            for token in exclude
        ):
            return False

    return True


def collect_runs(
    root: Path,
    include=None,
    exclude=None,
):
    """
    Returns:

        {
            base_experiment_name: [
                {
                    "run_name": ...,
                    "test_metric": ...,
                    "history": ...,
                },
                ...
            ]
        }
    """

    groups = {}

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        results_dir = run_dir / "results"

        test_path = results_dir / "test_metric.txt"
        history_path = results_dir / "training_history.txt"

        if not test_path.exists():
            print(
                f"[skip] Missing test metric: {run_dir.name}"
            )
            continue

        name = base_run_name(run_dir.name)

        if not matches_filters(
            name,
            include=include,
            exclude=exclude,
        ):
            continue

        try:
            test_metric = parse_test_metric(test_path)

            history = (
                parse_training_history(history_path)
                if history_path.exists()
                else None
            )

        except Exception as exc:
            print(
                f"[skip] Failed to parse "
                f"{run_dir.name}: {exc}"
            )
            continue

        groups.setdefault(name, []).append({
            "run_name": run_dir.name,
            "test_metric": test_metric,
            "history": history,
        })

    return groups


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def mean_std(values):
    values = np.asarray(values, dtype=float)

    mean = values.mean()

    # Sample standard deviation across seeds.
    std = (
        values.std(ddof=1)
        if len(values) > 1
        else 0.0
    )

    return mean, std


def summarize(groups, output_path: Path):
    lines = []

    for name in sorted(groups):
        runs = groups[name]

        values = [
            run["test_metric"]
            for run in runs
        ]

        mean, std = mean_std(values)

        lines.append(name)
        lines.append(
            f"  Test metric: "
            f"{mean:.4f} ± {std:.4f}"
        )
        lines.append(
            f"  Runs: {len(values)}"
        )
        lines.append(
            "  Individual: "
            + ", ".join(
                f"{v:.4f}"
                for v in values
            )
        )
        lines.append("")

    text = "\n".join(lines)

    print(text)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        f.write(text)

    print(
        f"\nSaved summary to {output_path}"
    )


# ---------------------------------------------------------------------
# Training-curve aggregation
# ---------------------------------------------------------------------

def aggregate_history(runs, key):
    histories = [
        run["history"]
        for run in runs
        if run["history"] is not None
    ]

    if not histories:
        raise ValueError(
            "No training histories available."
        )

    # Seeds should normally have the same number of epochs.
    # Taking the minimum makes this robust to interrupted runs.
    num_epochs = min(
        len(history[key])
        for history in histories
    )

    values = np.stack([
        history[key][:num_epochs]
        for history in histories
    ])

    epochs = histories[0]["epoch"][:num_epochs]

    mean = values.mean(axis=0)

    if len(values) > 1:
        std = values.std(
            axis=0,
            ddof=1,
        )
    else:
        std = np.zeros_like(mean)

    return epochs, mean, std


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def resolve_plot_names(groups, requested_names):
    """
    Resolve experiment names using prefix matching only.

    Examples
    --------
    Query:
        ViT_cifar-10_NO-MWU

    May match:
        ViT_cifar-10_NO-MWU_T1.0

    Will NOT match:
        MorphViT_cifar-10_NO-MWU_T1.0
    """

    available = list(groups.keys())
    resolved = []

    for query in requested_names:
        query_lower = query.lower()

        # Exact match always wins.
        exact_matches = [
            name
            for name in available
            if name.lower() == query_lower
        ]

        if len(exact_matches) == 1:
            resolved.append(exact_matches[0])
            continue
        
        prefix_matches = [
            name
            for name in available
            if name.lower().startswith(query_lower)
        ]

        if len(prefix_matches) == 0:
            raise ValueError(
                f"No experiment starts with {query!r}.\n\n"
                "Available experiment names:\n"
                + "\n".join(
                    f"  {name}"
                    for name in sorted(available)
                )
            )

        if len(prefix_matches) > 1:
            raise ValueError(
                f"Ambiguous experiment prefix {query!r}.\n"
                "Be more specific. Matches:\n"
                + "\n".join(
                    f"  {name}"
                    for name in sorted(prefix_matches)
                )
            )

        resolved.append(prefix_matches[0])

    return resolved


def plot_histories(
    groups,
    names,
    output_path=None,
    error_every=5,
):
    names = resolve_plot_names(
        groups,
        names,
    )

    fig, ax = plt.subplots(
        # figsize=(10, 6)
        figsize=(4, 4)
    )

    # Assign one default matplotlib color per experiment,
    colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    linestyles = ['-', '--', '-.'] 

    for idx, name in enumerate(names):
        runs = groups[name]

        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]

        train_epoch, train_mean, train_std = (
            aggregate_history(
                runs,
                "train_metric",
            )
        )

        val_epoch, val_mean, val_std = (
            aggregate_history(
                runs,
                "val_metric",
            )
        )

        ax.plot(
            val_epoch,
            val_mean,
            linewidth=1.5,
            color=color,
            linestyle=linestyle,
            label=f"{name} — val",
        )

        # Fill between standard deviations
        ax.fill_between(
            val_epoch, 
            val_mean - val_std, 
            val_mean + val_std, 
            linewidth=1.5,
            color=color, 
            linestyle='-',
            alpha=0.3, 
        )


    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.grid(
        alpha=0.25,
    )
    ax.legend(
        fontsize=8,
    )

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved plot to {output_path}"
        )

    else:
        plt.show()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs"),
        help="Root experiment directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results_summary.txt"
        ),
        help=(
            "Output text file in normal "
            "summary mode."
        ),
    )

    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help=(
            "Only include names containing "
            "all specified strings."
        ),
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help=(
            "Exclude names containing any "
            "specified string."
        ),
    )

    parser.add_argument(
        "--plot",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Plot training histories for "
            "the specified experiment names "
            "or unique name substrings."
        ),
    )

    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help=(
            "Save plot instead of showing it."
        ),
    )

    parser.add_argument(
        "--error-every",
        type=int,
        default=5,
        help=(
            "Show error bars every N epochs."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    groups = collect_runs(
        args.root,
        include=args.include,
        exclude=args.exclude,
    )

    if not groups:
        raise RuntimeError(
            f"No valid experiments found "
            f"under {args.root}"
        )

    if args.plot is None:
        summarize(
            groups,
            args.output,
        )

    else:
        plot_histories(
            groups,
            args.plot,
            output_path=args.plot_output,
            error_every=args.error_every,
        )


if __name__ == "__main__":
    main()