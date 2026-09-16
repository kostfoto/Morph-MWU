import re
from pathlib import Path


RESULT_RE = re.compile(
    r"^(?P<name>.+)\n"
    r"\s+Test metric:\s+"
    r"(?P<mean>[-+0-9.eE]+)\s+±\s+"
    r"(?P<std>[-+0-9.eE]+)",
    re.MULTILINE,
)


def parse_name(name):
    match = re.fullmatch(
        r"(?P<model>.+?)_"
        r"(?P<dataset>.+?)_"
        r"(?P<method>MWU|NO-MWU)_"
        r"T(?P<temperature>[-+0-9.]+)",
        name,
    )

    if match is None:
        raise ValueError(f"Could not parse: {name}")

    return match.groupdict()


def pretty_dataset(dataset):
    if dataset == "mnist":
        return "MNIST"
    if dataset == "fashion-mnist":
        return "Fashion-MNIST"
    if dataset == "cifar-10":
        return "CIFAR-10"

    if dataset.startswith("pmlb-"):
        dataset = dataset[len("pmlb-"):]

    return dataset.replace("-", " ").title()


def format_result(mean, std):
    return f"${100*mean:.2f} \\pm {100*std:.2f}$"


def main(path="results_summary.txt"):
    text = Path(path).read_text()

    results = {}

    for match in RESULT_RE.finditer(text):
        name = match.group("name")
        mean = float(match.group("mean"))
        std = float(match.group("std"))

        info = parse_name(name)

        model = info["model"]
        dataset = info["dataset"]
        method = info["method"]

        results.setdefault(dataset, {})

        # Conventional baseline
        if model in {"mlp", "ViT"}:
            results[dataset]["baseline"] = (mean, std)

        # Morphological + ordinary BP
        elif model in {"min-max-plus", "MorphViT"} and method == "NO-MWU":
            results[dataset]["morph_bp"] = (mean, std)

        # Morphological + proposed method
        elif model in {"min-max-plus", "MorphViT"} and method == "MWU":
            results[dataset]["morph_corr"] = (mean, std)

    for dataset in sorted(results):
        row = results[dataset]

        baseline = (
            format_result(*row["baseline"])
            if "baseline" in row else "--"
        )

        morph_bp = (
            format_result(*row["morph_bp"])
            if "morph_bp" in row else "--"
        )

        morph_corr = (
            format_result(*row["morph_corr"])
            if "morph_corr" in row else "--"
        )

        if "morph_bp" in row and "morph_corr" in row:
            gain_pp = 100 * (
                row["morph_corr"][0]
                - row["morph_bp"][0]
            )
            gain = f"{gain_pp:+.2f}"
        else:
            gain = "--"

        print(
            f"{pretty_dataset(dataset)} & "
            f"{baseline} & "
            f"{morph_bp} & "
            f"{morph_corr} & "
            f"{gain} \\\\"
        )


if __name__ == "__main__":
    main()