from pathlib import Path

def log(path: Path, prefix: str, placeholder: str, *args):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as txt:
        txt.write(prefix + "\n")
        for line in zip(*args):
            txt.write(placeholder.format(*line) + "\n")