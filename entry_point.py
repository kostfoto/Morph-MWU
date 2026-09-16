import argparse
import torch
from pathlib import Path
import yaml
import hashlib
import json

from reproducibility import set_seed, generate_seed
from dataloading import make_dataloaders
from models import make_model, param_count
from training import train
from evaluation import evaluate
from losses import make_loss
from metrics import make_metric
from optimizer import make_optimizer
import mylogging

def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--hidden_dims", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use_cheap", action="store_true")

    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=128)

    # Dataset
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset_root", type=str, default="./data")

    parser.add_argument("--pmlb_name", type=str, default="adult")
    parser.add_argument("--test_split", type=float, default=0.2)

    # Optimizer
    parser.add_argument("--optim", type=str, default="adam")

    # Loss
    parser.add_argument("--loss", type=str, default="cross_entropy")

    # Metrics
    parser.add_argument("--train_metric", type=str, default="accuracy")
    parser.add_argument("--test_metric", type=str, default="accuracy")

    # Proposed MWU updates
    parser.add_argument("--use_mwu", action="store_true")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Outputing
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--runs_root", type=str, default='runs')
    parser.add_argument("--results_root", type=str, default=None)

    return parser.parse_args()

def make_run_name(config):
    model = config["model"]["name"]
    dataset = config["dataset"]["name"]
    seed = config["seed"]
    temperature = config["model"]["temperature"]
    mwu = "MWU" if config["use_mwu"] else "NO-MWU"

    if dataset == "pmlb":
        dataset = f"pmlb-{config['dataset']['pmlb_name']}"

    prefix = (
        f"{model}_{dataset}_{mwu}_"
        f"T{temperature}_seed{seed}"
    )

    return f"{prefix}__{config_hash(config)}"

def build_config(args):
    seed = args.seed if args.seed is not None else generate_seed()

    hash_config = {
        "model": {
            "name": args.model, 
            "hidden_dims": args.hidden_dims, 
            "num_layers": args.num_layers, 
            "temperature": args.temperature, 
            "cheap": args.use_cheap, 
            "patch_size": args.patch_size, 
            "embed_dim": args.embed_dim, 
        },  
        "dataset": {
            "name": args.dataset, 
            "val_split": args.val_split, 
            "batch_size": args.batch_size, 
            "num_workers": args.num_workers, 
            "root": args.dataset_root, 

            "pmlb_name": args.pmlb_name, 
            "test_split": args.test_split, 
        }, 
        "use_mwu": args.use_mwu, 
        "optim": {
            "name": args.optim, 
            "lr": args.lr, 
            "weight_decay": args.weight_decay, 
        }, 
        "loss": args.loss,
        "metrics": {
            "train": args.train_metric, 
            "test": args.test_metric, 
        }, 
        "epochs": args.epochs, 
        "seed": seed, 
        "device": args.device, 
    }

    run_name = args.run_name
    if args.run_name is None:
        run_name = make_run_name(hash_config)
    results_root = Path(args.runs_root)/run_name/"results" if args.results_root is None else args.results_root
    
    hash_config["output"] = {
            "run_name": run_name, 
            "runs_root": Path(args.runs_root), 
            "results_root": Path(results_root),
    }
    return hash_config

def make_serializable(obj):
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {
            key: make_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [make_serializable(value) for value in obj]

    return obj

def save_config(config, path):
    with open(path, "w") as f:
        yaml.safe_dump(
            make_serializable(config),
            f,
            sort_keys=False,
        )

def config_hash(config):
    serializable = make_serializable(config)

    canonical = json.dumps(
        serializable,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()[:8]

def run_experiment(config):

    print(config)

    run_dir = (
        config["output"]["runs_root"]
        / config["output"]["run_name"]
    )

    results_dir = config["output"]["results_root"]
    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_config(
        config,
        run_dir / "config.yaml",
    )

    set_seed(config['seed'])

    train_loader, val_loader, test_loader, dataset_metadata = make_dataloaders(config)

    print(dataset_metadata)

    model = make_model(config, dataset_metadata).to(config['device'])

    total_params = param_count(model)
    print(f"Parameter count of model: {total_params}\n")

    optimizer = make_optimizer(model.parameters(), config['optim'])

    criterion = make_loss(config['loss'])

    train_metric = make_metric(config['metrics']['train'])
    test_metric = make_metric(config['metrics']['test'])

    history = train(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        metric=train_metric, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        config=config
    )

    result = evaluate(model, test_loader, test_metric, config)

    print(f"Test_metric: {result:.4f}\n")

    mylogging.log(
        config["output"]["results_root"] / "training_history.txt",
        f"Total number of parameters: {total_params}", 
        "Epoch: {}, Train_loss: {:.4f}, Train metric: {:.4f}, Val metric: {:.4f}",
        range(1,len(history["train_loss"])+1),
        history["train_loss"],
        history["train_metric"],
        history["val_metric"], 
    )

    with open(
        config["output"]["results_root"] / "test_metric.txt",
        "w",
    ) as f:
        f.write(f"Test_metric: {result:.4f}\n")

def main():
    args = parse_args()
    config = build_config(args)

    run_experiment(config)



if __name__ == '__main__':
    main()